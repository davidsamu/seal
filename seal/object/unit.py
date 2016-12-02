# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 14:06:14 2016

Class representing a (spike sorted) unit (single or multi).

@author: David Samu
"""

from datetime import datetime as dt
from itertools import product
from collections import OrderedDict as OrdDict

import numpy as np
import pandas as pd
from quantities import s, ms, us, deg, Hz
from neo import SpikeTrain

from seal.util import plot, util
from seal.object import constants
from seal.object.rate import Rate
from seal.object.spikes import Spikes
from seal.object.trials import Trials
from seal.analysis import tuning

# TODO: add receptive field coverage information!
# TODO: put stuff into Series/DFs, especially composite return values!


class Unit:
    """Generic class to store data of a unit (neuron or group of neurons)."""

    # %% Constructor
    def __init__(self, t_start=None, t_stop=None, kernels=None, step=10*ms,
                 TPLCell=None, tr_params=None):
        """Create Unit instance, optionally from TPLCell data structure."""

        # Create empty instance.
        self.Name = ''
        self.SessParams = pd.Series()
        self.Waveforms = pd.Series()
        self.TrialParams = pd.DataFrame()
        self.Events = pd.DataFrame()
        self.Spikes = Spikes([])
        self.Rates = pd.Series()
        self.QualityMetrics = pd.Series()
        self.Selectivity = OrdDict()

        self.t_start = t_start
        self.t_stop = t_stop

        # Return if no TPLCell is passed.
        if TPLCell is None:
            return

        # %% Session parameters.

        # Prepare session params.
        monkey, date, probe, exp, sortno = util.params_from_fname(TPLCell.File)
        task, task_idx = exp[:-1], int(exp[-1])
        [chan, un] = TPLCell.ChanUnit
        sampl_per = (1 / (TPLCell.Info.Frequency * Hz)).rescale(us)
        pinfo = [p.tolist() if isinstance(p, np.ndarray)
                 else p for p in TPLCell.PInfo]
        sess_date = dt.date(dt.strptime(date, '%m%d%y'))

        # Name unit.
        self.Name = (' '.join([task, monkey, date, probe]) +
                     ' Ch{:02}/{} ({})'.format(chan, un, sortno))

        # Assign session params.
        sp_list = [('task', task),
                   ('task_idx', task_idx),
                   ('monkey', monkey),
                   ('date', sess_date),
                   ('probe', probe),
                   ('channel #', chan),
                   ('unit #', un),
                   ('sort #', sortno),
                   ('filepath', TPLCell.Filename),
                   ('filename', TPLCell.File),
                   ('paraminfo', pinfo),
                   ('SamplPer', sampl_per)]
        self.SessParams = util.series_from_tuple_list(sp_list)

        # %% Waveform data.

        # Prepare waveform data.
        wfs = TPLCell.Waves
        if wfs.ndim == 1:  # there is only a single spike: extend it to matrix
            wfs = np.reshape(wfs, (1, len(wfs)))
        wf_time = range(wfs.shape[1]) * self.SessParams['SamplPer']

        t_min = np.min(TPLCell.Spikes)
        t_max = np.max(TPLCell.Spikes)
        sp_times = SpikeTrain(TPLCell.Spikes*s, t_start=t_min, t_stop=t_max)

        # Assign waveform data.
        wf_list = [('WaveformTime', wf_time),
                   ('SpikeWaveforms', wfs),
                   ('SpikeDuration', util.fill_dim(TPLCell.Spikes_dur * s)),
                   ('MeanSpikeDur', TPLCell.MeanSpikeDur * s),
                   ('SpikeTimes', util.fill_dim(sp_times)),
                   ('DirSelectivity', pd.DataFrame())]
        self.Waveforms = util.series_from_tuple_list(wf_list)

        # %% Trial parameters.

        trp_df = pd.DataFrame(TPLCell.TrialParams, columns=TPLCell.Header)

        if tr_params is not None:
            for name, (nnew, dim) in tr_params.items():
                if name not in trp_df.columns:
                    print('Warning: Parameter {0} not found!'.format(name))
                    continue
                # Rename column.
                trp_df.rename(columns={name: nnew}, inplace=True)
                if dim is not None:  # add dimension
                    trp_df[nnew] = util.add_dim_to_series(trp_df[nnew], dim)

        self.TrialParams = trp_df

        # Add column for subject response (saccade direction).
        if 'AnswCorr' in self.TrialParams.columns:
            self.TrialParams['AnswCorr'] = self.TrialParams['AnswCorr'] == 1
            same_dir = self.TrialParams['S1Dir'] == self.TrialParams['S2Dir']
            corr_ans = self.TrialParams['AnswCorr']
            self.TrialParams['Answ'] = ((same_dir & corr_ans) |
                                        (~same_dir & ~corr_ans))

        # Start and end times of each trials.
        tstamps = TPLCell.Timestamps
        tr_times = np.array([(tstamps[i1-1], tstamps[i2-1]) for i1, i2
                                in TPLCell.Info.successfull_trials_indices]) * s
        self.TrialParams['TrialStart'] = tr_times[:, 0]
        self.TrialParams['TrialStop'] = tr_times[:, 1]
        self.TrialParams['TrialLength'] = tr_times[:, 1] - tr_times[:, 0]

        # %% Trial events.

        # Timestamps of events. Only S1 offset and S2 onset are reliable!
        # Watch out: indexing starting with 0! (not with 1 as in Matlab)
        S1_len, S2_len = constants.stim_prds.dur()
        iS1off = TPLCell.Patterns.matchedPatterns[:, 2]-1
        iS2on = TPLCell.Patterns.matchedPatterns[:, 3]-1
        event_cols = ['S1 onset', 'S1 offset', 'S2 onset', 'S2 offset']
        self.Events = pd.DataFrame([TPLCell.Timestamps[iS1off]*s-S1_len,
                                    TPLCell.Timestamps[iS1off]*s,
                                    TPLCell.Timestamps[iS2on]*s,
                                    TPLCell.Timestamps[iS2on]*s+S2_len],
                                   index=event_cols).T
        # Align trial events to S1 onset.
        S1on = self.Events['S1 onset']
        self.Events = self.Events.subtract(S1on, axis=0)

        # %% Spikes and rates.

        # Trials spikes, aligned to S1 onset.
        spk_trains = [TS*s-S1on[i] for i, TS in enumerate(TPLCell.TrialSpikes)]
        self.Spikes = Spikes(spk_trains, t_start, t_stop)

        # Estimate firing rate per trial.
        spikes = self.Spikes.get_spikes()
        rate_list = [Rate(name, kernel, spikes, step)
                     for name, kernel in kernels.items()]
        self.Rates = pd.Series(rate_list, index=kernels.keys())

    # %% Utility methods.

    def is_empty(self):
        """Checks if unit is empty."""

        im_empty = self.Name == ''
        return im_empty

    def name_to_fname(self):
        """Return filename compatible name string."""

        fname = util.format_to_fname(self.Name)
        return fname

    def get_recording_name(self):
        """Return name of recording ([monkey_date])."""

        date_str = util.date_to_str(self.SessParams['date'])
        rec_str = self.SessParams['monkey'] + '_' + date_str
        return rec_str

    def get_rec_ch_un_task_index(self):
        """Return (recording, channel #, unit #, task) index quadruple."""

        rec = self.get_recording_name()
        task, ich, iunit = [self.SessParams[p]
                            for p in ('task', 'channel #', 'unit #')]
        uidx = (rec, ich, iunit, task)

        return uidx

    def get_unit_params(self, rem_dims=True):
        """Return main unit parameters."""

        unit_params = pd.Series()

        # Recording parameters.
        unit_params['Session information'] = ''
        unit_params.append(util.get_scalar_vals(self.SessParams, rem_dims))

        # Quality metrics.
        unit_params['Quality metrics'] = ''
        unit_params.append(util.get_scalar_vals(self.QualityMetrics, rem_dims))

        # Stimulus selectivity.
        unit_params['Selectivity'] = ''
        unit_params.append(util.get_scalar_vals(self.Selectivity, rem_dims))

        return unit_params

    def get_trial_params(self, trs=None, t1=None, t2=None):
        """Return default values of some common parameters."""

        if trs is None:
            trs = [self.all_trials()]
        if t1 is None:
            t1 = self.t_start
        if t2 is None:
            t2 = self.t_stop

        return trs, t1, t2

    def pref_dir(self, stim='S1'):
        """Return preferred direction."""

        pdir = self.Selectivity['PrefDirCoarse'][stim]
        return pdir

    def anti_pref_dir(self, stim='S1'):
        """Return anti-preferred direction."""

        adir = self.Selectivity['AntiPrefDirCoarse'][stim]
        return adir

    # %% Generic methods to get various set of trials.

    def included_trials(self):
        """Return included trials (i.e. not rejected after quality test)."""

        if 'IncludedTrials' in self.QualityMetrics:
            included_trials = self.QualityMetrics['IncludedTrials']
        else:
            included_trials = self.all_trials(filtered=False)
        return included_trials

    def filter_trials(self, trs):
        """Filter trials by excluding rejected ones."""

        tr_idxs = np.logical_and(trs.trials, self.included_trials().trials)
        filtered_trials = Trials(tr_idxs, trs.value, trs.name)
        return filtered_trials

    # TODO: distinguish between list and object of trials!
    def ftrials(self, trs, value=None, name=None, filtered=True):
        """
        Create and return trial object from list of trial indices
        after excluding unit's rejected trials.
        """

        trs = Trials(trs, value, name)
        if filtered:
            trs = self.filter_trials(trs)

        return trs

    def all_trials(self, filtered=True):
        """Return indices of all trials."""

        tr_idxs = np.ones(self.Spikes.n_trials(), dtype=bool)
        trs = self.ftrials(tr_idxs, 'all trials', None, filtered)

        return trs

    def param_values_in_trials(self, trs, pnames=None):
        """Return list of parameter values during given trials."""

        if pnames is None:
            pnames = self.TrialParams.columns.values
        pvals = self.TrialParams[pnames][trs.trials]

        return pvals

    def trials_by_param_values(self, pname, pvals=None, comb_values=False):
        """Return trials grouped by (requested) values of parameter."""

        # All unique values of parameter and their trial indices.
        vals, idxs = np.unique(self.TrialParams[pname], return_inverse=True)

        # Set default values: all values of parameter.
        if pvals is None:
            pvals = vals

        # Assamble list of trial names
        tr_names = ['{} = {}'.format(pname, v) for v in pvals]

        # Helper function to return indices of parameter value.
        def trials_idxs(v):
            ival = np.where(vals == v)[0] if v in vals else -1
            return idxs == ival

        # Create Trials object for each parameter value, separately.
        ptrials = [self.ftrials(trials_idxs(v), v, n)
                   for i, (n, v) in enumerate(zip(tr_names, pvals))]

        # Optionally, combine trials across parameter values.
        if comb_values:
            ptrials = [Trials.combine_trials(ptrials, 'or')]

        return ptrials

    def trials_by_comb_params(self, pdict, comb_params='all',
                              comb_values=False):
        """Return trials grouped by combination of values of parameters."""

        # First, collect requested trials from each parameter separately.
        ptrials = [self.trials_by_param_values(pname, pvals, comb_values)
                   for pname, pvals in pdict.items()]

        # Set up trial combinations across parameter names.
        if len(pdict) > 1:

            if comb_params == 'one2one':
                tr_combs = zip(*ptrials)  # one-to-one mapping
            else:
                tr_combs = product(*ptrials)  # all combination of values

            # Combine trails across parameters.
            ptrials = [Trials.combine_trials(trc, 'and') for trc in tr_combs]

        else:
            ptrials = ptrials[0]  # single parameter value

        return ptrials

    def correct_incorrect_trials(self):
        """Return indices of correct and incorrect trials."""

        ctrs = OrdDict()
        ctrs['correct'] = self.trials_by_param_values('AnswCorr', [True])[0]
        ctrs['error'] = self.trials_by_param_values('AnswCorr', [False])[0]

        return ctrs

    # %% Methods that provide interface to Unit's Spikes data.

    def get_rates_by_trial(self, trs=None, t1=None, t2=None):
        """Return spike statistics of time interval in given trials."""

        if self.is_empty():
            return None

        # Init trials.
        if trs is None:
            trs = self.included_trials()

        # Get rates.
        frate = self.Spikes.spike_stats_in_prd(trs, t1, t2)[1]

        # Put them into a Series with trials start times.
        tr_time = self.TrialParams['TrialStart'][trs.trials]
        frate_tr_time = pd.Series(frate, index=tr_time, name='FR (1/s)')

        return frate_tr_time

    # %% Methods to trials with specific directions.

    def dir_trials(self, direction, pname=['S1Dir', 'S2Dir'], offsets=[0*deg],
                   comb_params='all', comb_values=False):
        """Return trials with some direction +- offset during S1 and/or S2."""

        if not util.is_iterable(pname):
            pname = [pname]

        # Get trials for direction + each offset value.
        degs = [util.deg_mod(direction + offset) for offset in offsets]
        deg_dict = OrdDict([(pn, degs) for pn in pname])
        trs = self.trials_by_comb_params(deg_dict, comb_params, comb_values)

        return trs

    def dir_pref_trials(self, stim='S1', pname=['S1Dir', 'S2Dir'],
                        offsets=[0*deg], comb_params='all', comb_values=False):
        """Return trials with preferred direction."""

        # Get trials with preferred direction.
        pdir = self.pref_dir(stim)
        trs = self.dir_trials(pdir, pname, offsets, comb_params, comb_values)

        # Rename trials.
        if len(trs) == 1:
            trs[0].name = ' & '.join([pn[0:2] for pn in pname]) + ' pref'

        return trs

    def dir_anti_trials(self, stim='S1', pname=['S1Dir', 'S2Dir'],
                        offsets=[0*deg], comb_params='all', comb_values=False):
        """Return trials with anti-preferred direction."""

        # Get trials with anti-preferred direction.
        adir = self.anti_pref_dir(stim)
        trs = self.dir_trials(adir, pname, offsets, comb_params, comb_values)

        # Rename trials.
        if len(trs) == 1:
            trs[0].name = ' & '.join([pn[0:2] for pn in pname]) + ' anti'

        return trs

    def dir_pref_anti_trials(self, stim='S1', pname=['S1Dir', 'S2Dir'],
                             offsets=[0*deg], comb_params='all',
                             comb_values=False):
        """Return trials with preferred and antipreferred direction."""

        pref_trials = self.dir_pref_trials(stim, pname, offsets,
                                           comb_params, comb_values)
        apref_trials = self.dir_anti_trials(stim, pname, offsets,
                                            comb_params, comb_values)
        pref_apref_trials = pref_trials + apref_trials

        return pref_apref_trials

    def S_D_trials(self, stim='S1', offsets=[0*deg], combine=True):
        """
        Return trials for S1 = S2 (same) and S1 =/= S2 (different)
        with S2 being at given offset from the unit's preferred direction.
        """

        # Collect S- and D-trials for all offsets.
        trS = []
        trD = []
        for offset in offsets:

            # Trials to given offset to preferred direction.
            pS2 = self.dir_pref_trials(stim, 'S2Dir', [offset])[0]
            pS1 = self.dir_pref_trials(stim, 'S1Dir', [offset])[0]

            # S- and D-trials.
            trS.append(Trials.combine_trials([pS2, pS1], 'and', 'S trials'))
            trD.append(Trials.combine_trials([pS2, pS1], 'diff', 'D trials'))

        # Combine S- and D-trials across offsets.
        if combine:
            trS = Trials.combine_trials(trS, 'or', 'S trials')
            trD = Trials.combine_trials(trD, 'or', 'D trials')

        trs = [trS, trD]

        return trs

    # %% Methods to calculate tuning curves and preferred values of features.

    def calc_response_stats(self, pname, t1, t2):
        """Calculate mean response to different values of trial parameter."""

        # Get trials for each parameter value.
        trs = self.trials_by_param_values(pname)

        # Calculate spike count and stats for each value of parameter.
        p_values = [float(tr.value) for tr in trs]
        sp_stats = pd.DataFrame([self.Spikes.spike_count_stats(tr, t1, t2)
                                 for tr in trs], index=p_values)

        return sp_stats

    def calc_dir_response(self, stim, t1=None, t2=None):
        """Calculate mean response to each direction during given stimulus."""

        # Init stimulus.
        pname = stim + 'Dir'

        # Init time period.
        t1_stim, t2_stim = constants.del_stim_prds.periods(stim)
        if t1 is None:
            t1 = t1_stim
        if t2 is None:
            t2 = t2_stim

        # Calculate response statistics.
        response_stats = self.calc_response_stats(pname, t1, t2)

        return response_stats

    def calc_DS(self, stim, t1=None, t2=None):
        """Calculate direction selectivity (DS)."""

        # Get response stats to each direction.
        resp_stats = self.calc_dir_response(stim, t1, t2)
        dirs = np.array(resp_stats.index) * deg
        meanFR, stdFR, semFR = [util.dim_series_to_array(resp_stats[stat])
                                for stat in ('mean', 'std', 'sem')]

        # Calculate preferred direction PD and direction selectivity index DSI.
        DSI, PD, PDc = util.deg_w_mean(dirs, meanFR)

        # Calculate antipreferred direction.
        ADc = util.deg_mod(PDc+180*deg)

        # Calculate legacy direction selectivity index,
        # based on just two directions: FRpref - FRanti / (FRpref + FRanti)
        pFR = meanFR[np.where(dirs == PDc)[0]]
        aFR = meanFR[np.where(dirs == ADc)[0]]
        DS2 = float(util.modulation_index(pFR, aFR)) if aFR.size else np.nan

        # Calculate parameters of Gaussian tuning curve.
        # Center stimulus - response firts.
        tun_res = tuning.center_pref_dir(dirs, PD, meanFR, semFR)
        dirs_cntr, meanFR_cntr, semFR_cntr = tun_res

        # Fit Gaussian tuning curve to stimulus - response.
        fit_params, fit_res = tuning.fit_gaus_curve(dirs_cntr, meanFR_cntr,
                                                    semFR_cntr)

        # Prepare results.
        res = {'dirs': dirs, 'meanFR': meanFR, 'stdFR': stdFR, 'semFR': semFR,
               'DSI': DSI, 'DS2': DS2, 'PD': PD, 'PDc': PDc, 'ADc': ADc,
               'dirs_cntr': dirs_cntr, 'meanFR_cntr': meanFR_cntr,
               'semFR_cntr': semFR_cntr, 'fit_params': fit_params,
               'fit_res': fit_res}

        return res

    def test_DS(self, stims=['S1', 'S2'], no_labels=False, do_plot=True,
                ffig_tmpl=None, **kwargs):
        """
        Test DS of unit by calculating
          - DS index and PD, and
          - parameters of Gaussian tuning curve.
        """

        # Init field to store DS results.
        DSres_plot = {}
        df_temp = pd.DataFrame(index=stims)
        DSres = OrdDict([('DSIs', df_temp.copy()), ('PDs', df_temp.copy()),
                         ('TPs', df_temp.copy())])

        for stim in stims:

            res = self.calc_DS(stim, t1=None, t2=None)

            # Generate data points for plotting fitted tuning curve.
            a, b, x0, sigma = res['fit_params'].loc['fit']
            x, y = tuning.gen_fit_curve(tuning.gaus, deg, -180*deg, 180*deg,
                                        a=a, b=b, x0=x0, sigma=sigma)

            # Add calculated values to unit.

            # DSIs
            DSres['DSIs'].loc[stim, 'DSm'] = res['DS2']
            DSres['DSIs'].loc[stim, 'DSw'] = res['DSI']

            # PDs
            # By maximum rate.
            DSres['PDs'].loc[stim, 'PDm'] = res['PD']
            DSres['PDs'].loc[stim, 'cPDm'] = res['PDc']
            DSres['PDs'].loc[stim, 'cADm'] = res['ADc']
            # By weighted vector sum.
            DSres['PDs'].loc[stim, 'PDw'] = res['PD']
            DSres['PDs'].loc[stim, 'cPDw'] = res['PDc']
            DSres['PDs'].loc[stim, 'cADw'] = res['ADc']
            # By Gaussian tuning curve fit.
            # TODO: add preferred direction found by tuning curve fit???
            # Would it worth it?

            # TPs
            for fres, val in res['fit_res'].items():
                DSres['TPs'].loc[stim, fres] = val

            # Collect data for plotting.
            DSres_plot[stim] = res
            DSres_plot[stim]['xfit'] = x
            DSres_plot[stim]['yfit'] = y

        DSres_plot = pd.DataFrame(DSres_plot).T
        self.Selectivity['DS'] = DSres

        # Plot direction selectivity results.
        if do_plot:

            # Minimise labels on plot.
            if no_labels:
                title = None
                kwargs['labels'] = False
                kwargs['polar_legend'] = True
                kwargs['tuning_legend'] = False

            ffig = None
            if ffig_tmpl is not None:
                ffig = ffig_tmpl.format(self.name_to_fname())
            title = self.Name
            plot.direction_selectivity(DSres_plot, title=title,
                                       ffig=ffig, **kwargs)

    # %% Plotting methods.

    def prep_plot_params(self, trs, t1, t2, nrate=None):
        """Prepare plotting parameters."""

        # Get trial params.
        trs, t1, t2 = self.get_trial_params(trs, t1, t2)
        names = [tr.name for tr in trs]

        # Get spikes.
        spikes = [self.Spikes.get_spikes(tr, t1, t2) for tr in trs]

        # Get rates and rate times.
        rates, times = None, None
        if nrate is not None:
            rates = [self.Rates[nrate].get_rates(tr.trials, t1, t2)
                     for tr in trs]
            times = self.Rates[nrate].get_times(t1, t2)

        return trs, t1, t2, spikes, rates, times, names

    def plot_raster(self, trs=None, t1=None, t2=None, **kwargs):
        """Plot raster plot of unit for specific trials."""

        if self.is_empty():
            return

        # Set up params.
        plot_params = self.prep_plot_params(trs, t1, t2, nrate=None)
        trs, t1, t2, spikes, rates, times, names = plot_params
        spikes = spikes[0]
        names = names[0]

        # Plot raster.
        ax = plot.raster(spikes, t1, t2,
                         segments=constants.stim_prds,
                         title=self.Name, **kwargs)
        return ax

    def plot_rate(self, nrate, trs=None, t1=None, t2=None, **kwargs):
        """Plot rate plot of unit for specific trials."""

        if self.is_empty():
            return

        # Set up params.
        plot_params = self.prep_plot_params(trs, t1, t2, nrate)
        trs, t1, t2, spikes, rates, times, names = plot_params

        # Plot rate.
        ax = plot.rate(rates, times, t1, t2, names,
                       segments=constants.stim_prds,
                       title=self.Name, **kwargs)
        return ax

    def plot_raster_rate(self, nrate, trs=None, t1=None, t2=None,
                         no_labels=False, **kwargs):
        """Plot raster and rate plot of unit for specific trials."""

        if self.is_empty():
            return

        # Set up params.
        plot_params = self.prep_plot_params(trs, t1, t2, nrate)
        trs, t1, t2, spikes, rates, times, names = plot_params

        title = self.Name

        # Minimise labels on plot.
        if no_labels:
            title = None
            kwargs['xlab'] = None
            kwargs['ylab_rate'] = None
            kwargs['add_ylab_raster'] = False

        # Plot raster and rate.
        res = plot.raster_rate(spikes, rates, times, t1, t2, names,
                               segments=constants.stim_prds,
                               title=title, **kwargs)
        fig, raster_axs, rate_ax = res

        return fig, raster_axs, rate_ax
