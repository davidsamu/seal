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
from pandas import DataFrame, Series
from quantities import Quantity, s, ms, us, deg, Hz
from neo import SpikeTrain

from seal.util import plot, util
from seal.object import constants
from seal.object.rate import Rate
from seal.object.spikes import Spikes
from seal.object.trials import Trials

# TODO: add receptive field coverage information!

class Unit:
    """Generic class to store data of a unit (neuron or group of neurons)."""

    # %% Constructor
    def __init__(self, t_start=None, t_stop=None, kernels=None, step=10*ms,
                 TPLCell=None, params_info=None):
        """Create Unit instance, optionally from TPLCell data structure."""

        # Create empty instance.
        self.Name = ''
        self.SessParams = OrdDict()
        self.UnitParams = OrdDict()
        self.QualityMetrics = OrdDict()
        self.TrialParams = DataFrame()
        self.Events = DataFrame()
        self.Spikes = []
        self.Rates = OrdDict()
        self.t_start = t_start
        self.t_stop = t_stop

        # Return if no TPLCell is passed.
        if TPLCell is None:
            return

        # Extract session parameters.
        monkey, date, probe, exp, sortno = util.params_from_fname(TPLCell.File)

        [chan, un] = TPLCell.ChanUnit
        self.Name = ' '.join([exp, monkey, date, probe])
        self.Name += ' Ch{:02}/{} ({})'.format(chan, un, sortno)

        self.SessParams['experiment'] = exp
        self.SessParams['monkey'] = monkey
        self.SessParams['date'] = dt.date(dt.strptime(date, '%m%d%y'))
        self.SessParams['probe'] = probe
        self.SessParams['channel #'] = chan
        self.SessParams['unit #'] = un
        self.SessParams['sort #'] = sortno
        self.SessParams['filepath'] = TPLCell.Filename
        self.SessParams['filename'] = TPLCell.File
        self.SessParams['paraminfo'] = [p.tolist() if isinstance(p, np.ndarray)
                                        else p for p in TPLCell.PInfo]
        sampl_per = (1 / (TPLCell.Info.Frequency * Hz)).rescale(us)
        self.SessParams['SamplPer'] = sampl_per

        # Unit parameters/stats.
        wfs = TPLCell.Waves
        if wfs.ndim == 1:  # there is only a single spike: extend it to matrix
            wfs = np.reshape(wfs, (1, len(wfs)))
        wf_time = range(wfs.shape[1]) * self.SessParams['SamplPer']
        self.UnitParams['WaveformTime'] = wf_time
        self.UnitParams['SpikeWaveforms'] = wfs
        self.UnitParams['SpikeDuration'] = util.fill_dim(TPLCell.Spikes_dur * s)
        self.UnitParams['MeanSpikeDur'] = TPLCell.MeanSpikeDur * s

        t_min = np.min(TPLCell.Spikes)
        t_max = np.max(TPLCell.Spikes)
        sp_train = SpikeTrain(TPLCell.Spikes*s, t_start=t_min, t_stop=t_max)
        self.UnitParams['SpikeTimes'] = util.fill_dim(sp_train)

        self.UnitParams['PrefDir'] = OrdDict()
        self.UnitParams['PrefDirCoarse'] = OrdDict()
        self.UnitParams['DirSelectivity'] = OrdDict()

        # Trial parameters.
        self.TrialParams = DataFrame(TPLCell.TrialParams,
                                     columns=TPLCell.Header)
        if params_info is not None:  # This takes a lot of time, speed it up?
            for name, (new_name, dim) in params_info.items():
                if name not in self.TrialParams.columns:
                    print('Warning: Parameter {0} not found!'.format(name))
                    continue
                self.TrialParams.rename(columns={name: new_name}, inplace=True)
                if dim is not None:  # add physical dimension
                    c = util.add_dim_to_df_col(self.TrialParams[new_name], dim)
                    self.TrialParams[new_name] = c

        # Add column for subject response (saccade direction).
        if 'AnswCorr' in self.TrialParams.columns:
            self.TrialParams['AnswCorr'] = self.TrialParams['AnswCorr'] == 1
            same_dir = self.TrialParams['S1Dir'] == self.TrialParams['S2Dir']
            corr_ans = self.TrialParams['AnswCorr']
            self.TrialParams['Answ'] = ((same_dir & corr_ans) |
                                        (~same_dir & ~corr_ans))

        # Start and end times of each trials.
        trial_times = [(TPLCell.Timestamps[i1-1]*s, TPLCell.Timestamps[i2-1]*s)
                       for i1, i2 in TPLCell.Info.successfull_trials_indices]
        self.TrialParams['TrialStart'] = [tr_t[0] for tr_t in trial_times]
        self.TrialParams['TrialStop'] = [tr_t[1] for tr_t in trial_times]
        self.TrialParams['TrialLength'] = [tr_t[1] - tr_t[0]
                                           for tr_t in trial_times]

        # Timestamps of events. Only S1 offset and S2 onset are reliable!
        # Take care of indexing starting with 0! (not with 1 as in Matlab)
        S1_len, S2_len = constants.stim_prds.dur()
        iS1off = TPLCell.Patterns.matchedPatterns[:, 2]-1
        iS2on = TPLCell.Patterns.matchedPatterns[:, 3]-1
        self.Events = DataFrame([TPLCell.Timestamps[iS1off]*s-S1_len,
                                 TPLCell.Timestamps[iS1off]*s,
                                 TPLCell.Timestamps[iS2on]*s,
                                 TPLCell.Timestamps[iS2on]*s+S2_len]).T
        self.Events.columns = ['S1 onset', 'S1 offset',
                               'S2 onset', 'S2 offset']
        # Align trial events to S1 onset.
        S1on = self.Events['S1 onset']
        self.Events = self.Events.subtract(S1on, axis=0)

        # Trials spikes, aligned to S1 onset.
        spk_trains = [TS*s-S1on[i] for i, TS in enumerate(TPLCell.TrialSpikes)]
        self.Spikes = Spikes(spk_trains, t_start, t_stop)

        # Estimate firing rate per trial.
        spikes = self.Spikes.get_spikes()
        self.Rates = OrdDict([(name, Rate(name, kernel, spikes, step))
                              for name, kernel in kernels.items()])

        # Calculate preferred direction.
        # self.test_direction_selectivity()

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

    def get_unit_params(self, remove_dimensions=True):
        """Return main unit parameters."""

        # TODO: remove this by exporting all available metrics automatically?

        # Function to get value from dictionary, or None if key does not exist.
        # Optionally, remove dimension from quantity values.
        def get_val(dic, key):
            val = None
            if dic is not None and key in dic.keys():
                val = dic[key]
                if remove_dimensions and isinstance(val, Quantity):
                    val = float(val)
            return val

        # Recording parameters.
        unit_params = OrdDict()
        sp = self.SessParams
        unit_params['Session information'] = ''
        unit_params['Name'] = self.Name
        unit_params['experiment'] = get_val(sp, 'experiment')
        unit_params['monkey'] = get_val(sp, 'monkey')
        unit_params['date'] = get_val(sp, 'date')
        unit_params['probe'] = get_val(sp, 'probe')
        unit_params['channel #'] = get_val(sp, 'channel #')
        unit_params['unit #'] = get_val(sp, 'unit #')
        unit_params['sort #'] = get_val(sp, 'sort #')
        unit_params['filepath'] = get_val(sp, 'filepath')
        unit_params['filename'] = get_val(sp, 'filename')

        # Quality metrics.
        qm = self.QualityMetrics
        unit_params['Quality metrics'] = ''
        unit_params['MeanWfAmplitude'] = get_val(qm, 'MeanWfAmplitude')
        unit_params['MeanWfDuration (us)'] = get_val(qm, 'MeanWfDuration')
        unit_params['SNR'] = get_val(qm, 'SNR')
        unit_params['MeanFiringRate (sp/s)'] = get_val(qm, 'MeanFiringRate')
        unit_params['ISIviolation (%)'] = get_val(qm, 'ISIviolation')
        unit_params['TrueSpikes (%)'] = get_val(qm, 'TrueSpikes')
        unit_params['UnitType'] = get_val(qm, 'UnitType')

        # Trial stats.
        unit_params['Trial stats'] = ''
        unit_params['total # trials'] = get_val(qm, 'NTrialsTotal')
        unit_params['# rejected trials'] = get_val(qm, 'NTrialsExcluded')
        unit_params['# remaining trials'] = get_val(qm, 'NTrialsIncluded')

        # Stimulus response properties.
        up = self.UnitParams
        dsi = get_val(up, 'DirSelectivity')
        pd = get_val(up, 'PrefDir')
        pdc = get_val(up, 'PrefDirCoarse')
        unit_params['Direction selectivity'] = ''
        unit_params['DSI S1'] = get_val(dsi, 'S1')
        unit_params['DSI S2'] = get_val(dsi, 'S2')
        unit_params['PD S1 (deg)'] = get_val(pd, 'S1')
        unit_params['PD S2 (deg)'] = get_val(pd, 'S2')
        unit_params['PD8 S1 (deg)'] = get_val(pdc, 'S1')
        unit_params['PD8 S2 (deg)'] = get_val(pdc, 'S2')

        return unit_params

    def get_trial_params(self, trials=None, t1=None, t2=None):
        """Return default values of some common parameters."""

        if trials is None:
            trials = [self.all_trials()]
        if t1 is None:
            t1 = self.t_start
        if t2 is None:
            t2 = self.t_stop

        return trials, t1, t2

    def pref_dir(self, stim='S1'):
        """Return preferred direction."""

        pdir = self.UnitParams['PrefDirCoarse'][stim]
        return pdir

    def anti_pref_dir(self, stim='S1'):
        """Return anti-preferred direction."""

        pdir = self.pref_dir(stim)
        adir = util.deg_mod(pdir+180*deg)
        return adir

    # %% Generic methods to get various set of trials.

    def included_trials(self):
        """Return included trials (i.e. not rejected after quality test)."""

        if 'IncludedTrials' in self.QualityMetrics:
            included_trials = self.QualityMetrics['IncludedTrials']
        else:
            included_trials = self.all_trials(filtered=False)
        return included_trials

    def filter_trials(self, trials):
        """Filter trials by excluding rejected ones."""

        tr_idxs = np.logical_and(trials.trials, self.included_trials().trials)
        filtered_trials = Trials(tr_idxs, trials.value, trials.name)
        return filtered_trials

    def ftrials(self, trials, value=None, name=None, filtered=True):
        """
        Create and return trial object from list of trial indices
        after excluding unit's rejected trials.
        """

        trials = Trials(trials, value, name)
        if filtered:
            trials = self.filter_trials(trials)

        return trials

    def all_trials(self, filtered=True):
        """Return indices of all trials."""

        tr_idxs = np.ones(self.Spikes.n_trials(), dtype=bool)
        trials = self.ftrials(tr_idxs, 'all trials', None, filtered)

        return trials

    def param_values_in_trials(self, trials, pnames=None):
        """Return list of parameter values during given trials."""

        if pnames is None:
            pnames = self.TrialParams.columns.values
        pvals = self.TrialParams[pnames][trials.trials]

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

    # %% Methods that provide interface to Unit's Spikes data.

    def get_rates_by_trial(self, trials=None, t1=None, t2=None):
        """Return spike statistics of time interval in given trials."""

        if self.is_empty():
            return None

        if trials is None:
            trials = self.included_trials()

        frate = self.Spikes.spike_stats_in_prd(trials, t1, t2)[1]
        tr_time = self.TrialParams['TrialStart'][trials.trials]
        tr_time = util.remove_dim_to_df_col(tr_time)
        frate_tr_time = Series(frate, index=tr_time, name='FR (1/s)')

        return frate_tr_time

    # %% Methods to trials with specific directions.

    def dir_trials(self, direction, stim='S1', pname=['S1Dir', 'S2Dir'],
                   offsets=[0*deg], comb_params='all', comb_values=False):
        """Return trials with preferred or antipreferred direction."""

        if not util.is_iterable(pname):
            pname = [pname]

        # Get trials for direction + each offset value.
        degs = [util.deg_mod(direction + offset) for offset in offsets]
        deg_dict = OrdDict([(pn, degs) for pn in pname])
        trials = self.trials_by_comb_params(deg_dict, comb_params, comb_values)

        return trials

    def dir_pref_trials(self, stim='S1', pname=['S1Dir', 'S2Dir'],
                        offsets=[0*deg], comb_params='all', comb_values=False):
        """Return trials with preferred direction."""

        # Get trials with preferred direction.
        pdir = self.pref_dir(stim)
        trials = self.dir_trials(pdir, stim, pname, offsets,
                                 comb_params, comb_values)

        # Rename trials.
        if len(trials) == 1:
            trials[0].name = ' & '.join([pn[0:2] for pn in pname]) + ' pref'

        return trials

    def dir_anti_trials(self, stim='S1', pname=['S1Dir', 'S2Dir'],
                        offsets=[0*deg], comb_params='all', comb_values=False):
        """Return trials with anti-preferred direction."""

        # Get trials with anti-preferred direction.
        adir = self.anti_pref_dir(stim)
        trials = self.dir_trials(adir, stim, pname, offsets,
                                 comb_params, comb_values)

        # Rename trials.
        if len(trials) == 1:
            trials[0].name = ' & '.join([pn[0:2] for pn in pname]) + ' anti'

        return trials

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
        with S2 being at given offset from the units preferred direction.
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

        trials = [trS, trD]

        return trials

    # %% Methods to calculate tuning curves and preferred feature values.

    def calc_response_stats(self, pname, t1, t2):
        """Calculate mean response to different values of trial parameter."""

        # Get trials for each parameter value.
        trials = self.trials_by_param_values(pname)

        # Calculate binned spike count per value.
        p_values = util.list_to_quantity([tr.value for tr in trials])
        sp_stats = [self.Spikes.spike_count_stats(tr, t1, t2) for tr in trials]
        mean_rate, std_rate, sem_rate = zip(*sp_stats)
        mean_rate = util.list_to_quantity(mean_rate)
        std_rate = util.list_to_quantity(std_rate)
        sem_rate = util.list_to_quantity(sem_rate)

        return p_values, mean_rate, std_rate, sem_rate

    def calc_dir_response(self, stim):
        """Calculate mean response to each direction during given stimulus."""

        # Calculate binned spike count per direction.
        pname = stim + 'Dir'
        t1, t2 = constants.stim_prds.periods(stim)
        response_stats = self.calc_response_stats(pname, t1, t2)
        dirs, mean_rate, std_rate, sem_rate = response_stats

        return dirs, mean_rate, std_rate, sem_rate

    def test_direction_selectivity(self, stims=['S1', 'S2'], no_labels=False,
                                   do_plot=True, ffig_tmpl=None, **kwargs):
        """
        Test direction selectivity of unit by
          - calculating direction selectivity index, and
          - plotting response to each direction on polar plot.
        """

        # Init for multi-stimulus plotting.
        dir_select_dict = OrdDict()
        for stim in stims:

            # Get mean response to each direction.
            dirs, mean_resp, std_resp, sem_resp = self.calc_dir_response(stim)

            # Calculate preferred direction and direction selectivity index.
            dsi, pref_dir, pref_dir_c = util.deg_w_mean(dirs, mean_resp)
            dir_select_dict[stim] = (dirs, mean_resp, sem_resp,
                                     dsi, pref_dir, pref_dir_c)

            # Add calculated values to unit.
            self.UnitParams['PrefDir'][stim] = pref_dir
            self.UnitParams['PrefDirCoarse'][stim] = pref_dir_c
            self.UnitParams['DirSelectivity'][stim] = dsi

        title = self.Name

        # Minimise labels on plot.
        if no_labels:
            title = None
            kwargs['labels'] = False
            kwargs['polar_legend'] = True
            kwargs['tuning_legend'] = False

        # Plot direction response and selectivity results.
        if do_plot:
            ffig = None
            if ffig_tmpl is not None:
                ffig = ffig_tmpl.format(self.name_to_fname())
            plot.direction_selectivity(dir_select_dict, title=title, ffig=ffig,
                                       **kwargs)

    # %% Plotting methods.

    def prep_plot_params(self, trials, t1, t2, nrate=None):
        """Prepare plotting parameters."""

        # Get trial params.
        trials, t1, t2 = self.get_trial_params(trials, t1, t2)
        names = [tr.name for tr in trials]

        # Get spikes.
        spikes = [self.Spikes.get_spikes(tr, t1, t2) for tr in trials]

        # Get rates and rate times.
        rates, times = None, None
        if nrate is not None:
            rates = [self.Rates[nrate].get_rates(tr.trials, t1, t2)
                     for tr in trials]
            times = self.Rates[nrate].get_times(t1, t2)

        return trials, t1, t2, spikes, rates, times, names

    def plot_raster(self, trials=None, t1=None, t2=None, **kwargs):
        """Plot raster plot of unit for specific trials."""

        if self.is_empty():
            return

        # Set up params.
        plot_params = self.prep_plot_params(trials, t1, t2, nrate=None)
        trials, t1, t2, spikes, rates, times, names = plot_params
        spikes = spikes[0]
        names = names[0]

        # Plot raster.
        ax = plot.raster(spikes, t1, t2,
                         segments=constants.stim_prds,
                         title=self.Name, **kwargs)
        return ax

    def plot_rate(self, nrate, trials=None, t1=None, t2=None, **kwargs):
        """Plot rate plot of unit for specific trials."""

        if self.is_empty():
            return

        # Set up params.
        plot_params = self.prep_plot_params(trials, t1, t2, nrate)
        trials, t1, t2, spikes, rates, times, names = plot_params

        # Plot rate.
        ax = plot.rate(rates, times, t1, t2, names,
                       segments=constants.stim_prds,
                       title=self.Name, **kwargs)
        return ax

    def plot_raster_rate(self, nrate, trials=None, t1=None, t2=None,
                         no_labels=False, **kwargs):
        """Plot raster and rate plot of unit for specific trials."""

        if self.is_empty():
            return

        # Set up params.
        plot_params = self.prep_plot_params(trials, t1, t2, nrate)
        trials, t1, t2, spikes, rates, times, names = plot_params

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
