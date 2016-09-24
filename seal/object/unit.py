# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 14:06:14 2016

Class representing a unit, after spike sorting and preprocessing.

@author: David Samu
"""

from datetime import datetime as dt
from itertools import product
from collections import OrderedDict as OrdDict

import numpy as np
from pandas import DataFrame
from quantities import Quantity, s, ms, us, deg, Hz
from neo import SpikeTrain

from seal.util import plot, util
from seal.object.trials import Trials
from seal.object.spikes import Spikes
from seal.object.rate import Rate


class Unit:
    """Generic class to store data of a unit (neuron or group of neurons)."""

    # %% Constructor
    def __init__(self, TPLCell, t_start, t_stop, kernels, step=10*ms,
                 params_info=None, i=None):
        """Create Unit instance, optionally from TPLCell data structure."""

        # Create empty instance.
        self.Name = ''
        self.SessParams = OrdDict()
        self.UnitParams = OrdDict()
        self.QualityMetrics = OrdDict()
        self.TrialParams = DataFrame()
        self.ExpSegments = OrdDict()
        self.Events = DataFrame()
        self.Spikes = []
        self.Rates = OrdDict()
        self.t_start = t_start
        self.t_stop = t_stop

        # Return if no data is passed.
        if TPLCell is None:
            return

        # Extract session parameters.
        file_parts = TPLCell.File[:-4].split('_')
        if len(file_parts) == 4:
            [monkey, dateprobe, exp, sortno] = file_parts
        else:  # this a test case
            [monkey, dateprobe, exp, sortno, my_name, test] = file_parts

        [date, probe] = [dateprobe[0:6], dateprobe[6:].upper()]
        [chan, un] = TPLCell.ChanUnit
        self.Name = 'Unit {:0>3}  '.format(i)
        self.Name += ' '.join([exp, monkey, date, probe])
        self.Name += ' Ch{:02}/{} ({})'.format(chan, un, sortno)

        self.SessParams = OrdDict()
        self.SessParams['experiment'] = exp
        self.SessParams['monkey'] = monkey
        self.SessParams['date'] = dt.date(dt.strptime(date, '%m%d%y'))
        self.SessParams['probe'] = probe
        self.SessParams['channel #'] = chan
        self.SessParams['unit #'] = un
        self.SessParams['sort #'] = int(sortno)
        self.SessParams['filepath'] = TPLCell.Filename
        self.SessParams['filename'] = TPLCell.File
        self.SessParams['paraminfo'] = [p.tolist() if isinstance(p, np.ndarray)
                                        else p for p in TPLCell.PInfo]
        sampl_per = (1 / (TPLCell.Info.Frequency * Hz)).rescale(us)
        self.SessParams['SamplPer'] = sampl_per

        # Unit parameters/stats.
        self.UnitParams = OrdDict()
        wf_time = range(TPLCell.Waves.shape[1]) * self.SessParams['SamplPer']
        self.UnitParams['WaveformTime'] = wf_time
        self.UnitParams['SpikeWaveforms'] = TPLCell.Waves
        self.UnitParams['SpikeDuration'] = TPLCell.Spikes_dur * s
        self.UnitParams['MeanSpikeDur'] = TPLCell.MeanSpikeDur * s
        t_max = np.max(TPLCell.Spikes)
        self.UnitParams['SpikeTimes'] = SpikeTrain(TPLCell.Spikes*s,
                                                   t_start=0*s, t_stop=t_max)
        self.UnitParams['PrefDir'] = OrdDict()
        self.UnitParams['PrefDirCoarse'] = OrdDict()
        self.UnitParams['DirSelectivity'] = OrdDict()

        # Trial parameters.
        self.TrialParams = DataFrame(TPLCell.TrialParams,
                                     columns=TPLCell.Header)
        if params_info is not None:  # This takes a lot of time, speed it up?
            for name, (new_name, unit) in params_info.items():
                if name not in self.TrialParams.columns:
                    print('Warning: Parameter {0} not found!'.format(name))
                    continue
                self.TrialParams.rename(columns={name: new_name}, inplace=True)
                new_col = np.asarray(self.TrialParams[new_name]) * unit
                self.TrialParams[new_name] = DataFrame([new_col]).T
        # Start and end times of each trials.
        trial_times = [(TPLCell.Timestamps[i1-1]*s, TPLCell.Timestamps[i2-1]*s)
                       for i1, i2 in TPLCell.Info.successfull_trials_indices]
        self.TrialParams['TrialStart'] = [tr_t[0] for tr_t in trial_times]
        self.TrialParams['TrialStop'] = [tr_t[1] for tr_t in trial_times]
        self.TrialParams['TrialLength'] = [tr_t[1] - tr_t[0]
                                           for tr_t in trial_times]

        # Define experiment parameters and segments.
        S1 = np.array([0.0, 0.5]) * s
        S2 = np.array([2.0, 2.5]) * s
        S1_dur = S1[1]-S1[0]
        S2_dur = S2[1]-S2[0]
        self.ExpSegments['S1'] = S1
        self.ExpSegments['S2'] = S2

        # Timestamps of events. Only S1 offset and S2 onset are reliable!
        # Take care of indexing starting with 0! (not with 1 as in Matlab)
        iS1off = TPLCell.Patterns.matchedPatterns[:, 2]-1
        iS2on = TPLCell.Patterns.matchedPatterns[:, 3]-1
        self.Events = DataFrame([TPLCell.Timestamps[iS1off]*s-S1_dur,
                                 TPLCell.Timestamps[iS1off]*s,
                                 TPLCell.Timestamps[iS2on]*s,
                                 TPLCell.Timestamps[iS2on]*s+S2_dur]).T
        self.Events.columns = ['S1 onset', 'S1 offset',
                               'S2 onset', 'S2 offset']
        # Align trial events to S1 onset.
        S1on = self.Events['S1 onset']
        self.Events = self.Events.subtract(S1on, axis=0)

        # Trials spikes, aligned to S1 onset.
        spk_trains = [TS*s-S1on[i] for i, TS in enumerate(TPLCell.TrialSpikes)]
        self.Spikes = Spikes(spk_trains, t_start, t_stop)

        # Estimate firing rate per trial.
        self.Rates = OrdDict([(name, Rate(kernel, self.Spikes.get_spikes(), step))
                              for name, kernel in kernels.items()])

        # Calculate preferred direction.
        # self.test_direction_selectivity()

    # %% Utility methods.

    def name_to_fname(self):
        """Return filename compatible name string."""

        fname = util.format_to_fname(self.Name)
        return fname

    def get_unit_params(self, remove_dimensions=True):
        """Return main unit parameters."""

        # TODO: remove this by exporting all available metrics automatically?

        # Function to get value from dictionary, or None if key does not exist.
        # Optionally, remove dimension from quantity values.
        def get_val(dic, key):
            val = None
            if key in dic.keys():
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

        # Stimulus response properties.
        up = self.UnitParams
        dsi = get_val(up, 'DirSelectivity')
        pd = get_val(up, 'PrefDir')
        pdc = get_val(up, 'PrefDirCoarse')
        unit_params['Direction selectivity'] = ''
        unit_params['DSI S1'] = get_val(dsi, 'S1')
        unit_params['DSI S2 (deg)'] = get_val(dsi, 'S2')
        unit_params['PD S1 (deg)'] = get_val(pd, 'S1')
        unit_params['PD S2 (deg)'] = get_val(pd, 'S2')
        unit_params['PD8 S1 (deg)'] = get_val(pdc, 'S1')
        unit_params['PD8 S2 (deg)'] = get_val(pdc, 'S2')

        return unit_params

    def get_trial_params(self, trials=None, t1=None, t2=None):
        """Return default values of some common parameters."""

        if trials is None:
            trials = self.all_trials()
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

    # %% Generic methods to get various set of trials

    def all_trials(self):
        """Return indices of all trials."""

        tr_idxs = np.ones(self.Spikes.n_trials(), dtype=bool)
        trials = [Trials(tr_idxs, 'all trials')]
        return trials

    def param_values_by_trials(self, trials, pnames=None):
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
        ptrials = [Trials(trials_idxs(v), v, n)
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

    def calc_mean_response(self, pname, t1, t2):
        """Calculate mean response to different values of trial parameter."""

        # Get trials for each parameter value.
        trials = self.trials_by_param_values(pname)

        # Calculate binned spike count per value.
        p_values = [tr.value for tr in trials]
        mean_sp_count = np.array([self.Spikes.avg_n_spikes(tr, t1, t2)
                                  for tr in trials])

        return p_values, mean_sp_count

    def calc_dir_response(self, stim):
        """Calculate mean response to each direction during given stimulus."""

        # Calculate binned spike count per direction.
        pname = stim + 'Dir'
        t1, t2 = self.ExpSegments[stim]
        dirs, mean_sp_count = self.calc_mean_response(pname, t1, t2)

        return dirs, mean_sp_count

    def test_direction_selectivity(self, stims=['S1', 'S2'], ffig_tmpl=None):
        """
        Test direction selectivity of unit by
        - calculating direction selectivity index, and
        - plotting response to each direction on polar plot.
        """

        # Init for multi-stimulus plotting.
        ax = plot.axes(polar=True)
        colors = ['m', 'g']  # plot.get_colors_from_cycle()
        patches = []
        for stim, color in zip(stims, colors):

            # Init of direction.
            self.UnitParams['PrefDir'][stim] = OrdDict()
            self.UnitParams['PrefDirCoarse'][stim] = OrdDict()

            # Get mean response to each direction.
            dirs, mean_sp_count = self.calc_dir_response(stim)

            # Calculate preferred direction and direction selectivity index.
            ds_idx, pref_dir, pref_dir_c = util.deg_w_mean(dirs, mean_sp_count)

            # Add calculated values to unit.
            self.UnitParams['PrefDir'][stim] = pref_dir
            self.UnitParams['PrefDirCoarse'][stim] = pref_dir_c
            self.UnitParams['DirSelectivity'][stim] = ds_idx

            # Plot direction selectivity.
            plot.direction_selectivity(dirs, mean_sp_count, ds_idx, pref_dir,
                                       color=color, ax=ax)

            # Collect stimulus params.
            s_pd = str(float(round(pref_dir, 1)))
            s_pd_c = str(int(pref_dir_c))
            lgd_lbl = '{}:   {:.3f}'.format(stim, ds_idx)
            lgd_lbl += '      {:>5}     {:>3}'.format(s_pd, s_pd_c)
            patches.append(plot.get_proxy_patch(lgd_lbl, color))

        # Set labels
        plot.set_labels(title=self.Name, ytitle=1.10, ax=ax)

        # Set legend
        lgd_ttl = 'DSI'.rjust(30) + 'PD (deg)'.rjust(16) + 'PD8 (deg)'.rjust(12)
        lgd = plot.set_legend(ax, handles=patches, title=lgd_ttl,
                              bbox_to_anchor=(0., -0.30, 1., .0),
                              loc='lower center', prop={'family': 'monospace'})
        lgd.get_title().set_ha('left')

        # Save figure
        if ffig_tmpl is not None:
            ffig = ffig_tmpl.format(self.name_to_fname())
            plot.save_fig(ffig=ffig, bbox_extra_artists=(lgd,))

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

        # Set up params.
        plot_params = self.prep_plot_params(trials, t1, t2, nrate=None)
        trials, t1, t2, spikes, rates, times, names = plot_params
        spikes = spikes[0]
        names = names[0]

        # Plot raster.
        ax = plot.raster(spikes, t1, t2,
                         segments=self.ExpSegments, title=self.Name, **kwargs)
        return ax

    def plot_rate(self, nrate, trials=None, t1=None, t2=None, **kwargs):
        """Plot rate plot of unit for specific trials."""

        # Set up params.
        plot_params = self.prep_plot_params(trials, t1, t2, nrate)
        trials, t1, t2, spikes, rates, times, names = plot_params

        # Plot rate.
        ax = plot.rate(rates, times, t1, t2, names,
                       segments=self.ExpSegments, title=self.Name, **kwargs)
        return ax

    def plot_raster_rate(self, nrate, trials=None, t1=None, t2=None, **kwargs):
        """Plot raster and rate plot of unit for specific trials."""

        # Set up params.
        plot_params = self.prep_plot_params(trials, t1, t2, nrate)
        trials, t1, t2, spikes, rates, times, names = plot_params

        # Plot raster and rate.
        fig = plot.raster_rate(spikes, rates, times, t1, t2, names,
                               segments=self.ExpSegments,
                               title=self.Name, **kwargs)
        return fig
