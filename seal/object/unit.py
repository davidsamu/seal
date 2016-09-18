# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 14:06:14 2016

Class representing a unit, after spike sorting and preprocessing.

@author: David Samu
"""

from itertools import product
from collections import OrderedDict as OrdDict

import numpy as np
from pandas import DataFrame
from quantities import s, ms, us, deg, rad, Hz
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
        self.QualityMetrics = dict()
        self.TrialParams = DataFrame()
        self.ExpSegments = OrdDict()
        self.Events = DataFrame()
        self.Spikes = []
        self.Rates = dict()
        self.t_start = t_start
        self.t_stop = t_stop

        # Return if no data is passed.
        if TPLCell is None:
            return

        # Extract session parameters.
        [monkey, dateprobe, exp, sortno] = TPLCell.File[:-4].split('_')
        [date, probe] = [dateprobe[0:6], dateprobe[6:].upper()]
        [chan, un] = TPLCell.ChanUnit
        self.Name = 'Unit {:0>3}  '.format(i)
        self.Name += ' '.join([exp, monkey, date, probe])
        self.Name += ' Ch{:02}/{} ({})'.format(chan, un, sortno)

        self.SessParams = OrdDict()
        self.SessParams['monkey'] = monkey
        self.SessParams['date'] = date
        self.SessParams['probe'] = probe
        self.SessParams['experiment'] = exp
        self.SessParams['channel #'] = chan
        self.SessParams['unit #'] = un
        self.SessParams['sort #'] = int(sortno)
        self.SessParams['Filepath'] = TPLCell.Filename
        self.SessParams['Filename'] = TPLCell.File
        self.SessParams['PInfo'] = [p.tolist() if isinstance(p, np.ndarray)
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
        self.Rates = dict([(name, Rate(kernel, self.Spikes.get_spikes(), step))
                           for name, kernel in kernels.items()])

        # Calculate preferred direction.
        self.calc_pref_dir()

    # %% Utility methods.

    def name_to_fname(self):
        """Return filename compatible name string."""

        fname = util.format_to_fname(self.Name)
        return fname

    def set_params(self, trials=None, t1=None, t2=None):
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

        tr_idxs = np.ones(len(self.Spikes.n_trials), dtype=bool)
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
        deg_dict = dict([(pn, degs) for pn in pname])
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
            trials[0].name = trials[0].name[0:2] + ' pref'

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
            trials[0].name = trials[0].name[0:2] + ' anti'

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

    def calc_pref_dir(self):
        """
        Calculate 'precise' and 'coarse' preferred directions
        during each segment.
        """

        self.UnitParams['PrefDir'] = dict()
        self.UnitParams['PrefDirCoarse'] = dict()

        for nstim, (t1, t2) in self.ExpSegments.items():

            self.UnitParams['PrefDir'][nstim] = dict()
            self.UnitParams['PrefDirCoarse'][nstim] = dict()

            # TODO: this should not be hardcoded!
            pname = 'S1Dir' if nstim == 'S1' else 'S2Dir'

            # Get list of all directions and their trials.
            trials = self.trials_by_param_values(pname)
            dirs = [pt.value for pt in trials]

            # Calculate binned spike count per direction.
            mean_sp_count = [self.Spikes.avg_n_spikes(tr, t1, t2)
                             for tr in trials]

            # Convert directions to Cartesian unit vectors.
            cart_dirs = np.array([util.pol2cart(1, d.rescale(rad))
                                  for d in dirs])
            # Calculate mean along each dimension.
            x_mean = np.average(cart_dirs[:, 0], weights=mean_sp_count)
            y_mean = np.average(cart_dirs[:, 1], weights=mean_sp_count)
            # Re-convert into angle in degrees.
            rho, phi = util.cart2pol(x_mean, y_mean)
            phi_deg = (phi*rad).rescale(deg)
            pdir = phi_deg if phi_deg >= 0 else phi_deg + 360*deg
            # Coarse to one of the directions used in the experiment.
            deg_diffs = np.array([util.deg_diff(d, pdir) for d in dirs])
            pdir_c = dirs[np.argmin(deg_diffs)]

            # Register values.
            self.UnitParams['PrefDir'][nstim] = pdir
            self.UnitParams['PrefDirCoarse'][nstim] = pdir_c

    # %% Plotting methods.

    def prep_plot_params(self, trials, t1, t2, nrate=None):
        """Prepare plotting parameters."""

        # Get trial params.
        trials, t1, t2 = self.set_params(trials, t1, t2)
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
