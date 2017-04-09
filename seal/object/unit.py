# -*- coding: utf-8 -*-
"""
Class representing a (spike sorted) unit (single or multi).

@author: David Samu
"""

import warnings

from datetime import datetime as dt

import numpy as np
import pandas as pd
from quantities import s, ms, us, deg, Hz

from seal.util import util, constants
from seal.object.rate import Rate
from seal.object.spikes import Spikes
from seal.analysis import direction, stats


class Unit:
    """Generic class to store data of a unit (neuron or group of neurons)."""

    # %% Constructor
    def __init__(self, TPLCell=None, rec_info=None, kset=None):
        """Create Unit instance from TPLCell data structure."""

        # Create empty instance.
        self.Name = ''
        self.UnitParams = pd.Series()
        self.SessParams = pd.Series()
        self.Waveforms = pd.DataFrame()
        self.SpikeParams = pd.DataFrame()
        self.Events = pd.DataFrame()
        self.TrData = pd.DataFrame()
        self._Spikes = Spikes([])
        self._Rates = pd.Series()
        self.QualityMetrics = pd.Series()
        self.DS = pd.Series()
        self.TaskRelPrds = pd.Series()

        # Default unit params.
        self.UnitParams['empty'] = True
        self.UnitParams['excluded'] = True

        # Return if no TPLCell is passed.
        if TPLCell is None:
            return

        # %% Session parameters.

        # Prepare session params.
        fname_pars = util.params_from_fname(TPLCell.File)
        subj, date, probe = fname_pars[['monkey', 'date', 'probe']]
        task, task_idx, sortno = fname_pars[['task', 'idx', 'sortno']]
        [chan, un] = TPLCell.ChanUnit
        sampl_prd = (1 / (TPLCell.Info.Frequency * Hz)).rescale(us)
        pinfo = [p.tolist() if isinstance(p, np.ndarray)
                 else p for p in TPLCell.PInfo]
        sess_date = dt.date(dt.strptime(date, '%m%d%y'))
        recording = subj + '_' + util.date_to_str(sess_date)

        # Assign session params.
        sp_list = [('task', task),
                   ('task #', task_idx),
                   ('subject', subj),
                   ('date', sess_date),
                   ('recording', recording),
                   ('probe', probe),
                   ('channel #', chan),
                   ('unit #', un),
                   ('sort #', sortno),
                   ('filepath', TPLCell.Filename),
                   ('filename', TPLCell.File),
                   ('paraminfo', pinfo),
                   ('sampl_prd', sampl_prd)]
        self.SessParams = util.series_from_tuple_list(sp_list)
        self.SessParams = self.SessParams.append(rec_info)

        # Name unit.
        self.set_name()

        # Unit params.
        self.UnitParams['empty'] = False
        self.UnitParams['excluded'] = False

        # %% Waveforms.

        if 'Waves' in TPLCell._fieldnames:
            wfs = TPLCell.Waves
            if wfs.ndim == 1:  # there is only a single spike
                wfs = np.reshape(wfs, (1, len(wfs)))  # extend it to matrix
                wf_sampl_t = float(sampl_prd) * np.arange(wfs.shape[1])
                self.Waveforms = pd.DataFrame(wfs, columns=wf_sampl_t)

        # %% Spike params.

        if 'Spikes' in TPLCell._fieldnames:
            spk_pars = [('time', util.fill_dim(np.array(TPLCell.Spikes))),
                        ('included', True)]
            self.SpikeParams = pd.DataFrame.from_items(spk_pars)

        # %% Stimulus parameters.

        stim_params = constants.stim_params

        # Extract all trial parameters.
        trpars = pd.DataFrame(TPLCell.TrialParams, columns=TPLCell.Header)

        # Extract stimulus parameters.
        StimParams = trpars[stim_params.name]
        StimParams.columns = stim_params.index

        # Change type if required.
        stim_pars = StimParams.copy()
        for stim_par in stim_pars:
            stim_type = stim_params.type[stim_par]
            if stim_type is not None:
                stim_pars[stim_par] = stim_pars[stim_par].astype(stim_type)
        StimParams = stim_pars

        # Combine x and y stimulus coordinates into a single location variable.
        stim_pars = StimParams.copy()
        for stim in stim_pars.columns.levels[0]:
            pstim = stim_pars[stim]
            if ('LocX' in pstim.columns) and ('LocY' in pstim.columns):
                lx, ly = pstim.LocX, pstim.LocY
                stim_pars[stim, 'Loc'] = [(x, y) for x, y in zip(lx, ly)]
        StimParams = stim_pars.sort_index(axis=1)

        # Add same-different columns (S/D trials).
        feats = np.unique([f[1] for f in StimParams.columns
                           if util.is_iterable(f) and len(f) == 2])
        for feat in feats:
                s1f, s2f, dsf = ('S1', feat), ('S2', feat), ('S_D', feat)
                if (s1f in StimParams) and (s2f in StimParams):
                    StimParams[dsf] = 'diff'
                    isame = (StimParams[s1f] == StimParams[s2f])
                    StimParams.loc[isame, dsf] = 'same'

        # %% Subject answer parameters.

        Answer = pd.DataFrame()

        # Recode correct/incorrect answer column.
        corr_ans = trpars['subjectAnswer']
        if len(corr_ans.unique()) > 2:
            warnings.warn(('More than 2 unique values for correct answer: ' +
                           corr_ans.unique()))
        corr_ans = corr_ans == corr_ans.max()  # higher value is correct!
        Answer['correct'] = corr_ans

        # Add column for subject response (saccade direction).
        same_dir = StimParams['S1', 'Dir'] == StimParams['S2', 'Dir']
        # This is not actually correct for passive task!
        Answer['saccade'] = ((same_dir & corr_ans) | (~same_dir & ~corr_ans))

        # %% Trial events.

        # Timestamps of events. Only S1 offset and S2 onset are reliable!
        # S1 onset and S2 offset are fixed to these two.
        # Altogether these four are called anchor events.

        # Watch out: indexing starting with 1 in TPLCell (Matlab)!
        # Everything is in seconds below!

        if 'rel_times' in TPLCell._fieldnames:
            rel_times = TPLCell.rel_times
            anchor_evts = [('S1 on', rel_times.S1_on),
                           ('S1 off', rel_times.S1_off),
                           ('S2 on', rel_times.S2_on),
                           ('S2 off', rel_times.S2_off)]
            anchor_evts = pd.DataFrame.from_items(anchor_evts)
        else:
            # Use absolute times.
            S1dur = float(constants.stim_dur['S1'].rescale(s))
            S2dur = float(constants.stim_dur['S2'].rescale(s))
            iS1off = TPLCell.Patterns.matchedPatterns[:, 2]-1
            iS2on = TPLCell.Patterns.matchedPatterns[:, 3]-1
            ts = TPLCell.Timestamps
            anchor_evts = [('S1 on', ts[iS1off]-S1dur),
                           ('S1 off', ts[iS1off]),
                           ('S2 on', ts[iS2on]),
                           ('S2 off', ts[iS2on]+S2dur)]
            anchor_evts = pd.DataFrame.from_items(anchor_evts)

            # Align trial events to S1 onset.
            abs_S1_onset = anchor_evts['S1 on']  # this is also used below!
            anchor_evts = anchor_evts.subtract(abs_S1_onset, axis=0)

        # Add additional trial events, relative to anchor events.
        evts = [(evt, anchor_evts[rel]+float(offset.rescale(s)))
                for evt, (rel, offset) in constants.tr_evts.iterrows()]
        evts = pd.DataFrame.from_items(evts)

        # Update saccade (end of recording) if info available.
        if ('rel_times' in TPLCell._fieldnames and
            'saccade' in TPLCell.rel_times._fieldnames):
            evts['saccade'] = TPLCell.rel_times.saccade

        # Add dimension to timestamps (ms).
        for evt in evts:
            evts[evt] = util.add_dim_to_series(1000*evts[evt], ms)  # s --> ms
        self.Events = evts

        # %% Trial parameters

        TrialParams = pd.DataFrame()

        # Add start time, end time and length of each trials.
        if 'Timestamps' in TPLCell._fieldnames:
            tstamps = TPLCell.Timestamps
            tr_times = np.array([(tstamps[i1-1], tstamps[i2-1]) for i1, i2
                                 in TPLCell.Info.successfull_trials_indices])
            tr_times = tr_times * s
            for name, col in [('TrialStart', tr_times[:, 0]),
                              ('TrialStop', tr_times[:, 1]),
                              ('TrialLength', tr_times[:, 1] - tr_times[:, 0])]:
                util.add_quant_col(TrialParams, col, name)

        # Add trial period lengths to trial params.
        TrialParams['S1Len'] = evts['S1 off'] - evts['S1 on']
        TrialParams['S2Len'] = evts['S2 off'] - evts['S2 on']
        TrialParams['DelayLenPrec'] = evts['S2 on'] - evts['S1 off']

        # "Categorical" (rounded) delay length variable.
        delay_lens = util.dim_series_to_array(TrialParams['DelayLenPrec'])
        len_diff = [(i, np.abs(delay_lens - dl))
                    for i, dl in enumerate(constants.del_lens)]
        min_diff = pd.DataFrame.from_items(len_diff).idxmin(1)
        dlens = constants.del_lens[min_diff]
        TrialParams['DelayLen'] = list(util.remove_dim_from_series(dlens))

        # Add target feature to be reported.
        if task == 'com':  # Combined task: target feature varies.
            to_report = trpars.TrialType.replace([0, 1], ['loc', 'dir'])
        else:
            to_report = constants.to_report(task)
        TrialParams['ToReport'] = to_report

        # Init included trials (all trials included initially).
        TrialParams['included'] = np.array(True, dtype=bool)

        # %% Assamble full trial data frame.

        StimParams.columns = StimParams.columns.tolist()
        self.TrData = pd.concat([TrialParams, StimParams, Answer], axis=1)

        # %% Spikes.

        # Trials spikes, aligned to S1 onset.
        if 'RelTrialSpikes' in TPLCell._fieldnames:
            RelTrSpikes = TPLCell.RelTrialSpikes
        else:
            TrSpikes = TPLCell.TrialSpikes
            RelTrSpikes = [(spk_train - abs_S1_onset[i])  # align to S1 on
                           for i, spk_train in enumerate(TrSpikes)]

        spk_trains = [spk_train * s for spk_train in RelTrSpikes]  # add dim
        t_starts = self.ev_times('fixate')  # start of trial
        t_stops = self.ev_times('saccade')  # end of trial
        self._Spikes = Spikes(spk_trains, t_starts, t_stops)

        # %% Rates.

        # Estimate firing rate in each trial.
        for name, (kernel, step) in kset.iterrows():
            self.add_rate(name, kernel, step)

    # %% Utility methods.

    def is_empty(self):
        """Is unit empty? (return 1 if empty, 0 if not)"""

        im_empty = self.UnitParams['empty']
        return im_empty

    def is_excluded(self):
        """Is unit excluded? (return 1 if excluded, 0 if not)"""

        im_excluded = self.UnitParams['excluded']
        return im_excluded

    def n_inc_trials(self):
        """Return number of included trials."""

        if 'NTrialsInc' in self.QualityMetrics:
            n_inc_trs = self.QualityMetrics['NTrialsInc']
        else:
            n_inc_trs = len(self.TrData.index)

        return n_inc_trs

    def to_plot(self):
        """Can unit be plotted? (return 1 if plotable, 0 if not)"""

        to_plot = (not self.is_empty()) and bool(self.n_inc_trials())
        return to_plot

    def get_region(self):
        """Return unit's region of origin."""

        my_region = self.SessParams['region']
        return my_region

    def get_hemisphere(self):
        """Return unit's hemisphere of origin."""

        my_hemisphere = self.SessParams['hemisphere']
        return my_hemisphere

    def get_task(self):
        """Return task."""

        task = self.SessParams['task']
        return task

    def set_excluded(self, to_excl):
        """Set unit's exclude flag."""

        self.UnitParams['excluded'] = to_excl

    def set_name(self, name=None):
        """Set/update unit's name."""

        self.Name = name
        if name is None:  # set name to session, channel and unit parameters
            params = self.SessParams[['task', 'recording', 'probe',
                                      'channel #', 'unit #', 'sort #']]
            task, rec, probe, chan, un, isort = params
            subj, date = rec.split('_')
            self.Name = (' '.join([task, subj, date, probe]) +
                         ' Ch{:02}/{} ({})'.format(chan, un, isort))

    def name_to_fname(self):
        """Return filename compatible name string."""

        fname = util.format_to_fname(self.Name)
        return fname

    def add_index_to_name(self, i):
        """Add index to unit's name."""

        idx_str = 'Unit {:0>3}  '.format(i)

        # Remove previous index, if present.
        if self.Name[:5] == 'Unit ':
            self.Name = self.Name[len(idx_str):]

        self.Name = idx_str + self.Name

    def get_uid(self):
        """Return (recording, channel #, unit #) index triple."""

        uid = self.SessParams[['recording', 'channel #', 'unit #']]
        return uid

    def get_utid(self):
        """Return (recording, channel #, unit #, task) index quadruple."""

        utid = self.SessParams[['recording', 'channel #', 'unit #', 'task']]
        return utid

    def get_unit_params(self, rem_dims=True):
        """Return main unit parameters."""

        upars = pd.Series()

        # Basic params.
        upars['Name'] = self.Name
        upars['hemisphere'] = self.get_hemisphere()
        upars['region'] = self.get_region()
        upars['included'] = not self.is_excluded()

        # Recording params.
        upars['Session information'] = ''
        upars = upars.append(util.get_scalar_vals(self.SessParams, rem_dims))

        # Quality metrics.
        upars['Quality metrics'] = ''
        upars = upars.append(util.get_scalar_vals(self.QualityMetrics,
                                                  rem_dims))

        # Direction selectivity.
        upars['DS'] = ''
        for pname, pdf in self.DS.items():
            if pname == 'DR':
                continue
            upars[pname] = ''
            pdf_melt = pd.melt(pdf)
            idxs = list(pdf.index)
            if isinstance(pdf.index, pd.MultiIndex):
                idxs = [' '.join(idx) for idx in idxs]
            stim_list = pdf.shape[1] * idxs
            pdf_melt.index = [' '.join([pr, st]) for pr, st
                              in zip(pdf_melt.variable, stim_list)]
            upars = upars.append(util.get_scalar_vals(pdf_melt.value,
                                                      rem_dims))

        upars.drop_duplicates(inplace=True)

        return upars

    def update_included_trials(self, tr_inc):
        """Update fields related to included/excluded trials and spikes."""

        # Init.
        tr_inc = np.array(tr_inc, dtype=bool)
        tr_exc = np.invert(tr_inc)

        # Update included trials.
        self.TrData['included'] = tr_inc

        # Statistics on trial inclusion.
        self.QualityMetrics['NTrialsTotal'] = len(self.TrData.index)
        self.QualityMetrics['NTrialsInc'] = np.sum(tr_inc)
        self.QualityMetrics['NTrialsExc'] = np.sum(tr_exc)

        # Update included spikes.
        if tr_inc.all():  # all trials included: include full recording
            t1 = self.SpikeParams['time'].min()
            t2 = self.SpikeParams['time'].max()
        else:
            # Include spikes occurring between first and last included trials.
            t1 = self.TrData.TrialStart[tr_inc].min()
            t2 = self.TrData.TrialStop[tr_inc].max()

        spk_inc = util.indices_in_window(self.SpikeParams['time'], t1, t2)
        self.SpikeParams['included'] = spk_inc

    def get_trial_params(self, trs=None, t1s=None, t2s=None):
        """Return default values of trials, start times and stop times."""

        if trs is None:
            trs = self.inc_trials()         # default: all included trials
        if t1s is None:
            t1s = self.ev_times('fixate')   # default: start of fixation
        if t2s is None:
            t2s = self.ev_times('saccade')  # default: saccade time

        return trs, t1s, t2s

    def init_nrate(self, nrate=None):
        """Initialize rate name."""

        if nrate is None:
            nrate = self._Rates.index[0]

        elif nrate not in self._Rates:
            warnings.warn('Rate name: ' + str(nrate) + ' not found in unit.')
            # Return default (or first available) rate name.
            nrate = self._Rates.index[0]

        return nrate

    def sort_feature_values(self, feat, vals):
        """Sort feature values."""

        if not util.is_iterable(feat) or len(feat) != 2:
            # Non-stimulus prefixed feature.
            sorted_vals = np.sort(vals)

        elif feat[1] == 'Dir':
            # Direction: from lower to higher.
            sorted_vals = np.sort(vals)

        elif feat[1] == 'Loc':
            # Location: first up - down (y descending),
            # then left - right (x ascending).
            reord_vs = [[-v[1], v[0]] for v in vals]
            idx = sorted(range(len(reord_vs)), key=reord_vs.__getitem__)
            sorted_vals = vals[idx]

        else:
            # Unknown feature.
            warnings.warn('Sorting values of unknown feature...')
            sorted_vals = np.sort(vals)

        return sorted_vals

    # %% Methods to get times of trial events and periods.

    def ev_times(self, evname, trs=None, add_latency=False):
        """Return timing of events across trials."""

        if trs is None:
            trs = self.inc_trials()

        evt = self.Events.loc[trs, evname].copy()
        if add_latency:
            evt = evt + constants.nphy_cons.latency[self.get_region()]

        return evt

    def pr_times(self, prname, trs=None, add_latency=False, concat=True):
        """Return timing of period (start event, stop event) across trials."""

        ev1, ev2 = constants.tr_prds.loc[prname]
        evt1 = self.ev_times(ev1, trs, add_latency)
        evt2 = self.ev_times(ev2, trs, add_latency)

        prt = pd.concat([evt1, evt2], axis=1) if concat else [evt1, evt2]

        return prt

    def pr_dur(self, prname, trs=None, add_latency=False):
        """Return duration of period (maximum duration across trials)."""

        evt1, evt2 = self.pr_times(prname, trs, add_latency, False)
        dur = (evt2 - evt1).max()

        return dur

    def get_analysis_prds(self, prds=None, trs=None):
        """Get parameters of periods to analyse."""

        is_combined = self.SessParams.combined

        if prds is None:
            prds = (constants.tr_third_prds.copy() if is_combined
                    else constants.tr_half_prds.copy())

        # Update cue plotting.
        if is_combined:
            to_report = self.TrData.ToReport.unique()
            if len(to_report) == 1:   # feature to be reported is constant

                # Set cue to beginning of trial.
                S1_prds = (prds.ref == 'S1 on')
                cue_to_S1_on = constants.tr_evts.loc['fixate', 'shift']
                cue_to_S1_on = cue_to_S1_on + 100*ms  # add a bit to avoid axis
                prds.loc[S1_prds, 'cue'] = cue_to_S1_on

                # Remove cue from rest of the periods.
                prds.loc[~S1_prds, 'cue'] = None
        else:
            # Remove cue from all periods.
            prds.cue = None
            prds.loc['S1 half', 'max_len'] = 3000*ms

        # Add period duration (specific to unit).
        prds['dur'] = [self.pr_dur(prd, trs) for prd in prds.index]

        return prds

    def get_stim_prds(self):
        """Get simulus periods from standard trial alignment."""

        prd_pars = self.get_analysis_prds()
        stim_prds = []
        for stim in prd_pars.stim.unique():
            stim_idx = (prd_pars.stim == stim) & (prd_pars.ref == (stim+' on'))
            tstart = prd_pars.lbl_shift[stim_idx][0]
            tend = tstart + constants.stim_dur[stim]
            stim_prds.append((stim, float(tstart), float(tend)))

        return stim_prds

    # %% Generic methods to get various set of trials.

    def inc_trials(self):
        """Return included trials (i.e. not rejected after quality test)."""

        inc_trs = self.TrData.index[self.TrData[('included')]]
        return inc_trs

    def ser_inc_trials(self):
        """Return included trials in pandas Series."""

        ser_inc_trs = pd.Series([self.inc_trials()], ['included'])
        return ser_inc_trs

    def filter_trials(self, trs_ser):
        """Filter out excluded trials."""

        ftrs_ser = util.filter_lists(trs_ser, self.inc_trials())
        return ftrs_ser

    def trials_by_param(self, param, vals=None, comb_vals=False):
        """Return trials grouped by (selected) values of trial parameter."""

        # Error check.
        if param not in self.TrData.columns:
            warnings.warn('Unknown trial parameter {}'.format(param))
            return pd.Series()

        # Group indices by values of parameter.
        tr_grps = pd.Series(self.TrData.groupby([param]).groups)
        tr_grps = self.filter_trials(tr_grps)

        # Default: all values of parameter.
        if vals is None:
            vals = self.sort_feature_values(param, tr_grps.keys().to_series())
        else:
            # Remove any physical quantity.
            dtype = self.TrData[param].dtype
            vals = np.array(vals, dtype=dtype)

        # Convert to Series of trial list per parameter value.
        v_trs = [(v, np.array(tr_grps[v]) if v in tr_grps else np.empty(0))
                 for v in vals]
        tr_grps = util.series_from_tuple_list(v_trs)

        # Optionally, combine trials across feature values.
        if comb_vals:
            tr_grps = util.union_lists(tr_grps)

        return tr_grps

    def trials_by_params(self, params):
        """Return trials grouped by value combinations of trial parameters."""

        # Error check.
        for param in params:
            if param not in self.TrData.columns:
                warnings.warn('Unknown trial parameter {}'.format(param))
                return pd.Series()

        # Group indices by values of parameter.
        tr_grps = pd.Series(self.TrData.groupby(params).groups)
        tr_grps = self.filter_trials(tr_grps)

        val_combs = tr_grps.keys().to_series()

        # Convert to Series of trial list per parameter value.
        v_trs = [(vc, np.array(tr_grps[vc]) if vc in tr_grps else np.empty(0))
                 for vc in val_combs]
        tr_grps = util.series_from_tuple_list(v_trs)

        return tr_grps

    # %% Methods that provide interface to Unit's Spikes and Rates data.

    def add_rate(self, name, kernel, step):
        """Calculate and add specified firing rate estimate to unit."""

        trs = self.TrData.index
        self._Rates[name] = Rate(name, kernel, self._Spikes.get_spikes(trs),
                                 step)

    def get_time_rates(self, trs=None, t1s=None, t2s=None, tr_time_idx=False):
        """Return rates within time window in given trials."""

        if self.is_empty():
            return

        # Init trials.
        trs, t1s, t2s = self.get_trial_params(trs, t1s, t2s)

        # Get rates.
        rates = self._Spikes.rates(trs, t1s, t2s)

        # Change index from trial index to trials start times.
        if tr_time_idx:
            tr_time = self.TrData.TrialStart[trs]
            rates.index = util.remove_dim_from_series(tr_time)

        return rates

    def get_prd_rates(self, prd, trs=None, add_latency=False,
                      tr_time_idx=False):
        """Return rates within named period in given trials."""

        t1s, t2s = self.pr_times(prd, trs, add_latency, concat=False)
        rates = self.get_time_rates(trs, t1s, t2s, tr_time_idx)
        return rates

    # %% Methods to get trials with specific stimulus directions.

    def dir_trials(self, direc, stims=['S1', 'S2'], offsets=[0*deg]):
        """Return trials with some direction +- offset during S1 and/or S2."""

        # Init list of directions.
        dirs = [float(direction.deg_mod(direc + offset)) for offset in offsets]

        # Get trials for direction + each offset value.
        sd_trs = [self.trials_by_param((stim, 'Dir'), dirs, True)[0]
                  for stim in stims]
        names = [(stim+' '+str(dirs)) for stim in stims]
        sd_trs = pd.Series(sd_trs, index=names)

        return sd_trs

    def dir_pref_trials(self, pref_of, stims=None, offsets=[0*deg]):
        """Return trials with preferred direction."""

        if stims is None:
            stims = [pref_of]

        pdir = self.pref_dir(pref_of)
        trs = self.dir_trials(pdir, stims, offsets)

        return trs

    def dir_anti_trials(self, anti_of, stims=None, offsets=[0*deg]):
        """Return trials with anti-preferred direction."""

        if stims is None:
            stims = [anti_of]

        adir = self.anti_pref_dir(anti_of)
        trs = self.dir_trials(adir, stims, offsets)

        return trs

    def dir_pref_anti_trials(self, pref_anti_of, stims=None, offsets=[0*deg]):
        """Return trials with preferred and antipreferred direction."""

        if stims is None:
            stims = [pref_anti_of]

        pref_trials = self.dir_pref_trials(pref_anti_of, stims, offsets)
        apref_trials = self.dir_anti_trials(pref_anti_of, stims, offsets)
        pref_apref_trials = pref_trials.append(apref_trials)

        return pref_apref_trials

    def S_D_trials(self, pref_of, offsets=[0*deg]):
        """
        Return trials for S1 = S2 (same) and S1 =/= S2 (different)
        with S2 being at given offset from the unit's preferred direction.
        """

        # Collect S- and D-trials for all offsets.
        trS, trD = pd.Series(), pd.Series()
        for offset in offsets:

            # Trials to given offset to preferred direction.
            # stimulus order must be first S2, then S1!
            trs = self.dir_pref_trials(pref_of=pref_of, stims=['S2', 'S1'],
                                       offsets=[offset])

            # S- and D-trials.
            trS[float(offset)] = util.intersect_lists(trs)[0]
            trD[float(offset)] = util.diff_lists(trs)[0]

        # Combine S- and D-trials across offsets.
        trS = util.union_lists(trS, 'S trials')
        trD = util.union_lists(trD, 'D trials')

        # Check that there's no overlap between trials.
        if len(np.intersect1d(trS[0], trD[0])):
            warnings.warn('S and D trails overlap!')

        trsSD = trS.append(trD)

        return trsSD

    # %% Methods to calculate tuning curves and preferred values of features.

    def get_stim_resp_vals(self, stim, feat, t1s=None, t2s=None,
                           add_latency=True):
        """Return response to different values of stimulus feature."""

        # Init time period (stimulus on).
        if t1s is None and t2s is None:
            t1s, t2s = self.pr_times(stim, None, add_latency, concat=False)

        # Get response (firing rates) in each trial.
        trs = self.trials_by_param((stim, feat))
        if not len(trs):
            return pd.DataFrame(columns=['vals', 'resp'])
        vals = pd.concat([pd.Series(v, index=tr)
                          for v, tr in trs.iteritems()])
        resp = self._Spikes.rates(self.inc_trials(), t1s, t2s)
        stim_resp = pd.DataFrame({'vals': vals, 'resp': resp})
        stim_resp.resp = util.remove_dim_from_series(stim_resp.resp)

        return stim_resp

    def calc_DS(self, stim):
        """Calculate direction selectivity (DS)."""

        ltncy, wndw_len = constants.nphy_cons.loc[self.get_region()]

        # Find maximal stimulus response across all directions combined.
        # WARNING: This assumes DS is maximal at time of maximum response!
        nrate, trs = self.init_nrate(), self.inc_trials()
        t1s, t2s = self.pr_times(stim, add_latency=False, concat=False)
        rates = self._Rates[nrate].get_rates(trs, t1s, t2s, t1s)

        # Get period around tmax (but within full time window of stimulus).
        tmax = rates.mean().argmax()
        tvec = rates.columns
        TW = stats.prd_in_window(tmax, tvec.min(), tvec.max(), wndw_len, ms)

        # Get direction response values and stats.
        stim_resp = self.get_stim_resp_vals(stim, 'Dir', t1s+TW.tstart,
                                            t1s+TW.tend)
        resp_stats = util.calc_stim_resp_stats(stim_resp, constants.all_dirs)

        # Convert each result to Numpy array.
        dirs = np.array(resp_stats.index) * deg
        rstats = [np.array(resp_stats[stat])
                  for stat in ('mean', 'std', 'sem')]
        mean_resp, std_resp, sem_resp = rstats

        # DS based on maximum rate only (legacy method).
        mPDres, mDS = direction.max_DS(dirs, mean_resp)

        # DS based on weighted sum of all rates & directions.
        wPDres, wDS = direction.weighted_DS(dirs, mean_resp)

        # DS based on Gaussian tuning curve fit.
        dir0 = wPDres.PD  # reference direction (to start curve fitting from)
        dirs, resp = stim_resp['vals'], stim_resp['resp']
        tPDres, tune_pars, tune_res = direction.tuned_DS(dirs, resp, dir0)

        # Prepare results.
        PD = pd.concat([mPDres, wPDres, tPDres], axis=1,
                       keys=('max', 'weighted', 'tuned'))
        DSI = pd.Series([mDS, wDS], index=['mDS', 'wDS'])

        return TW, resp_stats, PD, DSI, tune_pars, tune_res

    def test_DS(self, stims=['S1', 'S2']):
        """Test unit's direction selectivit."""

        if not self.n_inc_trials():
            return

        # Init field to store DS results.
        lTW, lDR, lDSI, lPD, lTP = [], [], [], [], []
        for stim in stims:

            # Calculate DS during stimulus.
            TW, DR, PD, DSI, tune_pars, tune_res = self.calc_DS(stim)

            # Collect DR values.
            lTW.append(TW)
            lDR.append(DR)

            # Collect DS params.
            lDSI.append(DSI)
            lPD.append(PD)

            # Collect tuning params.
            tun_fit = tune_pars.loc['fit']
            lTP.append(tun_fit.append(tune_res))

        # Convert each to a DataFrame.
        TW, DR, DSI, PD, TP = [pd.concat(rlist, axis=1, keys=stims).T
                               for rlist in (lTW, lDR, lDSI, lPD, lTP)]

        # Save DS results.
        self.DS['TW'] = TW
        self.DS['DR'] = DR.T
        self.DS['DSI'] = DSI
        self.DS['PD'] = PD
        self.DS['TP'] = TP

    def pref_dir(self, stim='S1', method='max', pd_type='cPD'):
        """Return preferred direction."""

        pdir = self.DS['PD'].loc[(stim, method), pd_type]
        return pdir

    def anti_pref_dir(self, stim='S1', method='max', pd_type='cAD'):
        """Return anti-preferred direction."""

        adir = self.DS['PD'].loc[(stim, method), pd_type]
        return adir

    # %% Functions to return unit parameters.

    def get_baseline(self):
        """Return baseline firing rate."""

        baseline = (self.QualityMetrics['baseline']
                    if 'baseline' in self.QualityMetrics else None)
        return baseline
