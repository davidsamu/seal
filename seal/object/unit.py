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

from seal.util import util
from seal.object import constants
from seal.object.rate import Rate
from seal.object.spikes import Spikes
from seal.analysis import direction


class Unit:
    """Generic class to store data of a unit (neuron or group of neurons)."""

    # %% Constructor
    def __init__(self, TPLCell=None, kernels=None, step=None, stim_params=None,
                 answ_params=None, stim_dur=None, tr_evt=None, taskname=None,
                 region=None):
        """Create Unit instance from TPLCell data structure."""

        # Create empty instance.
        self.Name = ''
        self.UnitParams = pd.Series()
        self.SessParams = pd.Series()
        self.Waveforms = pd.DataFrame()
        self.SpikeParams = pd.DataFrame()
        self.StimParams = pd.DataFrame()
        self.Answer = pd.DataFrame()
        self.Events = pd.DataFrame()
        self.TrialParams = pd.DataFrame()
        self._Spikes = Spikes([])
        self._Rates = pd.Series()
        self.QualityMetrics = pd.Series()
        self.DS = pd.Series()

        # Default unit params.
        self.UnitParams['region'] = region
        self.UnitParams['empty'] = True
        self.UnitParams['excluded'] = True

        # Return if no TPLCell is passed.
        if TPLCell is None:
            return

        # %% Session parameters.

        # Prepare session params.
        subj, date, probe, exp, isort = util.params_from_fname(TPLCell.File)
        task, task_idx = exp[:-1], int(exp[-1])
        task = task if taskname is None else taskname  # use provided name
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
                   ('sort #', isort),
                   ('filepath', TPLCell.Filename),
                   ('filename', TPLCell.File),
                   ('paraminfo', pinfo),
                   ('sampl_prd', sampl_prd)]
        self.SessParams = util.series_from_tuple_list(sp_list)

        # Name unit.
        self.set_name()

        # %% Waveforms.

        wfs = TPLCell.Waves
        if wfs.ndim == 1:  # there is only a single spike: extend it to matrix
            wfs = np.reshape(wfs, (1, len(wfs)))
        wf_sampl_t = float(sampl_prd) * np.arange(wfs.shape[1])
        self.Waveforms = pd.DataFrame(wfs, columns=wf_sampl_t)

        # %% Spike params.

        spk_pars = [('time', util.fill_dim(TPLCell.Spikes)),
                    ('dur', util.fill_dim(TPLCell.Spikes_dur * s.rescale(us))),
                    ('included', True)]
        self.SpikeParams = pd.DataFrame.from_items(spk_pars)

        # %% Stimulus parameters.

        # Extract all trial parameters.
        trpars = pd.DataFrame(TPLCell.TrialParams, columns=TPLCell.Header)

        # Extract stimulus parameters.
        self.StimParams = trpars[stim_params.name]
        self.StimParams.columns = stim_params.index

        # Combine x and y stimulus coordinates into a single location variable.
        stim_pars = self.StimParams.copy()
        for stim in stim_pars.columns.levels[0]:
            pstim = stim_pars[stim]
            if ('LocX' in pstim.columns) and ('LocY' in pstim.columns):
                stim_pars[stim, 'Loc'] = list(zip(pstim.LocX, pstim.LocY))
        self.StimParams = stim_pars.sort_index(axis=1)

        # %% Subject answer parameters.

        # Recode correct/incorrect answer column.
        corr_ans = trpars[answ_params['AnswCorr']]
        if len(corr_ans.unique()) > 2:
            warnings.warn(('More than 2 unique values in AnswCorr: ' +
                           corr_ans.unique()))
        corr_ans = corr_ans == corr_ans.max()  # higher value is correct!
        self.Answer['AnswCorr'] = corr_ans

        # Add column for subject response (saccade direction).
        same_dir = self.StimParams['S1', 'Dir'] == self.StimParams['S2', 'Dir']
        # This is not actually correct for passive task!
        self.Answer['SaccadeDir'] = ((same_dir & corr_ans) |
                                     (~same_dir & ~corr_ans))

        # %% Trial events.

        # Timestamps of events. Only S1 offset and S2 onset are reliable!
        # S1 onset and S2 offset are fixed to these two.
        # Altogether these four are called anchor events.

        # Watch out: indexing starting with 1 in TPLCell (Matlab)!
        # Everything is in seconds below!

        S1dur = float(stim_dur['S1'].rescale(s))
        S2dur = float(stim_dur['S2'].rescale(s))
        iS1off = TPLCell.Patterns.matchedPatterns[:, 2]-1
        iS2on = TPLCell.Patterns.matchedPatterns[:, 3]-1
        anchor_evts = [('S1 on', TPLCell.Timestamps[iS1off]-S1dur),
                       ('S1 off', TPLCell.Timestamps[iS1off]),
                       ('S2 on', TPLCell.Timestamps[iS2on]),
                       ('S2 off', TPLCell.Timestamps[iS2on]+S2dur)]
        anchor_evts = pd.DataFrame.from_items(anchor_evts)

        # Align trial events to S1 onset.
        abs_S1_onset = anchor_evts['S1 on']  # this is also used below!
        anchor_evts = anchor_evts.subtract(abs_S1_onset, axis=0)

        # Add additional trial events, relative to anchor events.
        evts = [(evt, anchor_evts[rel]+float(offset.rescale(s)))
                for evt, (rel, offset) in tr_evt.iterrows()]
        evts = pd.DataFrame.from_items(evts)

        # Add dimension to timestamps (ms).
        for evt in evts:
            evts[evt] = util.add_dim_to_series(1000*evts[evt], ms)  # s --> ms
        self.Events = evts

        # %% Trial parameters.

        # Add start time, end time and length of each trials.
        tstamps = TPLCell.Timestamps
        tr_times = np.array([(tstamps[i1-1], tstamps[i2-1]) for i1, i2
                             in TPLCell.Info.successfull_trials_indices]) * s
        for colname, col in [('TrialStart', tr_times[:, 0]),
                             ('TrialStop', tr_times[:, 1]),
                             ('TrialLength', tr_times[:, 1] - tr_times[:, 0])]:
            util.add_quant_col(self.TrialParams, col, colname)

        # Add trial period lengths to trial params.
        self.TrialParams['S1Len'] = evts['S1 off'] - evts['S1 on']
        self.TrialParams['S2Len'] = evts['S2 off'] - evts['S2 on']
        self.TrialParams['DelayLenPrec'] = evts['S2 on'] - evts['S1 off']
        self.TrialParams['DelayLen'] = [np.round(v, 1) for v in
                                        self.TrialParams['DelayLenPrec']]

        # Init included trials (all trials included).
        self.TrialParams['included'] = np.array(True, dtype=bool)

        # %% Spikes.

        # Trials spikes, aligned to S1 onset.
        spk_trains = [(spk_train - abs_S1_onset[i]) * s
                      for i, spk_train in enumerate(TPLCell.TrialSpikes)]
        t_starts = self.ev_times('fixate')  # start of trial
        t_stops = self.ev_times('saccade')  # end of trial
        self._Spikes = Spikes(spk_trains, t_starts, t_stops)

        # %% Rates.

        # Estimate firing rate in each trial.
        spikes = self._Spikes.get_spikes()
        rate_list = [Rate(name, kernel, spikes, step)
                     for name, kernel in kernels.items()]
        self._Rates = pd.Series(rate_list, index=kernels.keys())

        # %% Unit params.

        self.UnitParams['empty'] = False
        self.UnitParams['excluded'] = False

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
            n_inc_trs = len(self.TrialParams.index)

        return n_inc_trs

    def to_plot(self):
        """Can unit be plotted? (return 1 if plotable, 0 if not)"""

        to_plot = (not self.is_empty()) and bool(self.n_inc_trials())
        return to_plot

    def get_region(self):
        """Return unit's region of origin."""

        my_region = self.UnitParams['region']
        return my_region

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
        upars['region'] = self.get_region()
        upars['excluded'] = self.is_excluded()

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

        return upars

    def update_included_trials(self, tr_inc):
        """Update fields related to included/excluded trials and spikes."""

        # Init included and excluded trials.
        tr_inc = np.array(tr_inc, dtype=bool)
        tr_exc = np.invert(tr_inc)

        # Update included trials.
        self.TrialParams['included'] = tr_inc

        # Statistics on trial inclusion.
        self.QualityMetrics['NTrialsTotal'] = len(self.TrialParams.index)
        self.QualityMetrics['NTrialsInc'] = np.sum(tr_inc)
        self.QualityMetrics['NTrialsExc'] = np.sum(tr_exc)

        # Update included spikes.
        if tr_inc.all():  # all trials included: include full recording
            t1 = self.SpikeParams['time'].min()
            t2 = self.SpikeParams['time'].max()
        else:
            # Include spikes occurring between first and last included trials.
            t1 = self.TrialParams.loc[tr_inc, 'TrialStart'].min()
            t2 = self.TrialParams.loc[tr_inc, 'TrialStop'].max()

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

        def_nrate = constants.def_nrate

        if nrate is None:
            nrate = (def_nrate if def_nrate in self._Rates
                     else self._Rates.index[0])

        elif nrate not in self._Rates:
            warnings.warn('Rate name: ' + str(nrate) + ' not found in unit.')
            self.init_nrate()  # return default (or first available) rate name

        return nrate

    # %% Methods to get times of trial events and periods.

    def ev_times(self, evname, add_latency=False):
        """Return timing of events across trials."""

        evt = self.Events[evname].copy()
        if add_latency:
            evt += constants.latency[self.get_region()]

        return evt

    def pr_times(self, prname, add_latency=False, concat=True):
        """Return timing of period (start event, stop event) across trials."""

        ev1, ev2 = constants.tr_prd.loc[prname]
        evt1 = self.ev_times(ev1, add_latency)
        evt2 = self.ev_times(ev2, add_latency)

        prt = pd.concat([evt1, evt2], axis=1) if concat else [evt1, evt2]

        return prt

    def pr_dur(self, prname, add_latency=False):
        """Return duration of period (maximum duration across trials)."""

        evt1, evt2 = self.pr_times(prname, add_latency, False)
        dur = (evt2 - evt1).max()

        return dur

    # %% Generic methods to get various set of trials.

    def inc_trials(self):
        """Return included trials (i.e. not rejected after quality test)."""

        inc_trs = self.TrialParams.index[self.TrialParams['included']]
        return inc_trs

    def ser_inc_trials(self):
        """Return included trials in pandas Series."""

        ser_inc_trs = pd.Series([self.inc_trials()], ['included'])
        return ser_inc_trs

    def filter_trials(self, trs_ser):
        """Filter out excluded trials."""

        ftrs_ser = trs_ser.apply(np.intersect1d, args=(self.inc_trials(),))
        ftrs_ser = ftrs_ser[[len(ftrs) > 0 for ftrs in ftrs_ser]]

        return ftrs_ser

    def correct_incorrect_trials(self):
        """Return indices of correct and incorrect trials."""

        corr = self.Answer['AnswCorr']
        ctrs = pd.Series([corr.index[idxs] for idxs in (corr, ~corr)])
        ctrs.index = ['correct', 'error']

        ctrs = self.filter_trials(ctrs)

        return ctrs

    def trials_by_features(self, stim, feat, vals=None, comb_vals=False):
        """Return trials grouped by (selected) values of stimulus param."""

        # Error check.
        if feat not in self.StimParams.columns.levels[1]:
            warnings.warn('Requested feature "{}" '.format(feat) +
                          'not found in StimParams table.')
            return pd.Series()

        # Group indices by stimulus feature value.
        tr_grps = pd.Series(self.StimParams.groupby([(stim, feat)]).groups)
        tr_grps = self.filter_trials(tr_grps)

        # Default: all values of stimulus feature.
        if vals is None:
            vals = sorted(tr_grps.keys())
        else:
            vals = [float(v) for v in vals]  # remove dimension

        # Convert to Series of trial list per feature value.
        v_trs = [(v, np.array(tr_grps[v]) if v in tr_grps else np.empty(0))
                 for v in vals]
        tr_grps = util.series_from_tuple_list(v_trs)

        # Optionally, combine trials across feature values.
        if comb_vals:
            tr_grps = util.union_lists(tr_grps)

        return tr_grps

    # %% Methods that provide interface to Unit's Spikes data.

    def get_time_rates(self, trs=None, t1s=None, t2s=None, tr_time_idx=False):
        """Return rates within time window in given trials."""

        if self.is_empty():
            return None

        # Init trials.
        trs, t1s, t2s = self.get_trial_params(trs, t1s, t2s)

        # Get rates.
        rates = self._Spikes.rates(trs, t1s, t2s)

        # Change index from trial index to trials start times.
        if tr_time_idx:
            tr_time = self.TrialParams.loc[trs, 'TrialStart']
            rates.index = util.remove_dim_from_series(tr_time)

        return rates

    def get_prd_rates(self, prd, trs=None, add_latency=False,
                      tr_time_idx=False):
        """Return rates within named period in given trials."""

        t1s, t2s = self.pr_times(prd, add_latency, concat=False)
        rates = self.get_time_rates(trs, t1s, t2s, tr_time_idx)
        return rates

    # %% Methods to get trials with specific stimulus directions.

    def dir_trials(self, direc, stims=['S1', 'S2'], offsets=[0*deg],
                   comb_trs=False):
        """Return trials with some direction +- offset during S1 and/or S2."""

        # Init list of directions.
        direcs = [float(direction.deg_mod(direc + offset))
                  for offset in offsets]

        # Get trials for direction + each offset value.
        sd_trs = [((stim, d), self.trials_by_features(stim, 'Dir', [d]).loc[d])
                  for d in direcs for stim in stims]
        sd_trs = util.series_from_tuple_list(sd_trs)

        # Combine values across trials.
        if comb_trs:
            sd_trs = util.combine_lists(sd_trs)

        return sd_trs

    def dir_pref_trials(self, pref_of, **kwargs):
        """Return trials with preferred direction."""

        pdir = self.pref_dir(pref_of)
        trs = self.dir_trials(pdir, **kwargs)

        return trs

    def dir_anti_trials(self, anti_of, **kwargs):
        """Return trials with anti-preferred direction."""

        adir = self.anti_pref_dir(anti_of)
        trs = self.dir_trials(adir, **kwargs)

        return trs

    def dir_pref_anti_trials(self, pref_anti_of, **kwargs):
        """Return trials with preferred and antipreferred direction."""

        pref_trials = self.dir_pref_trials(pref_anti_of, **kwargs)
        apref_trials = self.dir_anti_trials(pref_anti_of, **kwargs)
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
            # stims order must be first S2, then S1!
            trs = self.dir_pref_trials(pref_of=pref_of, stims=['S2', 'S1'],
                                       offsets=[offset])

            # S- and D-trials.
            trS[float(offset)] = util.intersect_lists(trs)[0]
            trD[float(offset)] = util.diff_lists(trs)[0]

        # Combine S- and D-trials across offsets.
        trS = util.union_lists(trS, 'S trials')
        trD = util.union_lists(trD, 'D trials')

        trsSD = trS.append(trD)

        return trsSD

    # %% Methods to calculate tuning curves and preferred values of features.

    def get_stim_resp_vals(self, stim, feat, t1s=None, t2s=None,
                           add_latency=True):
        """Return response to different values of stimulus feature."""

        # Init time period (stimulus on).
        if t1s is None and t2s is None:
            t1s, t2s = self.pr_times(stim, add_latency, concat=False)

        # Get response (firing rates) in each trial.
        trs = self.trials_by_features(stim, feat)
        if not len(trs):
            return pd.DataFrame(columns=['vals', 'resp'])
        vals = pd.concat([pd.Series(v, index=tr)
                          for v, tr in trs.iteritems()])
        resp = self._Spikes.rates(self.inc_trials(), t1s, t2s)
        stim_resp = pd.DataFrame({'vals': vals, 'resp': resp})
        stim_resp.resp = util.remove_dim_from_series(stim_resp.resp)

        return stim_resp

    def calc_DS(self, stim, **kwargs):
        """Calculate direction selectivity (DS)."""

        # Get direction response values and stats.
        stim_resp = self.get_stim_resp_vals(stim, 'Dir', **kwargs)
        resp_stats = util.calc_stim_resp_stats(stim_resp)

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
        dirs = stim_resp['vals']
        resp = util.remove_dim_from_series(stim_resp['resp'])
        tPDres, tune_pars, tune_res = direction.tuned_DS(dirs, resp, dir0)

        # Prepare results.
        PD = pd.concat([mPDres, wPDres, tPDres], axis=1,
                       keys=('max', 'weighted', 'tuned'))
        DSI = pd.Series([mDS, wDS], index=['mDS', 'wDS'])

        return resp_stats, PD, DSI, tune_pars, tune_res

    def test_DS(self, stims=['S1', 'S2'], **kwargs):
        """Test unit's direction selectivit."""

        if not self.n_inc_trials():
            return

        # Init field to store DS results.
        lDR, lDSI, lPD, lTP = [], [], [], []
        for stim in stims:

            # Calculate DS during stimulus.
            DR, PD, DSI, tune_pars, tune_res = self.calc_DS(stim, **kwargs)

            # Collect DR values.
            lDR.append(DR)

            # Collect DS params.
            lDSI.append(DSI)
            lPD.append(PD)

            # Collect tuning params.
            tun_fit = tune_pars.loc['fit']
            lTP.append(tun_fit.append(tune_res))

        # Convert each to a DataFrame.
        DR, DSI, PD, TP = [pd.concat(rlist, axis=1, keys=stims).T
                           for rlist in (lDR, lDSI, lPD, lTP)]

        # Save DS results.
        self.DS['DR'] = DR.T
        self.DS['DSI'] = DSI
        self.DS['PD'] = PD
        self.DS['TP'] = TP

    def pref_dir(self, stim='S1', method='weighted', pd_type='cPD'):
        """Return preferred direction."""

        pdir = self.DS['PD'].loc[(stim, method), pd_type]
        return pdir

    def anti_pref_dir(self, stim='S1', method='weighted', pd_type='cAD'):
        """Return anti-preferred direction."""

        adir = self.DS['PD'].loc[(stim, method), pd_type]
        return adir

    # %% Functions to return unit parameters.

    def get_baseline(self):
        """Return baseline firing rate."""

        baseline = (self.QualityMetrics['baseline']
                    if 'baseline' in self.QualityMetrics else None)
        return baseline
