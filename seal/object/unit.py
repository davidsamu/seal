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
from seal.plot import prate, ptuning, putil
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
        """Return 1 if unit is empty, 0 if not empty."""

        im_empty = self.UnitParams['empty']
        return im_empty

    def is_excluded(self):
        """Return 1 if unit is excluded, 0 if included."""

        im_excluded = self.UnitParams['excluded']
        return im_excluded

    def n_inc_trials(self):
        """Return number of included trials."""

        if 'NTrialsInc' in self.QualityMetrics:
            n_inc_trs = self.QualityMetrics['NTrialsInc']
        else:
            n_inc_trs = len(self.TrialParams.index)

        return n_inc_trs

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

    def pvals_in_trials(self, trs=None, pnames=None):
        """
        Return selected stimulus params during given trials.
        pnames is list of (stim, feature) pairs.
        """

        # Defaults.
        if trs is None:  # all trials
            trs = self.inc_trials()
        if pnames is None:  # all stimulus params
            pnames = self.StimParams.columns.values

        pvals = self.StimParams.loc[trs, pnames]

        return pvals

    def trials_by_pvals(self, stim, feat, vals=None, comb_vals=False):
        """Return trials grouped by (selected) values of stimulus param."""

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

    def get_prd_rates(self, trs=None, t1s=None, t2s=None, tr_time_idx=False):
        """Return rates within time periods in given trials."""

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

    # %% Methods to get trials with specific stimulus directions.

    def dir_trials(self, direc, stims=['S1', 'S2'], offsets=[0*deg],
                   comb_trs=False):
        """Return trials with some direction +- offset during S1 and/or S2."""

        # Init list of directions.
        direcs = [float(direction.deg_mod(direc + offset))
                  for offset in offsets]

        # Get trials for direction + each offset value.
        sd_trs = [((stim, d), self.trials_by_pvals(stim, 'Dir', [d]).loc[d])
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
        trs = self.trials_by_pvals(stim, feat)
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

    # %% Plotting methods.

    def plot_DS(self, no_labels=False, ftempl=None, **kwargs):
        """Plot direction selectivity results."""

        if self.is_empty() or not self.n_inc_trials():
            return

        # Test DS if it hasn't been yet.
        if not len(self.DS.index):
            self.test_DS()

        # Set up plot params.
        if no_labels:  # minimise labels on plot
            title = None
            kwargs['labels'] = False
            kwargs['polar_legend'] = True
            kwargs['tuning_legend'] = False
        else:
            title = self.Name

        ffig = None if ftempl is None else ftempl.format(self.name_to_fname())
        ax_polar, ax_tuning = ptuning.plot_DS(self.DS, title=title,
                                              ffig=ffig, **kwargs)
        return ax_polar, ax_tuning

    def prep_rr_plot_params(self, prd, ref, nrate=None, trs=None):
        """Prepare plotting parameters."""

        # Get trial params.
        t1s, t2s = self.pr_times(prd, concat=False)
        ref_ts = self.ev_times(ref)
        if trs is None:
            trs = self.ser_inc_trials()

        # Get spikes.
        spikes = [self._Spikes.get_spikes(tr, t1s, t2s, ref_ts) for tr in trs]

        # Get rates and rate times.
        nrate = self.init_nrate(nrate)
        rates = [self._Rates[nrate].get_rates(tr, t1s, t2s, ref_ts)
                 for tr in trs]

        # Get stimulus periods.
        stim_prd = constants.ev_stim.loc[ref]

        # Get trial set names.
        names = trs.index

        return trs, spikes, rates, stim_prd, names

    def plot_raster(self, prd, ref, nrate=None, trs=None, **kwargs):
        """Plot raster plot of unit for specific trials."""

        if self.is_empty() or not self.n_inc_trials():
            return

        # Set up params.
        plot_params = self.prep_rr_plot_params(prd, ref, nrate, trs)
        trs, spikes, rates, stim_prd, names = plot_params

        # Plot raster.
        ax = prate.raster(spikes[0], prds=[stim_prd], xlab=ref,
                          title=self.Name, **kwargs)

        return ax

    def plot_rate(self, prd, ref, nrate=None, trs=None, **kwargs):
        """Plot rate plot of unit for specific trials."""

        if self.is_empty() or not self.n_inc_trials():
            return

        # Set up params.
        plot_params = self.prep_rr_plot_params(prd, ref, nrate, trs)
        trs, spikes, rates, stim_prd, names = plot_params

        # Plot rate.
        ax = prate.rate(rates, names, prds=[stim_prd], xlab=ref,
                        title=self.Name, **kwargs)

        return ax

    def plot_rr(self, prd, ref, nrate=None, trs=None, no_labels=False,
                rate_kws=dict(), **kwargs):
        """Plot raster and rate plot of unit for selected sets of trials."""

        if self.is_empty() or not self.n_inc_trials():
            return

        # Set up params.
        plot_params = self.prep_rr_plot_params(prd, ref, nrate, trs)
        trs, spikes, rates, stim_prd, names = plot_params

        # Set labels.
        if no_labels:
            title = None
            rate_kws.update({'xlab': None, 'ylab': None, 'add_lgn': False})
        else:
            title = self.Name
            rate_kws['xlab'] = prd

        # Plot raster and rate.
        res = prate.raster_rate(spikes, rates, names, prds=[stim_prd],
                                title=title, rate_kws=rate_kws, **kwargs)
        fig, raster_axs, rate_ax = res

        return fig, raster_axs, rate_ax

    def plot_SR(self, stims, feat=None, vals=None, nrate=None, colors=None,
                add_stim_name=True, fig=None, sps=None, **kwargs):
        """Plot stimulus response (raster and rate) for mutliple stimuli."""

        if self.is_empty() or not self.n_inc_trials():
            return

        # Init figure and gridspec.
        fig = putil.figure(fig)
        if sps is None:
            sps = putil.gridspec(1, 1)[0]

        # Create a gridspec for each stimulus.
        wratio = [float(dur) for dur in stims.dur]
        stim_rr_gsp = putil.embed_gsp(sps, 1, len(stims.index),
                                      width_ratios=wratio, wspace=0.1)

        # Init params.
        if colors is None:
            # One trial set: stimulus specific color.
            if vals is None or len(vals) == 1:
                colors = pd.DataFrame(putil.stim_colors[stims.index])
            else:  # more than one trial set: value specific color
                colcyc = putil.get_colors()
                col_vec = [next(colcyc) for i in range(len(vals))]
                colors = pd.DataFrame(np.tile(col_vec, (len(stims.index), 1)),
                                      index=stims.index)

        axes_raster, axes_rate = [], []
        for i, stim in enumerate(stims.index):

            s_rr_gsp = stim_rr_gsp[i]
            cols = colors.loc[stim]

            # Prepare plot params.
            if feat is not None and vals is not None:
                trs = self.trials_by_pvals(stim, feat, vals)
            else:
                trs = self.ser_inc_trials()

            rr_gsp = putil.embed_gsp(s_rr_gsp, 2, 1)

            # Plot response on raster and rate plots.
            prd, ref = stims.loc[stim, 'prd'], stim + ' on'
            res = self.plot_rr(prd, ref, nrate, trs, cols=cols,
                               fig=fig, gsp=rr_gsp, **kwargs)
            _, raster_axs, rate_ax = res

            # Add stimulus name to rate plot.
            if add_stim_name:
                color = (putil.stim_colors[stim]
                         if vals is None or len(vals) == 1 else 'k')
                rate_ax.text(0.02, 0.95, stim, fontsize=10, color=color,
                             va='top', ha='left',
                             transform=rate_ax.transAxes)

            axes_raster.extend(raster_axs)
            axes_rate.append(rate_ax)

        # Format rate plots.
        for ax in axes_rate[1:]:  # second and later stimuli
            putil.set_spines(ax, bottom=True, left=False)
            putil.hide_ticks(ax, show_x_ticks=True, show_y_ticks=False)

        # Match scale of y axes.
        putil.sync_axes(axes_rate, sync_y=True)

        return axes_raster, axes_rate

    def plot_DR(self, nrate=None, fig=None, sps=None):
        """Plot 3x3 direction response plot, with polar plot in center."""

        if self.is_empty() or not self.n_inc_trials():
            return

        # Set up stimulus parameters.
        stims = pd.DataFrame(index=constants.stim_dur.index)
        stims['prd'] = ['around ' + stim for stim in stims.index]
        stims['dur'] = [self.pr_dur(stims.loc[stim, 'prd'])
                        for stim in stims.index]

        # Init figure and gridspec.
        fig = putil.figure(fig)
        if sps is None:
            sps = putil.gridspec(1, 1)[0]
        gsp = putil.embed_gsp(sps, 3, 3)  # inner gsp with subplots

        # Polar plot.
        ax_polar = fig.add_subplot(gsp[4], polar=True)
        for stim in stims.index:  # for each stimuli
            stim_resp = self.get_stim_resp_vals(stim, 'Dir')
            resp_stats = util.calc_stim_resp_stats(stim_resp)
            dirs, resp = np.array(resp_stats.index) * deg, resp_stats['mean']
            c = putil.stim_colors[stim]
            ptuning.plot_DR(dirs, resp, color=c, ax=ax_polar)
        putil.hide_ticks(ax_polar, 'y')

        # Raster-rate plots.
        rr_pos = [5, 2, 1, 0, 3, 6, 7, 8]  # Position of each direction.
        rr_dir_pos = pd.Series(constants.all_dirs, index=rr_pos)

        rate_axs = []
        for isp, d in rr_dir_pos.iteritems():

            # Prepare plot formatting.
            first_dir = (isp == 0)

            # Plot stimulus response across stimuli.
            res = self.plot_SR(stims, 'Dir', [d], nrate, None, first_dir,
                               fig, gsp[isp], no_labels=True)
            draster_axs, drate_axs = res

            # Remove axis ticks.
            for i, ax in enumerate(drate_axs):
                first_stim = (i == 0)
                show_x_ticks = first_dir
                show_y_ticks = first_dir & first_stim
                putil.hide_ticks(ax, show_x_ticks, show_y_ticks)

            # Add task name as title (to top center axes).
            if isp == 1:
                putil.set_labels(draster_axs[0], title=self.get_task(),
                                 ytitle=1.10, title_kws={'loc': 'right'})

            rate_axs.extend(drate_axs)

        # Match scale of y axes.
        putil.sync_axes(rate_axs, sync_y=True)

        return ax_polar, rate_axs

    def plot_rate_DS(self, nrate=None, fig=None, sps=None):
        """Plot rate and direction selectivity summary plot."""

        if self.is_empty() or not self.n_inc_trials():
            return

        # Set up stimulus parameters.
        stims = pd.DataFrame(index=constants.stim_dur.index)
        stims['prd'] = ['around ' + stim for stim in stims.index]
        stims['dur'] = [self.pr_dur(stims.loc[stim, 'prd'])
                        for stim in stims.index]

        # Init figure and gridspec.
        fig = putil.figure(fig)
        if sps is None:
            sps = putil.gridspec(1, 1)[0]

        gsp = putil.embed_gsp(sps, 4, 1, height_ratios=[0.25, 1, 1, 1])
        info_sps, all_rr_sps, ds_sps, pa_rr_sps = [g for g in gsp]
        rate_axs = []

        # Info header.
        gsp_info = putil.embed_gsp(info_sps, 1, 1)
        info_ax = fig.add_subplot(gsp_info[0, 0])
        putil.unit_info(self, ax=info_ax)

        # Raster & rate over all trials.
        res = self.plot_SR(stims, nrate=nrate, fig=fig, sps=all_rr_sps,
                           no_labels=True)
        sraster_axs, srate_axs = res
        rate_axs.extend(srate_axs)

        # Direction tuning.
        ds_gsp = putil.embed_gsp(ds_sps, 1, 2)
        ax_polar, ax_tuning = self.plot_DS(no_labels=True, fig=fig, gsp=ds_gsp)

        # Raster & rate in pref and anti trials.
        stim = stims.index[0]
        pa_dir = [self.pref_dir(stim), self.anti_pref_dir(stim)]
        res = self.plot_SR(stims, 'Dir', pa_dir, nrate, None, True, fig,
                           pa_rr_sps, no_labels=True)
        draster_axs, drate_axs = res
        rate_axs.extend(drate_axs)

        # Match scale of y axes.
        putil.sync_axes(rate_axs, sync_y=True)
        [putil.move_signif_lines(ax) for ax in rate_axs]

        return rate_axs, ax_polar, ax_tuning
