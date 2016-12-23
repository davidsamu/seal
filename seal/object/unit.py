# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 14:06:14 2016

Class representing a (spike sorted) unit (single or multi).

@author: David Samu
"""

import warnings

from datetime import datetime as dt
from itertools import product
from collections import OrderedDict as OrdDict

import numpy as np
import pandas as pd
from quantities import s, ms, us, deg, Hz
from neo import SpikeTrain

from seal.util import util
from seal.plot import prate, ptuning
from seal.object import constants
from seal.object.rate import Rate
from seal.object.spikes import Spikes
from seal.object.trials import Trials
from seal.analysis import tuning


class Unit:
    """Generic class to store data of a unit (neuron or group of neurons)."""

    # %% Constructor
    def __init__(self, TPLCell=None, kernels=None, step=None, stim_params=None,
                 answ_params=None, taskname=None, region=None):
        """Create Unit instance from TPLCell data structure."""

        # Create empty instance.
        self.Name = ''
        self.region = region
        self.SessParams = pd.Series()
        self.Waveforms = pd.Series()
        self.StimParams = pd.DataFrame()
        self.Answer = pd.DataFrame()
        self.Events = pd.DataFrame()
        self.TrialParams = pd.DataFrame()
        self.Spikes = Spikes([])
        self.Rates = pd.Series()
        self.QualityMetrics = pd.Series()
        self.DS = pd.Series()

        self.is_empty = True
        self.is_excluded = True

        # Return if no TPLCell is passed.
        if TPLCell is None:
            return

        # %% Session parameters.

        # Prepare session params.
        subj, date, probe, exp, isort = util.params_from_fname(TPLCell.File)
        task, task_idx = exp[:-1], int(exp[-1])
        task = task if taskname is None else taskname  # use provided name
        [chan, un] = TPLCell.ChanUnit
        sampl_per = (1 / (TPLCell.Info.Frequency * Hz)).rescale(us)
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
                   ('sampl_prd', sampl_per)]
        self.SessParams = util.series_from_tuple_list(sp_list)

        # Name unit.
        self.set_name()

        # %% Waveform data.

        # Prepare waveform data.
        wfs = TPLCell.Waves
        if wfs.ndim == 1:  # there is only a single spike: extend it to matrix
            wfs = np.reshape(wfs, (1, len(wfs)))
        wf_sampl_t = range(wfs.shape[1]) * self.SessParams['sampl_prd']

        t_min = np.min(TPLCell.Spikes)
        t_max = np.max(TPLCell.Spikes)
        sp_times = SpikeTrain(TPLCell.Spikes*s, t_start=t_min, t_stop=t_max)

        # Assign waveform data.
        wf_data = [('tWF', wf_sampl_t),
                   ('WF', wfs),
                   ('SpkDur', util.fill_dim(TPLCell.Spikes_dur * s)),
                   ('mSpkDur', TPLCell.MeanSpikeDur * s),
                   ('tSpk', util.fill_dim(sp_times))]
        self.Waveforms = util.series_from_tuple_list(wf_data)

        # %% Stimulus parameters.

        # Extract all trial parameters.
        trpars = pd.DataFrame(TPLCell.TrialParams, columns=TPLCell.Header)

        # Process stimulus parameters.
        self.StimParams = pd.DataFrame(columns=stim_params.index,
                                       index=trpars.index)
        for sf, (pname, dim) in stim_params.iterrows():
            self.StimParams[sf] = util.add_dim_to_series(trpars[pname], dim)

        # %% Subject answer parameters.

        # Recode correct/incorrect answer column.
        corr_ans = trpars[answ_params.loc['AnswCorr', 'name']]
        if len(corr_ans.unique()) > 2:
            warnings.warn(('More than 2 unique values in AnswCorr: ' +
                           corr_ans.unique()))
        corr_ans = corr_ans == corr_ans.max()  # higher value is correct!
        self.Answer['AnswCorr'] = corr_ans

        # Add column for subject response (saccade direction).
        same_dir = self.StimParams['S1', 'Dir'] == self.StimParams['S2', 'Dir']
        self.Answer['SaccadeDir'] = ((same_dir & corr_ans) |
                                     (~same_dir & ~corr_ans))

        # %% Trial events.

        # Timestamps of events. Only S1 offset and S2 onset are reliable!
        # S1 onset and S2 offset are fixed to these two.
        # Altogether these four are called anchor events.

        # Watch out: indexing starting with 1 in TPLCell (Matlab)!
        # Everything is in seconds below!

        S1dur, S2dur = [float(l.rescale(s)) for l in (constants.S1_dur,
                                                      constants.S2_dur)]
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
                for evt, (rel, offset) in constants.tr_evt.iterrows()]
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

        # %% Spikes and rates.

        # Trials spikes, aligned to S1 onset.
        spk_trains = [(spk_train - abs_S1_onset[i]) * s
                      for i, spk_train in enumerate(TPLCell.TrialSpikes)]
        t_starts = self.ev_times('fixate')
        t_stops = self.ev_times('saccade')
        self.Spikes = Spikes(spk_trains, t_starts, t_stops)

        # Estimate firing rate per trial.
        if step is None:
            step = constants.step
        spikes = self.Spikes.get_spikes()
        rate_list = [Rate(name, kernel, spikes, step)
                     for name, kernel in kernels.items()]
        self.Rates = pd.Series(rate_list, index=kernels.keys())

        self.is_empty = False
        self.is_excluded = False

    # %% Utility methods.

    def set_excluded(self, to_excl):
        """Set unit's exclude flag."""

        self.is_excluded = to_excl

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
        upars['excluded'] = self.is_excluded

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

        # Update included trials.
        tr_exc = np.invert(tr_inc)
        self.QualityMetrics['NTrialsTotal'] = len(self.TrialParams.index)
        self.QualityMetrics['NTrialsInc'] = np.sum(tr_inc)
        self.QualityMetrics['NTrialsExc'] = np.sum(tr_exc)
        self.QualityMetrics['IncTrials'] = Trials(tr_inc, 'included trials')
        self.QualityMetrics['ExcTrials'] = Trials(tr_exc, 'excluded trials')

        # Update included spikes.
        t1 = self.TrialParams.TrialStart[tr_inc].min()
        t2 = self.TrialParams.TrialStop[tr_inc].max()
        spk_inc = util.indices_in_window(self.Waveforms['tSpk'], t1, t2)
        self.QualityMetrics['IncSpikes'] = spk_inc

    def get_trial_params(self, trs=None, t1s=None, t2s=None):
        """Return default values of some common parameters."""

        if trs is None:
            trs = [self.all_trials()]  # default: all (included) trials
        if t1s is None:
            t1s = self.timing('fixate')   # default: start of fixation
        if t2s is None:
            t2s = self.timing('saccade')  # default: saccade time

        return trs, t1s, t2s

    def pref_dir(self, stim='S1', method='weighted', pd_type='cPD'):
        """Return preferred direction."""

        pdir = self.DS['PD'].loc[(stim, method), pd_type]
        return pdir

    def anti_pref_dir(self, stim='S1', method='weighted', pd_type='cAD'):
        """Return anti-preferred direction."""

        adir = self.DS['PD'].loc[(stim, method), pd_type]
        return adir

    def init_nrate(self, nrate=None):
        """Initialize rate name."""

        def_nrate = constants.def_nrate

        if nrate is None:
            nrate = (def_nrate if def_nrate in self.Rates
                     else self.Rates.index[0])

        elif nrate not in self.Rates:
            warnings.warn('Rate name: ' + str(nrate) + ' not found in unit.')
            self.init_nrate()  # return default (or first available) rate name

        return nrate


    # %% Methods to get times of trial events and periods.

    def ev_times(self, evname):
        """Return timing of events across trials."""

        evt = self.Events[evname]
        return evt


    def pr_times(self, prname):
        """Return timing of period (start event, stop event) across trials."""

        ev1, ev2 = constants.tr_prd.loc[prname]
        evt1, evt2 = self.ev_times(ev1), self.ev_times(ev2)
        prt = pd.concat([evt1, evt2], axis=1)

        return prt

    # %% Generic methods to get various set of trials.

    def included_trials(self):
        """Return included trials (i.e. not rejected after quality test)."""

        if 'IncTrials' in self.QualityMetrics:
            included_trials = self.QualityMetrics['IncTrials']
        else:
            included_trials = self.all_trials(filtered=False)
        return included_trials

    def filter_trials(self, trs):
        """Filter trials by excluding rejected ones."""

        tr_idxs = np.logical_and(trs.trials, self.included_trials().trials)
        filtered_trials = Trials(tr_idxs, trs.value, trs.name)
        return filtered_trials

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

    def get_prd_rates(self, trs=None, t1s=None, t2s=None,
                      index_by_tr_time=False):
        """Return rates within time periods in given trials."""

        if self.is_empty:
            return None

        # Init trials.
        if trs is None:
            trs = self.included_trials().trials

        # Get rates.
        rates = self.Spikes.rates(trs, t1s, t2s)

        # Change index from trial index to trials start times.
        if index_by_tr_time:
            tr_time = self.TrialParams['TrialStart'][trs]
            rates.index = util.remove_dim_from_series(tr_time)

        return rates

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

    def calc_response_stats(self, pname, t1s, t2s):
        """Calculate mean response to different values of trial parameter."""

        # Get trials for each parameter value.
        trs = self.trials_by_param_values(pname)

        # Calculate spike count and stats for each value of parameter.
        par_vals = [float(tr.value) for tr in trs]
        sp_stats = pd.DataFrame([self.Spikes.spike_rate_stats(tr, t1s, t2s)
                                 for tr in trs], index=par_vals)

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

        pd_idx = ['PD', 'cPD', 'AD', 'cAD']

        # Get response stats to each direction.
        resp_stats = self.calc_dir_response(stim, t1, t2)
        dirs = np.array(resp_stats.index) * deg
        meanFR, stdFR, semFR = [util.dim_series_to_array(resp_stats[stat])
                                for stat in ('mean', 'std', 'sem')]

        # DS based on maximum rate only (legacy method).
        mPD = dirs[np.argmax(meanFR)]
        mAD = util.deg_mod(mPD+180*deg)
        cmPD, cmAD = mPD, mAD
        mPR, mAR = [meanFR[np.where(dirs == d)[0]] for d in (mPD, mAD)]
        mDS = float(util.modulation_index(mPR, mAR)) if mAR.size else np.nan

        mPDres = pd.Series([mPD, mAD, cmPD, cmAD], pd_idx)

        # DS based on weighted sum of all rates & directions.
        wDS, wPD, cwPD = util.deg_w_mean(dirs, meanFR, constants.all_dirs)
        wAD, cwAD = [util.deg_mod(d+180*deg) for d in (wPD, cwPD)]

        wPDres = pd.Series([wPD, cwPD, wAD, cwAD], pd_idx)

        # Calculate parameters of Gaussian tuning curve.
        # Start by centering stimulus - response.
        tun_res = tuning.center_pref_dir(dirs, wPD, meanFR, semFR)
        dirs_cntr, meanFR_cntr, semFR_cntr = tun_res
        # Fit Gaussian tuning curve to stimulus - response.
        fit_params, fit_res = tuning.fit_gaus_curve(dirs_cntr, meanFR_cntr,
                                                    semFR_cntr)

        # DS based on Gaussian tuning curve fit.
        tPD = wPD + fit_params.loc['fit', 'x0']
        ctPD = util.coarse_dir(tPD, constants.all_dirs)
        tAD, ctAD = [util.deg_mod(d+180*deg) for d in (tPD, ctPD)]

        tPDres = pd.Series([tPD, ctPD, tAD, ctAD], pd_idx)

        PD = pd.concat([mPDres, wPDres, tPDres], axis=1,
                       keys=('max', 'weighted', 'tuned'))
        DSI = pd.Series([mDS, wDS], index=['mDS', 'wDS'])

        # Prepare results.
        res = {'dirs': dirs, 'meanFR': meanFR, 'stdFR': stdFR, 'semFR': semFR,
               'dirs_cntr': dirs_cntr, 'meanFR_cntr': meanFR_cntr,
               'semFR_cntr': semFR_cntr, 'fit_params': fit_params,
               'fit_res': fit_res, 'PD': PD, 'DSI': DSI}

        return res

    def test_DS(self, stims=['S1', 'S2'], no_labels=False, do_plot=True,
                ftempl=None, **kwargs):
        """
        Test DS of unit by calculating
          - DS index and PD, and
          - parameters of Gaussian tuning curve.
        """

        # Init field to store DS results.
        DSres_plot = {}
        lDSI, lPD, lTP = [], [], []
        for stim in stims:

            res = self.calc_DS(stim, t1=None, t2=None)

            # Generate data points for plotting fitted tuning curve.
            a, b, x0, sigma = res['fit_params'].loc['fit']
            x, y = tuning.gen_fit_curve(tuning.gaus, deg, -180*deg, 180*deg,
                                        a=a, b=b, x0=x0, sigma=sigma)

            # Collect calculated DS results param values.
            lDSI.append(res['DSI'])
            lPD.append(res['PD'])

            # TPs
            lTP.append(res['fit_params'].loc['fit'].append(res['fit_res']))

            # Collect data for plotting.
            DSres_plot[stim] = res
            DSres_plot[stim]['xfit'] = x
            DSres_plot[stim]['yfit'] = y

        # Convert each to a DataFrame.
        DSI, PD, TP = [pd.concat(rlist, axis=1, keys=stims).T
                       for rlist in (lDSI, lPD, lTP)]

        # Save DS results.
        self.DS['DSI'] = DSI
        self.DS['PD'] = PD
        self.DS['TP'] = TP

        # Plot direction selectivity results.
        if do_plot:
            DSres_plot = pd.DataFrame(DSres_plot).T
            title = self.Name

            # Minimise labels on plot.
            if no_labels:
                title = None
                kwargs['labels'] = False
                kwargs['polar_legend'] = True
                kwargs['tuning_legend'] = False

            ffig = (None if ftempl is None
                    else ftempl.format(self.name_to_fname()))
            ptuning.direction_selectivity(DSres_plot, title=title,
                                          ffig=ffig, **kwargs)

    # %% Plotting methods.

    def prep_plot_params(self, nrate, trs, t1s, t2s):
        """Prepare plotting parameters."""

        # Get trial params.
        trs, t1s, t2s = self.get_trial_params(trs, t1s, t2s)
        names = [tr.name for tr in trs]

        # Get spikes.
        spikes = [self.Spikes.get_spikes(tr, t1s, t2s) for tr in trs]

        # Get rates and rate times.
        nrate = self.init_nrate(nrate)
        rates, time = None, None
        if nrate is not None:
            rates = [self.Rates[nrate].get_rates(tr.trials, t1s, t2s)
                     for tr in trs]
            time = self.Rates[nrate].get_sample_times(t1s, t2s)

        return trs, t1s, t2s, spikes, rates, time, names

    def plot_raster(self, nrate=None, trs=None, t1=None, t2=None, **kwargs):
        """Plot raster plot of unit for specific trials."""

        if self.is_empty:
            return

        # Set up params.
        plot_params = self.prep_plot_params(nrate, trs, t1, t2)
        trs, t1, t2, spikes, rates, tvec, names = plot_params
        spikes = spikes[0]
        names = names[0]

        # Plot raster.
        ax = prate.raster(spikes, t1, t2, prds=constants.stim_prds,
                          title=self.Name, **kwargs)

        return ax

    def plot_rate(self, nrate=None, trs=None, t1=None, t2=None, **kwargs):
        """Plot rate plot of unit for specific trials."""

        if self.is_empty:
            return

        # Set up params.
        plot_params = self.prep_plot_params(nrate, trs, t1, t2)
        trs, t1, t2, spikes, rates, tvec, names = plot_params

        # Plot rate.
        ax = prate.rate(rates, tvec, names, t1, t2, prds=constants.stim_prds,
                        title=self.Name, **kwargs)

        return ax

    def plot_raster_rate(self, nrate=None, trs=None, t1=None, t2=None,
                         no_labels=False, rate_kws=dict(), **kwargs):
        """Plot raster and rate plot of unit for specific trials."""

        if self.is_empty:
            return

        # Set up params.
        plot_params = self.prep_plot_params(nrate, trs, t1, t2)
        trs, t1, t2, spikes, rates, tvec, names = plot_params

        # Set labels.
        title = self.Name if not no_labels else None
        if no_labels:
            rate_kws.update({'xlab': None, 'ylab': None, 'add_lgn': False})

        # Plot raster and rate.
        res = prate.raster_rate(spikes, rates, tvec, names, t1, t2,
                                prds=constants.stim_prds, title=title,
                                rate_kws=rate_kws, **kwargs)
        fig, raster_axs, rate_ax = res

        return fig, raster_axs, rate_ax

    def plot_dir_resp(self):
        """Plot response to all directions + polar plot in center."""

        # TODO: to be moved here from quality. Along with RR/DS summary plot!
        pass
