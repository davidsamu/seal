#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to calculate quality metrics of units  after spike sorting
(SNR, ISIvr, etc), and to exclude trials / units not meeting QC criteria.

@author: David Samu
"""


import numpy as np
import scipy as sp
import pandas as pd
from quantities import s, ms

import elephant

from seal.analysis import stats
from seal.util import util, constants


# %% Constants.

# Recording constants.
VMIN = -2048                   # minimum voltage gain of recording
VMAX =  2047                   # maximum voltage gain of recording
CENSORED_PRD_LEN = 0.675*ms    # length of censored period
WF_T_START = 9                 # start index of spikes (aligned by Plexon)

# Constants related to quality metrics calculation.
ISI_TH = 1.0*ms      # ISI violation threshold
NTR_STEPS = 5        # number of trials to step by when detecting signal drift
NTR_WINDOW = 20      # number of trials to average when detecting signal drift

# Quality control thresholds per brain region.
# Minimum window length and firing rate of task related activity.
QC_THs = pd.DataFrame.from_items([('MT', [200*ms, 10]),
                                  ('PFC', [100*ms, 5]),
                                  ('MT/PFC', [100*ms, 5])], orient='index',
                                 columns=['wndw_len', 'minFR'])

# Constants related to unit exclusion.
min_SNR = 1.0           # min. SNR
max_ISIvr = 1.5         # max. ISI violation ratio (%)
min_n_trs = 50          # min. number of trials (in case subject quit)
min_inc_trs_ratio = 50  # min. ratio of included trials out of all (%)


# %% Core methods.

def get_start_stop_times(spk_times, tr_starts, tr_stops):
    """Return start and stop times of recording."""

    tfirst_spk, tlast_spk = spk_times.min()*s, spk_times.max()*s
    tfirst_trl, tlast_trl = tr_starts.min(), tr_stops.max()
    t_start, t_stop = min(tfirst_spk, tfirst_trl), max(tlast_spk, tlast_trl)

    return t_start, t_stop


def has_signal_difted(r1, r2):
    """Test whether firing rate difference out of tolerable range."""

    # Params to set max tolerable drift.
    rlow, dlow = 0.1, 1000   # at 0.5 sp/s: 500%
    rhigh, dhigh = 50, 125   # at 50 sp/s: 150%
    # With exponential decay between them.
    mr = np.min([np.max([np.mean([r1, r2]), rlow]), rhigh])
    rr = (np.log(mr)-np.log(rlow)) / (np.log(rhigh)-np.log(rlow))
    max_ratio = (dlow-dhigh) * (1-rr)**1.5 + dhigh

    rmin, rmax = np.min([r1, r2]), np.max([r1, r2])
    has_drifted = (rmax/np.max([rmin,rlow])) > (max_ratio/100)

    return has_drifted


def calc_waveform_stats(waveforms):
    """Calculate waveform duration and amplitude."""

    # Init.
    wfs = np.array(waveforms)
    minV, maxV = wfs.min(), wfs.max()

    # Spline fit and interpolation parameters.
    step = 1  # interpolation step in microseconds
    k = 3     # spline degree
    smoothing_fac = 0     # smoothing factor, 0: no smoothing

    # Is waveform set truncated?
    is_truncated = np.sum(wfs == minV) > 1 or np.sum(wfs == maxV) > 1

    # Init waveform data and time vector.
    x = np.array(waveforms.columns)

    def calc_wf_stats(x, y):

        # Remove truncated data points.
        ivalid = (y != minV) & (y != maxV)
        xv, yv = x[ivalid], y[ivalid]

        # Check that enough data points remaining for fitting.
        if len(xv) <= k or sum(ivalid[WF_T_START:]) < 2:
            return np.nan, np.nan, True, len(xv)

        # Fit cubic spline and get fitted y values.
        tck = sp.interpolate.splrep(xv, yv, s=smoothing_fac, k=k)
        xfit = np.arange(x[WF_T_START-2], xv[-1], step)
        yfit = sp.interpolate.splev(xfit, tck)

        # Index of first local minimum and the first following local maximum.
        # If no local min/max is found, then get global min/max.
        imins = sp.signal.argrelextrema(yfit, np.less)[0]
        imin = imins[0] if len(imins) else np.argmin(yfit)
        imaxs = sp.signal.argrelextrema(yfit[imin:], np.greater)[0]
        imax = (imaxs[0] if len(imaxs) else np.argmax(yfit[imin:])) + imin

        # Calculate waveform duration, amplitude and # of valid samples.
        dur = xfit[imax] - xfit[imin] if imin < imax else np.nan
        amp = yfit[imax] - yfit[imin] if imin < imax else np.nan
        truncated = (len(x) != len(xv))
        nvalid = len(xv)

        return dur, amp, truncated, nvalid

    # Calculate duration, amplitude and number of valid samples."""
    res = [calc_wf_stats(x, wfs[i, :]) for i in range(wfs.shape[0])]
    wfstats = pd.DataFrame(res, columns=['duration', 'amplitude',
                                         'truncated', 'nvalid'])

    for i in range(wfs.shape[0]):
        calc_wf_stats(x, wfs[i, :])

    return wfstats, is_truncated, minV, maxV


def isi_stats(spk_times):
    """Returns ISIs and some related statistics."""

    # No spike: ISI v.r. and TrueSpikes no calculable.
    if not spk_times.size:
        return np.nan, np.nan

    # Only one spike: ISI v.r. is 0%, TrueSpikes is 100%.
    if spk_times.size == 1:
        return 0, 100

    isi = elephant.statistics.isi(spk_times).rescale(ms)

    # Percent of spikes violating ISI treshold.
    n_ISI_vr = sum(isi < ISI_TH)
    percent_ISI_vr = 100 * n_ISI_vr / isi.size

    # Percent of spikes estimated to originate from the sorted single unit.
    # Based on refractory period violations.
    # See Hill et al., 2011: Quality Metrics to Accompany Spike Sorting of
    # Extracellular Signals
    N = spk_times.size
    T = (spk_times.max() - spk_times.min()).rescale(ms)
    r = n_ISI_vr
    tmax = ISI_TH
    tmin = CENSORED_PRD_LEN
    tdif = tmax - tmin

    det = 1/4 - float(r*T / (2*tdif*N**2))  # determinant
    true_spikes = 100*(1/2 + np.sqrt(det)) if det >= 0 else np.nan

    return percent_ISI_vr, true_spikes


def calc_snr(waveforms):
    """
    Calculate signal to noise ratio (SNR) of waveforms.

    SNR: std of mean waveform divided by std of residual waveform (noise).
    """

    # Handling extreme case of only a single spike in Unit.
    if waveforms.shape[0] < 2:
        return np.nan

    # Mean, residual and the ratio of their std.
    wf_mean = waveforms.mean()
    wf_res = waveforms - wf_mean
    snr = wf_mean.std() / np.array(wf_res).std()

    return snr


def test_drift(u):
    """Test drift (gradual or abrupt) in baseline activity of unit."""

    # Baseline rate averaged over every n consecutive trials.
    trs = list(u.TrData.index)
    tr_len = float(constants.fixed_tr_len.rescale(s))
    bs_rate = u.get_prd_rates('fixation', trs=trs, add_latency=False,
                              tr_time_idx=True)

    # Get trial indices to average over.
    itrs = []
    for i in range(int(len(bs_rate) / NTR_STEPS)):
        cntr = i * NTR_STEPS
        hl = int(NTR_WINDOW / 2)
        idxs = list(range(cntr-hl, cntr+hl))
        if min(idxs) >= 0 and max(idxs) <= max(trs):
            itrs.append(idxs)
    if not len(itrs):  # in case there's not enough trials for a single window
        itrs = [list(range(len(bs_rate)))]
    else:
        itrs[-1] = list(range(itrs[-1][0], trs[-1]+1))  # add modulo trials

    # Get baseline activity stats in each window.
    bs_stats = pd.DataFrame()
    bs_stats['trials'] = itrs
    bs_stats['tstart'] = [bs_rate.index[idx[0]] for idx in itrs]
    bs_stats['tmean'] = [np.mean(bs_rate.index[idx]) for idx in itrs]
    bs_stats['tstop'] = [bs_rate.index[idx[-1]]+tr_len for idx in itrs]
    bs_stats['rate'] = [np.mean(bs_rate.iloc[idx]) for idx in itrs]

    # Adjust start and end times of session.
    spk_times = u.SpikeParams['time']
    t_start, t_stop = get_start_stop_times(spk_times, bs_rate.index,
                                           bs_rate.index+tr_len)
    bs_stats.loc[0,'tstart'] = t_start
    bs_stats.loc[bs_stats.index[-1],'tstop'] = t_stop

    # Find period within acceptable drift range for each bin.
    res = []
    for i in bs_stats.index:
        rmin = rmax = bs_stats.loc[i, 'rate']
        for j in bs_stats.index[i:]:
            r = bs_stats.loc[j, 'rate']
            # Update extreme values.
            rmin = min(rmin, r)
            rmax = max(rmax, r)
            # If difference becomes unacceptable, terminate period.
            if has_signal_difted(rmin, rmax):
                j -= 1
                break
        # Collect results.
        tstart = bs_stats.loc[i, 'tstart']
        tstop = bs_stats.loc[j, 'tstop']
        first_tr = bs_stats.loc[i, 'trials'][0]
        last_tr = bs_stats.loc[j, 'trials'][-1]
        ntrs = last_tr - first_tr + 1
        res.append([i, j, tstart, tstop, first_tr, last_tr, ntrs])
    cols = ['istart', 'istop', 'tstart', 'tstop',
            'first_tr', 'last_tr', 'ntrs']
    prd_res = pd.DataFrame.from_records(res, columns=cols)

    # Get params of longest stable period.
    stab_prd_res = prd_res.loc[prd_res.ntrs.argmax()]

    # Return included trials and spikes.
    prd_inc = util.indices_in_window(bs_stats.index, stab_prd_res.istart,
                                     stab_prd_res.istop)
    tstart, tstop = stab_prd_res[['tstart', 'tstop']]
    tr_inc = ((bs_rate.index >= tstart) & (bs_rate.index <= tstop))
    spk_inc = util.indices_in_window(spk_times, tstart, tstop)

    return bs_stats, stab_prd_res, prd_inc, tr_inc, spk_inc


def is_isolated(snr, true_spikes):
    """Classify unit as single or multi-unit."""

    if true_spikes >= 90 and snr >= 2.0:
        isolation = 'single unit'
    else:
        isolation = 'multi unit'

    return isolation


def calc_baseline_rate(u):
    """Calculate baseline firing rate of unit."""

    base_rate = u.get_prd_rates('baseline').mean()
    return base_rate


def test_task_relatedness(u, p_th=0.05):
    """Test if unit has any task related activity."""

    # Init.
    nrate = u.init_nrate()
    wndw_len, minFR = QC_THs.loc[u.get_region()]
    if not len(u.inc_trials()):
        return False

    # Get baseline rate per trial.
    baseline = util.remove_dim_from_series(u.get_prd_rates('baseline'))

    # Init periods and trials sets to test.
    feats = ('Dir',)  # ('Dir', 'Loc')
    prds_trs = [('S1', [('S1', 'early delay', 'late delay'), feats]),
                ('S2', [('S2', 'post-S2'), feats])]
    prds_trs = pd.DataFrame.from_items(prds_trs, orient='index',
                                       columns=['prds', 'trpars'])

    # Go through each stimulus, period and trial parameter to be tested.
    pval = []
    mean_rate = []
    for stim, (prds, trpars) in prds_trs.iterrows():

        for prd in prds:
            t1s, t2s = u.pr_times(prd, add_latency=False, concat=False)

            for par in trpars:
                ptrs = u.trials_by_param((stim, par))

                for vpar, trs in ptrs.iteritems():

                    # Get rates during period on trials with given param value.
                    rates = u._Rates[nrate].get_rates(trs, t1s, t2s)
                    bs_rates = baseline[trs]

                    # No rates available.
                    if rates.empty:
                        continue

                    # Get sub-period around time with maximal rate.
                    tmax = rates.mean().argmax()
                    tmin, tmax = rates.columns.min(), rates.columns.max()
                    tstart, tend = stats.prd_in_window(tmax, tmin, tmax,
                                                       wndw_len, ms)
                    tidx = (rates.columns >= tstart) & (rates.columns <= tend)

                    # Test difference from baseline rate.
                    wnd_rates = rates.loc[:, tidx].mean(1)
                    stat, p = stats.mann_whithney_u_test(wnd_rates, bs_rates)
                    pval.append(((stim, prd, par, str(vpar)), p))

                    # Mean rate.
                    mrate = rates.mean().mean()
                    mean_rate.append(((stim, prd, par, str(vpar)), mrate))

    # Format results.
    names = ['stim', 'prd', 'par', 'vpar']
    pval, mean_rate = [util.series_from_tuple_list(res, names)
                       for res in (pval, mean_rate)]

    # Save results to unit.
    u.PrdParTests = pd.concat([mean_rate, pval], axis=1,
                              keys=['mean_rate', 'pval'])
    u.PrdParTests['sign'] = u.PrdParTests['pval'] < p_th

    # Save test parameters.
    u.PrdParTests.test = 'mann_whithney_u_test'
    u.PrdParTests.p_th = p_th

    # Is there any task- (stimulus-parameter-) related period?
    has_min_rate = (u.PrdParTests.mean_rate >= minFR).any()
    is_task_related = u.PrdParTests.sign.any()

    return has_min_rate, is_task_related


# %% Calculate quality metrics, and find trials and units to be excluded.

def test_qm(u, include=None, first_tr=None, last_tr=None):
    """
    Test ISI, SNR and stationarity of FR and spike waveforms.
    Find trials with unacceptable drift.

    Non-stationarities can happen due to e.g.:
    - electrode drift, or
    - change in the state of the neuron.

    Optionally, user can provide trials to be selected and whether to include
    unit.
    """

    if u.is_empty():
        return

    # Init values.
    waveforms = u.Waveforms
    spk_times = u.SpikeParams['time']

    # Calculate waveform statistics of each spike.
    wf_stats, is_truncated, minV, maxV = calc_waveform_stats(waveforms)
    u.SpikeParams['duration'] = wf_stats['duration']
    u.SpikeParams['amplitude'] = wf_stats['amplitude']
    u.SpikeParams['truncated'] = wf_stats['truncated']
    u.SpikeParams['nvalid'] = wf_stats['nvalid']
    u.UnitParams['truncated'] = is_truncated
    u.SessParams['minV'] = min(minV, VMIN)
    u.SessParams['maxV'] = max(maxV, VMAX)

    # Trial exclusion.
    # TODO: implement manual trial selection.
    # if (first_tr is not None) and (last_tr is not None):
        # Use passed parameters.
        # res = set_inc_trials(u, first_tr, last_tr)
    # else:  # Automatic trial selection.

    # Test drifts and reject trials if necessary.
    bs_stats, stab_prd_res, prd_inc, tr_inc, spk_inc = test_drift(u)

    u.update_included_trials(tr_inc)

    # SNR.
    snr = calc_snr(waveforms[spk_inc])

    # ISI statistics.
    ISIvr, true_spikes = isi_stats(np.array(spk_times[spk_inc])*s)
    isolation = is_isolated(snr, true_spikes)

    # Minimum firing rate and task-related activity.
    has_min_rate, is_task_related = test_task_relatedness(u)

    # Add quality metrics to unit.
    u.QualityMetrics['SNR'] = snr
    u.QualityMetrics['mWfDur'] = u.SpikeParams.duration[spk_inc].mean()
    u.QualityMetrics['ISIvr'] = ISIvr
    u.QualityMetrics['TrueSpikes'] = true_spikes
    u.QualityMetrics['isolation'] = isolation
    u.QualityMetrics['baseline'] = calc_baseline_rate(u)
    u.QualityMetrics['has_min_rate'] = has_min_rate
    u.QualityMetrics['task_related'] = is_task_related

    # Run unit exclusion test.
    QC_tests = test_rejection(u)
    if include is None:
        include = QC_tests['include']
    u.set_excluded(not include)

    # Return all results (for plotting).
    res = {'bs_stats': bs_stats, 'stab_prd_res': stab_prd_res,
           'prd_inc': prd_inc, 'tr_inc': tr_inc, 'spk_inc': spk_inc,
           'QC_tests': QC_tests}

    return res


def test_rejection(u):
    """Check whether unit is to be rejected from analysis."""

    qm = u.QualityMetrics
    QC_tests = pd.Series()

    # Insufficient receptive field coverage.
    # th_passed.append(qm['RC_coverage'] < min_RF_coverage)

    # Extremely low waveform consistency (SNR).
    QC_tests['SNR'] = qm['SNR'] > min_SNR

    # Extremely high ISI violation ratio (ISIvr).
    QC_tests['ISI'] = qm['ISIvr'] < max_ISIvr

    # Insufficient total number of trials (subject quit).
    QC_tests['NTotalTrs'] = qm['NTrialsTotal'] > min_n_trs

    # Insufficient amount of included trials.
    inc_trs_ratio = 100 * qm['NTrialsInc'] / qm['NTrialsTotal']
    QC_tests['IncTrsRatio'] = inc_trs_ratio > min_inc_trs_ratio

    # Extremely low unit activity (FR).
    QC_tests['has_min_rate'] = qm['has_min_rate']

    # Not task-related.
    QC_tests['task_related'] = qm['task_related']

    # Include unit if all criteria met.
    QC_tests['include'] = QC_tests.all()

    return QC_tests
