#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to calculate quality metrics of units  after spike sorting
(SNR, ISIvr, etc), and to exclude trials / units not meeting QC criteria.

@author: David Samu
"""

import warnings

import numpy as np
import scipy as sp
import pandas as pd
from quantities import s, ms

import elephant

from seal.analysis import stats
from seal.util import util


# %% Constants.

# Recording constants.
VMIN = -2048                   # minimum voltage gain of recording
VMAX =  2047                   # maximum voltage gain of recording
CENSORED_PRD_LEN = 0.675*ms    # length of censored period
WF_T_START = 9                 # start index of spikes (aligned by Plexon)

# Constants related to quality metrics calculation.
ISI_TH = 1.0*ms               # ISI violation threshold
MAX_DRIFT_PCT = 200           # maximum tolerable drift percentage (max/min FR)
MIN_BIN_LEN = 120*s           # (minimum) window length for firing binned stats

# Quality control thresholds per brain region.
# Minimum window length and firing rate of task related activity.
QC_THs = pd.DataFrame.from_items([('MT', [300*ms, 20]),
                                  ('PFC', [100*ms, 10]),
                                  ('MT/PFC', [100*ms, 10])], orient='index',
                                 columns=['wndw_len', 'minFR'])

# Constants related to unit exclusion.
min_SNR = 1.0           # min. SNR
max_ISIvr = 1.0         # max. ISI violation ratio (%)
min_n_trs = 20          # min. number of trials (in case monkey quit)
min_inc_trs_ratio = 50  # min. ratio of included trials out of all (%)


# %% Utility functions.

def get_start_stop_times(spk_times, tr_starts, tr_stops):
    """Return start and stop times of recording."""

    tfirst_spk, tlast_spk = spk_times.min()*s, spk_times.max()*s
    tfirst_trl, tlast_trl = tr_starts.min(), tr_stops.max()
    t_start, t_stop = min(tfirst_spk, tfirst_trl), max(tlast_spk, tlast_trl)

    return t_start, t_stop


def time_bin_data(spk_times, waveforms, tr_starts, tr_stops):
    """Return time binned data for statistics over session time."""

    t_start, t_stop = get_start_stop_times(spk_times, tr_starts, tr_stops)

    # Time bins and binned waveforms and spike times.
    nbins = max(int(np.floor((t_stop - t_start) / MIN_BIN_LEN)), 1)
    tbin_lims = util.quantity_linspace(t_start, t_stop, nbins+1, s)
    tbins = [(tbin_lims[i], tbin_lims[i+1]) for i in range(len(tbin_lims)-1)]
    tbin_vmid = np.array([np.mean([t1, t2]) for t1, t2 in tbins])*s
    spk_idx_binned = [util.indices_in_window(spk_times, float(t1), float(t2))
                      for t1, t2 in tbins]
    wf_binned = [waveforms[spk_idx] for spk_idx in spk_idx_binned]
    spk_times_binned = [spk_times[spk_idx] for spk_idx in spk_idx_binned]

    return tbins, tbin_vmid, wf_binned, spk_times_binned


# %% Core methods.

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
        truncated = len(x) == len(xv)
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


def test_drift(t, v, tbins, tr_starts, spk_times):
    """Test drift (gradual, or more instantaneous jump or drop) in variable."""

    # Number of trials from beginning of session
    # until start and end of each period.
    tr_starts_arr = util.list_to_quantity(tr_starts)
    n_tr_prd_start = [np.sum(tr_starts_arr < t1) for t1, t2 in tbins]
    n_tr_prd_end = [np.sum(tr_starts_arr < t2) for t1, t2 in tbins]

    # Find period within acceptible drift range for each bin.
    cols = ['prd_start_i', 'prd_end_i', 'n_prd',
            't_start', 't_end', 't_len',
            'tr_start_i', 'tr_end_i', 'n_tr']
    prd_res = pd.DataFrame(index=range(len(v)), columns=cols)
    for i, v1 in enumerate(v):
        vmin, vmax = v1, v1
        for j, v2 in enumerate(v[i:]):
            # Update extreme values.
            vmin = min(vmin, v2)
            vmax = max(vmax, v2)
            # If difference becomes unacceptable, terminate period.
            if vmax > MAX_DRIFT_PCT/100*v2 or v2 > MAX_DRIFT_PCT/100*vmin:
                j -= 1
                break
        end_i = i + j
        prd_res.prd_start_i[i] = i
        prd_res.prd_end_i[i] = end_i
        prd_res.n_prd[i] = j + 1
        prd_res.t_start[i] = tbins[i][0]
        prd_res.t_end[i] = tbins[end_i][1]
        prd_res.t_len[i] = tbins[end_i][1] - tbins[i][0]
        prd_res.tr_start_i[i] = n_tr_prd_start[i]
        prd_res.tr_end_i[i] = n_tr_prd_end[end_i]
        prd_res.n_tr[i] = n_tr_prd_end[end_i] - n_tr_prd_start[i]

    # Find bin with longest period.
    idx = prd_res.n_tr.argmax()
    # Indices of longest period.
    prd1 = prd_res.prd_start_i[idx]
    prd2 = prd_res.prd_end_i[idx]
    # Times of longest period.
    t1_inc = prd_res.t_start[idx]
    t2_inc = prd_res.t_end[idx]
    # Trial indices within longest period.
    first_tr = prd_res.tr_start_i[idx]
    last_tr = prd_res.tr_end_i[idx]

    # Return included trials and spikes.
    prd_inc = util.indices_in_window(np.arange(len(tbins)), prd1, prd2)
    tr_inc = (tr_starts.index >= first_tr) & (tr_starts.index < last_tr)
    spk_inc = util.indices_in_window(spk_times, float(t1_inc), float(t2_inc))

    return t1_inc, t2_inc, prd_inc, tr_inc, spk_inc


def set_inc_trials(first_tr, last_tr, tr_starts, tr_stops, spk_times,
                   tbin_vmid):
    """Set included trials by values provided."""

    # Included trial range.
    tr_inc = (tr_starts.index >= first_tr) & (tr_starts.index < last_tr)

    # Start and stop times of included period.
    tstart, tstop = get_start_stop_times(spk_times, tr_starts, tr_stops)
    t1_inc = tstart if first_tr == 0 else tr_starts[first_tr]
    t2_inc = tstop if last_tr == len(tr_starts) else tr_starts[last_tr]

    # Included time periods.
    prd_inc = (tbin_vmid >= t1_inc) & (tbin_vmid <= t2_inc)

    # Included spikes.
    spk_inc = util.indices_in_window(spk_times, t1_inc, t2_inc)

    return t1_inc, t2_inc, prd_inc, tr_inc, spk_inc


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


def test_task_relatedness(u, p_th=0.01):
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
    has_min_rate = (u.PrdParTests.mean_rate > minFR).any()
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

    Optionally, user can provide whether to include unit and selected trials.
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

    tr_starts, tr_stops = u.TrData.TrialStart, u.TrData.TrialStop

    # Time binned statistics.
    tbinned_stats = time_bin_data(spk_times, waveforms, tr_starts, tr_stops)
    tbins, tbin_vmid, wf_binned, spk_times_binned = tbinned_stats

    rate_t = np.array([spkt.size/(t2-t1).rescale(s)
                       for spkt, (t1, t2) in zip(spk_times_binned, tbins)]) / s

    # Trial exclusion.
    if (first_tr is not None) and (last_tr is not None):
        # Use passed parameters.
        res = set_inc_trials(first_tr, last_tr, tr_starts, tr_stops,
                             spk_times, tbin_vmid)
    else:
        # Test drifts and reject trials if necessary.
        res = test_drift(tbin_vmid, rate_t, tbins, tr_starts, spk_times)
    t1_inc, t2_inc, prd_inc, tr_inc, spk_inc = res

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
    if include is None:
        include = test_rejection(u)
    u.set_excluded(not include)

    # Return all results (for plotting).
    res = {'tbin_vmid': tbin_vmid, 'rate_t': rate_t,
           't1_inc': t1_inc, 't2_inc': t2_inc, 'prd_inc': prd_inc,
           'tr_inc': tr_inc, 'spk_inc': spk_inc}
    return res


def test_rejection(u):
    """Check whether unit is to be rejected from analysis."""

    qm = u.QualityMetrics
    test_passed = pd.Series()

    # Insufficient receptive field coverage.
    # th_passed.append(qm['RC_coverage'] < min_RF_coverage)

    # Extremely low waveform consistency (SNR).
    test_passed['SNR'] = qm['SNR'] > min_SNR

    # Extremely high ISI violation ratio (ISIvr).
    test_passed['ISI'] = qm['ISIvr'] < max_ISIvr

    # Insufficient total number of trials (monkey quit).
    test_passed['NTotalTrs'] = qm['NTrialsTotal'] > min_n_trs

    # Insufficient amount of included trials.
    inc_trs_ratio = 100 * qm['NTrialsInc'] / qm['NTrialsTotal']
    test_passed['IncTrsRatio'] = inc_trs_ratio > min_inc_trs_ratio

    # Extremely low unit activity (FR).
    test_passed['has_min_rate'] = qm['has_min_rate']

    # Not task-related.
    test_passed['task_related'] = qm['task_related']

    # Include unit if all criteria met.
    include = test_passed.all()

    return include
