#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:04:07 2016

Functions to calculate and plot quality metrics of units  after spike sorting
(SNR, ISIvr, etc), and to exclude trials / units not meeting QC criteria.

@author: David Samu
"""

import numpy as np
import pandas as pd
from quantities import s, ms, us

from elephant import statistics

from seal.util import util
from seal.plot import putil, pplot
from seal.object.trials import Trials
from seal.object.periods import Periods


# %% Constants.

# Recording constants.
REC_GAIN = 4100                # gain of recording
CENSORED_PRD_LEN = 0.675 * ms  # length of censored period
WF_T_START = 9                 # start index of spikes (aligned by Plexon)


# Constants related to quality metrics calculation.
ISI_TH = 1.0 * ms      # ISI violation threshold
MAX_DRIFT_RATIO = 2    # maximum tolerable drift ratio
MIN_BIN_LEN = 120 * s  # minimum window length for firing binned statistics


# Constants related to unit exclusion.
min_RF_coverage = 0.5  # min. receptive field coverage
min_SNR = 1.0          # min. SNR
min_FR = 1.0           # min. firing rate (sp/s)
max_ISIvr = 1.0        # max. ISI violation ratio (%)
min_inc_trs_rat = 50   # min. ratio of included trials out of all recorded (%)
min_DSI = 0.1          # min. direction selectivity index (8-dir weighted)


# %% Utility functions.

def get_base_data(u):
    """Return base data of unit for quality metrics calculation."""

    # Init values.
    waveforms = u.Waveforms['SpikeWaveforms']
    wavetime = u.Waveforms['WaveformTime']
    spike_dur = u.Waveforms['SpikeDuration']
    spike_times = u.Waveforms['SpikeTimes']
    sampl_per = u.SessParams['sampl_prd']

    return waveforms, wavetime, spike_dur, spike_times, sampl_per


def time_bin_data(spike_times, waveforms):
    """Return time binned data for statistics over session time."""

    # Time bins and binned waveforms and spike times.
    t_start, t_stop = spike_times.t_start, spike_times.t_stop
    nbins = max(int(np.floor((t_stop - t_start) / MIN_BIN_LEN)), 1)
    tbin_lims = util.quantity_linspace(t_start, t_stop, nbins+1, s)
    tbins = [(tbin_lims[i], tbin_lims[i+1]) for i in range(len(tbin_lims)-1)]
    tbin_vmid = np.array([np.mean([t1, t2]) for t1, t2 in tbins])*s
    sp_idx_binned = [util.indices_in_window(spike_times, t1, t2)
                     for t1, t2 in tbins]
    wf_binned = [waveforms[sp_idx] for sp_idx in sp_idx_binned]
    sp_times_binned = [spike_times[sp_idx] for sp_idx in sp_idx_binned]

    return tbins, tbin_vmid, wf_binned, sp_times_binned


# %% Core methods .

def waveform_stats(wfs, wtime):
    """Calculates SNR, amplitude and durations of spike waveforms."""

    # No waveforms: waveform stats are uninterpretable.
    if not wfs.size:
        return np.nan, np.array([]) * us, np.array([]) * us

    # SNR: std of mean waveform divided by std of residual waveform (noise).
    wf_mean = np.mean(wfs, 0)
    wf_std = wfs - wf_mean
    if wfs.shape[0] > 1:
        snr = np.std(wf_mean) / np.std(wf_std)
    else:
        snr = 10  # extreme case of only a single spike in Unit.

    # Indices of minimum and maximum times.
    # imin = np.argmin(wfs, 1)  # actual minimum of waveform
    imin = wfs.shape[0] * [WF_T_START]  # crossing of threshold (Plexon value)
    imax = [np.argmax(w[imin[i]:]) + imin[i] for i, w in enumerate(wfs)]

    # Duration: time difference between times of minimum and maximum values.
    wf_tmin = wtime[imin]
    wf_tmax = wtime[imax]
    wf_dur = wf_tmax - wf_tmin

    # Amplitude: value difference between mininum and maximum values.
    wmin = wfs[np.arange(len(imin)), imin]
    wmax = wfs[np.arange(len(imax)), imax]
    wf_amp = wmax - wmin

    return snr, wf_amp, wf_dur


def isi_stats(spike_times):
    """Returns ISIs and some related statistics."""

    # No spike: ISI v.r. and TrueSpikes are uninterpretable.
    if not spike_times.size:
        return np.nan, np.nan

    # Only one spike: ISI v.r. is 0%, TrueSpikes is 100%.
    if spike_times.size == 1:
        return 100, 0

    isi = statistics.isi(spike_times).rescale(ms)

    # Percent of spikes violating ISI treshold.
    n_ISI_vr = sum(isi < ISI_TH)
    percent_ISI_vr = 100 * n_ISI_vr / isi.size

    # Percent of spikes estimated to originate from the sorted single unit.
    # Based on refractory period violations.
    # See Hill et al., 2011: Quality Metrics to Accompany Spike Sorting of
    # Extracellular Signals
    N = spike_times.size
    T = (spike_times.t_stop - spike_times.t_start).rescale(ms)
    r = n_ISI_vr
    tmax = ISI_TH
    tmin = CENSORED_PRD_LEN
    tdif = tmax - tmin
    true_spikes = 100*(1/2 + np.sqrt(1/4 - float(r*T / (2*tdif*N**2))))

    return true_spikes, percent_ISI_vr


def classify_unit(snr, true_spikes):
    """Classify unit as single or multi-unit."""

    if true_spikes >= 90 and snr >= 2.0:
        unit_type = 'single unit'
    else:
        unit_type = 'multi unit'

    return unit_type


def test_drift(t, v, tbins, tr_starts, spike_times, rej_trials):
    """Test drift (gradual, or more instantaneous jump or drop) in variable."""

    # Return full task length if not rejecting trials.
    if not rej_trials:
        t1 = spike_times[0]
        t2 = spike_times[-1]
        first_tr_inc = 0
        last_tr_inc = len(tr_starts)
        prd1 = 0
        prd2 = len(tbins)

    else:

        # Number of trials from beginning of session
        # until start and end of each period.
        tr_starts = util.list_to_quantity(tr_starts)
        n_tr_prd_start = [np.sum(util.indices_in_window(tr_starts, vmax=t1))
                          for t1, t2 in tbins]
        n_tr_prd_end = [np.sum(util.indices_in_window(tr_starts, vmax=t2))
                        for t1, t2 in tbins]

        # Find period within acceptible drift range for each bin.
        cols = ['prd_start_i', 'prd_end_i', 'n_prd',
                't_start', 't_end', 't_len',
                'tr_start_i', 'tr_end_i', 'n_tr']
        period_res = pd.DataFrame(index=range(len(v)), columns=cols)
        for i, v1 in enumerate(v):
            vmin, vmax = v1, v1
            for j, v2 in enumerate(v[i:]):
                # Update extreme values.
                vmin = min(vmin, v2)
                vmax = max(vmax, v2)
                # If difference becomes unacceptable, terminate period.
                if vmax > MAX_DRIFT_RATIO*v2 or v2 > MAX_DRIFT_RATIO*vmin:
                    j -= 1
                    break
            end_i = i + j
            period_res.prd_start_i[i] = i
            period_res.prd_end_i[i] = end_i
            period_res.n_prd[i] = j + 1
            period_res.t_start[i] = tbins[i][0]
            period_res.t_end[i] = tbins[end_i][1]
            period_res.t_len[i] = tbins[end_i][1] - tbins[i][0]
            period_res.tr_start_i[i] = n_tr_prd_start[i]
            period_res.tr_end_i[i] = n_tr_prd_end[end_i]
            period_res.n_tr[i] = n_tr_prd_end[end_i] - n_tr_prd_start[i]

        # Find bin with longest period.
        idx = period_res.n_tr.argmax()
        # Indices of longest period.
        prd1 = period_res.prd_start_i[idx]
        prd2 = period_res.prd_end_i[idx]
        # Times of longest period.
        t1 = period_res.t_start[idx]
        t2 = period_res.t_end[idx]
        # Trial indices within longest period.
        first_tr_inc = period_res.tr_start_i[idx]
        last_tr_inc = period_res.tr_end_i[idx] - 1

    # Return included trials and spikes.
    prd_inc = util.indices_in_window(np.array(range(len(tbins))), prd1, prd2)
    tr_inc = util.indices_in_window(np.array(range(len(tr_starts))),
                                    first_tr_inc, last_tr_inc)
    spk_inc = util.indices_in_window(spike_times, t1, t2)

    return t1, t2, prd_inc, tr_inc, spk_inc


# %% Calculate quality metrics, and find trials and units to be excluded.

def test_qm(u, rej_trials=True, ftempl=None):
    """
    Test ISI, SNR and stationarity of spikes and spike waveforms.
    Optionally find and reject trials with unacceptable
    drift (if do_trial_rejection is True).

    Non-stationarities can happen due to e.g.:
    - electrode drift, or
    - change in the state of the neuron.
    """

    # Init values.
    waveforms, wavetime, spike_dur, spike_times, sampl_per = get_base_data(u)

    # Time binned statistics.
    tbinned_stats = time_bin_data(spike_times, waveforms)
    tbins, tbin_vmid, wf_binned, sp_times_binned = tbinned_stats

    snr_t = [waveform_stats(wfb, wavetime)[0] for wfb in wf_binned]
    rate_t = np.array([spt.size/(t2-t1).rescale(s)
                       for spt, (t1, t2) in zip(sp_times_binned, tbins)]) / s

    # Test drifts and reject trials if necessary.
    tr_starts = u.TrialParams.TrialStart
    test_res = test_drift(tbin_vmid, rate_t, tbins, tr_starts,
                          spike_times, rej_trials)
    t1_inc, t2_inc, prd_inc, tr_inc, spk_inc = test_res

    # Waveform statistics of included spikes only.
    snr, wf_amp, wf_dur = waveform_stats(waveforms[spk_inc], wavetime)

    # Firing rate.
    mean_rate = float(np.sum(spk_inc) / (t2_inc - t1_inc))

    # ISI statistics.
    true_spikes, ISI_vr = isi_stats(spike_times[spk_inc])
    unit_type = classify_unit(snr, true_spikes)

    # Add quality metrics to unit.
    u.QualityMetrics['SNR'] = snr
    u.QualityMetrics['MeanWfAmplitude'] = np.mean(wf_amp)
    u.QualityMetrics['MeanWfDur'] = np.mean(spike_dur[spk_inc]).rescale(us)
    u.QualityMetrics['MeanFiringRate'] = mean_rate
    u.QualityMetrics['ISIviolation'] = ISI_vr
    u.QualityMetrics['TrueSpikes'] = true_spikes
    u.QualityMetrics['UnitType'] = unit_type

    # Trial removal info.
    tr_exc = np.invert(tr_inc)
    u.QualityMetrics['NTrialsTotal'] = len(tr_starts)
    u.QualityMetrics['NTrialsIncluded'] = np.sum(tr_inc)
    u.QualityMetrics['NTrialsExcluded'] = np.sum(tr_exc)
    u.QualityMetrics['IncludedTrials'] = Trials(tr_inc, 'included trials')
    u.QualityMetrics['ExcludedTrials'] = Trials(tr_exc, 'excluded trials')
    u.QualityMetrics['IncludedSpikes'] = spk_inc

    # Plot quality metric results.
    if ftempl is not None:
        plot_qm(u, mean_rate, ISI_vr, true_spikes, unit_type, tbin_vmid, tbins,
                snr_t, rate_t, t1_inc, t2_inc, prd_inc, tr_inc, spk_inc, ftempl)


def test_rejection(u):
    """Check whether unit is to be rejected from analysis."""

    qm = u.QualityMetrics
    test_passed = pd.Series()

    # Insufficient receptive field coverage.
    # th_passed.append(qm['RC_coverage'] < min_RF_coverage)

    # Extremely low waveform consistency (SNR).
    test_passed['SNR'] = qm['SNR'] > min_SNR

    # Extremely low unit activity (FR).
    test_passed['FR'] = qm['MeanFiringRate'] > min_FR

    # Extremely high ISI violation ratio (ISIvr).
    test_passed['ISI'] = qm['ISIviolation'] < max_ISIvr

    # Insufficient number of trials (ratio of included trials).
    inc_trs_ratio = 100 * qm['NTrialsIncluded'] / qm['NTrialsTotal']
    test_passed['IncTrsRatio'] = inc_trs_ratio > min_inc_trs_rat

    # Insufficient direction selectivity (DSI).
    DSIs = u.DS.loc['DSI'].wDS   # 8-direction weighted DSI
    test_passed['DSI'] = (DSIs > min_DSI).any()

    # Exclude unit if any of the criteria is not met.
    exclude = not test_passed.all()

    return exclude


# %% Plot quality metrics.

def plot_qm(u, mean_rate, ISI_vr, true_spikes, unit_type, tbin_vmid, tbins,
            snr_t, rate_t, t1_inc, t2_inc, prd_inc, tr_inc, spk_inc, ftempl):
    """Plot quality metrics related figures."""

    # Init values.
    waveforms, wavetime, spike_dur, spike_times, sampl_per = get_base_data(u)

    # Get waveform stats of included and excluded spikes.
    wf_inc = waveforms[spk_inc]
    wf_exc = waveforms[np.invert(spk_inc)]
    snr_all, wf_amp_all, wf_dur_all = waveform_stats(waveforms, wavetime)
    snr_inc, wf_amp_inc, wf_dur_inc = waveform_stats(wf_inc, wavetime)
    snr_exc, wf_amp_exc, wf_dur_exc = waveform_stats(wf_exc, wavetime)

    # Minimum and maximum gain.
    gmin = min(-REC_GAIN/2, np.min(waveforms))
    gmax = max(REC_GAIN/2, np.max(waveforms))

    # %% Init plots.

    # Init plotting theme.
    putil.set_style('notebook', 'ticks')

    # Init plotting objects.
    fig, gsp, sbp = putil.get_gs_subplots(nrow=3, ncol=3, subw=4, subh=4)
    ax_wf_inc, ax_wf_exc, ax_filler1 = sbp[0, 0], sbp[1, 0], sbp[2, 0]
    ax_wf_amp, ax_wf_dur, ax_amp_dur = sbp[0, 1], sbp[1, 1], sbp[2, 1]
    ax_snr, ax_rate, ax_filler2 = sbp[0, 2], sbp[1, 2], sbp[2, 2]

    ax_filler1.axis('off')
    ax_filler2.axis('off')

    # Trial markers.
    trial_starts = u.TrialParams.TrialStart
    trms = trial_starts[9::10]
    tr_markers = {tr_i+1: tr_t for tr_i, tr_t in zip(trms.index, trms)}

    # Common variables, limits and labels.
    spk_i = range(-WF_T_START, waveforms.shape[1]-WF_T_START)
    spk_t = spk_i * sampl_per
    ses_t_lim = [min(spike_times.t_start, trial_starts.iloc[0]),
                 max(spike_times.t_stop, trial_starts.iloc[-1])]
    ss = 1.0  # marker size on scatter plot
    sa = .80  # marker alpha on scatter plot
    glim = [gmin, gmax]  # gain axes limit
    wf_t_lim = [min(spk_t), max(spk_t)]
    dur_lim = [0*us, wavetime[-1]-wavetime[WF_T_START]]  # same across units
    amp_lim = [0, gmax-gmin]  # [np.min(wf_ampl), np.max(wf_ampl)]

    # Color spikes by their occurance over session time.
    my_cmap = putil.get_cmap('jet')
    spk_cols = np.tile(np.array([.25, .25, .25, .25]), (len(spike_times), 1))
    if not np.all(np.invert(spk_inc)):  # check if there is any spike included
        spk_t_inc = np.array(spike_times[spk_inc])
        spk_t_inc_shifted = spk_t_inc - spk_t_inc.min()
        spk_t_inc_max = spk_t_inc_shifted.max()
        spk_cols[spk_inc, :] = my_cmap(spk_t_inc_shifted/spk_t_inc_max)
    # Put excluded trials to the front, and randomise order of included trials
    # so later spikes don't systematically cover earlier ones.
    sp_order = np.hstack((np.where(np.invert(spk_inc))[0],
                          np.random.permutation(np.where(spk_inc)[0])))

    # Common labels for plots
    wf_t_lab = 'WF time ($\mu$s)'
    ses_t_lab = 'Recording time (s)'
    volt_lab = 'Voltage'
    amp_lab = 'Amplitude'
    dur_lab = 'Duration ($\mu$s)'

    # %% Waveform shape analysis.

    # Plot excluded and included waveforms on different axes.
    # Color included by occurance in session time to help detect drifts.
    wfs = np.transpose(waveforms)
    for i in sp_order:
        ax = ax_wf_inc if spk_inc[i] else ax_wf_exc
        ax.plot(spk_t, wfs[:, i], color=spk_cols[i, :], alpha=0.05)

    # Format waveform plots
    n_sp_inc, n_sp_exc = sum(spk_inc), sum(np.invert(spk_inc))
    n_tr_inc, n_tr_exc = sum(tr_inc), sum(np.invert(tr_inc))
    for ax, st, n_sp, n_tr in [(ax_wf_inc, 'Included', n_sp_inc, n_tr_inc),
                               (ax_wf_exc, 'Excluded', n_sp_exc, n_tr_exc)]:
        title = '{} WFs, {} spikes, {} trials'.format(st, n_sp, n_tr)
        putil.format_plot(ax, wf_t_lim, glim, wf_t_lab, volt_lab, title)

    # %% Waveform summary metrics.

    # Waveform amplitude across session time.
    m_amp, sd_amp = float(np.mean(wf_amp_inc)), float(np.std(wf_amp_inc))
    title = 'WF amplitude: {:.1f} $\pm$ {:.1f}'.format(m_amp, sd_amp)
    pplot.scatter(spike_times, wf_amp_all, spk_inc, c='m', bc='grey', s=ss,
                  xlab=ses_t_lab, ylab=amp_lab, xlim=ses_t_lim, ylim=amp_lim,
                  edgecolors='', alpha=sa, title=title, ax=ax_wf_amp)

    # Waveform duration across session time.
    wf_dur_all = spike_dur.rescale(us)  # to use TPLCell's waveform duration
    wf_dur_inc = wf_dur_all[spk_inc]
    mdur, sdur = float(np.mean(wf_dur_inc)), float(np.std(wf_dur_inc))
    title = 'WF duration: {:.1f} $\pm$ {:.1f} $\mu$s'.format(mdur, sdur)
    pplot.scatter(spike_times, wf_dur_all, spk_inc, c='c', bc='grey', s=ss,
                  xlab=ses_t_lab, ylab=dur_lab, xlim=ses_t_lim, ylim=dur_lim,
                  edgecolors='', alpha=sa, title=title, ax=ax_wf_dur)

    # Waveform duration against amplitude.
    title = 'WF duration - amplitude'
    pplot.scatter(wf_dur_all[sp_order], wf_amp_all[sp_order], c=spk_cols[sp_order],
                  s=ss, xlab=dur_lab, ylab=amp_lab, xlim=dur_lim, ylim=amp_lim,
                  edgecolors='', alpha=sa, title=title, ax=ax_amp_dur)

    # %% SNR, firing rate and spike timing.

    # Color segments depending on whether they are included / excluded.
    def plot_periods(v, color, ax):
        # Plot line segments.
        for i in range(len(prd_inc[:-1])):
            col = color if prd_inc[i] and prd_inc[i+1] else 'grey'
            x, y = [(tbin_vmid[i], tbin_vmid[i+1]), (v[i], v[i+1])]
            ax.plot(x, y, color=col)
        # Plot line points.
        for i in range(len(prd_inc)):
            col = color if prd_inc[i] else 'grey'
            x, y = [tbin_vmid[i], v[i]]
            ax.plot(x, y, color=col, marker='o',
                    markersize=3, markeredgecolor=col)

    # SNR over session time.
    title = 'SNR: {:.2f}'.format(snr_inc)
    ylim = [0, 1.1*np.max(snr_t)]
    plot_periods(snr_t, 'y', ax_snr)
    pplot.lines([], [], c='y', xlim=ses_t_lim, ylim=ylim, title=title,
                xlab=ses_t_lab, ylab='SNR', ax=ax_snr)

    # Firing rate over session time.
    title = 'Firing rate: {:.1f} spike/s'.format(mean_rate)
    ylim = [0, 1.1*np.max(rate_t.magnitude)]
    plot_periods(rate_t, 'b', ax_rate)
    pplot.lines([], [], c='b', xlim=ses_t_lim, ylim=ylim, title=title,
                xlab=ses_t_lab, ylab=putil.FR_lbl, ax=ax_rate)

    # Add trial markers and highlight included period in each plot.
    for ax in [ax_snr, ax_rate]:

        # Trial markers.
        putil.plot_events(tr_markers, t_unit=s, lw=0.5, ls='--', alpha=0.35,
                          ax=ax)

        # Included period.
        if not np.all(np.invert(tr_inc)):  # check if there is any trials
            incl_segment = Periods([('selected', [t1_inc, t2_inc])])
            putil.plot_periods(incl_segment, t_unit=s, ymax=0.96, ax=ax)

    # %% Save figure and metrics.

    # Create title
    title = ('{}: "{}"'.format(u.Name, unit_type) +
             # '\n\nSNR: {:.2f}%'.format(snr_inc) +
             ',   ISI vr: {:.2f}%'.format(ISI_vr) +
             ',   Tr Sp: {:.1f}%'.format(true_spikes))
    fname = ftempl.format(u.name_to_fname())
    putil.save_gsp_figure(fig, gsp, fname, title, rect_height=0.92)
