#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:04:07 2016

Collection of functions related to quality metrics of recording.

@author: David Samu
"""

import numpy as np
import pandas as pd
from quantities import s, ms, us

from matplotlib import pyplot as plt
from matplotlib import gridspec as gs

from elephant import statistics

from seal.util import plot, util


# %% Constants.

REC_GAIN = 4100  # gain of recording
CENSORED_PRD_LEN = 0.675 * ms  # width of censored period
ISI_TH = 1.0 * ms  # ISI violation threshold
WF_T_START = 9  # start INDEX of spiked (aligned by Plexon)
MAX_DRIFT_RATIO = 2  # maximum tolerable drift ratio

MIN_RATIO_TRIALS = 0.5  # minimum ratio of trials to keep

MIN_BIN_LEN = 120 * s  # minimum window length for firing binned statistics


# %% Utility functions.

def get_base_data(u):
    """Return base data of unit for quality metrics calculation."""

    # Init values.
    waveforms = u.UnitParams['SpikeWaveforms']
    wavetime = u.UnitParams['WaveformTime']
    spike_dur = u.UnitParams['SpikeDuration']
    spike_times = u.UnitParams['SpikeTimes']
    sampl_per = u.SessParams['SamplPer']

    return waveforms, wavetime, spike_dur, spike_times, sampl_per


def time_bin_data(spike_times, waveforms):
    """Return time binned data for statistics over session time."""

    # Time bins and binned waveforms and spike times.
    t_start, t_stop = spike_times.t_start, spike_times.t_stop
    nbins = int(np.floor((t_stop - t_start) / MIN_BIN_LEN))
    tbin_lims = util.quantity_linspace(t_start, t_stop, s, nbins)
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

    # No waveforms passed.
    if not wfs.size:
        return np.nan, np.array([]) * us, np.array([]) * us

    # SNR: std of mean waveform divided by std of residual waveform (noise).
    wf_mean = np.mean(wfs, 0)
    wf_std = wfs - wf_mean
    snr = np.std(wf_mean) / np.std(wf_std)

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

    # No spikes passed.
    if not spike_times.size:
        return np.nan, np.nan

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


def classify_unit(snr, true_spikes=np.nan):
    """Classify unit as single or multi-unit."""

    # Check if we have true spikes ratio calculated.
    if np.isnan(true_spikes):
        true_spikes = 100

    if true_spikes >= 90 and snr >= 2.0:
        unit_type = 'single unit'
    else:
        unit_type = 'multi unit'

    return unit_type


def test_drift(t, v, tbins, tr_starts, spike_times):
    """Test drift (gradual, or more instantaneous jump or drop) in variable."""

    # Number of trials from session start until start and end of each period.
    tr_starts = util.pd_to_np_quantity(tr_starts)
    n_tr_prd_start = [np.sum(util.indices_in_window(tr_starts, vmax=t1))
                      for t1, t2 in tbins]
    n_tr_prd_end = [np.sum(util.indices_in_window(tr_starts, vmax=t2))
                    for t1, t2 in tbins]

    # Find longest period with acceptible drift.
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

    # Keep it only if it contains more trials than minimum limit.
    n_tr = len(tr_starts)
    t1, t2 = tbins[0][0], tbins[0][0]
    first_tr_inc, last_tr_inc = -1, -1
    prd1, prd2 = -1, -1
    most_trs = period_res.n_tr.max()
    if most_trs >= MIN_RATIO_TRIALS * n_tr:
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
    tr_inc = util.indices_in_window(np.array(range(n_tr)),
                                    first_tr_inc, last_tr_inc)
    sp_inc = util.indices_in_window(spike_times, t1, t2)

    return t1, t2, prd_inc, tr_inc, sp_inc


# %% Calculate quality metrics.

def test_qm(u, ffig_template):
    """
    Test ISI, SNR and stationarity of spikes and spike waveforms.
    Exclude trials with unacceptable drift.

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
    test_res = test_drift(tbin_vmid, rate_t, tbins, tr_starts, spike_times)
    t1_inc, t2_inc, prd_inc, tr_inc, sp_inc = test_res

    # Waveform statistics of included spikes only.
    snr, wf_amp, wf_dur = waveform_stats(waveforms[sp_inc], wavetime)

    # Firing rate.
    mean_rate = float(np.sum(sp_inc) / (t2_inc - t1_inc))

    # ISI statistics.
    true_spikes, ISI_vr = isi_stats(spike_times[sp_inc])
    unit_type = classify_unit(snr, true_spikes)

    # Add quality metrics to unit.
    u.QualityMetrics['SNR'] = snr
    u.QualityMetrics['MeanWfAmplitude'] = np.mean(wf_amp)
    u.QualityMetrics['MeanWfDuration'] = np.mean(spike_dur[sp_inc]).rescale(us)
    u.QualityMetrics['MeanFiringRate'] = mean_rate
    u.QualityMetrics['ISIviolation'] = ISI_vr
    u.QualityMetrics['TrueSpikes'] = true_spikes
    u.QualityMetrics['UnitType'] = unit_type

    # Trial removal info.
    u.QualityMetrics['NTrialsTotal'] = len(tr_starts)
    u.QualityMetrics['NTrialsIncluded'] = np.sum(tr_inc)
    u.QualityMetrics['NTrialsExcluded'] = len(tr_starts) - np.sum(tr_inc)
    u.QualityMetrics['TrialsIncluded'] = tr_inc
    u.QualityMetrics['TrialsExcluded'] = np.invert(tr_inc)

    # Plot quality metric results.
    if ffig_template is not None:
        plot_qm(u, mean_rate, ISI_vr, true_spikes, unit_type, tbin_vmid, tbins,
                snr_t, rate_t, t1_inc, t2_inc, prd_inc, tr_inc, sp_inc,
                ffig_template)

    return u


# %% Plot quality metrics.

def plot_qm(u, mean_rate, ISI_vr, true_spikes, unit_type, tbin_vmid, tbins,
            snr_t, rate_t, t1_inc, t2_inc, prd_inc, tr_inc, sp_inc,
            ffig_template):
    """Plot quality metrics related figures."""

    # Init values.
    waveforms, wavetime, spike_dur, spike_times, sampl_per = get_base_data(u)

    # Get waveform stats of included and excluded spikes.
    wf_inc = waveforms[sp_inc]
    wf_exc = waveforms[np.invert(sp_inc)]
    snr_all, wf_amp_all, wf_dur_all = waveform_stats(waveforms, wavetime)
    snr_inc, wf_amp_inc, wf_dur_inc = waveform_stats(wf_inc, wavetime)
    snr_exc, wf_amp_exc, wf_dur_exc = waveform_stats(wf_exc, wavetime)

    # Minimum and maximum gain.
    gmin = min(-REC_GAIN/2, np.min(waveforms))
    gmax = max(REC_GAIN/2, np.max(waveforms))

    # %% Init plots.

    # Init plotting.
    fig = plot.figure(figsize=(12, 12))
    gs1 = gs.GridSpec(3, 3)
    sp = np.array([plt.subplot(gs) for gs in gs1]).reshape(gs1.get_geometry())
    ax_wf_inc, ax_wf_exc, ax_filler1 = sp[0, 0], sp[1, 0], sp[2, 0]
    ax_wf_amp, ax_wf_dur, ax_amp_dur = sp[0, 1], sp[1, 1], sp[2, 1]
    ax_snr, ax_rate, ax_filler2 = sp[0, 2], sp[1, 2], sp[2, 2]

    ax_filler1.axis('off')
    ax_filler2.axis('off')

    # Common variables, limits and labels.
    sp_i = range(-WF_T_START, waveforms.shape[1]-WF_T_START)
    sp_t = sp_i * sampl_per
    ses_t_lim = [spike_times.t_start, spike_times.t_stop]
    ss = 1.0  # marker size on scatter plot
    sa = .8  # marker alpha on scatter plot
    glim = [gmin, gmax]  # gain axes limit
    wf_t_lim = [min(sp_t), max(sp_t)]
    dur_lim = [0*us, wavetime[-1]-wavetime[WF_T_START]]  # same across units
    amp_lim = [0, gmax-gmin]  # [np.min(wf_ampl), np.max(wf_ampl)]

    tr_alpha = 0.25  # alpha of trial event lines

    # Color spikes by their occurance over session time.
    my_cmap = plt.get_cmap('jet')
    sp_cols = np.tile(np.array([.25, .25, .25, .25]), (len(spike_times), 1))
    if not np.all(np.invert(sp_inc)):  # check if there is any spike included
        sp_t_inc = np.array(spike_times[sp_inc])
        sp_t_inc_shifted = sp_t_inc - sp_t_inc.min()
        sp_cols[sp_inc, :] = my_cmap(sp_t_inc_shifted/sp_t_inc_shifted.max())
    # Put excluded trials to the front, and randomise order of included trials
    # so later spikes don't cover earlier ones.
    sp_order = np.hstack((np.where(np.invert(sp_inc))[0],
                          np.random.permutation(np.where(sp_inc)[0])))

    # Trial markers.
    trial_starts = u.TrialParams.TrialStart
    trms = trial_starts[9::10]
    tr_markers = dict((tr_i+1, tr_t) for tr_i, tr_t in zip(trms.index, trms))

    # Common labels for plots
    wf_t_lab = 'Waveform time ($\mu$s)'
    ses_t_lab = 'Session time (s)'
    volt_lab = 'Voltage (normalized)'
    amp_lab = 'Amplitude'
    dur_lab = 'Duration ($\mu$s)'

    # %% Waveform shape analysis.

    # Plot excluded and included waveforms on different axes.
    # Color included by occurance in session time to help detect drifts.
    wfs = np.transpose(waveforms)
    for i in sp_order:
        ax = ax_wf_inc if sp_inc[i] else ax_wf_exc
        ax.plot(sp_t, wfs[:, i], color=sp_cols[i, :], alpha=0.05)

    # Format waveform plots
    n_sp_inc, n_sp_exc = sum(sp_inc), sum(np.invert(sp_inc))
    n_tr_inc, n_tr_exc = sum(tr_inc), sum(np.invert(tr_inc))
    for ax, st, n_sp, n_tr in [(ax_wf_inc, 'Included', n_sp_inc, n_tr_inc),
                               (ax_wf_exc, 'Excluded', n_sp_exc, n_tr_exc)]:
        title = '{} waveforms, {} spikes, {} trials'.format(st, n_sp, n_tr)
        plot.set_limits(xlim=wf_t_lim, ylim=glim, ax=ax)
        plot.show_ticks(xtick_pos='none', ytick_pos='none', ax=ax)
        plot.show_spines(True, False, False, False, ax=ax)
        plot.set_labels(title=title, xlab=wf_t_lab, ylab=volt_lab, ax=ax)

    # %% Waveform summary metrics.

    # Function to return colors of spikes / waveforms
    # based on whether they are included / excluded.
    def get_color(col_incld, col_bckgrnd='grey'):
        cols = np.array(len(sp_inc) * [col_bckgrnd])
        cols[sp_inc] = col_incld
        return cols

    # Waveform amplitude across session time.
    m_amp, sd_amp = float(np.mean(wf_amp_inc)), float(np.std(wf_amp_inc))
    title = 'Waveform amplitude: {:.1f} $\pm$ {:.1f}'.format(m_amp, sd_amp)
    plot.scatter(spike_times, wf_amp_all, add_r=False, c=get_color('m'), s=ss,
                 xlab=ses_t_lab, ylab=amp_lab, xlim=ses_t_lim, ylim=amp_lim,
                 edgecolors='none', alpha=sa, title=title, ax=ax_wf_amp)

    # Waveform duration across session time.
    wf_dur_all = spike_dur.rescale(us)  # to use TPLCell's waveform duration
    wf_dur_inc = wf_dur_all[sp_inc]
    mdur, sdur = float(np.mean(wf_dur_inc)), float(np.std(wf_dur_inc))
    title = 'Waveform duration: {:.1f} $\pm$ {:.1f} $\mu$s'.format(mdur, sdur)
    plot.scatter(spike_times, wf_dur_all, add_r=False, c=get_color('c'), s=ss,
                 xlab=ses_t_lab, ylab=dur_lab, xlim=ses_t_lim, ylim=dur_lim,
                 edgecolors='none', alpha=sa, title=title, ax=ax_wf_dur)

    # Waveform duration against amplitude.
    title = 'Waveform duration - amplitude'
    plot.scatter(wf_dur_all[sp_order], wf_amp_all[sp_order], c=sp_cols[sp_order],
                 s=ss, xlab=dur_lab, ylab=amp_lab, xlim=dur_lim, ylim=amp_lim,
                 edgecolors='none', alpha=sa, title=title, ax=ax_amp_dur)

    # %% SNR, firing rate and spike timing.

    # Color segments depending on wether they are included / excluded.
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
    plot.lines([], [], c='y', xlim=ses_t_lim, ylim=ylim,
               title=title, xlab=ses_t_lab, ylab='SNR', ax=ax_snr)

    # Firing rate over session time.
    title = 'Firing rate: {:.1f} spike/s'.format(mean_rate)
    ylim = [0, 1.1*np.max(rate_t.magnitude)]
    plot_periods(rate_t, 'b', ax_rate)
    plot.lines([], [], c='b', xlim=ses_t_lim, ylim=ylim,
               title=title, xlab=ses_t_lab, ylab='Firing rate (spike/s)',
               ax=ax_rate)

    # Add trial markers and highlight included period in each plot.
    for ax in [ax_snr, ax_rate]:

        # Trial markers.
        plot.plot_events(tr_markers, t_unit=s, lw=0.5, ls='--',
                         alpha=tr_alpha, ax=ax)

        # Included period.
        if not np.all(np.invert(tr_inc)):  # check if there is any trials
            incl_segment = dict(selected=[t1_inc, t2_inc])
            plot.plot_segments(incl_segment, t_unit=s, alpha=0.2,
                               color='grey', ymax=0.96, ax=ax)

    # %% Save figure and metrics

    # Create title
    fig_title = ('{}: "{}"'.format(u.Name, unit_type) +
                 # '\n\nSNR: {:.2f}%'.format(snr_inc) +
                 ',   ISI vr: {:.2f}%'.format(ISI_vr) +
                 ',   Tr Sp: {:.1f}%'.format(true_spikes))

    fig.suptitle(fig_title, y=0.98, fontsize='xx-large')
    gs1.tight_layout(fig, rect=[0, 0.0, 1, 0.92])
    ffig = ffig_template.format(u.name_to_fname())
    plot.save_fig(fig, ffig)
