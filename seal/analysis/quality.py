#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:04:07 2016

Collection of functions related to quality metrics of recording.

@author: David Samu
"""

import numpy as np
from quantities import s, ms, us

from matplotlib import pyplot as plt
from matplotlib import gridspec as gs

from elephant import statistics

from seal.util import plot, util


# %% Constants.

REC_GAIN = 2050  # gain of recording
CENSORED_PRD_LEN = 0.675 * ms  # width of censored period
ISI_TH = 1.0 * ms  # ISI violation threshold
WF_T_START = 9  # start INDEX of spiked (aligned by Plexon)
MAX_DRIFT_RATIO = 2  # maximum tolerable drift ratio
MIN_STABLE_PERIOD_LENGTH = 1000 * s  # length of minimal stable period to keep

BIN_LEN = 120 * s  # window length for firing binned statistics


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
    tbin_lims = util.quantity_arange(spike_times.t_start,
                                     spike_times.t_stop, BIN_LEN)
    tbin_lims = [(tbin_lims[i], tbin_lims[i+1])
                 for i in range(len(tbin_lims)-1)]
    tbin_vals = np.array([np.mean([t1, t2]) for t1, t2 in tbin_lims])*s
    sp_idx_binned = [util.indices_in_window(spike_times, t1, t2)
                     for t1, t2 in tbin_lims]
    wf_binned = [waveforms[sp_idx] for sp_idx in sp_idx_binned]
    sp_times_binned = [spike_times[sp_idx] for sp_idx in sp_idx_binned]

    return tbin_lims, tbin_vals, wf_binned, sp_times_binned


# %% Core methods .

def waveform_stats(wfs, wtime):
    """Calculates SNR, amplitude and durations of spike waveforms."""

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

    isi = statistics.isi(spike_times).rescale(ms)

    # Percent of spikes violating ISI treshold.
    n_isi_viol = sum(isi < ISI_TH)
    percent_isi_viol = 100 * n_isi_viol / isi.size

    # Percent of spikes estimated to originate from the sorted single unit.
    # Based on refractory period violations.
    # See Hill et al., 2011: Quality Metrics to Accompany Spike Sorting of
    # Extracellular Signals
    N = spike_times.size
    T = (spike_times.t_stop - spike_times.t_start).rescale(ms)
    r = n_isi_viol
    tmax = ISI_TH
    tmin = CENSORED_PRD_LEN
    tdif = tmax - tmin
    true_spikes = 100*(1/2 + np.sqrt(1/4 - float(r*T / (2*tdif*N**2))))

    return true_spikes, percent_isi_viol


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


def test_drift(t, v, tbin_lims, trial_start, spike_times):
    """Test drift (gradual, or more instantaneous jump or drop) in variable."""

    # Find longest period with acceptible drift.
    prd_lens = np.zeros(len(v)) * t.units
    prd_end_idx = np.zeros(len(v), dtype=int)
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
        prd_lens[i] = t[i+j] - t[i]
        prd_end_idx[i] = i+j

    # Keep it only if it's longer than minimum limit of stable period length.
    t1, t2 = tbin_lims[0][0], tbin_lims[0][0]
    first_tr_incl, first_tr_excl = 0, 0
    longest_period_length = np.max(prd_lens)
    if longest_period_length >= MIN_STABLE_PERIOD_LENGTH:
        # Get indices of longest period.
        start_i = np.argmax(prd_lens)
        end_i = prd_end_idx[start_i]
        # Get times of longest period.
        t1 = tbin_lims[start_i][0]
        t2 = tbin_lims[end_i][1]
        # Get trial indices within longest period.
        tr_start = util.pd_to_np_quantity(trial_start)
        first_tr_incl = np.argmax(tr_start > t1)
        first_tr_excl = first_tr_incl + (np.argmax(tr_start[first_tr_incl:] > t2))

    # Return included trials and spikes.
    tr_incl = list(range(first_tr_incl, first_tr_excl))
    sp_incl = util.indices_in_window(spike_times, t1, t2)

    return tr_incl, sp_incl


# %% Calculate quality metrics.

def test_qm(u, ffig_template):
    """
    Test ISI, SNR and stationarity of spikes and spike waveforms.
    Non-stationarities can happen due to e.g.:
    - electrode drift, or
    - change in the state of the neuron.
    """

    # Init values.
    waveforms, wavetime, spike_dur, spike_times, sampl_per = get_base_data(u)

    # Waveform statistics.
    snr, wf_amp, wf_dur = waveform_stats(waveforms, wavetime)

    # Firing rate.
    ses_length = spike_times.t_stop - spike_times.t_start
    mean_rate = float(spike_times.size / ses_length)

    # ISI statistics.
    true_spikes, isi_viol = isi_stats(spike_times)
    unit_type = classify_unit(snr, true_spikes)

    # Time binned statistics.
    tbinned_stats = time_bin_data(spike_times, waveforms)
    tbin_lims, tbin_vals, wf_binned, sp_times_binned = tbinned_stats
    snr_t = [waveform_stats(wfb, wavetime)[0] for wfb in wf_binned]
    rate_t = [spt.size for spt in sp_times_binned] / BIN_LEN
    ISIvr_t = np.array([isi_stats(spt)[1] for spt in sp_times_binned])
    tr_starts = u.TrialParams.TrialStart
    tr_incl, sp_incl = test_drift(tbin_vals, rate_t, tbin_lims,
                                  tr_starts, spike_times)

    # Add quality metrics to unit.
    u.QualityMetrics['SNR'] = snr
    u.QualityMetrics['MeanWfAmplitude'] = np.mean(wf_amp)
    u.QualityMetrics['MeanWfDuration'] = np.mean(spike_dur).rescale(us)
    u.QualityMetrics['MeanFiringRate'] = mean_rate
    u.QualityMetrics['ISIviolation'] = isi_viol
    u.QualityMetrics['TrueSpikes'] = true_spikes
    u.QualityMetrics['UnitType'] = unit_type

    # Trial removal info.
    ntr = len(tr_starts)
    ntr_inc = len(tr_incl)
    u.QualityMetrics['NumTrialsRecorded'] = ntr
    u.QualityMetrics['NumTrialsKept'] = ntr_inc
    u.QualityMetrics['NumTrialsExcluded'] = ntr - ntr_inc
    u.QualityMetrics['TrialsKept'] = tr_incl
    u.QualityMetrics['NumTrialsExcluded'] = set(range(ntr)) - set(tr_incl)

    # Plot quality metric results.
    if ffig_template is not None:
        plot_qm(u, snr, wf_amp, wf_dur, mean_rate, isi_viol, true_spikes,
                unit_type, tbin_vals, snr_t, rate_t, ISIvr_t,
                tr_incl, sp_incl, ffig_template)

    return u


# %% Plot quality metrics.

def plot_qm(u, snr, wf_amp, wf_dur, mean_rate, isi_viol, true_spikes,
            unit_type, tbin_vals, snr_t, rate_t, ISIvr_t,
            tr_incl, sp_incl, ffig_template):
    """Plot quality metrics related figures."""

    # Init values.
    waveforms, wavetime, spike_dur, spike_times, sampl_per = get_base_data(u)

    # Minimum and maximum gain.
    gmin = min(-REC_GAIN/2, np.min(waveforms))
    gmax = max(REC_GAIN/2, np.max(waveforms))

    # %% Init plots.

    # Init plotting.
    fig = plot.figure(figsize=(12, 12))
    gs1 = gs.GridSpec(3, 3)
    sp = np.array([plt.subplot(gs) for gs in gs1]).reshape(gs1.get_geometry())
    ax_wf_inc, ax_wf_exc, ax_filler = sp[0, 0], sp[1, 0], sp[2, 0]
    ax_wf_amp, ax_wf_dur, ax_amp_dur = sp[0, 1], sp[1, 1], sp[2, 1]
    ax_snr, ax_rate, ax_isi = sp[0, 2], sp[1, 2], sp[2, 2]

    ax_filler.axis('off')

    # Common variables, limits and labels.
    sp_i = range(-WF_T_START, waveforms.shape[1]-WF_T_START)
    sp_t = sp_i * sampl_per
    ses_t_lim = [spike_times.t_start, spike_times.t_stop]
    ss = 1.0  # marker size on scatter plot
    sa = .8  # marker alpha on scatter plot
    glim = [gmin, gmax]  # gain axes limit
    wf_t_lim = [min(sp_t), max(sp_t)]
    dur_lim = [0*us, wavetime[-1] - wavetime[WF_T_START]]  # same across units
    amp_lim = [0, gmax - gmin]  # [np.min(wf_ampl), np.max(wf_ampl)]

    tr_alpha = 0.25  # alpha of trial event lines

    # Color spikes by their occurance over session time.
    my_cmap = plt.get_cmap('jet')
    sp_cols = np.tile(np.array([0., 0., 0., 0.]), (len(spike_times), 1))
    sp_t_inc = np.array(spike_times[sp_incl])
    sp_t_inc_shifted = sp_t_inc - sp_t_inc.min()
    sp_cols[sp_incl, :] = my_cmap(sp_t_inc_shifted/sp_t_inc_shifted.max())
    # Put excluded trials to the front, and randomise order of included trials
    # so later spikes don't cover earlier ones.
    sp_order = np.hstack((np.where(np.invert(sp_incl))[0],
                          np.random.permutation(np.where(sp_incl)[0])))

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

    # All waveforms over spike time.
#    wfs = np.transpose(waveforms)
#    title = 'All waveforms, n = {} spikes'.format(waveforms.shape[0])
#    plot.lines(sp_t, wfs, xlim=wf_t_lim, ylim=glim, color='g', alpha=0.005,
#               title=title, xlab=wf_t_lab, ylab=volt_lab,
#               ax=ax_wf_all)
#
#    # Waveform density: 2-D histrogram of waveforms over spike time.
#    x = np.tile(sp_t, waveforms.shape[0])
#    y = waveforms.reshape((1, waveforms.size))[0, ]
#    title = 'Waveform density'
#    nbins = waveforms.shape[1]
#    plot.histogram2D(x, y, nbins, hist_type='hist2d', xlim=wf_t_lim, ylim=glim,
#                     xlab=wf_t_lab, ylab=volt_lab, title=title, ax=ax_wf_den)

    title = 'All waveforms, n = {} spikes'.format(waveforms.shape[0])

    # Plot excluded and included waveforms on different axes.
    # Color included by occurance in session time to help detect drifts.
    wfs = np.transpose(waveforms)
    for i in sp_order:
        ax = ax_wf_inc if sp_incl[i] else ax_wf_exc
        ax.plot(sp_t, wfs[:, i], color=sp_cols[i, :], alpha=0.05)

    # Format waveform plots
    inc_ttl = 'Included waveforms, n = {} spikes'.format(sum(sp_incl))
    exl_ttl = 'Excluded waveforms, n = {} spikes'.format(sum(np.invert(sp_incl)))
    for ax, ttl in [(ax_wf_inc, inc_ttl), (ax_wf_exc, exl_ttl)]:
        plot.set_limits(xlim=wf_t_lim, ylim=glim, ax=ax)
        plot.show_ticks(xtick_pos='none', ytick_pos='none', ax=ax)
        plot.show_spines(True, False, False, False, ax=ax)
        plot.set_labels(title=ttl, xlab=wf_t_lab, ylab=volt_lab, ax=ax)

    # %% Waveform summary metrics.

    # Function to return colors of spikes / waveforms
    # based on whether they are included / excluded.
    def get_color(col_incld, col_bckgrnd='grey'):
        cols = np.array(len(sp_incl) * [col_bckgrnd])
        cols[sp_incl] = col_incld
        return cols

    # Waveform amplitude across session time.
    m_amp, sd_amp = np.mean(wf_amp), np.std(wf_amp)
    title = 'Waveform amplitude: {:.1f} $\pm$ {:.1f}'.format(m_amp, sd_amp)
    plot.scatter(spike_times, wf_amp, add_r=False, c=get_color('m'), s=ss,
                 xlab=ses_t_lab, ylab=amp_lab, xlim=ses_t_lim, ylim=amp_lim,
                 edgecolors='none', alpha=sa, title=title, ax=ax_wf_amp)

    # Waveform duration across session time.
    wf_dur = spike_dur.rescale(us)  # to use TPLCell's waveform duration
    mdur, sdur = float(np.mean(wf_dur)), float(np.std(wf_dur))
    title = 'Waveform duration: {:.1f} $\pm$ {:.1f} $\mu$s'.format(mdur, sdur)
    plot.scatter(spike_times, wf_dur, add_r=False, c=get_color('c'), s=ss,
                 xlab=ses_t_lab, ylab=dur_lab, xlim=ses_t_lim, ylim=dur_lim,
                 edgecolors='none', alpha=sa, title=title, ax=ax_wf_dur)

    # Waveform duration against amplitude.
    title = 'Waveform duration - amplitude'
    plot.scatter(wf_dur[sp_order], wf_amp[sp_order], c=sp_cols[sp_order], s=ss,
                 xlab=dur_lab, ylab=amp_lab, xlim=dur_lim, ylim=amp_lim,
                 edgecolors='none', alpha=sa, title=title, ax=ax_amp_dur)

    # %% SNR, firing rate and spike timing.

    # SNR over session time.
    snr_sd = np.std(snr_t)
    title = 'SNR: {:.2f} $\pm$ {:.2f}'.format(snr, snr_sd)
    ylim = [0, 1.1*np.max(snr_t)]
    plot.lines(tbin_vals, snr_t, c='y', xlim=ses_t_lim, ylim=ylim,
               title=title, xlab=ses_t_lab, ylab='SNR',
               ax=ax_snr)

    # Firing rate over session time.
    srate = float(np.std(rate_t))
    title = 'Firing rate: {:.1f} $\pm$ {:.1f} spike/s'.format(mean_rate, srate)
    ylim = [0, 1.1*np.max(rate_t.magnitude)]
    plot.lines(tbin_vals, rate_t, c='b', xlim=ses_t_lim, ylim=ylim,
               title=title, xlab=ses_t_lab, ylab='Firing rate (spike/s)',
               ax=ax_rate)

    # Interspike interval distribution.
    ylab = 'ISI violations (%)'
    title = ('ISI violations: {:.2f}%'.format(isi_viol) +
             ', true spikes: {:.1f}%'.format(true_spikes))
    ylim = [0, 1.1*np.max(ISIvr_t)]
    plot.lines(tbin_vals, ISIvr_t, c='r', xlim=ses_t_lim, ylim=ylim,
               xlab=ses_t_lab, ylab=ylab, title=title, ax=ax_isi)

    # Add trial markers and highlight included period in each plot.
    for ax in [ax_snr, ax_rate, ax_isi]:
        plot.plot_events(tr_markers, t_unit=s, lw=0.5, ls='--',
                         alpha=tr_alpha, ax=ax)
        t1, t2 = trial_starts[min(tr_incl)], trial_starts[max(tr_incl)]
        incl_segment = dict(selected=[t1, t2])
        plot.plot_segments(incl_segment, t_unit=s, alpha=0.2,
                           color='grey', ymax=0.96, ax=ax)

    # %% Save figure and metrics

    fig_title = '{}: "{}"'.format(u.Name, unit_type)
    fig.suptitle(fig_title, y=0.98, fontsize='xx-large')
    gs1.tight_layout(fig, rect=[0, 0.0, 1, 0.94])
    ffig = ffig_template.format(u.name_to_fname())
    plot.save_fig(fig, ffig)
