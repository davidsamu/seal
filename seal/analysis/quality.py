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


# %% Core functions.

def waveform_stats(wfs, wtime, t_start=None):
    """Calculates SNR, amplitude and durations of spike waveforms."""

    # SNR: std of mean waveform divided by std of residual waveform (noise).
    wf_mean = np.mean(wfs, 0)
    wf_std = wfs - wf_mean
    snr = np.std(wf_mean) / np.std(wf_std)

    # Indices of minimum and maximum times.
    imin = np.argmin(wfs, 1) if t_start is None else wfs.shape[0] * [t_start]
    imax = [np.argmax(w[imin[i]:]) + imin[i] for i, w in enumerate(wfs)]

    # Duration: time difference between times of minimum and maximum values.
    wf_tmin = wtime[imin]
    wf_tmax = wtime[imax]
    wf_dur = wf_tmax - wf_tmin

    # Amplitude: value difference between mininum and maximum values.
    wmin = wfs[np.arange(len(wtime)), imin]
    wmax = wfs[np.arange(len(wtime)), imax]
    wf_amp = wmax - wmin

    return snr, wf_amp, wf_dur


def isi_stats(spike_times, isi_th=1.0*ms):
    """Returns ISIs and some related statistics."""

    isi = statistics.isi(spike_times).rescale(ms)

    # Percent of spikes violating ISI treshold.
    n_isi_viol = sum(isi < isi_th)
    percent_isi_viol = 100 * n_isi_viol / isi.size

    # Percent of spikes estimated to originate from the sorted single unit.
    # Based on refractory period violations.
    # See Hill et al., 2011: Quality Metrics to Accompany Spike Sorting of
    # Extracellular Signals
    N = spike_times.size
    T = (spike_times.t_stop - spike_times.t_start).rescale(ms)
    r = n_isi_viol
    tmax = isi_th
    tmin = 0.675 * ms  # width of censored period
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


def test(unit, ffig_template):
    """
    Test ISI, SNR and stationarity of spikes and spike waveforms.
    Non-stationarities can happen due to e.g.:
    - electrode drift, or
    - change in the state of the neuron.
    """

    # %% Init analyses.

    # Init values.
    waveforms = unit.UnitParams['SpikeWaveforms']
    wavetime = unit.UnitParams['WaveformTime']
    spike_dur = unit.UnitParams['SpikeDuration']
    spike_times = unit.UnitParams['SpikeTimes']
    sampl_per = unit.SessParams['SamplPer']

    # Waveform statistics.
    wf_t_start = 9  # align values by threshold crossing (not at minimum time)
    snr, wf_amp, wf_dur = waveform_stats(waveforms, wavetime, wf_t_start)
    gmin = min(-2050, np.min(waveforms))  # minimum gain
    gmax = max(2050, np.max(waveforms))  # maximum gain

    # ISI statistics.
    isi_th = 1.0*ms
    true_spikes, isi_viol = isi_stats(spike_times, isi_th)
    unit_type = classify_unit(snr, true_spikes)

    # Time bins and binned waveforms and spike times
    # for statistics over session time.
    bin_length = 60*s
    t_start = spike_times.t_start
    t_stop = spike_times.t_stop
    ses_length = t_stop - t_start
    nbins = int((ses_length) / bin_length)
    tbin_lims = util.quantity_linspace(t_start, t_stop, s, nbins)
    tbin_lims = [(tbin_lims[i], tbin_lims[i+1])
                 for i in range(len(tbin_lims)-1)]
    tbin_vals = [np.mean([t1, t2]) for t1, t2 in tbin_lims]
    sp_idx_binned = [util.indices_in_window(spike_times, t1, t2)
                     for t1, t2 in tbin_lims]
    wf_binned = [waveforms[sp_idx] for sp_idx in sp_idx_binned]
    sp_times_binned = [spike_times[sp_idx] for sp_idx in sp_idx_binned]

    # Trial markers
    trms = unit.TrialParams.TrialStart[9::10]
    tr_markers = dict((tr_i+1, tr_t) for tr_i, tr_t in zip(trms.index, trms))

    # %% Init plots.

    # Init plotting.
    fig = plot.figure(figsize=(12, 12))
    gs1 = gs.GridSpec(3, 3)
    sp = np.array([plt.subplot(gs) for gs in gs1]).reshape(gs1.get_geometry())
    ax_wf_all, ax_wf_den, ax_wf_ses = sp[0, 0], sp[1, 0], sp[2, 0]
    ax_wf_amp, ax_wf_dur, ax_amp_dur = sp[0, 1], sp[1, 1], sp[2, 1]
    ax_snr, ax_rate, ax_isi = sp[0, 2], sp[1, 2], sp[2, 2]

    # Common variables, limits and labels.
    sp_t = range(-wf_t_start, waveforms.shape[1]-wf_t_start) * sampl_per
    ses_t_lim = [t_start, t_stop]
    ss = 1.0  # marker size on scatter plot
    sa = 0.5  # marker alpha on scatter plot
    glim = [gmin, gmax]  # gain axes limit
    # g_lim = [np.min(waveforms), np.max(waveforms)] # alternative gain limit
    wf_t_lim = [min(sp_t), max(sp_t)]
    dur_lim = [0*us, wavetime[-1] - wavetime[wf_t_start]]  # same across units
    # dur_lim = [0*us, np.max(wf_dur)]  # for unit-wise duration limit
    amp_lim = [0, gmax - gmin]  # [np.min(wf_ampl), np.max(wf_ampl)]

    tr_alpha = 0.25  # alpha of trial event lines

    # Colors of spikes by their occurance over session time.
    my_cmap = plt.get_cmap('jet')
    cols = my_cmap(spike_times.magnitude/max(spike_times.magnitude))
    # Random order so later spikes don't cover earlier ones.
    sp_order = np.array(range(len(spike_times)))
    np.random.shuffle(sp_order)

    wf_t_lab = 'Waveform time ($\mu$s)'
    ses_t_lab = 'Session time (s)'
    volt_lab = 'Voltage (normalized)'
    amp_lab = 'Amplitude'
    dur_lab = 'Duration ($\mu$s)'

    # %% Waveform shape analysis.

    # All waveforms over spike time.
    wfs = np.transpose(waveforms)
    title = 'All waveforms, n = {} spikes'.format(waveforms.shape[0])
    plot.lines(wfs, sp_t, xlim=wf_t_lim, ylim=glim, color='g', alpha=0.005,
               title=title, xlab=wf_t_lab, ylab=volt_lab,
               ax=ax_wf_all)

    # Waveform density: 2-D histrogram of waveforms over spike time.
    x = np.tile(sp_t, waveforms.shape[0])
    y = waveforms.reshape((1, waveforms.size))[0, ]
    title = 'Waveform density'
    nbins = waveforms.shape[1]
    plot.histogram2D(x, y, nbins, hist_type='hist2d', xlim=wf_t_lim, ylim=glim,
                     xlab=wf_t_lab, ylab=volt_lab, title=title, ax=ax_wf_den)

    # All waveforms over spike time, colored by occurance in session time.
    wfs = np.transpose(waveforms)
    title = 'All waveforms, colored by session time'
    for i in sp_order:
        ax_wf_ses.plot(sp_t, wfs[:, i], color=cols[i, :], alpha=0.05)
    plot.set_limits(xlim=wf_t_lim, ylim=glim, ax=ax_wf_ses)
    plot.show_ticks(xtick_pos='none', ytick_pos='none', ax=ax_wf_ses)
    plot.show_spines(True, False, False, False, ax=ax_wf_ses)
    plot.set_labels(title=title, xlab=wf_t_lab, ylab=volt_lab,
                    ax=ax_wf_ses)

    # %% Waveform summary metrics.

    # Waveform amplitude across session time.
    m_amp, sd_amp = np.mean(wf_amp), np.std(wf_amp)
    title = 'Waveform amplitude: {:.1f} $\pm$ {:.1f}'.format(m_amp, sd_amp)
    plot.scatter(spike_times, wf_amp, add_r=False, c='m', s=ss,
                 xlab=ses_t_lab, ylab=amp_lab, xlim=ses_t_lim, ylim=amp_lim,
                 edgecolors='none', alpha=sa, title=title, ax=ax_wf_amp)

    # Waveform duration across session time.
    wf_dur = spike_dur.rescale(us)  # to use TPLCell's waveform duration
    mdur, sdur = float(np.mean(wf_dur)), float(np.std(wf_dur))
    title = 'Waveform duration: {:.1f} $\pm$ {:.1f} $\mu$s'.format(mdur, sdur)
    plot.scatter(spike_times, wf_dur, add_r=False, c='c', s=ss,
                 xlab=ses_t_lab, ylab=dur_lab, xlim=ses_t_lim, ylim=dur_lim,
                 edgecolors='none', alpha=sa, title=title, ax=ax_wf_dur)

    # Waveform duration against amplitude.
    title = 'Waveform duration - amplitude'
    plot.scatter(wf_dur[sp_order], wf_amp[sp_order], c=cols[sp_order], s=ss,
                 xlab=dur_lab, ylab=amp_lab, xlim=dur_lim, ylim=amp_lim,
                 edgecolors='none', alpha=sa, title=title, ax=ax_amp_dur)

    # %% SNR, firing rate and spike timing.

    # SNR over session time.
    snr_t = [waveform_stats(wfb, wavetime)[0] for wfb in wf_binned]
    snr_sd = np.std(snr_t)
    title = 'SNR: {:.2f} $\pm$ {:.2f}'.format(snr, snr_sd)
    ylim = [0, 1.1*np.max(snr_t)]
    plot.lines(snr_t, tbin_vals, c='y', xlim=ses_t_lim, ylim=ylim,
               title=title, xlab=ses_t_lab, ylab='SNR',
               ax=ax_snr)
    plot.plot_events(tr_markers, t_unit=s, lw=0.5, ls='--', alpha=tr_alpha,
                     ax=ax_snr)

    # Firing rate across session time.
    rates = [spt.size for spt in sp_times_binned] / bin_length
    mrate = float(spike_times.size / ses_length)
    srate = float(np.std(rates))
    title = 'Firing rate: {:.1f} $\pm$ {:.1f} spike/s'.format(mrate, srate)
    ylim = [0, 1.1*np.max(rates.magnitude)]
    plot.lines(rates, tbin_vals, c='b', xlim=ses_t_lim, ylim=ylim,
               title=title, xlab=ses_t_lab, ylab='Firing rate (spike/s)',
               ax=ax_rate)
    plot.plot_events(tr_markers, t_unit=s, lw=0.5, ls='--', alpha=tr_alpha,
                     ax=ax_rate)

    # Interspike interval distribution
    isi_viols = np.array([isi_stats(spt, isi_th)[1]
                          for spt in sp_times_binned])
    ylab = 'ISI violations (%)'
    title = ('ISI violations: {:.2f}%'.format(isi_viol) +
             ', true spikes: {:.1f}%'.format(true_spikes))
    ylim = [0, 1.1*np.max(isi_viols)]
    plot.lines(isi_viols, tbin_vals, c='r', xlim=ses_t_lim, ylim=ylim,
               xlab=ses_t_lab, ylab=ylab, title=title, ax=ax_isi)
    plot.plot_events(tr_markers, t_unit=s, lw=0.5, ls='--', alpha=tr_alpha,
                     ax=ax_isi)

    # %% Save figure and metrics

    # Format and save figure
    fig_title = '{}: "{}"'.format(unit.Name, unit_type)
    fig.suptitle(fig_title, y=0.98, fontsize='xx-large')
    gs1.tight_layout(fig, rect=[0, 0.0, 1, 0.94])
    ffig = ffig_template.format(unit.name_to_fname())
    plot.save_fig(fig, ffig)

    # Add quality metrics to unit.
    unit.QualityMetrics['SNR'] = snr
    unit.QualityMetrics['MeanWfAmplitude'] = np.mean(wf_amp)
    unit.QualityMetrics['MeanWfDuration'] = np.mean(spike_dur)
    unit.QualityMetrics['ISIviolation'] = isi_viol
    unit.QualityMetrics['TrueSpikes'] = true_spikes
    unit.QualityMetrics['UnitType'] = unit_type

    return unit
