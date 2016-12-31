#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to calculate and plot quality metrics of units  after spike sorting
(SNR, ISIvr, etc), and to exclude trials / units not meeting QC criteria.

@author: David Samu
"""

import numpy as np
import pandas as pd
from quantities import s, ms, us

import elephant

from seal.util import util
from seal.plot import putil, pplot, pwaveform


# %% Constants.

# Recording constants.
REC_GAIN = 4100                # gain of recording
CENSORED_PRD_LEN = 0.675 * ms  # length of censored period
WF_T_START = 9                 # start index of spikes (aligned by Plexon)

# Constants related to quality metrics calculation.
ISI_TH = 1.0 * ms          # ISI violation threshold
MAX_DRIFT_RATIO = 3        # maximum tolerable drift ratio
MIN_BIN_LEN = 120 * s      # minimum window length for firing binned statistics

# Constants related to unit exclusion.
min_SNR = 1.0          # min. SNR
min_FR = 1.0           # min. firing rate (sp/s)
max_ISIvr = 1.0        # max. ISI violation ratio (%)
min_inc_trs_rat = 50   # min. ratio of included trials out of all recorded (%)


# %% Utility functions.

def get_base_data(u):
    """Return base data of unit for quality metrics calculation."""

    # Init values.
    waveforms = np.array(u.Waveforms)
    wavetime = np.array(u.Waveforms.columns) * us
    spk_dur = u.SpikeParams['dur']
    spk_times = u.SpikeParams['time']
    sampl_per = u.SessParams['sampl_prd']

    return waveforms, wavetime, spk_dur, spk_times, sampl_per


def time_bin_data(spk_times, waveforms):
    """Return time binned data for statistics over session time."""

    # Time bins and binned waveforms and spike times.
    t_start, t_stop = spk_times.min()*s, spk_times.max()*s
    nbins = max(int(np.floor((t_stop - t_start) / MIN_BIN_LEN)), 1)
    tbin_lims = util.quantity_linspace(t_start, t_stop, nbins+1, s)
    tbins = [(tbin_lims[i], tbin_lims[i+1]) for i in range(len(tbin_lims)-1)]
    tbin_vmid = np.array([np.mean([t1, t2]) for t1, t2 in tbins])*s
    spk_idx_binned = [util.indices_in_window(spk_times, t1, t2)
                      for t1, t2 in tbins]
    wf_binned = [waveforms[spk_idx] for spk_idx in spk_idx_binned]
    spk_times_binned = [spk_times[spk_idx] for spk_idx in spk_idx_binned]

    return tbins, tbin_vmid, wf_binned, spk_times_binned


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


def isi_stats(spk_times):
    """Returns ISIs and some related statistics."""

    # No spike: ISI v.r. and TrueSpikes are uninterpretable.
    if not spk_times.size:
        return np.nan, np.nan

    # Only one spike: ISI v.r. is 0%, TrueSpikes is 100%.
    if spk_times.size == 1:
        return 100, 0

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

    return true_spikes, percent_ISI_vr


def classify_unit(snr, true_spikes):
    """Classify unit as single or multi-unit."""

    if true_spikes >= 90 and snr >= 2.0:
        unit_type = 'single unit'
    else:
        unit_type = 'multi unit'

    return unit_type


def test_drift(t, v, tbins, tr_starts, spk_times):
    """Test drift (gradual, or more instantaneous jump or drop) in variable."""

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
    prd_res = pd.DataFrame(index=range(len(v)), columns=cols)
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
    t1 = prd_res.t_start[idx]
    t2 = prd_res.t_end[idx]
    # Trial indices within longest period.
    first_tr_inc = prd_res.tr_start_i[idx]
    last_tr_inc = prd_res.tr_end_i[idx] - 1

    # Return included trials and spikes.
    prd_inc = util.indices_in_window(np.array(range(len(tbins))), prd1, prd2)
    tr_inc = util.indices_in_window(np.array(range(len(tr_starts))),
                                    first_tr_inc, last_tr_inc)
    spk_inc = util.indices_in_window(spk_times, t1, t2)

    return t1, t2, prd_inc, tr_inc, spk_inc


# %% Calculate quality metrics, and find trials and units to be excluded.

def test_qm(u):
    """
    Test ISI, SNR and stationarity of FR and spike waveforms.
    Find trials with unacceptable drift.

    Non-stationarities can happen due to e.g.:
    - electrode drift, or
    - change in the state of the neuron.
    """

    if u.is_empty():
        return

    # Init values.
    waveforms, wavetime, spk_dur, spk_times, sampl_per = get_base_data(u)

    # Time binned statistics.
    tbinned_stats = time_bin_data(spk_times, waveforms)
    tbins, tbin_vmid, wf_binned, spk_times_binned = tbinned_stats

    rate_t = np.array([spkt.size/(t2-t1).rescale(s)
                       for spkt, (t1, t2) in zip(spk_times_binned, tbins)]) / s

    # Test drifts and reject trials if necessary.
    tr_starts = u.TrialParams.TrialStart
    test_res = test_drift(tbin_vmid, rate_t, tbins, tr_starts, spk_times)
    t1_inc, t2_inc, prd_inc, tr_inc, spk_inc = test_res

    # Waveform statistics of included spikes only.
    snr, wf_amp, wf_dur = waveform_stats(waveforms[spk_inc], wavetime)

    # Firing rate.
    mean_rate = float(np.sum(spk_inc) / (t2_inc - t1_inc))

    # ISI statistics.
    true_spikes, ISIvr = isi_stats(np.array(spk_times[spk_inc])*s)
    unit_type = classify_unit(snr, true_spikes)

    # Add quality metrics to unit.
    u.QualityMetrics['SNR'] = snr
    u.QualityMetrics['mWfAmpl'] = np.mean(wf_amp)
    u.QualityMetrics['mWfDur'] = np.mean(spk_dur[spk_inc])
    u.QualityMetrics['mFR'] = mean_rate
    u.QualityMetrics['ISIvr'] = ISIvr
    u.QualityMetrics['TrueSpikes'] = true_spikes
    u.QualityMetrics['UnitType'] = unit_type

    # Trial removal info.
    u.update_included_trials(tr_inc)

    # Run unit exclusion test.
    to_excl = test_rejection(u)
    u.set_excluded(to_excl)

    # Return all results.
    res = {'tbin_vmid': tbin_vmid, 'tbins': tbins, 'rate_t': rate_t,
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

    # Extremely low unit activity (FR).
    test_passed['FR'] = qm['mFR'] > min_FR

    # Extremely high ISI violation ratio (ISIvr).
    test_passed['ISI'] = qm['ISIvr'] < max_ISIvr

    # Insufficient number of trials (ratio of included trials).
    inc_trs_ratio = 100 * qm['NTrialsInc'] / qm['NTrialsTotal']
    test_passed['IncTrsRatio'] = inc_trs_ratio > min_inc_trs_rat

    # Exclude unit if any of the criteria is not met.
    exclude = not test_passed.all()

    return exclude


# %% Plot quality metrics.

def plot_qm(u, tbin_vmid, tbins, rate_t, t1_inc, t2_inc, prd_inc, tr_inc,
            spk_inc, add_lbls=False, ftempl=None, fig=None, sps=None):
    """Plot quality metrics related figures."""

    # Init values.
    mean_rate = u.QualityMetrics['mFR']
    waveforms, wavetime, spk_dur, spk_times, sampl_per = get_base_data(u)

    # Get waveform stats of included and excluded spikes.
    wf_inc = waveforms[spk_inc]
    snr_all, wf_amp_all, wf_dur_all = waveform_stats(waveforms, wavetime)
    snr_inc, wf_amp_inc, wf_dur_inc = waveform_stats(wf_inc, wavetime)

    # Minimum and maximum gain.
    gmin = min(-REC_GAIN/2, np.min(waveforms))
    gmax = max(REC_GAIN/2, np.max(waveforms))

    # %% Init plots.

    # Init plotting theme.
    putil.set_style('notebook', 'ticks')
    putil.inline_off()   # disable inline plotting to prevent memory leak

    # Init figure and gridspec.
    fig = putil.figure(fig)
    if sps is None:
        sps = putil.gridspec(1, 1)[0]
    ogsp = putil.embed_gsp(sps, 2, 1, height_ratios=[0.12, 1])

    info_sps, qm_sps = ogsp[0], ogsp[1]

    # Info header.
    gsp_info = putil.embed_gsp(info_sps, 1, 1)
    info_ax = fig.add_subplot(gsp_info[0, 0])
    putil.unit_info(u, ax=info_ax)

    # Core axes.
    gsp = putil.embed_gsp(qm_sps, 3, 2, wspace=0.3, hspace=0.4)
    ax_wf_inc, ax_wf_exc = [fig.add_subplot(gsp[0, i]) for i in (0, 1)]
    ax_wf_amp, ax_wf_dur = [fig.add_subplot(gsp[1, i]) for i in (0, 1)]
    ax_amp_dur, ax_rate = [fig.add_subplot(gsp[2, i]) for i in (0, 1)]

    # Trial markers.
    trial_starts = u.TrialParams.TrialStart
    tr_markers = pd.DataFrame({'time': trial_starts[9::10]})
    tr_markers['label'] = [str(itr+1) if i % 2 else ''
                           for i, itr in enumerate(tr_markers.index)]

    # Common variables, limits and labels.
    spk_i = range(-WF_T_START, waveforms.shape[1]-WF_T_START)
    spk_t = spk_i * sampl_per
    ses_t_lim = [min(spk_times.min() * s, trial_starts.iloc[0]),
                 max(spk_times.max() * s, trial_starts.iloc[-1])]
    ss = 1.0  # marker size on scatter plot
    sa = .80  # marker alpha on scatter plot
    g_lim = [gmin, gmax]  # gain axes limit
    wf_t_lim = [min(spk_t), max(spk_t)]
    dur_lim = [0*us, wavetime[-1]-wavetime[WF_T_START]]  # same across units
    amp_lim = [0, gmax-gmin]  # [np.min(wf_ampl), np.max(wf_ampl)]

    # Color spikes by their occurance over session time.
    my_cmap = putil.get_cmap('jet')
    spk_cols = np.tile(np.array([.25, .25, .25, .25]), (len(spk_times), 1))
    if np.any(spk_inc):  # check if there is any spike included
        spk_t_inc = np.array(spk_times[spk_inc])
        tmin, tmax = float(spk_times.min()), float(spk_times.max())
        spk_cols[spk_inc, :] = my_cmap((spk_t_inc-tmin) / (tmax-tmin))
    # Put excluded trials to the front, and randomise order of included trials
    # so later spikes don't systematically cover earlier ones.
    spk_order = np.hstack((np.where(np.invert(spk_inc))[0],
                           np.random.permutation(np.where(spk_inc)[0])))

    # Common labels for plots
    wf_t_lab = 'WF time ($\mu$s)'
    ses_t_lab = 'Recording time (s)'
    volt_lab = 'Voltage'
    amp_lab = 'Amplitude'
    dur_lab = 'Duration ($\mu$s)'

    # %% Waveform shape analysis.

    # Plot included and excluded waveforms on different axes.
    # Color included by occurance in session time to help detect drifts.
    s_waveforms, s_spk_cols = waveforms[spk_order, :], spk_cols[spk_order]
    for st in ('Included', 'Excluded'):
        ax = ax_wf_inc if st == 'Included' else ax_wf_exc
        spk_idx = spk_inc if st == 'Included' else np.invert(spk_inc)
        tr_idx = tr_inc if st == 'Included' else np.invert(tr_inc)

        title = '{} WFs, {} spikes, {} trials'.format(st, sum(spk_idx),
                                                      sum(tr_idx))
        # Select waveforms and colors.
        rand_spk_idx = spk_idx[spk_order]
        wfs = s_waveforms[rand_spk_idx, :]
        cols = s_spk_cols[rand_spk_idx]

        # Plot waveforms.
        xlab, ylab = (wf_t_lab, volt_lab) if add_lbls else (None, None)
        pwaveform.wfs(wfs, spk_t, cols=cols, lw=0.1, alpha=0.05,
                      xlim=wf_t_lim, ylim=g_lim, title=title,
                      xlab=xlab, ylab=ylab, ax=ax)

    # %% Waveform summary metrics.

    # Waveform amplitude across session time.
    m_amp, sd_amp = float(np.mean(wf_amp_inc)), float(np.std(wf_amp_inc))
    title = 'WF amplitude: {:.1f} $\pm$ {:.1f}'.format(m_amp, sd_amp)
    xlab, ylab = (ses_t_lab, amp_lab) if add_lbls else (None, None)
    pplot.scatter(spk_times, wf_amp_all, spk_inc, c='m', bc='grey', s=ss,
                  xlab=xlab, ylab=ylab, xlim=ses_t_lim, ylim=amp_lim,
                  edgecolors='', alpha=sa, title=title, ax=ax_wf_amp)

    # Waveform duration across session time.
    wf_dur_all = spk_dur  # to use TPLCell's waveform duration
    wf_dur_inc = wf_dur_all[spk_inc]
    mdur, sdur = float(np.mean(wf_dur_inc)), float(np.std(wf_dur_inc))
    title = 'WF duration: {:.1f} $\pm$ {:.1f} $\mu$s'.format(mdur, sdur)
    xlab, ylab = (ses_t_lab, dur_lab) if add_lbls else (None, None)
    pplot.scatter(spk_times, wf_dur_all, spk_inc, c='c', bc='grey', s=ss,
                  xlab=xlab, ylab=ylab, xlim=ses_t_lim, ylim=dur_lim,
                  edgecolors='', alpha=sa, title=title, ax=ax_wf_dur)

    # Waveform duration against amplitude.
    title = 'WF duration - amplitude'
    xlab, ylab = (dur_lab, amp_lab) if add_lbls else (None, None)
    pplot.scatter(wf_dur_all[spk_order], wf_amp_all[spk_order],
                  c=spk_cols[spk_order], s=ss, xlab=xlab, ylab=ylab,
                  xlim=dur_lim, ylim=amp_lim, edgecolors='', alpha=sa,
                  title=title, ax=ax_amp_dur)

    # %% Firing rate.

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

    # Firing rate over session time.
    title = 'Firing rate: {:.1f} spike/s'.format(mean_rate)
    xlab, ylab = (ses_t_lab, putil.FR_lbl) if add_lbls else (None, None)
    ylim = [0, 1.25*np.max(rate_t.magnitude)]
    plot_periods(rate_t, 'b', ax_rate)
    pplot.lines([], [], c='b', xlim=ses_t_lim, ylim=ylim, title=title,
                xlab=xlab, ylab=ylab, ax=ax_rate)

    # Trial markers.
    putil.plot_events(tr_markers, lw=0.5, ls='--', alpha=0.35,
                      lbl_height=0.92, ax=ax_rate)

    # Excluded periods.
    excl_prds = []
    tstart, tstop = spk_times.min(), spk_times.max()
    if tstart != t1_inc:
        excl_prds.append(('beg', tstart, t1_inc))
    if tstop != t2_inc:
        excl_prds.append(('end', t2_inc, tstop))
    putil.plot_periods(excl_prds, ymax=0.92, ax=ax_rate)

    # %% Save figure.
    if ftempl is not None:
        fname = ftempl.format(u.name_to_fname())
        putil.save_gsp_figure(fig, gsp, fname, title, rect_height=0.92)
        putil.inline_on()

    return [ax_wf_inc, ax_wf_exc], ax_wf_amp, ax_wf_dur, ax_amp_dur, ax_rate
