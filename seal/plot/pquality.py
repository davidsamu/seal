# -*- coding: utf-8 -*-
"""
Functions to plot quality metrics of units  after spike sorting.

@author: David Samu
"""

import numpy as np
import pandas as pd

from quantities import us

from seal.plot import putil, pplot, pwaveform
from seal.quality import test_sorting


# %% Plot quality metrics.

def plot_qm(u, tbin_vmid, rate_t, t1_inc, t2_inc, prd_inc, tr_inc, spk_inc,
            add_lbls=False, ftempl=None, fig=None, sps=None):
    """Plot quality metrics related figures."""

    # Init values.
    waveforms = np.array(u.Waveforms)
    wavetime = u.Waveforms.columns * us
    spk_times = np.array(u.SpikeParams['time'], dtype=float)
    mean_rate = u.QualityMetrics['mFR']

    # Minimum and maximum gain.
    gmin = u.UnitParams['minV']
    gmax = u.UnitParams['maxV']

    # %% Init plots.

    # Disable inline plotting to prevent memory leak.
    putil.inline_off()

    # Init figure and gridspec.
    fig = putil.figure(fig)
    if sps is None:
        sps = putil.gridspec(1, 1)[0]
    ogsp = putil.embed_gsp(sps, 2, 1, height_ratios=[0.12, 1])

    info_sps, qm_sps = ogsp[0], ogsp[1]

    # Info header.
    info_ax = fig.add_subplot(info_sps)
    putil.hide_axes(info_ax)
    title = putil.get_unit_info_title(u)
    putil.set_labels(ax=info_ax, title=title)

    # Create axes.
    gsp = putil.embed_gsp(qm_sps, 3, 2, wspace=0.3, hspace=0.4)
    ax_wf_inc, ax_wf_exc = [fig.add_subplot(gsp[0, i]) for i in (0, 1)]
    ax_wf_amp, ax_wf_dur = [fig.add_subplot(gsp[1, i]) for i in (0, 1)]
    ax_amp_dur, ax_rate = [fig.add_subplot(gsp[2, i]) for i in (0, 1)]

    # Trial markers.
    trial_starts, trial_stops = u.TrData.TrialStart, u.TrData.TrialStop
    tr_markers = pd.DataFrame({'time': trial_starts[9::10]})
    tr_markers['label'] = [str(itr+1) if i % 2 else ''
                           for i, itr in enumerate(tr_markers.index)]

    # Common variables, limits and labels.
    WF_T_START = test_sorting.WF_T_START
    spk_t = u.SessParams.sampl_prd * (np.arange(waveforms.shape[1])-WF_T_START)
    ses_t_lim = test_sorting.get_start_stop_times(spk_times, trial_starts,
                                                  trial_stops)
    ss, sa = 1.0, 0.8  # marker size and alpha on scatter plot

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
    ses_t_lab = 'Recording time (s)'

    # %% Waveform shape analysis.

    # Plot included and excluded waveforms on different axes.
    # Color included by occurance in session time to help detect drifts.
    s_waveforms, s_spk_cols = waveforms[spk_order, :], spk_cols[spk_order]
    wf_t_lim, glim = [min(spk_t), max(spk_t)], [gmin, gmax]
    wf_t_lab, volt_lab = 'WF time ($\mu$s)', 'Voltage'
    for st in ('Included', 'Excluded'):
        ax = ax_wf_inc if st == 'Included' else ax_wf_exc
        spk_idx = spk_inc if st == 'Included' else np.invert(spk_inc)
        tr_idx = tr_inc if st == 'Included' else np.invert(tr_inc)

        nspsk, ntrs = sum(spk_idx), sum(tr_idx)
        title = '{} WFs, {} spikes, {} trials'.format(st, nspsk, ntrs)

        # Select waveforms and colors.
        rand_spk_idx = spk_idx[spk_order]
        wfs = s_waveforms[rand_spk_idx, :]
        cols = s_spk_cols[rand_spk_idx]

        # Plot waveforms.
        xlab, ylab = (wf_t_lab, volt_lab) if add_lbls else (None, None)
        pwaveform.plot_wfs(wfs, spk_t, cols=cols, lw=0.1, alpha=0.05,
                           xlim=wf_t_lim, ylim=glim, title=title,
                           xlab=xlab, ylab=ylab, ax=ax)

    # %% Waveform summary metrics.

    # Init data.
    wf_amp_all = u.SpikeParams['amplitude']
    wf_amp_inc = wf_amp_all[spk_inc]
    wf_dur_all = u.SpikeParams['duration']
    wf_dur_inc = wf_dur_all[spk_inc]

    # Set common limits and labels.
    dur_lim = [0, wavetime[-2]-wavetime[WF_T_START]]  # same across units
    glim = max(wf_amp_all.max(), gmax-gmin)
    amp_lim = [0, glim]

    amp_lab = 'Amplitude'
    dur_lab = 'Duration ($\mu$s)'

    # Waveform amplitude across session time.
    m_amp, sd_amp = wf_amp_inc.mean(), wf_amp_inc.std()
    title = 'WF amplitude: {:.1f} $\pm$ {:.1f}'.format(m_amp, sd_amp)
    xlab, ylab = (ses_t_lab, amp_lab) if add_lbls else (None, None)
    pplot.scatter(spk_times, wf_amp_all, spk_inc, c='m', bc='grey', s=ss,
                  xlab=xlab, ylab=ylab, xlim=ses_t_lim, ylim=amp_lim,
                  edgecolors='', alpha=sa, title=title, ax=ax_wf_amp)

    # Waveform duration across session time.
    mdur, sdur = wf_dur_inc.mean(), wf_dur_inc.std()
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
    tstart, tstop = ses_t_lim
    if tstart != t1_inc:
        excl_prds.append(('beg', tstart, t1_inc))
    if tstop != t2_inc:
        excl_prds.append(('end', t2_inc, tstop))
    putil.plot_periods(excl_prds, ymax=0.92, ax=ax_rate)

    # %% Post-formatting.

    # Maximize number of ticks on recording time axes to prevent covering.
    for ax in (ax_wf_amp, ax_wf_dur, ax_rate):
        putil.set_max_n_ticks(ax, 6, 'x')

    # %% Save figure.
    if ftempl is not None:
        fname = ftempl.format(u.name_to_fname())
        putil.save_gsp_figure(fig, gsp, fname, title, rect_height=0.92)
        putil.inline_on()

    return [ax_wf_inc, ax_wf_exc], ax_wf_amp, ax_wf_dur, ax_amp_dur, ax_rate
