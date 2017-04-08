# -*- coding: utf-8 -*-
"""
Functions to prepare data for decoding analysis.

@author: David Samu
"""

import numpy as np
import pandas as pd
import seaborn as sns

from quantities import deg

from seal.plot import putil, pplot
from seal.util import util, constants, ua_query
from seal.analysis import direction

# Constants.
min_n_units = 5           # minimum number of units to keep (0: off)
min_n_trs_per_unit = 5    # minimum number of trials per unit to keep (0: off)


# %% Unit and trial selection.

def select_units_trials(UA, utids=None, fres=None, ffig=None):
    """Select optimal set of units and trials for population decoding."""

    print('Selecting optimal set of units and trials for decoding...')

    # Init.
    if utids is None:
        utids = UA.utids(as_series=True)
    u_rt_grpby = utids.groupby(level=['rec', 'task'])

    # Unit info frame.
    UInc = pd.Series(False, index=utids.index)

    # Included trials by unit.
    IncTrs = pd.Series([(UA.get_unit(utid[:3], utid[3]).inc_trials())
                        for utid in utids], index=utids.index)

    # Result DF.
    rec_task = pd.MultiIndex.from_tuples([rt for rt, _ in u_rt_grpby],
                                         names=['rec', 'task'])
    cols = ['units', 'nunits', 'nallunits', '% remaining units',
            'trials', 'ntrials', 'nalltrials', '% remaining trials']
    RecInfo = pd.DataFrame(index=rec_task, columns=cols)
    rt_utids = [utids.xs((r, t), level=('rec', 'task')) for r, t in rec_task]
    RecInfo.nallunits = [len(utids) for utids in rt_utids]
    rt_ulist = [UA.get_unit(utids[0][:3], utids[0][3]) for utids in rt_utids]
    RecInfo.nalltrials = [int(u.QualityMetrics['NTrialsTotal'])
                          for u in rt_ulist]

    # Function to plot matrix (DF) included/excluded trials.
    def plot_inc_exc_trials(IncTrsMat, ax, title=None, ytitle=None,
                            xlab='Trial #', ylab=None,):
        # Plot on heatmap.
        sns.heatmap(IncTrsMat, cmap='RdYlGn', center=0.5, cbar=False, ax=ax)
        # Set tick labels.
        putil.hide_tick_marks(ax)
        tr_ticks = [1] + list(np.arange(25, IncTrsMat.shape[1]+1, 25))
        ax.xaxis.set_ticks(tr_ticks)
        ax.set_xticklabels(tr_ticks)
        putil.rot_xtick_labels(ax, 0)
        putil.rot_ytick_labels(ax, 0, va='center')
        putil.set_labels(ax, xlab, ylab, title, ytitle)

    # Init plotting.
    ytitle = 1.40
    putil.set_style('notebook', 'whitegrid')
    fig, gsp, axs = putil.get_gs_subplots(nrow=len(rec_task), ncol=3,
                                          subw=6, subh=4, create_axes=True)

    for i_rt, ((rec, task), rt_utids) in enumerate(u_rt_grpby):
        print('{} / {}: {} - {}'.format(i_rt+1, len(u_rt_grpby), rec, task))

        # Create matrix of included trials of recording & task of units.
        ch_idxs = rt_utids.index.droplevel(3).droplevel(0)
        n_alltrs = RecInfo.nalltrials[(rec, task)]
        IncTrsMat = pd.DataFrame(np.zeros((len(ch_idxs), n_alltrs), dtype=int),
                                 index=ch_idxs, columns=np.arange(n_alltrs)+1)
        for ch_idx, utid in zip(ch_idxs, rt_utids):
            IncTrsMat.loc[ch_idx].iloc[IncTrs[utid]] = 1

        # Plot included/excluded trials after preprocessing.
        ax = axs[i_rt, 0]
        ylab = '{} {}'.format(rec, task)
        title = ('Included (green) and excluded (red) trials'
                 if i_rt == 0 else None)
        plot_inc_exc_trials(IncTrsMat, ax, title, ytitle, ylab=ylab)

        # Calculate and plot overlap of trials across units.
        # How many trials will remain if we iteratively excluding units
        # with the least overlap with the rest of the units?
        def n_cov_trs(df):  # return number of trials covered in df
            return sum(df.all())

        def calc_heuristic(df):
            return df.shape[0] * n_cov_trs(df)

        n_trs = IncTrsMat.sum(1)
        n_units = IncTrsMat.shape[0]

        # Init results DF.
        columns = ('uid', 'ntrs_cov', 'n_rem_u', 'trial x units')
        tr_covs = pd.DataFrame(columns=columns, index=range(n_units+1))
        tr_covs.loc[0] = ('none', n_cov_trs(IncTrsMat), n_units,
                          calc_heuristic(IncTrsMat))

        # Subset of included units (to be updated in each iteration).
        uinc = IncTrsMat.index.to_series()
        for iu in range(1, len(uinc)):

            # Number of covered trials after removing each unit.
            sntrscov = pd.Series([n_cov_trs(IncTrsMat.loc[uinc.drop(uid)])
                                  for uid in uinc], index=uinc.index)

            #########################################
            # Select and remove unit that           #
            # (a) yields maximum trial coverage,    #
            # (b) has minimum number of trials      #
            #########################################
            maxtrscov = sntrscov.max()
            worst_us = sntrscov[sntrscov == maxtrscov].index  # (a)
            utrs = n_trs.loc[worst_us]
            uid_remove = utrs[(utrs == min(utrs))].index[0]   # (b)

            # Update current subset of units and their trial DF.
            uinc.drop(uid_remove, inplace=True)
            tr_covs.loc[iu] = (uid_remove, maxtrscov, len(uinc),
                               calc_heuristic(IncTrsMat.loc[uinc]))

        # Add last unit.
        tr_covs.iloc[-1] = (uinc[0], 0, 0, 0)

        # Plot covered trials against each units removed.
        ax_trc = axs[i_rt, 1]
        sns.tsplot(tr_covs['ntrs_cov'], marker='o', ms=4, color='b',
                   ax=ax_trc)
        title = ('Trial coverage during iterative unit removal'
                 if i_rt == 0 else None)
        xlab, ylab = 'current unit removed', '# trials covered'
        putil.set_labels(ax_trc, xlab, ylab, title, ytitle)
        ax_trc.xaxis.set_ticks(tr_covs.index)
        x_ticklabs = ['none'] + ['{} - {}'.format(ch, ui)
                                 for ch, ui in tr_covs.uid.loc[1:]]
        ax_trc.set_xticklabels(x_ticklabs)
        putil.rot_xtick_labels(ax_trc, 45)
        ax_trc.grid(True)

        # Add # of remaining units to top.
        ax_remu = ax_trc.twiny()
        ax_remu.xaxis.set_ticks(tr_covs.index)
        ax_remu.set_xticklabels(list(range(len(x_ticklabs)))[::-1])
        ax_remu.set_xlabel('# units remaining')
        ax_remu.grid(None)

        # Add heuristic index.
        ax_heur = ax_trc.twinx()
        sns.tsplot(tr_covs['trial x units'], linestyle='--', marker='o',
                   ms=4, color='m',  ax=ax_heur)
        putil.set_labels(ax_heur, ylab='remaining units x covered trials')
        [tl.set_color('m') for tl in ax_heur.get_yticklabels()]
        [tl.set_color('b') for tl in ax_trc.get_yticklabels()]
        ax_heur.grid(None)

        # Decide on which units to exclude.
        min_n_trials = min_n_trs_per_unit * tr_covs['n_rem_u']
        sub_tr_covs = tr_covs[(tr_covs['n_rem_u'] >= min_n_units) &
                              (tr_covs['ntrs_cov'] >= min_n_trials)]

        # If any subset of units passed above criteria.
        rem_uids, exc_uids = pd.Series(), tr_covs.uid[1:]
        n_tr_rem, n_tr_exc = 0, IncTrsMat.shape[1]
        if len(sub_tr_covs.index):
            hmax_idx = sub_tr_covs['trial x units'].argmax()
            rem_uids = tr_covs.uid[(hmax_idx+1):]
            exc_uids = tr_covs.uid[1:hmax_idx+1]
            n_tr_rem = tr_covs.ntrs_cov[hmax_idx]
            n_tr_exc = IncTrsMat.shape[1] - n_tr_rem

            # Add to UnitInfo dataframe
            rt_utids = [(rec, ch, ui, task) for ch, ui in rem_uids]
            UInc[rt_utids] = True

        # Highlight selected point in middle plot.
        sel_seg = [('selection', exc_uids.shape[0]-0.4,
                    exc_uids.shape[0]+0.4)]
        putil.plot_periods(sel_seg, ax=ax_trc, alpha=0.3)
        [ax.set_xlim([-0.5, n_units+0.5]) for ax in (ax_trc, ax_remu)]

        # Generate remaining trials dataframe.
        RemTrsMat = IncTrsMat.copy().astype(float)
        for exc_uid in exc_uids:   # Remove all trials from excluded units.
            RemTrsMat.loc[exc_uid] = 0.5
        # Remove uncovered trials in remaining units.
        exc_trs = np.where(~RemTrsMat.loc[list(rem_uids)].all())[0]
        if exc_trs.size:
            RemTrsMat.iloc[:, exc_trs] = 0.5
        # Overwrite by trials excluded during preprocessing.
        RemTrsMat[IncTrsMat == False] = 0.0

        # Plot remaining trials.
        ax = axs[i_rt, 2]
        n_u_rem, n_u_exc = len(rem_uids), len(exc_uids)
        title = ('# units remaining: {}, excluded: {}'.format(n_u_rem,
                                                              n_u_exc) +
                 '\n# trials remaining: {}, excluded: {}'.format(n_tr_rem,
                                                                 n_tr_exc))
        plot_inc_exc_trials(RemTrsMat, ax, title=title, ylab='')

        # Add remaining units and trials to RecInfo.
        rt = (rec, task)
        RecInfo.loc[rt, ('units', 'nunits')] = list(rem_uids), len(rem_uids)
        cov_trs = RemTrsMat.loc[list(rem_uids)].all()
        inc_trs = pd.Int64Index(np.where(cov_trs)[0])
        RecInfo.loc[rt, ('trials', 'ntrials')] = inc_trs, sum(cov_trs)

    RecInfo['% remaining units'] = 100 * RecInfo.nunits / RecInfo.nallunits
    RecInfo['% remaining trials'] = 100 * RecInfo.ntrials / RecInfo.nalltrials

    # Save results.
    if fres is not None:
        results = {'RecInfo': RecInfo, 'UInc': UInc}
        util.write_objects(results, fres)

    # Save plot.
    title = 'Trial & unit selection prior decoding'
    putil.save_fig(ffig, fig, title, w_pad=3, h_pad=3)

    return RecInfo, UInc


# %% Direction selectivity analysis.

def PD_across_units(UA, UInc, utids=None, fres=None, ffig=None):
    """
    Test consistency/spread of PD across units per recording.
    What is the spread in the preferred directions across units?

    Return population level preferred direction (and direction selectivity),
    that can be used to determine dominant preferred direction to decode.
    """

    # Init.
    if utids is None:
        utids = UA.utids(as_series=True)
    tasks = utids.index.get_level_values('task').unique()
    recs = utids.index.get_level_values('rec').unique()

    # Get DS info frame.
    DSInfo = ua_query.get_DSInfo_table(UA, utids)
    DSInfo['include'] = UInc

    # Calculate population PD and DSI.
    dPPDres = {}
    for rec in recs:
        for task in tasks:

            # Init.
            rtDSInfo = DSInfo.xs((rec, task), level=[0, 3])
            if rtDSInfo.empty:
                continue

            # Calculate population PD and population DSI.
            res = direction.calc_PPD(rtDSInfo.loc[rtDSInfo.include])
            dPPDres[(rec, task)] = res

    PPDres = pd.DataFrame(dPPDres).T

    # Save results.
    if fres is not None:
        results = {'DSInfo': DSInfo, 'PPDres': PPDres}
        util.write_objects(results, fres)

    # Plot results.

    # Init plotting.
    putil.set_style('notebook', 'darkgrid')
    fig, gsp, axs = putil.get_gs_subplots(nrow=len(recs), ncol=len(tasks),
                                          subw=6, subh=6, create_axes=True,
                                          ax_kws_list={'projection': 'polar'})
    xticks = direction.deg2rad(constants.all_dirs + 360/8/2*deg)

    for ir, rec in enumerate(recs):
        for it, task in enumerate(tasks):

            # Init.
            rtDSInfo = DSInfo.xs((rec, task), level=[0, 3])
            ax = axs[ir, it]
            if rtDSInfo.empty:
                ax.set_axis_off()
                continue
            PDSI, PPD, PPDc, PADc = PPDres.loc[(rec, task)]

            # Plot PD - DSI on polar plot.
            sPPDc, sPADc = [int(v) if not np.isnan(v) else v
                            for v in (PPDc, PADc)]
            title = ('{} {}\n'.format(rec, task) +
                     'PPDc = {}$^\circ$ - {}$^\circ$'.format(PPDc, PADc) +
                     ', PDSI = {:.2f}'.format(PDSI))
            PDrad = direction.deg2rad(util.remove_dim_from_series(rtDSInfo.PD))
            pplot.scatter(PDrad, rtDSInfo.DSI, rtDSInfo.include, ylim=[0, 1],
                          title=title, ytitle=1.08, c='darkblue',
                          edgecolor='k', linewidth=1, s=80, alpha=0.8,
                          zorder=2, ax=ax)

            # Highlight PPD and PAD.
            offsets = np.array([-45, 0, 45]) * deg
            for D, c in [(PPDc, 'g'), (PADc, 'r')]:
                if np.isnan(D):
                    continue
                hlDs = direction.deg2rad(np.array(D+offsets))
                for hlD, alpha in [(hlDs, 0.2), ([hlDs[1]], 0.4)]:
                    pplot.bars(hlD, len(hlD)*[1], align='center',
                               alpha=alpha, color=c, zorder=1, ax=ax)

            # Format ticks.
            ax.set_xticks(xticks, minor=True)
            ax.grid(b=True, axis='x', which='minor')
            ax.grid(b=False, axis='x', which='major')
            putil.hide_tick_marks(ax)

    # Save plot.
    title = 'Population direction selectivity'
    putil.save_fig(ffig, fig, title, w_pad=12, h_pad=20)

    return DSInfo, PPDres


# %% Trial type distribution analysis.

def plot_trial_type_distribution(UA, RecInfo, utids=None, tr_par=('S1', 'Dir'),
                                 save_plot=False, fname=None):
    """Plot distribution of trial types."""

    # Init.
    par_str = util.format_to_fname(str(tr_par))
    if utids is None:
        utids = UA.utids(as_series=True)
    recs, tasks = [RecInfo.index.get_level_values(v).unique()
                   for v in ('rec', 'task')]
    # Reorder recordings and tasks.
    recs = [rec for rec in UA.recordings() if rec in recs]
    tasks = [task for task in UA.tasks() if task in tasks]

    # Init plotting.
    putil.set_style('notebook', 'darkgrid')
    fig, gsp, axs = putil.get_gs_subplots(nrow=len(recs), ncol=len(tasks),
                                          subw=4, subh=3, create_axes=True)

    for ir, rec in enumerate(recs):
        for it, task in enumerate(tasks):

            ax = axs[ir, it]

            if (rec, task) not in RecInfo.index:
                ax.set_axis_off()
                continue

            # Get includecd trials and their parameters.
            inc_trs = RecInfo.loc[(rec, task), 'trials']
            utid = utids.xs([rec, task], level=('rec', 'task'))[0]
            TrData = UA.get_unit(utid[:3], utid[3]).TrData.loc[inc_trs]

            # Create DF to plot.
            anw_df = TrData[[tr_par, 'correct']].copy()
            anw_df['answer'] = 'error'
            anw_df.loc[anw_df.correct, 'answer'] = 'correct'
            all_df = anw_df.copy()
            all_df.answer = 'all'
            comb_df = pd.concat([anw_df, all_df])

            if not TrData.size:
                ax.set_axis_off()
                continue

            # Plot as countplot.
            sns.countplot(x=tr_par, hue='answer', data=comb_df,
                          hue_order=['all', 'correct', 'error'], ax=ax)
            sns.despine(ax=ax)
            putil.hide_tick_marks(ax)
            putil.set_max_n_ticks(ax, 6, 'y')
            ax.legend(loc=[0.95, 0.7])

            # Add title.
            title = '{} {}'.format(rec, task)
            nce = anw_df.answer.value_counts()
            nc, ne = [nce[c] if c in nce else 0 for c in ('correct', 'error')]
            pnc, pne = 100*nc/nce.sum(), 100*ne/nce.sum()
            title += '\n\n# correct: {} ({:.0f}%)'.format(nc, pnc)
            title += '      # error: {} ({:.0f}%)'.format(ne, pne)
            putil.set_labels(ax, title=title, xlab=par_str)

            # Format legend.
            if (ir != 0) or (it != 0):
                ax.legend_.remove()

    # Save plot.
    if save_plot:
        title = 'Trial type distribution'
        if fname is None:
            fname = util.join(['results', 'decoding', 'prepare',
                               par_str + '_trial_type_distr.pdf'])

        putil.save_fig(fname, fig, title, w_pad=3, h_pad=3)
