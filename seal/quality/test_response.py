# -*- coding: utf-8 -*-

"""
Collection of functions for plotting unit activity and direction selectivity
for different sets of trials or trial periods.

@author: David Samu
"""

from seal.util import util
from seal.plot import putil, pselectivity


# Figure size constants
subw = 7


def DR_plot(UA, ftempl=None, match_scales=True):
    """Plot responses to all 8 directions and polar plot in the center."""

    # Init plotting theme.
    putil.set_style('notebook', 'ticks')

    # For each unit over all tasks.
    for uid in UA.uids():

        # Init figure.
        fig, gsp, _ = putil.get_gs_subplots(nrow=1, ncol=len(UA.tasks()),
                                            subw=subw, subh=subw)
        task_rate_axs, task_polar_axs = [], []

        # Plot direction response of unit in each task.
        for task, sps in zip(UA.tasks(), gsp):
            u = UA.get_unit(uid, task)

            # Plot DR of unit.
            res = (pselectivity.plot_DR_3x3(u, fig, sps)
                   if not u.is_excluded() and u.to_plot() else None)
            if res is not None:
                ax_polar, rate_axs = res
                task_rate_axs.extend(rate_axs)
                task_polar_axs.append(ax_polar)
            else:
                putil.add_mock_axes(fig, sps)

        # Match scale of y axes across tasks.
        if match_scales:
            putil.sync_axes(task_polar_axs, sync_y=True)
            putil.sync_axes(task_rate_axs, sync_y=True)
            [putil.adjust_decorators(ax) for ax in task_rate_axs]

        # Save figure.
        if ftempl is not None:
            uid_str = util.format_uid(uid)
            title = uid_str.replace('_', ' ')
            ffig = ftempl.format(uid_str)
            putil.save_fig(ffig, fig, title, ytitle=1.05)


def selectivity_summary(UA, ftempl=None, match_scales=True):
    """Test unit responses within trails."""

    # Init plotting theme.
    putil.set_style('notebook', 'ticks')

    # For each unit over all tasks.
    for uid in UA.uids():

        # Init figure.
        fig, gsp, _ = putil.get_gs_subplots(nrow=1, ncol=len(UA.tasks()),
                                            subw=8, subh=16)
        ls_axs, ds_axs = [], []

        # Plot stimulus response summary plot of unit in each task.
        for task, sps in zip(UA.tasks(), gsp):
            u = UA.get_unit(uid, task)

            res = (pselectivity.plot_selectivity(u, fig, sps)
                   if not u.is_excluded() and u.to_plot() else None)
            if res is not None:
                ls_axs.extend(res[0])
                ds_axs.extend(res[1])

            else:
                putil.add_mock_axes(fig, sps)

        # Match scale of y axes across tasks.
        if match_scales:
            for rate_axs in [ls_axs, ds_axs]:
                putil.sync_axes(rate_axs, sync_y=True)
                [putil.adjust_decorators(ax) for ax in rate_axs]

        # Save figure.
        if ftempl is not None:
            uid_str = util.format_uid(uid)
            title = uid_str.replace('_', ' ')
            ffig = ftempl.format(uid_str)
            putil.save_fig(ffig, fig, title, ytitle=1.00)


def plot_response(f_sealcombined, resp_plot_dir,
                  plot_DR=True, plot_sel=True):
    """Plot basic unit activity figures."""

    print('\nStarting plotting unit activity...')
    putil.inline_off()

    # Init folders.
    ftempl_dr = util.join([resp_plot_dir, 'direction_response', '{}.png'])
    ftempl_sel = util.join([resp_plot_dir, 'stimulus_selectivity', '{}.png'])

    # Read in Units.
    print('  Reading in UnitArray...')
    UA = util.read_objects(f_sealcombined, 'UnitArr')
    UA.clean_array(keep_excl=False)

    # Test stimulus response to all directions.
    if plot_DR:
        print('  Plotting direction response...')
        DR_plot(UA, ftempl_dr)

    # Plot feature selectivity summary plots.
    if plot_sel:
        print('  Plotting selectivity summary figures...')
        selectivity_summary(UA, ftempl_sel)

    # Re-enable inline plotting
    putil.inline_on()
