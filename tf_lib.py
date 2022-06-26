import numpy as np
from gwinc import const
from gwinc.ifo.noises import arm_cavity
# from qlance.plotting import plotTF
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import subprocess
import os


def set_fontsize(axes, fontsize):
    for ax in axes:
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        labels.extend([ax.xaxis.label, ax.yaxis.label])
        for label in labels:
            label.set_fontsize(fontsize)


def plot_tf2(
        F_Hz, tf_mats, *axes, is_to_hom=False, is_from_hom=False,
        remove_diag=False, fontsize=14, kind='2p', **kwargs):
    """Helper for plot_tf_homs
    """

    if kind == '2p':
        q_to = 0 + 2*is_to_hom
        p_to = 1 + 2*is_to_hom
        q_fr = 0 + 2*is_from_hom
        p_fr = 1 + 2*is_from_hom

        plotTF(F_Hz, tf_mats[:, q_to, q_fr], *axes, color='C1', ls='-',
               label='qq', zorder=11, **kwargs)
        plotTF(F_Hz, tf_mats[:, p_to, p_fr], *axes, color='C2', ls='--',
               label='pp', zorder=12, **kwargs)
        if not remove_diag:
            plotTF(F_Hz, tf_mats[:, p_to, q_fr], *axes, color='C0', ls='-',
                   label='pq', zorder=10, **kwargs)
            plotTF(F_Hz, tf_mats[:, q_to, p_fr], *axes, color='C3', ls='-.',
                   label='qp', zorder=13, **kwargs)

    elif kind == 'sb':
        upr_to = 0 + 2*is_to_hom
        lwr_to = 1 + 2*is_to_hom
        upr_fr = 0 + 2*is_from_hom
        lwr_fr = 1 + 2*is_from_hom

        plotTF(F_Hz, tf_mats[:, upr_to, upr_fr], *axes, ls='-',
               label='++', **kwargs)
        plotTF(F_Hz, tf_mats[:, lwr_to, lwr_fr], *axes, ls='--',
               label='--', **kwargs)
        if not remove_diag:
            plotTF(F_Hz, tf_mats[:, upr_to, lwr_fr], *axes, ls='-',
                   label='+-', **kwargs)
            plotTF(F_Hz, tf_mats[:, lwr_to, upr_fr], *axes, ls='-.',
                   label='-+', **kwargs)

    else:
        raise ValueError('unknown type of transfer function ' + kind)

    leg = axes[0].legend(fontsize=fontsize)
    leg.set_zorder(20)
    for ax in axes:
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        labels.extend([ax.xaxis.label, ax.yaxis.label])
        for label in labels:
            label.set_fontsize(fontsize)


def plot_tf(
        F_Hz, tf_mats, is_to_hom=False, is_from_hom=False, remove_diag=False):
    """Plot a 2x2 transfer matrix

    The transfer matrices should be (nf, 2, 2)
    where nf is the number of frequency points. The second index is the
    quadrature the quadrature to and the third the quadrature from.
      0: amplitude quadrature
      1: phase quadrature

    Inputs:
      F_Hz: a (nf,) frequency array [Hz]
      tf_mats: a (nf, 2, 2) array transfer matrix
      is_to_hom: if true, the first index is a HOM (Default: False)
      is_from_hom: if true, the second index is a HOM (Default: False)
      ifo: an optional ifo struct with auxiliary information (Default: None)
      plot_freqs: if True, plot FSR, arm pole, DARM pole, and transverse mode
        spacings (Default: False)

    Returns:
      fig: the figure
    """

    q_to = 0 + 2*is_to_hom
    p_to = 1 + 2*is_to_hom
    q_fr = 0 + 2*is_from_hom
    p_fr = 1 + 2*is_from_hom

    fig = plotTF(F_Hz, tf_mats[:, p_to, q_fr], label='pq')
    plotTF(F_Hz, tf_mats[:, q_to, q_fr], *fig.axes, label='qq')
    plotTF(F_Hz, tf_mats[:, p_to, p_fr], *fig.axes, ls='--', label='pp')
    plotTF(F_Hz, tf_mats[:, q_to, p_fr], *fig.axes, ls='-.', label='qp')

    if remove_diag:
        for ax in fig.axes:
            del ax.lines[0]
            del ax.lines[-1]

    fig.axes[0].legend()

    return fig


def plot_tf_homs(
        f_Hz, tf_mats, kind='2p', title='', remove_diag=False, fontsize=20, lw=4):
    """Plot either a 2x2 or 4x4 sideband transfer matrix

    If kind is 2p, then it's a two photon transfer matrix
    If kind is sb, then it's a sideband transfer matrix
    """
    if tf_mats.shape[1] > 2:
        fig = plt.figure(figsize=(20, 25))
        gs = fig.add_gridspec(2, 2)
        gs_ff = gs[0, 0].subgridspec(2, 1, hspace=0.05)
        gs_hh = gs[1, 1].subgridspec(2, 1, hspace=0.05)
        gs_fh = gs[0, 1].subgridspec(2, 1, hspace=0.05)
        gs_hf = gs[1, 0].subgridspec(2, 1, hspace=0.05)

        ff_a = fig.add_subplot(gs_ff[0])
        ff_p = fig.add_subplot(gs_ff[1], sharex=ff_a)
        plot_tf2(
            f_Hz, tf_mats, ff_a, ff_p, is_to_hom=False, is_from_hom=False,
            kind=kind, remove_diag=remove_diag, fontsize=fontsize, lw=lw)
        ff_a.set_title(title + ' Fundamental to Fundamental', fontsize=fontsize)

        hh_a = fig.add_subplot(gs_hh[0])
        hh_p = fig.add_subplot(gs_hh[1], sharex=hh_a)
        plot_tf2(
            f_Hz, tf_mats, hh_a, hh_p, is_to_hom=True, is_from_hom=True,
            kind=kind, remove_diag=remove_diag, fontsize=fontsize, lw=lw)
        hh_a.set_title(title + ' HOM to HOM', fontsize=fontsize)

        fh_a = fig.add_subplot(gs_fh[0])
        fh_p = fig.add_subplot(gs_fh[1], sharex=fh_a)
        plot_tf2(
            f_Hz, tf_mats, fh_a, fh_p, is_to_hom=False, is_from_hom=True,
            kind=kind, remove_diag=remove_diag, fontsize=fontsize, lw=lw)
        fh_a.set_title(title + ' HOM to Fundamental', fontsize=fontsize)

        hf_a = fig.add_subplot(gs_hf[0])
        hf_p = fig.add_subplot(gs_hf[1], sharex=hf_a)
        plot_tf2(
            f_Hz, tf_mats, hf_a, hf_p, is_to_hom=True, is_from_hom=False,
            kind=kind, remove_diag=remove_diag, fontsize=fontsize, lw=lw)
        hf_a.set_title(title + ' Fundamental to HOM', fontsize=fontsize)

    else:
        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.05)
        ff_a = fig.add_subplot(gs[0])
        ff_p = fig.add_subplot(gs[1], sharex=ff_a)
        plot_tf2(
            f_Hz, tf_mats, ff_a, ff_p, is_to_hom=False, is_from_hom=False,
            kind=kind, remove_diag=remove_diag, fontsize=fontsize, lw=lw)
        ff_a.set_title(title, fontsize=fontsize)

    return fig


def plot_frequencies(ax, ifo, plot_tms=True, **kwargs):
    La = ifo.Infrastructure.Length
    Ti = ifo.Optics.ITM.Transmittance
    Ts = ifo.Optics.SRM.Transmittance
    ua = 1 - np.sqrt(1 - Ti)
    us = 1 - np.sqrt(1 - Ts)

    fa = const.c*ua/(4*np.pi*La)
    fp = const.c*ua/(4*np.pi*La) * (2 - us)/us
    fsr = const.c/(2*La)

    ax.axvline(fa, ls=':', c='xkcd:slate', label=r'$f_\mathrm{a}$', **kwargs)
    ax.axvline(fp, ls=':', c='xkcd:kelly green', label=r'$f_\mathrm{p}$', **kwargs)
    ax.axvline(fsr, ls=':', c='xkcd:sienna', label=r'$f_\mathrm{fsr}$', **kwargs)
    if plot_tms:
        cav = arm_cavity(ifo)
        for nn in np.arange(6):
            if nn == 0:
                label = r'$|2f_\mathrm{tms} - nf_\mathrm{fsr}|$'
            else:
                label = None
            ax.axvline(
                np.abs(2*cav.tms_Hz - nn*fsr), ls=':', label=label,
                c='xkcd:bright blue', **kwargs)


def plotTF(ff, tf, mag_ax=None, phase_ax=None, **kwargs):
    """Plot a SISO transfer function
    """
    if not(mag_ax and phase_ax):
        if (mag_ax is not None) or (phase_ax is not None):
            msg = 'If one of the phase or magnitude axes is given,'
            msg += ' the other must be given as well.'
            raise ValueError(msg)
        newFig = True
    else:
        newFig = False

    if newFig:
        fig = plt.figure()
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.05)
        mag_ax = fig.add_subplot(gs[0])
        phase_ax = fig.add_subplot(gs[1], sharex=mag_ax)

    mag_ax.loglog(ff, np.abs(tf), **kwargs)
    mag_ax.set_ylabel('Magnitude')
    mag_ax.autoscale(enable=True, axis='y')

    # If the TF is close to being constant magnitude, increase ylims
    # in order to show y tick labels and avoid a misleading plot.
    ylim = mag_ax.get_ylim()
    if ylim[1]/ylim[0] < 10:
        mag_ax.set_ylim(ylim[0]/10.1, ylim[1]*10.1)

    mag_ax.set_xlim(min(ff), max(ff))
    phase_ax.set_ylim(-185, 185)
    # ticks = np.linspace(-180, 180, 7)
    ticks = np.arange(-180, 181, 45)
    phase_ax.yaxis.set_ticks(ticks)
    phase_ax.semilogx(ff, np.angle(tf, True), **kwargs)
    phase_ax.set_ylabel('Phase [deg]')
    phase_ax.set_xlabel('Frequency [Hz]')
    plt.setp(mag_ax.get_xticklabels(), visible=False)
    mag_ax.grid(True, which='both', alpha=0.5)
    mag_ax.grid(True, alpha=0.25, which='minor')
    phase_ax.grid(True, which='both', alpha=0.5)
    phase_ax.grid(True, alpha=0.25, which='minor')
    if newFig:
        return fig


def plot_relative_error(ff, arr1, arr2, mag_ax=None, phase_ax=None, **kwargs):
    """Plot the relative error between two functions
    The ratio of magnitude and difference in phase are plotted separately
    """
    if not(mag_ax and phase_ax):
        if (mag_ax is not None) or (phase_ax is not None):
            msg = 'If one of the phase or magnitude axes is given,'
            msg += ' the other must be given as well.'
            raise ValueError(msg)
        newFig = True
    else:
        newFig = False

    if newFig:
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.05)
        mag_ax = fig.add_subplot(gs[0])
        phase_ax = fig.add_subplot(gs[1], sharex=mag_ax)
    else:
        old_ylims = mag_ax.get_ylim()

    mag_diff = np.abs(np.abs(arr1) - np.abs(arr2)) / np.abs(arr1)
    phase_diff = np.angle(arr1, True) - np.angle(arr2, True)
    mag_ax.loglog(ff, mag_diff, **kwargs)
    mag_ax.set_ylabel('Magnitude')
    mag_ax.autoscale(enable=True, axis='y')

    # # If plotting ontop of an old TF, adjust the ylims so that the old TF
    # # is still visible
    # if not newFig:
    #     new_ylims = mag_ax.get_ylim()
    #     mag_ax.set_ylim(min(old_ylims[0], new_ylims[0]),
    #                     max(old_ylims[1], new_ylims[1]))

    mag_ax.set_xlim(min(ff), max(ff))
    # # phase_ax.set_ylim(-185, 185)
    # # ticks = np.linspace(-180, 180, 7)
    # ticks = np.arange(-180, 181, 45)
    # phase_ax.yaxis.set_ticks(ticks)
    phase_ax.semilogx(ff, phase_diff, **kwargs)
    phase_ax.set_ylabel('Phase [deg]')
    phase_ax.set_xlabel('Frequency [Hz]')
    plt.setp(mag_ax.get_xticklabels(), visible=False)
    mag_ax.grid(True, which='both', alpha=0.5)
    mag_ax.grid(True, alpha=0.25, which='minor')
    phase_ax.grid(True, which='both', alpha=0.5)
    phase_ax.grid(True, alpha=0.25, which='minor')
    if newFig:
        return fig


def plot_sb_error(f_Hz, tfs1, tfs2, show_upper=True, show_lower=True, ph_lim=[-2, 2]):
    """Plot the error between to sideband transfer functions

    tfs1 and tfs2 should be a (npts, 2, 2) matrix where [0, 1], and [1, 0] will
    be 0, [0, 0] is the h(+omega) and [1, 1] the h(-omega)* transfer functions

    The transfer functions are plotted on the left and the relative errors
    on the right

    When show_upper is true the h(+omega) sidebands are plotted and
    when show_lower is true the h(-omega)* sidebands are plotted

    Both are plotted by default and the errors are always plotted for both
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(1, 2)
    gs_tf = gs[0, 0].subgridspec(2, 1, hspace=0.05)
    gs_err = gs[0, 1].subgridspec(2, 1, hspace=0.05)
    mag_ax0 = fig.add_subplot(gs_tf[0])
    ph_ax0 = fig.add_subplot(gs_tf[1], sharex=mag_ax0)
    mag_ax1 = fig.add_subplot(gs_err[0])
    ph_ax1 = fig.add_subplot(gs_err[1], sharex=mag_ax1)

    fontsize = 20
    lw = 4

    if show_upper:
        plotTF(
            f_Hz, tfs1[:, 0, 0], mag_ax0, ph_ax0, label='TF1+', lw=lw)
        plotTF(
            f_Hz, tfs2[:, 0, 0], mag_ax0, ph_ax0, ls='--', label='TF2+', lw=lw)
    if show_lower:
        plotTF(
            f_Hz, tfs1[:, 1, 1], mag_ax0, ph_ax0, label='TF1-', lw=lw)
        plotTF(
            f_Hz, tfs2[:, 1, 1], mag_ax0, ph_ax0, ls='--', label='TF2-', lw=lw)

    mag_ax0.legend(fontsize=fontsize)
    mag_ax0.set_title('Sideband Transfer Function', fontsize=fontsize)

    plot_relative_error(
        f_Hz, tfs1[:, 0, 0], tfs2[:, 0, 0], mag_ax1, ph_ax1, label='+', lw=lw)
    plot_relative_error(
        f_Hz, tfs1[:, 1, 1], tfs2[:, 1, 1], mag_ax1, ph_ax1, ls='--', label='-', lw=lw)
    mag_ax1.legend(fontsize=fontsize)
    mag_ax1.set_title('Sideband Relative Error', fontsize=fontsize)
    ph_ax1.set_ylim(*ph_lim)


    set_fontsize(fig.axes, fontsize=fontsize)

    return fig


def plot_2p_error(f_Hz, tfs1, tfs2, show_diag=True, show_offdiag=True, ph_lim=[-2, 2]):
    """Plot the error between 2 photon transfer functions

    tfs1 and tfs2 should be (npts, 2, 2) matrices

    The transfer functions from amplitude are plotted in the upper left and
    the transfer functions from phase are plotted in the lower left.
    The relative errors are plotted on the right

    When show_diag is True, the amp->amp and phase->phase are plotted and
    when who_offdiag is True, the amp->phase and phase->amp are plotted.
    Both are plotted by default
    """
    fig = plt.figure(figsize=(20, 25))
    gs = fig.add_gridspec(2, 2)
    gs_amp = gs[0, 0].subgridspec(2, 1, hspace=0.05)
    gs_ph = gs[1, 0].subgridspec(2, 1, hspace=0.05)
    gs_amp_err = gs[0, 1].subgridspec(2, 1, hspace=0.05)
    gs_ph_err = gs[1, 1].subgridspec(2, 1, hspace=0.05)

    fontsize = 20
    lw = 4

    mag_amp = fig.add_subplot(gs_amp[0])
    mag_ph = fig.add_subplot(gs_amp[1], sharex=mag_amp)
    if show_diag:
        plotTF(
            f_Hz, tfs1[:, 0, 0], mag_amp, mag_ph, label='TF1 to amp', lw=lw)
        plotTF(
            f_Hz, tfs2[:, 0, 0], mag_amp, mag_ph, ls='--', label='TF2 to amp', lw=lw)
    if show_offdiag:
        plotTF(
            f_Hz, tfs1[:, 1, 0], mag_amp, mag_ph, label='TF1 to phase', lw=lw)
        plotTF(
            f_Hz, tfs2[:, 1, 0], mag_amp, mag_ph, ls='--', label='TF2 to phase', lw=lw)
    mag_amp.set_title('Amplitude', fontsize=fontsize)
    mag_amp.legend(fontsize=fontsize)

    mag_amp_err = fig.add_subplot(gs_amp_err[0])
    mag_ph_err = fig.add_subplot(gs_amp_err[1], sharex=mag_amp_err)
    plot_relative_error(
        f_Hz, tfs1[:, 0, 0], tfs2[:, 0, 0], mag_amp_err, mag_ph_err, label='to amp', lw=lw)
    plot_relative_error(
        f_Hz, tfs1[:, 1, 0], tfs2[:, 1, 0], mag_amp_err, mag_ph_err, ls='--',
        label='to phase', lw=lw)
    mag_amp_err.set_title('Amplitude Relative Error', fontsize=fontsize)
    mag_amp_err.legend(fontsize=fontsize)
    mag_ph_err.set_ylim(*ph_lim)

    ph_amp = fig.add_subplot(gs_ph[0])
    ph_ph = fig.add_subplot(gs_ph[1], sharex=ph_amp)
    if show_diag:
        plotTF(
            f_Hz, tfs1[:, 1, 1], ph_amp, ph_ph, label='TF1 to phase', lw=lw)
        plotTF(
            f_Hz, tfs2[:, 1, 1], ph_amp, ph_ph, ls='--', label='TF2 to phase', lw=lw)
    if show_offdiag:
        plotTF(
            f_Hz, tfs1[:, 0, 1], ph_amp, ph_ph, label='TF1 to amp', lw=lw)
        plotTF(
            f_Hz, tfs2[:, 0, 1], ph_amp, ph_ph, ls='--', label='TF2 to amp', lw=lw)
    ph_amp.set_title('Phase', fontsize=fontsize)
    ph_amp.legend(fontsize=fontsize)

    ph_amp_err = fig.add_subplot(gs_ph_err[0])
    ph_ph_err = fig.add_subplot(gs_ph_err[1], sharex=ph_amp_err)
    plot_relative_error(
        f_Hz, tfs1[:, 1, 1], tfs2[:, 1, 1], ph_amp_err, ph_ph_err, label='to phase', lw=lw)
    plot_relative_error(
        f_Hz, tfs1[:, 0, 1], tfs2[:, 0, 1], ph_amp_err, ph_ph_err, ls='--',
        label='to amp', lw=lw)
    ph_amp_err.set_title('Phase Relative Error', fontsize=fontsize)
    ph_amp_err.legend(fontsize=fontsize)
    ph_ph_err.set_ylim(*ph_lim)

    set_fontsize(fig.axes, fontsize=fontsize)

    return fig
