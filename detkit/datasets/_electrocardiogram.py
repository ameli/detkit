# SPDX-FileCopyrightText: Copyright 2022, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

import numpy
import scipy
from ._plot_utilities import plt, load_plot_settings, save_plot
from ._display_utilities import is_notebook

__all__ = ['electrocardiogram']


# =================
# Electrocardiogram
# =================

def electrocardiogram(
        start=0.0,
        end=30.0,
        bw_window=0.5,
        freq_cut=45,
        plot=False):
    """
    Load an electrocardiogram signal as an example for a 1D signal.

    Parameters
    ----------

    start : float, default=0.0
        Start of the signal in seconds.

    end : float, default=10.0
        End of the signal in seconds.

    bw_window : default=1.0
        Length of moving average filter (in seconds) to remove baseline wander
        (bw). If zero, BW is not removed. If set to zero, baseline is not
        removed.

    freq_cut : float, default=45
        Frequencies (in Hz) above this limit will be cut by low-pass filter. If
        `numpy.inf`, no filtering is performed.

    plot : bool, default=False
        If `True`, the signal is plotted.

    Returns
    -------

        ecg : numpy.array
            ECG signal.

        time : numpy.array
            Time axis corresponding to the ECG signal.

    Notes
    -----

    The signal is sampled at 360 Hz.

    Two filters are applied on the original ECG signal:

    * Removing baseline wander (BW) by moving average filter. BW is the trend
      of the signal caused by respiration and movements of the person. Usually,
      BW is on the frequency range of 0.1 HZ to 0.5 Hz. Unfortunately, a
      high-pass filter above 0.5 Hz does not cleanly remove the BW. The best
      approach so far was a moving average filter with the kernel duration of
      about 1 seconds.

    * Removing noise by low-pass filter with critical frequency of 45 Hz.
      This also removes the 60 Hz power-line frequency that interferes with the
      measurement device.

    References
    ----------

    .. [1] Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
           IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
           (PMID: 11446209); DOI:10.13026/C2F305

    .. [2] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark
           RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank,
           PhysioToolkit, and PhysioNet: Components of a New Research Resource
           for Complex Physiologic Signals. Circulation 101(23):e215-e220;
           DOI:10.1161/01.CIR.101.23.e215
    """

    # Read dataset
    if hasattr(scipy, 'datasets'):
        # Scipy version 1.10
        ecg = scipy.datasets.electrocardiogram()
    elif hasattr(scipy, 'misc'):
        # Scipy prior to version 1.10
        ecg = scipy.misc.electrocardiogram()
    else:
        raise RuntimeError('Cannot find electrocardiogram function in scipy ' +
                           'package.')

    # Sampling frequency in Hz for this dataset
    fs = 360

    # Time for ecg array based on sampling frequency
    time = numpy.arange(ecg.size) / fs

    # Remove baseline wander by moving average filter
    if bw_window > 0.0:
        ecg_filtered, ecg_bw = _remove_baseline_wander(ecg, fs, bw_window)
    else:
        ecg_filtered = ecg
        ecg_bw = numpy.zeros_like(ecg)

    # Remove high frequencies from ECG signal
    if not numpy.isinf(freq_cut):
        filter_order = 5
        ecg_filtered = _remove_noise(ecg_filtered, fs, freq_cut, filter_order)

    # Cut time
    start_index = int(start*fs)
    end_index = int(end*fs)
    time = time[start_index:end_index]
    ecg = ecg[start_index:end_index]
    ecg_bw = ecg_bw[start_index:end_index]
    ecg_filtered = ecg_filtered[start_index:end_index]

    # Plot
    if plot:
        if is_notebook():
            save = False
        else:
            save = True

        _plot(time, ecg, ecg_bw, ecg_filtered, save=save)

    return time, ecg_filtered


# ======================
# Remove Baseline Wander
# ======================

def _remove_baseline_wander(
        signal,
        fs,
        bw_window=1):
    """
    Using a moving average filter to remove baseline wander of the ECG signal.

    Parameters
    ----------

    signal : numpy.array
        The ECG signal.

    fs : float
        Sampling frequency of the signal

    window : float, default=1.0
        The duration of the moving average window in seconds.

    Returns
    -------

    signal_filtered : numpy.array
        Signal with baseline removed.

    signal_bw : numpy.array
        Baseline wander of the signal
    """

    # Length of window from seconds to index
    kernel_window = int(bw_window*fs)

    # Moving average kernel
    kernel = numpy.ones((kernel_window,)) / kernel_window

    # Baseline wander
    signal_bw = numpy.convolve(signal, kernel, mode='same')

    signal_filtered = signal - signal_bw

    return signal_filtered, signal_bw


# ============
# Remove noise
# ============

def _remove_noise(
        signal,
        fs,
        freq_cut=45,
        filter_order=5):
    """
    Remove high frequency noise from ECG signal.

    This function uses Butter filter to design a low-pass FIR filter.

    Parameters
    ----------

    signal : numpy.array
        ECG signal

    fs : float
        Sampling frequency of the signal.

    freq_cut : float, default=45
        Frequencies (in Hz) above this limit will be cut.

    order : int, default=5
        Order of the filter. Higher number means stronger filter.

    Returns
    -------

    signal_filtered : numpy.array
        Signal with baseline removed.
    """

    # Nyquist frequency is half of sampling frequency
    nyq = 0.5 * fs

    # Ratio of cut frequency
    cut = freq_cut / nyq

    # Design filter
    sos = scipy.signal.butter(filter_order, cut, 'lowpass', output='sos')

    # Apply filter
    signal_filtered = scipy.signal.sosfilt(sos, signal)

    return signal_filtered


# ====
# plot
# ====

def _plot(
        time,
        ecg,
        ecg_bw,
        ecg_filtered,
        save=True):
    """
    Plots the ECG signal.

    Parameters
    ----------
    """

    load_plot_settings()

    # Settings
    title_fontsize = 11
    label_fontsize = 10
    tick_fontsize = 10

    # Plots
    fig, ax = plt.subplots(nrows=2, figsize=(9.8, 3.4))
    ax[0].plot(time, ecg, color='black', label='Original')
    ax[0].plot(time, ecg_bw, color='orange', label='Baseline wander')
    ax[1].plot(time, ecg_filtered, color='black', label='Filtered')
    ax[0].set_xlabel(r"$t$ (sec)", fontsize=label_fontsize)
    ax[1].set_xlabel(r"$t$ (sec)", fontsize=label_fontsize)
    ax[0].set_ylabel("ECG (mV)", fontsize=label_fontsize)
    ax[1].set_ylabel("ECG (mV)", fontsize=label_fontsize)
    ax[0].set_xlim([time[0], time[-1]])
    ax[1].set_xlim([time[0], time[-1]])
    ax[0].legend(fontsize='x-small')
    ax[1].legend(fontsize='x-small')
    ax[0].set_title('ECG Signal', fontsize=title_fontsize)
    ax[0].tick_params(axis='both', labelsize=tick_fontsize)
    ax[1].tick_params(axis='both', labelsize=tick_fontsize)

    # Remove bottom axis
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False,
                      labelbottom=False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    # Save plot
    if save:
        filename = 'electrocardiogram'
        save_plot(filename, transparent_background=True, pdf=True,
                  bbox_extra_artists=None, verbose=True)
    else:
        plt.show()
