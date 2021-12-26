import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns

def double_gamma(x, lag=6, a2=12, b1=0.9, b2=0.9, c=0.35, scale=True):

    a1 = lag
    d1 = a1 * b1
    d2 = a2 * b2
    hrf = np.array([(t/(d1))**a1 * np.exp(-(t-d1)/b1) - c *
                   (t/(d2))**a2 * np.exp(-(t-d2)/b2) for t in x])

    if scale:
        hrf = (1 - hrf.min()) * (hrf - hrf.min()) / \
            (hrf.max() - hrf.min()) + hrf.min()
    return hrf


def make_stimulus_vector(onset_df, scan_length=None, TR=0.105, osf=None, type='event', block_length=None, amplitude=None):

    """
    make_stimulus_vector

    create a np.1darray for a given condition with the length of number of scans

    """

    # check if we should reset or not
    try:
        onset_df = onset_df.reset_index()
    except:
        onset_df = onset_df    

    # check conditions we have
    try:
        names_cond = onset_df['event_type'].unique()
        names_cond.sort()
    except:
        raise ValueError('Could not extract condition names; are you sure you formatted the dataframe correctly?')

    # check if we got multiple amplitudes
    if isinstance(amplitude, np.ndarray):
        ampl_array = amplitude
    elif isinstance(amplitude, list):
        ampl_array = np.array(amplitude)
    else:
        ampl_array = False

    # loop through unique conditions
    stim_vectors = {}
    for idx,condition in enumerate(names_cond):

        if isinstance(ampl_array, np.ndarray):
            if ampl_array.shape[0] == names_cond.shape[0]:
                ampl = amplitude[idx]
                print(f"Amplitude for event '{names_cond[idx]}' = {round(ampl,2)}")
            else:
                raise ValueError(f"Nr of amplitudes ({ampl_array.shape[0]}) does not match number of conditions ({names_cond.shape[0]})")
        else:
            ampl = 1

        Y = np.zeros(int((scan_length*TR)*osf))

        if type == "event":
            for rr, ii in enumerate(onset_df['onset']):
                if onset_df['event_type'][rr] == condition:
                    Y[int(ii*osf)] = ampl
        elif type == 'block':

            if not isinstance(block_length, int):
                raise ValueError("Please specify the length of the block in seconds (integer)")

            for rr, ii in enumerate(onset_df['onset']):
                if onset_df['event_type'][rr] == condition:
                    Y[int(ii*osf):int((ii+block_length)*osf)] = ampl

        stim_vectors[condition] = Y

    return stim_vectors

def convolve_hrf(hrf, stim_v, make_figure=False, xkcd=False, add_array1=None, add_array2=None):

    def plot(stim_v, hrf, convolved, add_array1=None, add_array2=None):

        plt.figure(figsize=(15, 6))
        plt.subplot2grid((3, 3), (0, 0), colspan=2)
        plt.plot(stim_v, color="#B1BDBD")
        plt.ylim((-.5, 1.5))
        plt.ylabel('Activity (A.U.)', fontsize=12)
        plt.xlabel('Time (ms)', fontsize=12)
        plt.title('Events', fontsize=16)

        if isinstance(add_array1, np.ndarray):
            plt.plot(add_array1, color="#E41A1C")

        plt.subplot2grid((3, 3), (0, 2), rowspan=2)
        plt.axhline(0, color='black', lw=0.5)
        plt.plot(hrf, color="#F4A460")
        plt.title('HRF', fontsize=16)
        # plt.xlim(0, 24)
        plt.xlabel("Time (ms)", fontsize=12)

        plt.subplot2grid((3, 3), (1, 0), colspan=2)
        plt.axhline(0, color='black', lw=0.5)
        plt.plot(convolved, color="#E1C16E")
        plt.title('Convolved stimulus-vector', fontsize=16)
        plt.ylabel('Activity (A.U.)', fontsize=12)
        plt.xlabel('Time (ms)', fontsize=12)

        if isinstance(add_array2, np.ndarray):
            plt.plot(add_array2, "#D2691E")

        plt.tight_layout()
        sns.despine(offset=10)

    convolved_stim_vector = np.convolve(stim_v, hrf, 'full')[:stim_v.shape[0]]

    if make_figure:
        if xkcd:
            with plt.xkcd():
                plot(stim_v, hrf, convolved_stim_vector, add_array1=add_array1, add_array2=add_array2)
        else:
            plot(stim_v, hrf, convolved_stim_vector)
        plt.show()

    return convolved_stim_vector


def resample_stim_vector(array, npts, interpolate='nearest'):

    interpolated = interp1d(np.arange(len(array)), array, kind=interpolate, axis=0, fill_value='extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))

    return downsampled


def fit_first_level(stim_vector, voxel_signal, make_figure=False, copes=None, xkcd=False, plot_vox=None):

    if stim_vector.ndim == 1:
        # Add back a singleton axis (which was removed before downsampling)
        # otherwise stacking will give an error
        stim_vector = stim_vector[:, np.newaxis]

    if voxel_signal.ndim == 1:
        # Add back a singleton axis (which was removed before downsampling)
        # otherwise stacking will give an error
        voxel_signal = voxel_signal[:, np.newaxis]

    if stim_vector.shape[0] != voxel_signal.shape[0]:
        stim_vector = stim_vector[:voxel_signal.shape[0],:]

    intercept = np.ones((stim_vector.size, 1))
    X_conv = np.hstack((intercept, stim_vector))

    if copes is None:
        C = np.identity(X_conv.shape[1])

    # print(voxel_signal.shape)

    # np.linalg.pinv(X) = np.inv(X.T @ X) @ X
    betas_conv  = np.linalg.inv(X_conv.T @ X_conv) @ X_conv.T @ voxel_signal

    # calculate some other relevant parameters
    cope        = C @ betas_conv
    r           = voxel_signal - X_conv@betas_conv
    dof         = X_conv.shape[0] - np.linalg.matrix_rank(X_conv)
    sig2        = np.sum(r**2,axis=0)/dof
    varcope     = np.outer(C@np.diag(np.linalg.inv(X_conv.T@X_conv))@C.T,sig2)

    # calculate t-stats
    tstat = cope / np.sqrt(varcope)

    def best_voxel(betas):
        return np.where(betas == np.amax(betas))[0][0]

    if not plot_vox:
        best_vox = best_voxel(betas_conv[-1])
    else:
        best_vox = plot_vox

    print(f"max tstat (vox {best_vox}) = {round(tstat[-1,best_vox],2)}")
    print(f"max beta (vox {best_vox}) = {round(betas_conv[-1,best_vox],2)}")

    if make_figure:

        if xkcd:
            # with plt.xkcd():
            #     plot(voxel_signal[:,best_vox], X_conv, betas_conv[:,best_vox])
            plot_array([voxel_signal[:, best_vox], X_conv@betas_conv[:,best_vox]],
                        y_label="Activity (A.U.)",
                        x_label="volumes",
                        title=f"Model fit vox {best_vox}",
                        labels=['True signal', 'Event signal'],
                        figsize=(20,5),
                        xkcd=True)

        else:
            plot_array([voxel_signal[:, best_vox], X_conv@betas_conv[:,best_vox]],
                        y_label="Activity (A.U.)",
                        x_label="volumes",
                        title=f"Model fit vox {best_vox}",
                        labels=['True signal', 'Event signal'],
                        figsize=(20,5))

    return betas_conv,X_conv


def plot_array(array,
               error=None,
               error_val=0.3,
               x_label=None, 
               y_label=None, 
               title=None, 
               xkcd=False, 
               axis=True, 
               color=None, 
               figsize=(12,5), 
               cmap='viridis',
               save_as=None, 
               labels=None,
               font_size=12,
               add_hline=None,
               add_vline=None):

    def plot(array, x_label, y_label, title, error=None, error_val=error_val, color=None, labels=None, legend=False, hline=None, vline=None, font_size=font_size):

        fig,axs = plt.subplots(figsize=figsize)

        if isinstance(array, list):

            if not isinstance(color, list):
                color_list = sns.color_palette(cmap, len(array))
            else:
                color_list = color

            for idx,el in enumerate(array):
                if labels:
                    axs.plot(el, color=color_list[idx], label=labels[idx])
                else:
                    axs.plot(el, color=color_list[idx])

                if isinstance(error, list) or isinstance(error, np.ndarray):
                    yerr = error[idx]
                    if np.isscalar(yerr) or len(yerr) == len(el):
                        ymin = el - yerr
                        ymax = el + yerr
                    elif len(yerr) == 2:
                        ymin, ymax = yerr
                    x = np.arange(0,len(el))
                    axs.fill_between(x, ymax, ymin, color=color_list[idx], alpha=error_val)
        else:
            if not color:
                color = sns.color_palette(cmap, 1)[0]

            axs.plot(array, color=color, label=labels)

            if isinstance(error, list) or isinstance(error, np.ndarray):
                if np.isscalar(error) or len(error) == len(array):
                    ymin = array - error
                    ymax = array + error
                elif len(error) == 2:
                    ymin, ymax = error
                x = np.arange(0,len(array))
                axs.fill_between(x, ymax, ymin, color=color, alpha=error_val)

        if labels:
            axs.legend(frameon=False)

        if x_label:
            axs.set_xlabel(x_label, fontname='Arial', fontsize=font_size)

        if y_label:
            axs.set_ylabel(y_label, fontname='Arial', fontsize=font_size)

        if title:
            axs.set_title(title, fontname='Arial', fontsize=font_size)

        if vline:
            axs.axvline(vline['pos'], color=vline['color'], lw=vline['lw'], ls=vline['ls'])

        if hline:
            axs.axhline(hline['pos'], color=hline['color'], lw=hline['lw'], ls=hline['ls'])

        sns.despine(offset=10)

    if xkcd:
        with plt.xkcd():
            plot(array, x_label, y_label, title, error=error, error_val=error_val, color=color, labels=labels, hline=add_hline, vline=add_vline)
    else:
        plot(array, x_label, y_label, title, error=error, error_val=error_val, color=color, labels=labels, hline=add_hline, vline=add_vline)

    if save_as:
        plt.savefig(save_as, transparent=True)
    else:
        plt.show()
