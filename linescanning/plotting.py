import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

def LazyPRF(prf, vf_extent, save_as=None, ax=None, cmap='magma', cross_color="white", alpha=None):
    """
    LazyPRF

    Plot the geometric location of the Gaussian pRF.

    Parameters
    ----------
    prf: numpy.ndarray
        instantiation of `gauss2D_iso_cart`; will be np.squeeze'ed over the first axis if `ndim >= 3`.
    vf_extent: list
        the space the pRF lives in
    save_as: str, optional
        file path to save the image (*.pdf is recommended for quality and compatibility with Inkscape)
    ax: <AxesSubplot:>, optional
        Matplotlib axis to store the figure on
    cmap: str, optional
        Colormap for imshow; accepts output from :func:`linescanning.utils.make_binary_cm`. Defaults to 'magma'
    cross_color: str, optional
        Color for the fixation cross; defaults to 'white'. You can set it to 'k' if you have a binary colormap as input
    alpha: float, optional
        Opacity for imshow

    Returns
    ----------
    matplotlib.pyplot plot
    """

    if ax == None:
        ax = plt.gca()

    if prf.ndim >= 3:
        prf = np.squeeze(prf, axis=0)

    if alpha == None:
        alpha = 1

    ax.axvline(0, color=cross_color, linestyle='dashed', lw=0.5)
    ax.axhline(0, color=cross_color, linestyle='dashed', lw=0.5)
    im = ax.imshow(prf, extent=vf_extent+vf_extent, cmap=cmap, alpha=alpha)
    patch = patches.Circle((0, 0),
                           radius=vf_extent[-1],
                           transform=ax.transData,
                           edgecolor=cross_color,
                           facecolor="None",
                           linewidth=0.5)
    ax.add_patch(patch)
    im.set_clip_path(patch)
    ax.axis('off')

    if save_as:
        plt.savefig(save_as, transparant=True)


class LazyPlot():
    """LazyPlot

    Class for plotting because I'm lazy and I don't want to go through the `matplotlib` motion everything I quickly want to visualize something. This class makes that a lot easier. It allows single inputs, lists with multiple timecourses, labels, error shadings, and much more.

    Parameters
    ----------
    ts: list, numpy.ndarray
        Input data. Can either be a single list, or a list of multiple numpy arrays. If you want labels, custom colors, or error bars, these inputs must come in lists of similar length as `ts`!
    xx: list, numpy.ndarray, optional
        X-axis array
    error: list, numpy.ndarray, optional
        Error data with the same length/shape as the input timeseries, by default None. Can be either a numpy.ndarray for 1 timeseries, or a list of numpy.ndarrays for multiple timeseries
    error_alpha: float, optional
        Opacity level for error shadings, by default 0.3
    x_label: str, optional
        Label of x-axis, by default None
    y_label: str, optional
        Label of y-axis, by default None
    labels: str, list, optional
        String (if 1 timeseries) or list (with the length of `ts`) of colors, by default None. Labels for the timeseries to be used in the legend
    title: str, optional
        Plot title, by default None
    xkcd: bool, optional
        Plot the figre in XKCD-style (cartoon), by default False
    color: str, list, optional
        String (if 1 timeseries) or list (with the length of `ts`) of colors, by default None. If nothing is specified, we'll use `cmap` to create a color palette
    cmap: str, optional
        Color palette to use for colors if no individual colors are specified, by default 'viridis'
    figsize: tuple, optional
        Figure dimensions as per usual matplotlib conventions, by default (30,5)
    save_as: str, optional
        Save the plot, by default None. If you want to use figures in Inkscape, save them as PDFs to retain high resolution
    font_size: int, optional
        Font size of titles and axis labels, by default 12
    add_hline: dict, optional
        Dictionary for a horizontal line through the plot, by default None. Collects the following items:

        >>> add_hline = {'pos' 0,       # position
        >>>              'color': 'k',  # color
        >>>              'lw': 1,       # linewidth
        >>>              'ls': '--'}    # linestyle
        You can get the settings above by specifying *add_hline='default'*
    add_vline: [type], optional
        Dictionary for a vertical line through the plot, by default None. Same keys as `add_hline`
    line_width: int, list, optional
        Line widths for either all graphs (then *int*) or a *list* with the number of elements as requested graphs, default = 1.
    axs: <AxesSubplot:>, optional
        Matplotlib axis to store the figure on
    y_lim: list, optional
        List for `axs.set_ylim`
    set_xlim_zero: bool, optional
        Reduces the space between y-axis and start of the plot. Is set before sns.despine. Default = False

    Example
    ----------
    >>> # create a bunch of timeseries
    >>> from linescanning import utils
    >>> ts = utils.random_timeseries(1.2, 0.0, 100)
    >>> ts1 = utils.random_timeseries(1.2, 0.3, 100)
    >>> ts2 = utils.random_timeseries(1.2, 0.5, 100)
    >>> ts3 = utils.random_timeseries(1.2, 0.8, 100)
    >>> ts4 = utils.random_timeseries(1.2, 1, 100)

    >>> # plot 1 timecourse
    >>> plotting.LazyPlot(ts2, figsize=(20, 5))
    <linescanning.plotting.LazyPlot at 0x7f839b0289d0>

    >>> # plot multiple timecourses, add labels, and save file
    >>> plotting.LazyPlot([ts, ts1, ts2, ts3, ts4], figsize=(20, 5), save_as="test_LazyPlot.pdf", labels=['vol=0', 'vol=0.3', 'vol=0.5', 'vol=0.8', 'vol=1.0'])
    <linescanning.plotting.LazyPlot at 0x7f839b2177c0>

    >>> # add horizontal line at y=0
    >>> hline = {'pos': 0, 'color': 'k', 'lw': 0.5, 'ls': '--'}
    >>> >>> plotting.LazyPlot(ts2, figsize=(20, 5), add_hline=hline)
    <linescanning.plotting.LazyPlot at 0x7f839b053580>

    >>> # add shaded error bars
    >>> from scipy.stats import sem
    # make some stack
    >>> stack = np.hstack((ts1[...,np.newaxis],ts2[...,np.newaxis],ts4[...,np.newaxis]))
    >>> avg = stack.mean(axis=-1) # calculate mean
    >>> err = sem(stack, axis=-1) # calculate error
    >>> plotting.LazyPlot(avg, figsize=(20, 5), error=err)
    <linescanning.plotting.LazyPlot at 0x7f839b0d5220>
    """

    def __init__(self,
                 ts,
                 xx=None,
                 error=None,
                 error_alpha=0.3,
                 x_label=None,
                 y_label=None,
                 title=None,
                 xkcd=False,
                 color=None,
                 figsize=(30, 5),
                 cmap='viridis',
                 save_as=None,
                 labels=None,
                 font_size=12,
                 add_hline=None,
                 add_vline=None,
                 line_width=1,
                 axs=None,
                 y_lim=None,
                 x_lim=None,
                 sns_offset=10,
                 sns_trim=True,
                 sns_rm_bottom=False,
                 set_xlim_zero=False):

        self.array = ts
        self.xx = xx
        self.error = error
        self.error_alpha = error_alpha
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.xkcd = xkcd
        self.color = color
        self.figsize = figsize
        self.cmap = cmap
        self.save_as = save_as
        self.labels = labels
        self.font_size = font_size
        self.add_hline = add_hline
        self.add_vline = add_vline
        self.axs = axs
        self.line_width = line_width
        self.y_lim = y_lim
        self.x_lim = x_lim
        self.sns_offset = sns_offset
        self.sns_trim = sns_trim
        self.sns_bottom = sns_rm_bottom
        self.set_xlim_zero = set_xlim_zero

        if self.xkcd:
            with plt.xkcd():
                self.fontname = "Humor Sans"
                self.plot()
        else:
            self.fontname = "Arial"
            self.plot()

        if save_as:
            plt.savefig(self.save_as, transparent=True)

    def plot(self):

        if self.axs == None:
            fig, axs = plt.subplots(figsize=self.figsize)
        else:
            axs = self.axs

        if isinstance(self.array, list):

            if not isinstance(self.color, list):
                self.color_list = sns.color_palette(self.cmap, len(self.array))
            else:
                self.color_list = self.color
                if len(self.color_list) != len(self.array):
                    raise ValueError(
                        f"Length color list {len(self.color_list)} does not match length of data list ({len(self.array)}")

            for idx, el in enumerate(self.array):

                if isinstance(self.line_width, list):
                    if len(self.line_width) != len(self.array):
                        raise ValueError(
                            f"Length of line width lenghts {len(self.line_width)} does not match length of data list ({len(self.array)}")

                    use_width = self.line_width[idx]
                elif isinstance(self.line_width, int) or isinstance(self.line_width, float):
                    use_width = self.line_width
                else:
                    use_width = ""

                # decide on x-axis
                if not isinstance(self.xx, np.ndarray) and not isinstance(self.xx, list):
                    x = np.arange(0, len(el))
                else:
                    x = self.xx.copy()

                if self.labels:
                    axs.plot(
                        x, el, color=self.color_list[idx], label=self.labels[idx], lw=use_width)
                else:
                    axs.plot(x, el, color=self.color_list[idx], lw=use_width)

                if isinstance(self.error, list) or isinstance(self.error, np.ndarray):
                    yerr = self.error[idx]
                    if np.isscalar(yerr) or len(yerr) == len(el):
                        ymin = el - yerr
                        ymax = el + yerr
                    elif len(yerr) == 2:
                        ymin, ymax = yerr
                    axs.fill_between(
                        x, ymax, ymin, color=self.color_list[idx], alpha=self.error_alpha)

        else:
            if not self.color:
                self.color = sns.color_palette(self.cmap, 1)[0]

            # decide on x-axis
            if not isinstance(self.xx, np.ndarray) and not isinstance(self.xx, list):
                x = np.arange(0, len(self.array))
            else:
                x = self.xx.copy()

            axs.plot(x, self.array, color=self.color,
                     label=self.labels, lw=self.line_width)

            if isinstance(self.error, list) or isinstance(self.error, np.ndarray):
                if np.isscalar(self.error) or len(self.error) == len(self.array):
                    ymin = self.array - self.error
                    ymax = self.array + self.error
                elif len(self.error) == 2:
                    ymin, ymax = self.error
                else:
                    raise ValueError(
                        f"Don't underestimate input: {self.error}")
                axs.fill_between(x, ymax, ymin, color=self.color,
                                 alpha=self.error_alpha)

        if isinstance(self.y_lim, list):
            axs.set_ylim(self.y_lim)

        if self.labels:
            axs.legend(frameon=False)

        if self.x_label:
            axs.set_xlabel(self.x_label, fontname=self.fontname,
                           fontsize=self.font_size)

        if self.y_label:
            axs.set_ylabel(self.y_label, fontname=self.fontname,
                           fontsize=self.font_size)

        if self.title:
            axs.set_title(self.title, fontname=self.fontname,
                          fontsize=self.font_size)

        if self.add_vline:
            if self.add_vline == "default":
                self.add_vline = {'pos': 0, 'color': 'k',
                                  'ls': 'dashed', 'lw': 0.5}

            if isinstance(self.add_vline['pos'], list) or isinstance(self.add_vline['pos'], np.ndarray):
                for line in self.add_vline['pos']:
                    axs.axvline(
                        line, color=self.add_vline['color'], lw=self.add_vline['lw'], ls=self.add_vline['ls'])
            else:
                axs.axvline(self.add_vline['pos'], color=self.add_vline['color'],
                            lw=self.add_vline['lw'], ls=self.add_vline['ls'])

        if self.add_hline:
            if self.add_hline == "default":
                self.add_hline = {'pos': 0, 'color': 'k',
                                  'ls': 'dashed', 'lw': 0.5}

            if isinstance(self.add_hline['pos'], list) or isinstance(self.add_hline['pos'], np.ndarray):
                for line in self.add_hline['pos']:
                    axs.axaxhlinevline(
                        line, color=self.add_hline['color'], lw=self.add_hline['lw'], ls=self.add_hline['ls'])
            else:
                axs.axhline(self.add_hline['pos'], color=self.add_hline['color'],
                            lw=self.add_hline['lw'], ls=self.add_hline['ls'])

        if self.set_xlim_zero:
            axs.set_xlim(0)
        else:
            if self.x_lim:
                axs.set_xlim(self.x_lim)

        sns.despine(offset=self.sns_offset, trim=self.sns_trim, bottom=self.sns_bottom)


class LazyCorr():
    """LazyCorr

    Wrapper around seaborn's regplot. Plot data and a linear regression model fit.

    Parameters
    ----------
    x: np.ndarray, list
        First variable to include in regression
    y: np.ndarray, list
        Second variable to include in regression
    x_label: str, optional
        Label of x-axis, by default None
    y_label: str, optional
        Label of y-axis, by default None
    title: str, optional
        Plot title, by default None
    xkcd: bool, optional
        Plot the figre in XKCD-style (cartoon), by default False
    color: str, list, optional
        String representing a color, by default "#ccccccc"
    figsize: tuple, optional
        Figure dimensions as per usual matplotlib conventions, by default (30,5)
    save_as: str, optional
        Save the plot, by default None. If you want to use figures in Inkscape, save them as PDFs to retain high resolution
    font_size: int, optional
        Font size of titles and axis labels, by default 12
    axs: <AxesSubplot:>, optional
        Matplotlib axis to store the figure on


    Returns
    ----------
    <linescanning.plotting.LazyCorr> object
        figure with the regression + confidence intervals for the given variables
    """    

    def __init__(self,
                 x, 
                 y, 
                 color="#cccccc", 
                 axs=None, 
                 title=None,
                 x_label=None, 
                 y_label=None, 
                 figsize=(8,8),
                 xkcd=False,
                 font_size=20,
                 sns_despine=5,
                 sns_trim=True,
                 save_as=None):

        self.x              = x
        self.y              = y
        self.axs            = axs
        self.x_label        = x_label
        self.y_label        = y_label
        self.font_size      = font_size
        self.xkcd           = xkcd
        self.sns_despine    = sns_despine
        self.sns_trim       = sns_trim
        self.color          = color
        self.figsize        = figsize
        self.title          = title

        if self.xkcd:
            with plt.xkcd():
                self.fontname = "Humor Sans"
                self.plot()
        else:
            self.fontname = "Arial"
            self.plot()

        if save_as:
            plt.savefig(self.save_as, transparent=True)

    def plot(self):

        if self.axs == None:
            fig, axs = plt.subplots(figsize=self.figsize)
        else:
            axs = self.axs        

        sns.regplot(x=self.x, y=self.y, color=self.color, ax=axs)

        if self.x_label:
            if self.x_label != 'none':
                axs.set_xlabel(self.x_label, fontsize=self.font_size)
            else:
                axs.set_xlabel(None)

        if self.y_label:
            if self.y_label != 'none':
                axs.set_ylabel(self.y_label, fontsize=self.font_size)
            else:
                axs.set_ylabel(None)

        if self.title:
            axs.set_title(self.title, fontname=self.fontname, fontsize=self.font_size)

        sns.despine(offset=self.sns_despine, trim=self.sns_trim)
