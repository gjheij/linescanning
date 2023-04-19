import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
import string
from typing import Union

class Defaults():
    def __init__(self):
        self.pad_title = 20
        self.font_size = 18
        self.label_size = 14
        self.tick_width = 0.5
        self.tick_length = 0
        self.axis_width = 0.5
        self.line_width = 1
        self.line_style = "solid"
        self.sns_offset = None
        self.sns_trim = False
        self.sns_bottom = False
        self.xkcd = False
        self.return_obj = False
        self.ylim_bottom = None
        self.ylim_top = None
        self.xlim_left = 0
        self.xlim_right = None
        self.set_xlim_zero = False
        self.legend_handletext = 0.05

        if self.xkcd:
            self.fontname = "Humor Sans"
        else:
            self.fontname = "Montserrat"
        
        self.update_rc(self.fontname)

    def update_rc(self, font):
        plt.rcParams.update({'font.family': font})

class LazyPRF(Defaults):
    """LazyPRF

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
    shrink_factor: float, optional
        When the background of the image is white, we create a black border around the Circle patch. If this is equal to `vf_extent`, the border is cut off at some points. This factor shrinks the radius of the Circle, so that we can have a nice border. When set to 0.9, it becomes sort of like a target. This is relevant for **all** non-`magma` color maps that you insert, specifically a :func:`linescanning.utils.make_binary_cm` object
    xkcd: bool, optional
        Plot in cartoon-format
    label_size: int, optional
        Set the font size of the labels (i.e., axes). Default = 10
    tick_width: float, optional
        Set the thickness of the ticks. Larger value means thicker tick. Default = 0.5 (thin'ish)
    tick_length: int, optional
        Set the length of the ticks. Larger values mean longer ticks. Default = 7 (long'ish)
    axis_width: float, optional
        Set the thickness of the spines of the plot. Larger values mean thicker spines. Default = 0.5 (thin'ish)
    full_axis: bool, optional
        If `True`, the entire axis of `vf_extent` will be used for the ticks (more detailed). If `False`, a truncated/trimmed version will be returned (looks cleaner). Default = False
    axis_off: bool, optional
        If `True` the x/y axis will be maintained, and the `vf_extent` will be given as ticks. If `False`, axis will be turned off. If `axis_off=True`, then `full_axis` and other label/axis parameters are ignored. Default = True
    sns_trim: bool, optional
        If `True`, limit spines to the smallest and largest major tick on each non-despined axis. Maps to `sns.despine(trim=sns_trim)`
    sns_offset: int, optional
        Offset in the origin of the plot. Maps to `sns.despine(offset=sns_offset)`. Default is 10
    vf_only: bool, optional
        Only show the outline of the the visual field, without pRF. You still need to specify the pRF as we'll `imshow` an empty array with the same shape rather than the pRF. Default = False
    line_width: float, optional
        Width of the outer border of the visual field if `cmap` is not *viridis* or *magma* (these color maps are quite default, and do not require an extra border like :func:`linescanning.utils.make_binary_cm`-objects do). Default is 0.5.
    cross_width: float, optional
        Width of the cross denoting the x/y axis. Default is 0.5, but can be increased if `cmap` is not *viridis* or *magma* to enhance visibility 
    z_lines: int, optional
        Set the order of the vertical/horizontal lines. Default is **on top** of the pRF (1)
    z_prf: int, optional
        Set the order of the pRF imshow. Default is below the axis lines, but can be changed to be on top of them. Default = 0

    Returns
    ----------
    matplotlib.pyplot plot
    """

    def __init__(
        self, 
        prf, 
        vf_extent, 
        save_as=None, 
        ax=None, 
        cmap='RdBu_r', 
        cross_color="white", 
        alpha=None,
        shrink_factor=1, 
        xkcd=False,
        title=None,
        axis_off=True,
        figsize=(8,8),
        full_axis=False,
        vf_only=False,
        cross_width=0.5,
        concentric=None,
        z_lines=1,
        z_prf=0,
        **kwargs):
        
        self.prf            = prf
        self.vf_extent      = vf_extent
        self.save_as        = save_as
        self.ax             = ax
        self.cmap           = cmap
        self.cross_color    = cross_color
        self.alpha          = alpha
        self.xkcd           = xkcd
        self.shrink_factor  = shrink_factor
        self.title          = title
        self.axis_off       = axis_off
        self.figsize        = figsize
        self.full_axis      = full_axis
        self.vf_only        = vf_only
        self.cross_width    = cross_width
        self.concentric     = concentric
        self.z_lines        = z_lines
        self.z_prf          = z_prf

        super().__init__()
        self.__dict__.update(kwargs)
        self.update_rc(self.fontname)

        if not hasattr(self, "edge_color"):
            self.edge_color = self.cross_color

        if self.xkcd:
            with plt.xkcd():
                self.plot()
        else:
            self.plot()

        if self.save_as:
            if isinstance(self.save_as, list):
                for ii in self.save_as:
                    plt.savefig(ii, transparent=True, dpi=300, bbox_inches='tight')
            elif isinstance(self.save_as, str):
                plt.savefig(self.save_as, transparent=True, dpi=300, bbox_inches='tight')
            else:
                raise ValueError(f"Unknown input '{self.save_as}' for 'save_as'")

    def plot(self):

        if self.ax == None:
            _,self.ax = plt.subplots(figsize=self.figsize)

        if self.prf.ndim >= 3:
            self.prf = np.squeeze(self.prf, axis=0)

        if self.alpha == None:
            self.alpha = 1

        # line on x-axis
        self.ax.axvline(
            0, 
            color=self.cross_color, 
            linestyle='dashed', 
            lw=self.cross_width,
            zorder=self.z_lines)

        # line on y-axis
        self.ax.axhline(
            0, 
            color=self.cross_color, 
            linestyle='dashed', 
            lw=self.cross_width,
            zorder=self.z_lines)

        if not self.vf_only:
            plot_obj = self.prf
        else:
            plot_obj = np.zeros_like(self.prf)

        # check if pRF has negatives
        if plot_obj.min() < 0:
            vmin = plot_obj.min()
            vmax = -plot_obj.min()
        else:
            vmin = -plot_obj.max()
            vmax = plot_obj.max()
                    
        im = self.ax.imshow(
            plot_obj, 
            extent=self.vf_extent+self.vf_extent, 
            cmap=self.cmap, 
            alpha=self.alpha,
            zorder=self.z_prf,
            vmin=vmin,
            vmax=vmax)
        
        # In case of a white background, the circle for the visual field is cut off, so we need to make an adjustment:
        if self.cmap != 'magma' and self.cmap != 'viridis':
            radius = self.vf_extent[-1]*self.shrink_factor
        else:
            radius = self.vf_extent[-1]

        if self.title != None:
            self.ax.set_title(
                self.title, 
                fontsize=self.font_size, 
                fontname=self.fontname,
                pad=self.pad_title)
            
        self.patch = patches.Circle(
            (0, 0),
            radius=radius,
            transform=self.ax.transData,
            edgecolor=self.edge_color,
            facecolor="None",
            linewidth=self.line_width)

        self.ax.add_patch(self.patch)
        im.set_clip_path(self.patch)

        if self.axis_off:
            self.ax.axis('off')
        else:
            self.ax.tick_params(
                width=self.tick_width, 
                length=self.tick_length,
                labelsize=self.label_size)

            for axis in ['top', 'bottom', 'left', 'right']:
                self.ax.spines[axis].set_linewidth(0)
            
            if self.full_axis:
                new_ticks = np.arange(self.vf_extent[0],self.vf_extent[1]+1, 1)
                self.ax.set_xticks(new_ticks)
                self.ax.set_yticks(new_ticks)
            else:
                self.ax.set_yticks(self.vf_extent)
                self.ax.set_xticks(self.vf_extent)

            if self.sns_trim:
                sns.despine(
                    offset=self.sns_offset, 
                    trim=self.sns_trim)

class LazyPlot(Defaults):
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
    title: str, dict, optional
        String of dictionary collecting the following keys representing information about the title:
        >>> title = {
        >>>     'title' "some title",       # title text
        >>>     'color': 'k',               # color (default = 'k')
        >>>     'fontweight': "bold"}       # fontweight (default = 'normal'), can be any of the matplotib fontweight options (e.g., 'italic', 'bold', 'normal' etc.)
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
        >>> add_hline = {
        >>>     'pos' 0,       # position
        >>>     'color': 'k',  # color
        >>>     'lw': 1,       # linewidth
        >>>     'ls': '--'}    # linestyle
        You can get the settings above by specifying *add_hline='default'*. Now also accepts *add_hline='mean'* for single inputs
    add_vline: [type], optional
        Dictionary for a vertical line through the plot, by default None. Same keys as `add_hline`
    line_width: int, list, optional
        Line widths for either all graphs (then *int*) or a *list* with the number of elements as requested graphs, default = 1.
    axs: <AxesSubplot:>, optional
        Matplotlib axis to store the figure on
    y_lim: list, optional
        List for `axs.set_ylim`
    x_lim: list, optional
        List for `axs.set_xlim`
    set_xlim_zero: bool, optional
        Reduces the space between y-axis and start of the plot. Is set before sns.despine. Default = False
    label_size: int, optional
        Set the font size of the labels (i.e., axes). Default = 10
    tick_width: float, optional
        Set the thickness of the ticks. Larger value means thicker tick. Default = 0.5 (thin'ish)
    tick_length: int, optional
        Set the length of the ticks. Larger values mean longer ticks. Default = 7 (long'ish)
    axis_width: float, optional
        Set the thickness of the spines of the plot. Larger values mean thicker spines. Default = 0.5 (thin'ish)
    sns_trim: bool, optional
        If `True`, limit spines to the smallest and largest major tick on each non-despined axis. Maps to `sns.despine(trim=sns_trim)`
    sns_offset: int, optional
        Offset in the origin of the plot. Maps to `sns.despine(offset=sns_offset)`. Default is 10
    sns_bottom: bool, optional
        Also remove the bottom (x) spine of the plot
    markers: str, list, optional
        Use markers during plotting. If `ts` is a list, a list of similar length should be specified. If one array in `ts` should not have markers, use `None`. E.g., if `len(ts) == 3`, and we want only the first timecourse to have markers use: `markers=['.',None,None]
    markersize: str, list, optional
        Specify marker sizes during plotting. If `ts` is a list, a list of similar length should be specified. If one array in `ts` should not have markers, use `None`. E.g., if `len(ts) == 3`, and we want only the first timecourse to have markers use: `markers=['.',None,None]
    x_ticks: list, optional
        Locations where to put the ticks on the x-axis
    y_ticks: list, optional
        Locations where to put the ticks on the y-axis

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

    Notes
    ----------
    See https://linescanning.readthedocs.io/en/latest/examples/lazyplot.html for more examples
    """

    def __init__(
        self,
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
        add_hline=None,
        add_vline=None,
        axs=None,
        y_lim=None,
        x_lim=None,
        markers=None,
        markersize=None,
        x_ticks=None,
        y_ticks=None,
        plot_alpha=None,
        **kwargs):

        self.array              = ts
        self.xx                 = xx
        self.error              = error
        self.error_alpha        = error_alpha
        self.x_label            = x_label
        self.y_label            = y_label
        self.title              = title
        self.xkcd               = xkcd
        self.plot_alpha         = plot_alpha
        self.color              = color
        self.figsize            = figsize
        self.cmap               = cmap
        self.save_as            = save_as
        self.labels             = labels
        self.add_hline          = add_hline
        self.add_vline          = add_vline
        self.axs                = axs
        self.y_lim              = y_lim
        self.x_lim              = x_lim
        self.markers            = markers
        self.markersize         = markersize
        self.x_ticks            = x_ticks
        self.y_ticks            = y_ticks

        super().__init__()
        self.__dict__.update(kwargs)
        self.update_rc(self.fontname)

        if self.xkcd:
            with plt.xkcd():
                self.plot()
        else:
            self.plot()
        
        if self.save_as:
            if isinstance(self.save_as, list):
                for ii in self.save_as:
                    plt.savefig(ii, transparent=True, dpi=300, bbox_inches='tight')
            elif isinstance(self.save_as, str):
                plt.savefig(self.save_as, transparent=True, dpi=300, bbox_inches='tight')
            else:
                raise ValueError(f"Unknown input '{self.save_as}' for 'save_as'")

    def plot(self):

        if self.axs == None:
            _, axs = plt.subplots(figsize=self.figsize)
        else:
            axs = self.axs

        if isinstance(self.array, np.ndarray):
            self.array = [self.array]
            if not self.color:
                self.color = sns.color_palette(self.cmap, 1)[0]
            else:
                self.color = [self.color]
            
        if isinstance(self.array, list):

            if not isinstance(self.plot_alpha, list):
                if self.plot_alpha == None:
                    self.plot_alpha = [1 for ii in range(len(self.array))]
                else:
                    self.plot_alpha = [self.plot_alpha]
                    if len(self.plot_alpha) != len(self.array):
                        raise ValueError(f"Alpha list ({len(self.plot_alpha)}) does not match length of data list ({len(self.array)})")                        

            if isinstance(self.color, str):
                self.color = [self.color]

            if not isinstance(self.markers, list):
                if self.markers == None:
                    self.markers = [None for ii in range(len(self.array))]
                else:
                    self.markers = [self.markers]

                if len(self.markers) != len(self.array):
                    raise ValueError(f"Marker list ({len(self.markers)}) does not match length of data list ({len(self.array)})")

            if not isinstance(self.markersize, list):
                if self.markersize == None:
                    self.markersize = [None for ii in range(len(self.array))]
                else:
                    self.markersize = [self.markersize]

                if len(self.markersize) != len(self.array):
                    raise ValueError(f"Markersize list ({len(self.markersize)}) does not match length of data list ({len(self.array)})")                            

            # decide on color scheme
            if not isinstance(self.color, list):
                self.color_list = sns.color_palette(self.cmap, len(self.array))
            else:
                self.color_list = self.color
                if len(self.color_list) != len(self.array):
                    raise ValueError(f"Length color list ({len(self.color_list)}) does not match length of data list ({len(self.array)})")
                        
            for idx,el in enumerate(self.array):
                
                # decide on line-width
                if isinstance(self.line_width, list):
                    if len(self.line_width) != len(self.array):
                        raise ValueError(f"Length of line width lenghts {len(self.line_width)} does not match length of data list ({len(self.array)}")

                    use_width = self.line_width[idx]
                elif isinstance(self.line_width, (int,float)):
                    use_width = self.line_width
                else:
                    use_width = ""

                # decide on line-style
                if isinstance(self.line_style, list):
                    if len(self.line_style) != len(self.array):
                        raise ValueError(f"Length of line width lenghts {len(self.line_style)} does not match length of data list ({len(self.array)}")

                    use_style = self.line_style[idx]
                elif isinstance(self.line_style, str):
                    use_style = self.line_style
                else:
                    use_style = "solid"                    

                # decide on x-axis
                if not isinstance(self.xx, (np.ndarray,list,range)):
                    x = np.arange(0, len(el))
                else:
                    # range has no copy attribute
                    if isinstance(self.xx, range):
                        x = self.xx
                    else:
                        x = self.xx.copy()

                if self.labels:
                    lbl = self.labels[idx]
                else:
                    lbl = None

                # plot
                axs.plot(
                    x, 
                    el, 
                    color=self.color_list[idx], 
                    label=lbl, 
                    lw=use_width, 
                    ls=use_style,
                    marker=self.markers[idx],
                    markersize=self.markersize[idx],
                    alpha=self.plot_alpha[idx])

                # plot shaded error bars
                if isinstance(self.error, (list,np.ndarray)):
                    yerr = self.error[idx]
                    if np.isscalar(yerr) or len(yerr) == len(el):
                        ymin = el - yerr
                        ymax = el + yerr
                    elif len(yerr) == 2:
                        ymin, ymax = yerr
                    axs.fill_between(x, ymax, ymin, color=self.color_list[idx], alpha=self.error_alpha)

        # axis labels and titles
        if self.labels:
            axs.legend(
                frameon=False, 
                fontsize=self.label_size)

        if self.x_label:
            axs.set_xlabel(
                self.x_label, 
                fontname=self.fontname, 
                fontsize=self.font_size)

        if self.y_label:
            axs.set_ylabel(
                self.y_label, 
                fontname=self.fontname, 
                fontsize=self.font_size)

        if isinstance(self.title, (str,dict)):
            default_dict = {
                'color': 'k', 
                'fontweight': 'normal'}

            if isinstance(self.title, str):
                self.title_dict = {"title": self.title}
            elif isinstance(self.title, dict):
                self.title_dict = self.title.copy()
            else:
                raise ValueError(f"title input must be a string or dictionary, not {type(self.title)}: '{self.title}'")
            
            # add default keys if they're missing in dictionary
            for key in list(default_dict.keys()):
                if key not in list(self.title_dict.keys()):
                    self.title_dict[key] = default_dict[key]

            axs.set_title(
                self.title_dict["title"],
                color=self.title_dict["color"] ,
                fontweight=self.title_dict["fontweight"],
                fontname=self.fontname, 
                fontsize=self.font_size,
                pad=self.pad_title)

        axs.tick_params(
            width=self.tick_width, 
            length=self.tick_length,
            labelsize=self.label_size)

        for axis in ['top', 'bottom', 'left', 'right']:
            axs.spines[axis].set_linewidth(self.axis_width)
        
        # give priority to specify x-lims rather than seaborn's xlim
        if not self.x_lim:

            if isinstance(self.xlim_left, float) or isinstance(self.xlim_left, int):
                axs.set_xlim(left=self.xlim_left)
            
            if self.xlim_right:
                axs.set_xlim(right=self.xlim_right)
        else:
            axs.set_xlim(self.x_lim)

        if not self.y_lim:
            if isinstance(self.ylim_bottom, float) or isinstance(self.ylim_bottom, int):
                axs.set_ylim(bottom=self.ylim_bottom)
            
            if self.ylim_top:
                axs.set_ylim(top=self.ylim_top)
        else:
            axs.set_ylim(self.y_lim)      

        # despine the axis
        if isinstance(self.x_ticks, list):
            axs.set_xticks(self.x_ticks)

        if isinstance(self.y_ticks, list):
            axs.set_yticks(self.y_ticks)
     
        old_xlim = axs.get_xlim()[-1]
        old_ylim = axs.get_ylim()
        sns.despine(
            offset=self.sns_offset, 
            trim=self.sns_trim, 
            bottom=self.sns_bottom)

        # correct for axis shortening induced by trimming with sns.despine
        set_xlim = 1
        set_ylim = 1
        if self.sns_trim:
            set_xlim = x[-1]/old_xlim
            
            if len(self.array) > 1:
                y_max = np.amax(np.array(self.array))
            else:
                y_max = max(self.array)

            set_ylim = y_max/old_ylim[1]

        # defaults for ax?lines
        default_dict = {
            'color': 'k', 
            'ls': 'dashed', 
            'lw': 0.5}
        
        # add vertical lines
        add_vline = True
        if self.add_vline == "default":
            self.add_vline = {'pos': 0}
        elif isinstance(self.add_vline, int) or isinstance(self.add_vline, list) or isinstance(self.add_vline, np.ndarray):
            self.add_vline = {"pos": self.add_vline}
        elif isinstance(self.add_vline, dict):
            add_vline = True            
        else:
            add_vline = False

        if add_vline:
            for key in list(default_dict.keys()):
                if key not in list(self.add_vline.keys()):
                    self.add_vline[key] = default_dict[key]

            if isinstance(self.add_vline['pos'], list) or isinstance(self.add_vline['pos'], np.ndarray):
                for ix,line in enumerate(self.add_vline['pos']):
                    if isinstance(self.add_vline['color'], list):
                        color = self.add_vline['color'][ix]
                    else:
                        color = self.add_vline['color']

                    axs.axvline(
                        line, 
                        color=color, 
                        lw=self.add_vline['lw'], 
                        ls=self.add_vline['ls'],
                        ymax=set_ylim)
            else:
                axs.axvline(
                    self.add_vline['pos'], 
                    color=self.add_vline['color'],
                    lw=self.add_vline['lw'], 
                    ls=self.add_vline['ls'],
                    ymax=set_ylim)

        add_hline = True
        if self.add_hline == "default":
            self.add_hline = {'pos': 0}
        elif self.add_hline == "mean" or self.add_hline == "average":
            if isinstance(self.array, list):
                if len(self.array) > 1:
                    raise ValueError("This option can't be used with multiple inputs..")
                
            self.add_hline = {'pos': np.array(self.array).mean()}
        elif isinstance(self.add_hline, int) or isinstance(self.add_hline, list) or isinstance(self.add_hline, np.ndarray) or isinstance(self.add_hline, float):
            self.add_hline = {"pos": self.add_hline}
        elif isinstance(self.add_hline, dict):
            add_hline = True
        else:
            add_hline = False

        if add_hline:
            for key in list(default_dict.keys()):
                if key not in list(self.add_hline.keys()):
                    self.add_hline[key] = default_dict[key]

            if isinstance(self.add_hline['pos'], list) or isinstance(self.add_hline['pos'], np.ndarray):
                for ix,line in enumerate(self.add_hline['pos']):
                    if isinstance(self.add_hline['color'], list):
                        color = self.add_hline['color'][ix]
                    else:
                        color = self.add_hline['color']

                    axs.axhline(
                        line,
                        color=color, 
                        lw=self.add_hline['lw'], 
                        ls=self.add_hline['ls'],
                        xmax=set_xlim)
            else:
                axs.axhline(
                    self.add_hline['pos'], 
                    color=self.add_hline['color'],
                    lw=self.add_hline['lw'], 
                    ls=self.add_hline['ls'],
                    xmax=set_xlim)

        if self.return_obj:
            return self

class LazyCorr(Defaults):
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

    def __init__(
        self,
        x, 
        y, 
        color="#cccccc", 
        axs=None, 
        title=None,
        x_label=None, 
        y_label=None, 
        figsize=(8,8),
        xkcd=False,
        sns_trim=True,
        y_lim=None,
        x_lim=None,                 
        save_as=None,
        x_ticks: list=None,
        y_ticks: list=None,        
        points=True,
        scatter_kwargs={},
        **kwargs):

        self.x              = x
        self.y              = y
        self.axs            = axs
        self.x_label        = x_label
        self.y_label        = y_label
        self.xkcd           = xkcd
        self.sns_trim       = sns_trim
        self.color          = color
        self.figsize        = figsize     
        self.title          = title
        self.y_lim          = y_lim
        self.x_lim          = x_lim        
        self.save_as        = save_as
        self.points         = points
        self.scatter_kwargs = scatter_kwargs
        self.x_ticks        = x_ticks
        self.y_ticks        = y_ticks

        super().__init__()
        self.__dict__.update(kwargs)
        self.update_rc(self.fontname)

        if self.xkcd:
            with plt.xkcd():
                self.plot()
        else:
            self.plot()

        if self.save_as:
            if isinstance(self.save_as, list):
                for ii in self.save_as:
                    plt.savefig(ii, transparent=True, dpi=300, bbox_inches='tight')
            elif isinstance(self.save_as, str):
                plt.savefig(self.save_as, transparent=True, dpi=300, bbox_inches='tight')
            else:
                raise ValueError(f"Unknown input '{self.save_as}' for 'save_as'")

    def plot(self):

        if self.axs == None:
            _, axs = plt.subplots(figsize=self.figsize)
        else:
            axs = self.axs        

        sns.regplot(
            x=self.x, 
            y=self.y, 
            color=self.color, 
            ax=axs,
            scatter=self.points,
            scatter_kws=self.scatter_kwargs)

        if isinstance(self.x_label, str):
            axs.set_xlabel(self.x_label, fontsize=self.font_size)

        if isinstance(self.y_label, str):
            axs.set_ylabel(self.y_label, fontsize=self.font_size)

        if self.title:
            axs.set_title(self.title, fontsize=self.font_size)

        axs.tick_params(
            width=self.tick_width, 
            length=self.tick_length,
            labelsize=self.label_size)

        for axis in ['top', 'bottom', 'left', 'right']:
            axs.spines[axis].set_linewidth(self.axis_width)

        if self.x_lim:
            axs.set_xlim(self.x_lim)

        if self.y_lim:
            axs.set_ylim(self.y_lim)

        if isinstance(self.x_ticks, list):
            self.axs.set_xticks(self.x_ticks)

        if isinstance(self.y_ticks, list):
            self.axs.set_yticks(self.y_ticks)  

        sns.despine(offset=self.sns_offset, trim=self.sns_trim)

        if self.return_obj:
            return self

class LazyBar():

    def __init__(
        self, 
        data: pd.DataFrame=None,
        x: Union[str,np.ndarray]=None, 
        y: Union[str,np.ndarray]=None, 
        axs=None,
        sns_ori: str='h', 
        labels: list=None,
        sns_rot: Union[int,float]=None,
        palette: Union[list,sns.palettes._ColorPalette]=None,
        cmap: str="inferno",
        save_as: str=None,
        title: str=None,
        add_labels: bool=False,
        add_axis: bool=True,
        lim: list=None,
        ticks: list=None,
        x_label2: str=None,
        y_label2: str=None,
        title2: str=None,
        add_points: bool=False,
        points_color: Union[str,tuple]=None,
        points_palette: Union[list,sns.palettes._ColorPalette]=None,
        points_cmap: str="viridis",
        points_legend: bool=False,
        points_alpha: float=1,
        error: str="sem",
        fancy: bool=False,
        fancy_rounding: float=0.15,
        fancy_pad: float=-0.004,
        fancy_aspect: float=None,
        fancy_denom: int=4,
        bar_legend: bool=False,
        **kwargs):

        self.data               = data
        self.x                  = x
        self.y                  = y
        self.sns_ori            = sns_ori
        self.labels             = labels
        self.axs                = axs
        self.sns_rot            = sns_rot
        self.palette            = palette
        self.cmap               = cmap
        self.title              = title
        self.add_labels         = add_labels
        self.add_axis           = add_axis
        self.lim                = lim
        self.ticks              = ticks
        self.bar_legend         = bar_legend
        self.x_label2           = x_label2
        self.y_label2           = y_label2
        self.title2             = title2
        self.add_points         = add_points
        self.points_color       = points_color
        self.points_palette     = points_palette
        self.points_cmap        = points_cmap
        self.points_legend      = points_legend
        self.points_alpha       = points_alpha
        self.error              = error
        self.fancy              = fancy
        self.fancy_rounding     = fancy_rounding
        self.fancy_pad          = fancy_pad
        self.fancy_aspect       = fancy_aspect
        self.fancy_denom        = fancy_denom
        self.kw_defaults = Defaults()

        # avoid that these kwargs are passed down to matplotlib.bar.. Throws errors
        ignore_kwargs = [
            "x_label",
            "y_label",
            "add_hline",
            "add_vline",
            "y_lim",
            "trim_left",
            "trim_bottom",
            "points_hue",
            "points_alpha",
            "bbox_to_anchor",
            "figsize",
            "fancy",
            "fancy_rounding",
            "fancy_pad",
            "fancy_aspect",
            "fancy_denom",
            "font_name",
            "x_ticks",
            "y_ticks",
            "bar_legend"
        ]

        kw_sns = {}
        for ii in kwargs:
            if ii not in list(self.kw_defaults.__dict__.keys())+ignore_kwargs:
                kw_sns[ii] = kwargs[ii]

        self.__dict__.update(**self.kw_defaults.__dict__)
        self.__dict__.update(**kwargs)
        self.kw_defaults.update_rc(self.fontname)

        if self.xkcd:
            with plt.xkcd():
                self.plot(**kw_sns)
        else:
            self.plot(**kw_sns)
        
        if save_as:
            plt.savefig(self.save_as, transparent=True, dpi=300, bbox_inches='tight')

        if not hasattr(self, "bbox_to_anchor"):
            self.bbox_to_anchor = None

    def plot(self, **kw_sns):

        if self.axs == None:
            _, axs = plt.subplots(figsize=self.figsize)
        else:
            axs = self.axs

        if self.sns_ori == "h":
            xx = self.y
            yy = self.x
            trim_bottom = False
            trim_left   = True
        elif self.sns_ori == "v":
            xx = self.x 
            yy = self.y
            trim_bottom = True
            trim_left   = False            
        else:
            raise ValueError(f"sns_ori must be 'v' or 'h', not '{self.sns_ori}'")
        
        if "color" in list(kw_sns.keys()):
            if isinstance(kw_sns["color"], (str,tuple)):
                self.palette = None
                self.cmap = None
                self.color = kw_sns["color"]
            elif isinstance(kw_sns["color"], list):
                self.palette = sns.color_palette(palette=kw_sns["color"])
                self.color = None
        else:
            self.color = None
            if isinstance(self.palette, list):
                self.palette = sns.color_palette(palette=self.palette)

            if not isinstance(self.palette, sns.palettes._ColorPalette):
                self.palette = sns.color_palette(self.cmap, len(self.x))

        # check if we can do sem
        self.sem = None
        self.ci = None
        if isinstance(self.error, str):
            if self.error == "sem":
                self.ci = None
                if isinstance(self.data, pd.DataFrame):

                    # filter out relevant colums
                    self.data = self.data[[self.x,self.y]]

                    # get relevant error
                    if hasattr(self, "hue"):
                        self.sem = self.data.groupby([self.hue,self.x]).sem()[self.y].values
                        n_x = len(np.unique(self.data[self.x].values))
                        n_h = len(np.unique(self.data[self.hue].values))
                        self.sem = self.sem.reshape(n_h,n_x)
                    else:
                        self.sem = self.data.groupby(self.x).sem()[self.y].values
            else:
                self.sem = None
                self.ci = self.error
        
        self.ff = sns.barplot(
            data=self.data,
            x=xx, 
            y=yy, 
            ax=axs, 
            orient=self.sns_ori,
            ci=self.ci,
            **dict(
                kw_sns,
                color=self.color,
                palette=self.palette,
                yerr=self.sem
            ))

        if self.add_points:

            if not hasattr(self, "points_hue"):
                self.points_hue = None
            
            if not self.points_palette:
                self.points_palette = self.points_cmap

            # give priority to given points_color
            if isinstance(self.points_color, (str,tuple)):
                self.points_palette = None
                self.points_hue = None

            multi_strip = False
            if hasattr(self, "hue"):

                if isinstance(self.points_hue, str):

                    if self.points_hue != self.hue:
                        multi_strip = True
                            
                        self.hue_items = list(np.unique(self.data[self.points_hue].values))
                        if isinstance(self.points_color, (str,tuple)):
                            self.hue_colors = [self.points_color for ii in range(len(self.hue_items))]
                        else:
                            self.hue_colors = sns.color_palette(self.points_palette, len(self.hue_items))

                        for it, color in zip(self.hue_items, self.hue_colors):
                            df_per_it = self.data[self.data[self.points_hue] == it]
                            sns.stripplot(
                                data=df_per_it, 
                                x=xx, 
                                y=yy, 
                                hue=self.hue,
                                dodge=True, 
                                palette=[color] * 2,
                                ax=self.ff)
                
            else:
                sns.stripplot(
                    data=self.data, 
                    x=xx, 
                    y=yy, 
                    hue=self.points_hue,
                    dodge=False, 
                    ax=self.ff,
                    color=self.points_color,
                    palette=self.points_palette,
                    alpha=self.points_alpha
                )

        if self.bar_legend:
            if hasattr(self, "hue"):

                # find categorical handles
                handles,labels = self.ff.get_legend_handles_labels()

                # find indices of categorical handles in list
                cc = self.data[self.hue].values
                indexes = np.unique(cc, return_index=True)[1]
                cond = [cc[index] for index in sorted(indexes)]
                
                if multi_strip:
                    handles = handles[-len(cond):]
                    labels = labels[-len(cond):]

                # filter out handles that correspond to labels
                if isinstance(self.bbox_to_anchor, tuple):
                    self.ff.legend(
                        handles,
                        labels,
                        frameon=False, 
                        fontsize=self.label_size,
                        bbox_to_anchor=self.bbox_to_anchor,
                        handletextpad=self.legend_handletext)
                else:
                    self.ff.legend(
                        handles,
                        labels,
                        frameon=False, 
                        fontsize=self.label_size,
                        handletextpad=self.legend_handletext)

            else:

                if isinstance(self.bbox_to_anchor, tuple):
                    self.ff.legend(
                        frameon=False,
                        fontsize=self.label_size,
                        bbox_to_anchor=self.bbox_to_anchor,
                        handletextpad=self.legend_handletext)
                else:
                    self.ff.legend(
                        frameon=False,
                        fontsize=self.label_size,
                        handletextpad=self.legend_handletext)  
        else:
            if self.points_legend:
                if self.add_points:
                    if isinstance(self.bbox_to_anchor, tuple):
                        self.ff.legend(
                            frameon=False,
                            fontsize=self.label_size,
                            bbox_to_anchor=self.bbox_to_anchor,
                            handletextpad=self.legend_handletext)
                    else:
                        self.ff.legend(
                            frameon=False,
                            fontsize=self.label_size,
                            handletextpad=self.legend_handletext)                        

            else:
                self.ff.legend([],[], frameon=False)
                
        # axis labels and titles
        if self.title:
            self.ff.set_title(
                self.title, 
                fontname=self.fontname, 
                fontsize=self.font_size)                    
        
        self.ff.tick_params(
            width=self.tick_width, 
            length=self.tick_length,
            labelsize=self.label_size)

        for axis in ['top', 'bottom', 'left', 'right']:
            self.ff.spines[axis].set_linewidth(self.axis_width)

        if not self.add_labels:
            if self.sns_ori == 'h':
                self.ff.set_yticks([])
            elif self.sns_ori == "v":                
                self.ff.set_xticks([])
            else:
                raise ValueError(f"sns_ori must be 'v' or 'h', not '{self.sns_ori}'")
        elif isinstance(self.add_labels,list):
            self.ff.set_xlabel(self.add_labels)

        if isinstance(self.sns_rot, (int,float)):
            if self.sns_ori == 'h':
                self.ff.set_yticklabels(self.ff.get_yticklabels(), rotation=self.sns_rot)
            elif self.sns_ori == "v":
                self.ff.set_xticklabels(self.ff.get_xticklabels(), rotation=self.sns_rot)
            else:
                raise ValueError(f"sns_ori must be 'v' or 'h', not '{self.sns_ori}'")

        if isinstance(self.lim, list):
            if self.sns_ori == 'h':
                self.ff.set_xlim(self.lim)
            elif self.sns_ori == "v":
                self.ff.set_ylim(self.lim)
            else:
                raise ValueError(f"sns_ori must be 'v' or 'h', not '{self.sns_ori}'")     
        
        if isinstance(self.ticks, list):
            if self.sns_ori == 'h':
                self.ff.set_xticks(self.ticks)
            elif self.sns_ori == "v":
                self.ff.set_yticks(self.ticks)
            else:
                raise ValueError(f"sns_ori must be 'v' or 'h', not '{self.sns_ori}'")

        # from: https://stackoverflow.com/a/61569240
        if self.fancy:
            new_patches = []

            for patch in reversed(self.ff.patches):

                bb = patch.get_bbox()
                color = patch.get_facecolor()

                # max of axis divided by 4 gives nice rounding
                if not isinstance(self.fancy_aspect, (int,float)):
                    y_limiter = patch._axes.get_ylim()[-1]
                    if isinstance(self.lim, list):
                        y_limiter-=self.lim[0]

                    self.fancy_aspect = y_limiter/self.fancy_denom
                
                # make rounding at limit
                if isinstance(self.lim, list):
                    ymin = self.lim[0]
                    height = bb.height - ymin
                else:
                    ymin = bb.ymin
                    height = bb.height

                p_bbox = patches.FancyBboxPatch(
                    (bb.xmin, ymin),
                    abs(bb.width), abs(height),
                    boxstyle=f"round,pad={self.fancy_pad},rounding_size={self.fancy_rounding}",
                    ec="none", 
                    fc=color,
                    mutation_aspect=self.fancy_aspect
                )

                patch.remove()
                new_patches.append(p_bbox)

            for patch in new_patches:
                self.ff.add_patch(patch)

        if isinstance(self.x, str) and not isinstance(self.x_label2, str):
            self.ff.set(xlabel=None)

        if isinstance(self.y, str) and not isinstance(self.y_label2, str):
            self.ff.set(ylabel=None)            

        if self.x_label2:
            self.ff.set_xlabel(
                self.x_label2, 
                fontname=self.fontname, 
                fontsize=self.font_size)

        if self.y_label2:
            self.ff.set_ylabel(
                self.y_label2, 
                fontname=self.fontname,
                fontsize=self.font_size)

        if hasattr(self, "trim_left"):
            trim_left = self.trim_left
        else:
            trim_left = False

        if hasattr(self, "trim_bottom"):
            trim_bottom = self.trim_bottom
        else:
            trim_bottom = False

        sns.despine(
            offset=self.sns_offset, 
            trim=self.sns_trim,
            left=trim_left, 
            bottom=trim_bottom, 
            ax=self.ff)

        if self.title2:
            self.ff.set_title(
                self.title2, 
                fontname=self.fontname, 
                fontsize=self.font_size,
                pad=self.pad_title)                    

        if self.return_obj:
            return self

class LazyHist(Defaults):
    """LazyHist

    Wrapper around seaborn's histogram plotter

    Parameters
    ----------
    data: numpy.ndarray, pandas.DataFrame
        Input for histogram. Can be either numpy array or pandas dataframe. In case of the latter, `x` and `y` need to be column names to put the correct data on the axes.
    save_as: str, optional
        file path to save the image (*.pdf is recommended for quality and compatibility with Inkscape)
    axs: <AxesSubplot:>, optional
        Matplotlib axis to store the figure on
    cmap: str, optional
        Colormap for imshow; accepts output from :func:`linescanning.utils.make_binary_cm`. Defaults to 'magma'
    alpha: float, optional
        Opacity for imshow
    xkcd: bool, optional
        Plot in cartoon-format
    label_size: int, optional
        Set the font size of the labels (i.e., axes). Default = 10
    tick_width: float, optional
        Set the thickness of the ticks. Larger value means thicker tick. Default = 0.5 (thin'ish)
    tick_length: int, optional
        Set the length of the ticks. Larger values mean longer ticks. Default = 7 (long'ish)
    axis_width: float, optional
        Set the thickness of the spines of the plot. Larger values mean thicker spines. Default = 0.5 (thin'ish)
    sns_trim: bool, optional
        If `True`, limit spines to the smallest and largest major tick on each non-despined axis. Maps to `sns.despine(trim=sns_trim)`
    sns_offset: int, optional
        Offset in the origin of the plot. Maps to `sns.despine(offset=sns_offset)`. Default is 10
    line_width: float, optional
        Width of the outer border of the visual field if `cmap` is not *viridis* or *magma* (these color maps are quite default, and do not require an extra border like :func:`linescanning.utils.make_binary_cm`-objects do). Default is 0.5.

    Returns
    ----------
    matplotlib.pyplot plot
    """

    def __init__(
        self, 
        data, 
        x=None,
        y=None,
        save_as=None, 
        axs=None, 
        xkcd=False,
        title=None,
        figsize=(8,8),
        kde=False,
        hist=True,
        bins="auto",
        fill=False,
        kde_kwargs={},
        hist_kwargs={},
        color="#cccccc",
        x_ticks: list=None,
        y_ticks: list=None,
        x_label2: str=None,
        y_label2: str=None,
        title2: str=None,
        return_obj: bool=False,
        x_lim: list=None,
        y_lim: list=None,
        fancy: bool=False,
        fancy_rounding: float=0.15,
        fancy_pad: float=-0.004,
        fancy_aspect: float=None,
        **kwargs):
        
        self.data           = data
        self.x              = x
        self.y              = y
        self.save_as        = save_as
        self.axs            = axs
        self.xkcd           = xkcd
        self.title          = title
        self.figsize        = figsize
        self.kde            = kde
        self.kde_kwargs     = kde_kwargs
        self.hist_kwargs    = hist_kwargs
        self.hist           = hist
        self.bins           = bins
        self.fill           = fill
        self.x_label2       = x_label2
        self.y_label2       = y_label2
        self.title2         = title2
        self.x_ticks        = x_ticks
        self.y_ticks        = y_ticks
        self.color          = color
        self.return_obj     = return_obj
        self.x_lim          = x_lim
        self.y_lim          = y_lim
        self.kwargs         = kwargs
        self.fancy          = fancy
        self.fancy_rounding = fancy_rounding
        self.fancy_pad      = fancy_pad
        self.fancy_aspect   = fancy_aspect        

        super().__init__()
        self.__dict__.update(kwargs)
        self.__dict__.update(self.kde_kwargs)
        self.update_rc(self.fontname)

        if self.xkcd:
            with plt.xkcd():
                self.plot()
        else:
            self.plot()

        # if self.kde:
        #     self.kde = self.return_kde()

        if self.save_as:
            if isinstance(self.save_as, list):
                for ii in self.save_as:
                    plt.savefig(ii, transparent=True, dpi=300, bbox_inches='tight')
            elif isinstance(self.save_as, str):
                plt.savefig(self.save_as, transparent=True, dpi=300, bbox_inches='tight')
            else:
                raise ValueError(f"Unknown input '{self.save_as}' for 'save_as'")

    def plot(self):

        if self.axs == None:
            _,self.axs = plt.subplots(figsize=self.figsize)

        if self.hist:
            self.vals, self.bins, self.patches = self.axs.hist(
                self.data,
                density=True,
                bins=self.bins,
                color=self.color,
                **self.hist_kwargs
            )

            # from: https://stackoverflow.com/a/61569240
            if self.fancy:
                new_patches = []

                for patch in reversed(self.patches):

                    # max of axis divided by 4 gives nice rounding
                    if not isinstance(self.fancy_aspect, (int,float)):
                        self.fancy_aspect = patch._axes.get_ylim()[-1]/4
                    
                    bb = patch.get_bbox()
                    color = patch.get_facecolor()
                    p_bbox = patches.FancyBboxPatch(
                        (bb.xmin, bb.ymin),
                        abs(bb.width), abs(bb.height),
                        boxstyle=f"round,pad={self.fancy_pad},rounding_size={self.fancy_rounding}",
                        ec="none", 
                        fc=color,
                        mutation_aspect=self.fancy_aspect
                    )

                    patch.remove()
                    new_patches.append(p_bbox)

                for patch in new_patches:
                    self.axs.add_patch(patch)

        if self.kde:
            
            self.ff = sns.kdeplot(
                data=self.data,
                x=self.x,
                y=self.y,
                ax=self.axs,
                fill=self.fill,
                color=self.color,
                **self.kde_kwargs
            )

        # there's no self.ff if kde=False
        if hasattr(self, "ff"):
            self.active_axs = self.ff
        else:
            self.active_axs = self.axs

        # axis labels and titles
        if self.title:
            self.active_axs.set_title(
                self.title, 
                fontname=self.fontname, 
                fontsize=self.font_size)                    
        
        self.active_axs.tick_params(
            width=self.tick_width, 
            length=self.tick_length,
            labelsize=self.label_size)

        for axis in ['top', 'bottom', 'left', 'right']:
            self.active_axs.spines[axis].set_linewidth(self.axis_width)

        # give priority to specify x-lims rather than seaborn's xlim
        if isinstance(self.x_lim, list):
            self.active_axs.set_xlim(self.x_lim)
        
        if isinstance(self.y_lim, list):
            self.active_axs.set_ylim(self.y_lim)   

        if isinstance(self.x_ticks, list):
            self.active_axs.set_xticks(self.x_ticks)

        if isinstance(self.y_ticks, list):
            self.active_axs.set_yticks(self.y_ticks)            

        if not isinstance(self.x_label2, str):
            self.active_axs.set(xlabel=None)

        if  not isinstance(self.y_label2, str):
            self.active_axs.set(ylabel=None)            

        if self.x_label2:
            self.active_axs.set_xlabel(
                self.x_label2, 
                fontname=self.fontname, 
                fontsize=self.font_size)

        if self.y_label2:
            self.active_axs.set_ylabel(
                self.y_label2, 
                fontname=self.fontname,
                fontsize=self.font_size)

        if hasattr(self, "trim_left"):
            trim_left = self.trim_left
        else:
            trim_left = False

        if "trim_bottom" in list(self.kwargs.keys()):
            trim_bottom = self.kwargs["trim_bottom"]
        else:
            trim_bottom = False

        sns.despine(
            offset=self.sns_offset, 
            trim=self.sns_trim,
            left=trim_left, 
            bottom=trim_bottom, 
            ax=self.active_axs)

        if self.title2:
            self.active_axs.set_title(
                self.title2, 
                fontname=self.fontname, 
                fontsize=self.font_size,
                pad=self.pad_title)

    def return_kde(self):
        return self.ff.get_lines()[0].get_data()

def conform_ax_to_obj(
    ax,
    obj=None,
    title=None,
    x_label=None,
    y_label=None):

    """

    Function to conform any plot to the aesthetics of this plotting module. Can be used when a plot is created with functions other than :class:`linescanning.plotting.LazyPlot`, :class:`linescanning.plotting.LazyCorr`, :class:`linescanning.plotting.LazyHist`, or any other function specified in this file. Assumes `ax` is a `matplotlib.axes._subplots.AxesSubplot` object, and `obj` a `linescanning.plotting.Lazy*`-object.

    Parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot
        input axis that needs to be modified
    obj: linescanning.plotting.Lazy*
        linecanning-specified plotting object containing the information with which `ax` will be conformed
    title: str
        overwrite any existing `title` in `ax`
    x_label: str
        overwrite any existing `x_label` in `ax`
    y_label: str
        overwrite any existing `y_label` in `ax`

    Returns
    ----------
    matplotlib.axes._subplots.AxesSubplot
        input axis with updated aesthetics

    Example
    ----------
    >>> # convert statsmodels's QQ-plot
    >>> from linescanning import plotting
    >>> import matplotlib as plt
    >>> from scipy import stats
    >>> #
    >>> fig,ax = plt.subplots(figsize=(8,8))
    >>> sm.qqplot(mdf.resid, dist=stats.norm, line='s', ax=ax)
    >>> ax = plotting.conform_ax_to_obj(ax,pl)
    """
    
    if obj == None:
        obj = Defaults()

    if not isinstance(title, str):
        title = ax.get_title()

    if not isinstance(y_label, str):
        y_label = ax.get_ylabel()

    if not isinstance(x_label, str):
        x_label = ax.get_xlabel()           

    if isinstance(title, str):
        ax.set_title(
        title,
        fontname=obj.fontname,
        fontsize=obj.font_size)
    
    if isinstance(y_label, str):
        ax.set_ylabel(
            y_label,
            fontname=obj.fontname,
            fontsize=obj.font_size)

    if isinstance(x_label, str):
        ax.set_xlabel(
            x_label,
            fontname=obj.fontname,
            fontsize=obj.font_size)        

    ax.tick_params(
        width=obj.tick_width, 
        length=obj.tick_length,
        labelsize=obj.label_size)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(obj.axis_width)

    sns.despine(
        offset=obj.sns_offset, 
        trim=obj.sns_trim,
        left=False, 
        bottom=False, 
        ax=ax)

    return ax

class LazyColorbar(Defaults):

    def __init__(
        self,
        axs=None,
        cmap="magma_r",
        txt=None,
        vmin=0,
        vmax=10,
        ori="vertical",
        ticks=None,
        flip_ticks=False,
        flip_label=False,
        figsize=(6,0.5),
        save_as=None,
        **kwargs):

        self.axs = axs
        self.cmap = cmap
        self.txt = txt
        self.vmin = vmin
        self.vmax = vmax
        self.ori = ori
        self.ticks = ticks
        self.flip_ticks = flip_ticks
        self.flip_label = flip_label
        self.figsize = figsize
        self.save_as = save_as

        super().__init__()
        self.__dict__.update(kwargs)
        self.update_rc(self.fontname)

        if self.axs == None:
            if isinstance(self.save_as, str):
                self.fig, self.axs = plt.subplots(figsize=self.figsize)
            else:
                _, self.axs = plt.subplots(figsize=self.figsize)
            
        # set ticks to integer intervals if nothing's specified
        if not isinstance(self.ticks, list):
            self.ticks = [int(ii) for ii in range(vmin,vmax+1)]

        # make colorbase instance
        if isinstance(self.cmap, str):
            self.cmap = mpl.cm.get_cmap(self.cmap, 256)

        # plop everything in class
        mpl.colorbar.ColorbarBase(
            self.axs, 
            orientation=self.ori, 
            cmap=self.cmap,
            norm=mpl.colors.Normalize(vmin,vmax),
            label=self.txt,
            ticks=self.ticks)

        if self.ori == "vertical":
            # set font stuff
            if self.flip_ticks:
                self.axs.yaxis.set_ticks_position("left")

            if self.flip_label:
                self.axs.yaxis.set_label_position("left")

            text = self.axs.yaxis.label
        else:
            if self.flip_ticks:
                self.axs.xaxis.set_ticks_position("top")

            if self.flip_label:
                self.axs.xaxis.set_label_position("top")                      
                
            text = self.axs.xaxis.label

        font = mpl.font_manager.FontProperties(size=self.font_size)
        text.set_font_properties(font)

        # fix ticks
        self.axs.tick_params(
            width=self.tick_width, 
            length=self.tick_length,
            labelsize=self.label_size)        

        # turn off frame
        self.axs.set_frame_on(False)

        if hasattr(self, "fig"):
            self.fig.savefig(
                self.save_as,
                facecolor="white",
                bbox_inches="tight")

def fig_annot(fig, y=1.01, x0_corr=0, x_corr=-0.09, fontsize=28):

    # get figure letters
    alphabet = list(string.ascii_uppercase)

    # make annotations
    for ix,ax in enumerate(fig.axes):
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        if ix == 0:
            move_frac = x0_corr/bbox.width
        else:
            move_frac = x_corr/bbox.width

        pos = move_frac

        ax.annotate(
            alphabet[ix], 
            (pos,y), 
            fontsize=fontsize, 
            xycoords="axes fraction")        
