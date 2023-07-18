import numpy as np
from linescanning import utils
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
import string
from typing import Union

class Defaults():
    """Defaults

    Default settings for plotting.

    Parameters
    ----------
    pad_title: int
        Set the distance between the title and the plot. Default = 20
    title_size: int
        Set the font size of titles. Default = 22
    font_size: int
        Set the font size of axis labels/titles. Default = 18
    label_size: int
        Set the font size of tick labels. Default = 14
    tick_width: float
        Set the line-width of the ticks. Default = 0.5
    tick_length: float
        Set the length of the ticks. Default = 0 (no ticks)
    axis_width: float
        Set the line-width of axes. Default = 0.5
    line_width: int
        Line widths for either all graphs (then *int*) or a *list* with the number of elements as requested graphs, default = 1.
    line_style: str
        Set the style of data in line-plots. Default = "solid"
    sns_offset: int
        Set the distance between y-axis and start of plot. Default = None
    sns_trim: bool
        Trim the axes following seaborn's convention. Default = False
    sns_bottom: bool
        Trim specifically the x-axis of plots. Default = False
    sns_ori: str, optional
        Default orientation for bar-plots. Default is up-right (vertical). Allowed options are "v" or "h". :class:`linescanning.plotting.LazyBar()`
    sns_rot: int, float, optional
        Rotation of labels in bar plot. Specific to :class:`linescanning.plotting.LazyBar()`
    xkcd: bool
        Plot with cartoon style. Default = False
    ylim_bottom: float
        Set the y-limiter at the bottom of the plot. Default = None
    ylim_top: float
        Set the y-limiter at the top of the plot. Default = None
    xlim_left: float
        Set the x-limiter at the left of the plot. Default = None
    xlim_right: float
        Set the x-limiter at the right of the plot. Default = None        
    set_xlim_zero: bool
        Set the distance between plot and y-axis to 0. Default = False
    legend_handletext: float
        Set the distance between the handle and text in legends. Default = 0.05 (bit closer than default)
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
    save_as: str, list, optional
        Save the plot, by default None. If you want to use figures in Inkscape, save them as PDFs to retain high resolution; specify a list of strings to save the plot with multiple extensions
    y_lim: list, optional
        List for `self._set_ylim`
    x_lim: list, optional
        List for `self_.set_xlim`       
    x_dec: int, optional
        Enforce `x_ticks` to have `x_dec` decimal accuracy. Default is whatever the data dictates
    y_dec: int, optional
        Enforce `y_ticks` to have `y_dec` decimal accuracy. Default is whatever the data dictates      
    add_hline: dict, optional
        Dictionary for a horizontal line through the plot, by default None. Collects the following items:

>>> add_hline = {
>>>     'pos' 0,       # position
>>>     'color': 'k',  # color
>>>     'lw': 1,       # linewidth
>>>     'ls': '--'}    # linestyle

        You can get the settings above by specifying *add_hline='default'*. Now also accepts *add_hline='mean'* for single inputs
    add_vline: dict, optional
        Dictionary for a vertical line through the plot, by default None. Same keys as `add_hline`       
    dpi: int, optional
        Save figures with DPI-value. Default is 300
    figure_background_color: str, optional
        Background of images. Default is "white"
    bbox_inches: str, optional
        Bounding box settings. Default is "tight"    
    """

    def __init__(self, **kwargs):

        self.ls_kwargs = [
            "pad_title",
            "font_size",
            "title_size",
            "label_size",
            "tick_width",
            "tick_length",
            "axis_width",
            "line_width",
            "line_style",
            "sns_offset",
            "sns_trim",
            "sns_bottom",
            "sns_ori",
            "sns_rot",
            "xkcd",
            "ylim_bottom",
            "ylim_top",
            "xlim_left",
            "xlim_right",
            "set_xlim_zero",
            "legend_handletext",
            "x_label",
            "y_label",
            "title",
            "save_as",
            "y_lim",
            "x_lim",
            "x_ticks",
            "y_ticks",
            "axs",
            "color",
            "y_dec",
            "x_dec",
            "add_vline",
            "add_hline",
            "dpi",
            "figure_background_color",
            "bbox_inches",
            "fontname"
        ]

        self.pad_title = 20
        self.font_size = 18
        self.title_size = 24
        self.label_size = 14
        self.tick_width = 0.5
        self.tick_length = 0
        self.axis_width = 0.5
        self.line_width = 1
        self.line_style = "solid"
        self.sns_offset = None
        self.sns_trim = False
        self.sns_bottom = False
        self.sns_ori = "v"
        self.sns_rot = None
        self.xkcd = False
        self.ylim_bottom = None
        self.ylim_top = None
        self.xlim_left = None
        self.xlim_right = None
        self.set_xlim_zero = False
        self.legend_handletext = 0.05
        self.x_label = None
        self.y_label = None
        self.title = None
        self.save_as = None
        self.y_lim = None
        self.x_lim = None
        self.x_ticks = None
        self.y_ticks = None
        self.axs = None
        self.color = None
        self.y_dec = None
        self.x_dec = None
        self.add_vline = None
        self.add_hline = None
        self.dpi = 300
        self.figure_background_color = "white"
        self.bbox_inches = "tight"
        self.fontname = None
        

        # update kwargs
        self.__dict__.update(kwargs)
        
        # set default font
        if self.xkcd:
            self.fontname = "Humor Sans"
        else:
            if not isinstance(self.fontname, str):
                self.fontname = "Montserrat"

        # update font widely
        self.update_rc(self.fontname)
            
    def update_rc(self, font):
        """update font"""
        plt.rcParams.update({'font.family': font})

    def _set_figure_axs(self, figsize=None):
        if not isinstance(self.axs, mpl.axes._axes.Axes):
            if not isinstance(figsize, tuple):
                figsize = self.figsize
            _, self.axs = plt.subplots(figsize=figsize)

    def _set_spine_width(self, ax):
        """update spine width"""
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(self.axis_width)

    def _set_xlabel(self, ax, lbl, **kwargs):
        """set x-label"""
        if isinstance(lbl, (str,list)):
            ax.set_xlabel(
                lbl, 
                fontsize=self.font_size,
                fontname=self.fontname,
                **kwargs)

    def _set_ylabel(self, ax, lbl, **kwargs):
        """set y-label"""
        if isinstance(lbl, (str,list)):
            ax.set_ylabel(
                lbl, 
                fontsize=self.font_size, 
                fontname=self.fontname,
                **kwargs)

    def _set_tick_params(self, ax, **kwargs):
        """set width/length/labelsize of ticks"""
        ax.tick_params(
            width=self.tick_width, 
            length=self.tick_length,
            labelsize=self.label_size,
            **kwargs)

    def _set_title(self, ax, title, **kwargs):
        """set title of plot"""

        if isinstance(title, (str,dict)):
            default_dict = {
                'color': 'k', 
                'fontweight': 'normal'}

            if isinstance(title, str):
                title_dict = {"title": self.title}
            elif isinstance(title, dict):
                title_dict = title.copy()
            else:
                raise ValueError(f"title input must be a string or dictionary, not {type(title)}: '{title}'")
            
            # add default keys if they're missing in dictionary
            for key in list(default_dict.keys()):
                if key not in list(title_dict.keys()):
                    title_dict[key] = default_dict[key]

            ax.set_title(
                title_dict["title"],
                color=title_dict["color"] ,
                fontweight=title_dict["fontweight"],
                fontname=self.fontname, 
                fontsize=self.title_size,
                pad=self.pad_title,
                **kwargs)
    
    def _set_bar_lim(self, ax, lim):
        if isinstance(lim, list):
            if self.sns_ori == 'h':
                fc = self._set_xlim
            elif self.sns_ori == "v":
                fc = self._set_ylim
            else:
                raise ValueError(f"sns_ori must be 'v' or 'h', not '{self.sns_ori}'")  

            # set
            fc(ax, lim)

    def _set_bar_ticks(self, ax, ticks):
        if isinstance(ticks, list):
            if self.sns_ori == 'h':
                fc = self._set_xticks
            elif self.sns_ori == "v":
                fc = self._set_yticks
            else:
                raise ValueError(f"sns_ori must be 'v' or 'h', not '{self.sns_ori}'")  

            # set
            fc(ax, ticks)

    @staticmethod
    def _set_xticks(ax, ticks):
        """set x-ticks"""
        if isinstance(ticks, list):
            ax.set_xticks(ticks)

    @staticmethod
    def _set_yticks(ax, ticks):
        """set y-ticks"""
        if isinstance(ticks, list):
            ax.set_yticks(ticks)

    @staticmethod
    def _set_ylim(ax,lim):
        """set y-limit"""
        if isinstance(lim, list):
            ax.set_ylim(lim)
        elif isinstance(lim, (int,float)):
            ax.set_xlim(bottom=lim)
            
    @staticmethod
    def _set_xlim(ax,lim):
        """set x-limit"""
        if isinstance(lim, list):
            ax.set_xlim(lim)   
        elif isinstance(lim, (int,float)):
            ax.set_xlim(left=lim)

    def _despine(self, ax, **kwargs):
        """despine plot"""
        sns.despine(
            ax=ax, 
            offset=self.sns_offset, 
            trim=self.sns_trim,
            **kwargs)

    @staticmethod
    def _set_y_ticker(ax, dec):
        """set all y-ticks to decimal"""
        if isinstance(dec, int):
            from matplotlib.ticker import FormatStrFormatter
            ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{dec}f"))

    @staticmethod
    def _set_x_ticker(ax, dec):
        """set all x-ticks to decimal"""
        if isinstance(dec, int):
            from matplotlib.ticker import FormatStrFormatter
            ax.xaxis.set_major_formatter(FormatStrFormatter(f"%.{dec}f"))

    def _set_shaded_error(
        self, 
        x: np.ndarray=None,
        tc: np.ndarray=None,
        ax: mpl.axes._axes.Axes=None, 
        yerr: np.ndarray=None,
        **kwargs):

        if np.isscalar(yerr) or len(yerr) == len(tc):
            ymin = tc - yerr
            ymax = tc + yerr
        elif len(yerr) == 2:
            ymin, ymax = yerr

        ax.fill_between(
            x, 
            ymax, 
            ymin, 
            **kwargs)

    def _set_legend_labels(self, ax, labels=None, **kwargs):
        if isinstance(labels, (list,np.ndarray)):
            ax.legend(
                frameon=False, 
                fontsize=self.label_size,
                fontname=self.fontname
                **kwargs)        

    def _save_as(self, save_as, **kwargs):
        """simple save function"""
        if isinstance(save_as, str):
            plt.savefig(
                save_as,
                bbox_inches=self.bbox_inches,
                dpi=self.dpi,
                facecolor=self.figure_background_color,
                **kwargs
            )

    def _save_figure(self, save_as):
        """save same figure with multiple extensions"""
        if isinstance(save_as, (list,str)):
            if isinstance(save_as, str):
                save_as = [save_as]
            
            for ii in save_as:
                self._save_as(ii)
  
    def _add_line(
        self,
        ax=None,
        **kwargs):

        for ii in ["hline","vline"]:

            test_attr = getattr(self, f"add_{ii}")
            if isinstance(test_attr, (float,int,dict,str,list)):

                add_line = True

                # define default dictionary
                default_dict = {
                    'color': 'k', 
                    'ls': 'dashed', 
                    'lw': 0.5
                }

                # set fixer depending on line being drawn
                if ii == "hline":
                    default_dict["min"] = 0
                    default_dict["max"] = 1
                else:
                    default_dict["min"] = 0                    
                    default_dict["max"] = 1

                # add lines
                if test_attr == "default":
                    test_attr = {'pos': 0}
                elif isinstance(test_attr, (float,int,list,np.ndarray)):
                    test_attr = {"pos": test_attr}
                elif isinstance(test_attr, dict):
                    add_line = True
                else:
                    add_line = False

                if add_line:
                    for key in list(default_dict.keys()):
                        if key not in list(test_attr.keys()):
                            test_attr[key] = default_dict[key]

                    # enforce list so we only need to call functions once
                    if not "pos" in list(test_attr.keys()):
                        raise ValueError(f"Need the 'pos' key to denote position..")
                    else:
                        if isinstance(test_attr['pos'], (int,float)):
                            test_attr['pos'] = [test_attr['pos']]

                    # loop through elements
                    if isinstance(test_attr['pos'], (list,np.ndarray)):
                        for ix,line in enumerate(test_attr['pos']):
                            if isinstance(test_attr['color'], list):
                                color = test_attr['color'][ix]
                            else:
                                color = test_attr['color']

                            if ii == "hline":
                                ax.axhline(
                                    line,
                                    color=color,
                                    lw=test_attr['lw'], 
                                    ls=test_attr['ls'],
                                    xmin=test_attr["min"],
                                    xmax=test_attr["max"],
                                    **kwargs
                                )
                            else:
                                ax.axvline(
                                    line,
                                    color=color,
                                    lw=test_attr['lw'], 
                                    ls=test_attr['ls'],
                                    ymin=test_attr["min"],
                                    ymax=test_attr["max"],
                                    **kwargs
                                )            

class LazyPRF(Defaults):
    """LazyPRF

    Plot the geometric location of the Gaussian pRF.

    Parameters
    ----------
    prf: numpy.ndarray
        instantiation of `gauss2D_iso_cart`; will be np.squeeze'ed over the first axis if `ndim >= 3`.
    vf_extent: list
        the space the pRF lives in
    cmap: str, optional
        Colormap for imshow; accepts output from :func:`linescanning.utils.make_binary_cm`. Defaults to 'magma'
    cross_color: str, optional
        Color for the fixation cross; defaults to 'white'. You can set it to 'k' if you have a binary colormap as input
    alpha: float, optional
        Opacity for imshow
    shrink_factor: float, optional
        When the background of the image is white, we create a black border around the Circle patch. If this is equal to `vf_extent`, the border is cut off at some points. This factor shrinks the radius of the Circle, so that we can have a nice border. When set to 0.9, it becomes sort of like a target. This is relevant for **all** non-`magma` color maps that you insert, specifically a :func:`linescanning.utils.make_binary_cm` object
    full_axis: bool, optional
        If `True`, the entire axis of `vf_extent` will be used for the ticks (more detailed). If `False`, a truncated/trimmed version will be returned (looks cleaner). Default = False
    axis_off: bool, optional
        If `True` the x/y axis will be maintained, and the `vf_extent` will be given as ticks. If `False`, axis will be turned off. If `axis_off=True`, then `full_axis` and other label/axis parameters are ignored. Default = True
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
    imshow_kw: dict, optional
        Additional kwargs passed on to `imshow`

    Returns
    ----------
    matplotlib.pyplot plot
    """

    def __init__(
        self, 
        prf, 
        vf_extent, 
        cmap='RdBu_r', 
        cross_color="white", 
        alpha=None,
        shrink_factor=1, 
        axis_off=True,
        figsize=(8,8),
        full_axis=False,
        vf_only=False,
        cross_width=0.5,
        concentric=None,
        z_lines=1,
        z_prf=0,
        edge_color=None,
        imshow_kw={},
        **kwargs):
        
        self.prf            = prf
        self.vf_extent      = vf_extent
        self.cmap           = cmap
        self.cross_color    = cross_color
        self.alpha          = alpha
        self.shrink_factor  = shrink_factor
        self.axis_off       = axis_off
        self.figsize        = figsize
        self.full_axis      = full_axis
        self.vf_only        = vf_only
        self.cross_width    = cross_width
        self.concentric     = concentric
        self.z_lines        = z_lines
        self.z_prf          = z_prf
        self.edge_color     = edge_color
        self.imshow_kw      = imshow_kw

        super().__init__()
        self.__dict__.update(kwargs)
        self.update_rc(self.fontname)

        if self.xkcd:
            with plt.xkcd():
                self.plot()
        else:
            self.plot()

        # save
        self._save_figure(self.save_as)

    def plot(self):

        # set figure axis
        self._set_figure_axs()

        if self.prf.ndim >= 3:
            self.prf = np.squeeze(self.prf, axis=0)

        if self.alpha == None:
            self.alpha = 1

        # add cross-hair
        for ii in ["hline","vline"]:
            self.line_kw = {
                "pos": 0,
                "color": self.cross_color,
                "lw": self.cross_width
            }

            setattr(self, f"add_{ii}", self.line_kw)
        
        self._add_line(
            self.axs,
            zorder=self.z_lines
        )

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
                    
        if len(self.vf_extent) < 4:
            self.use_extent = self.vf_extent+self.vf_extent
        else:
            self.use_extent = self.vf_extent

        im = self.axs.imshow(
            plot_obj, 
            extent=self.use_extent, 
            cmap=self.cmap, 
            alpha=self.alpha,
            zorder=self.z_prf,
            vmin=vmin,
            vmax=vmax,
            **self.imshow_kw)
        
        # In case of a white background, the circle for the visual field is cut off, so we need to make an adjustment:
        if self.cmap != 'magma' and self.cmap != 'viridis':
            radius = self.use_extent[-1]*self.shrink_factor
        else:
            radius = self.use_extent[-1]

        # set title
        self._set_title(self.axs, self.title)
        
        # set patch
        self.patch = patches.Circle(
            (0,0),
            radius=radius,
            transform=self.axs.transData,
            edgecolor=self.edge_color,
            facecolor="None",
            linewidth=self.line_width)

        self.axs.add_patch(self.patch)
        im.set_clip_path(self.patch)

        if self.axis_off:
            self.axs.axis('off')
        else:
            # set tick params
            self._set_tick_params(self.axs)
        
            # set spine widths
            self._set_spine_width(self.axs)
            
            if self.full_axis:
                self.use_ticks = np.arange(self.vf_extent[0],self.vf_extent[1]+1, 1)
            else:
                self.use_ticks = self.vf_extent

            # set ticks
            self._set_xticks(self.axs, self.use_ticks)
            self._set_yticks(self.axs, self.use_ticks)

            # set tickers & despine
            self._set_y_ticker(self.axs, self.y_dec)
            self._set_x_ticker(self.axs, self.x_dec)
            self._despine(self.axs)

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

    cmap: str, optional
        Color palette to use for colors if no individual colors are specified, by default 'viridis'
    figsize: tuple, optional
        Figure dimensions as per usual matplotlib conventions, by default (25,5)
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
        figsize=(25,5),
        cmap='viridis',
        labels=None,
        markers=None,
        markersize=None,
        plot_alpha=None,
        **kwargs):

        self.array = ts
        self.xx = xx
        self.error = error
        self.error_alpha = error_alpha
        self.plot_alpha = plot_alpha
        self.figsize = figsize
        self.cmap = cmap
        self.labels = labels
        self.markers = markers
        self.markersize = markersize

        super().__init__()
        self.__dict__.update(kwargs)
        self.update_rc(self.fontname)

        # plot
        if self.xkcd:
            with plt.xkcd():
                self.plot()
        else:
            self.plot()

        # save
        self._save_figure(self.save_as)

    def plot(self):
        """main plotting function"""

        # set figure axis
        self._set_figure_axs()

        # sort out color
        if isinstance(self.array, np.ndarray):
            self.array = [self.array]
            if not self.color:
                self.color = sns.color_palette(self.cmap, 1)[0]
            else:
                self.color = [self.color]
        
        # check if alpha's match nr of elements in array
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
                
                # squeeze dimensions
                if el.ndim > 1:
                    el = el.squeeze()

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
                if not isinstance(self.xx, (np.ndarray,list,range, pd.DataFrame, pd.Series)):
                    x = np.arange(0, len(el))
                else:
                    # range has no copy attribute
                    if isinstance(self.xx, range):
                        x = self.xx
                    elif isinstance(self.xx, (pd.DataFrame,pd.Series)):
                        x = self.xx.values
                    else:
                        x = self.xx.copy()

                if isinstance(self.labels, (list,np.ndarray)):
                    lbl = self.labels[idx]
                else:
                    lbl = None

                # plot
                self.axs.plot(
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
                if isinstance(self.error, (int,float,list,np.ndarray)):
                    self._set_shaded_error(
                        x=x,
                        ax=self.axs,
                        tc=el,
                        yerr=self.error,
                        color=self.color_list[idx],
                        alpha=self.error_alpha
                    )

        # axis labels and titles
        self._set_legend_labels(self.axs, labels=self.labels)

        # set x-label
        self._set_xlabel(self.axs, self.x_label)

        # set x-label
        self._set_ylabel(self.axs, self.y_label)

        # set title
        self._set_title(self.axs, self.title)
        
        # set tick params
        self._set_tick_params(self.axs)
    
        # set spine widths
        self._set_spine_width(self.axs)

        # give priority to specify x-lims rather than seaborn's xlim
        if not self.x_lim:
            if isinstance(self.xlim_left, (float,int)):
                self.axs.set_xlim(left=self.xlim_left)
            else:
                self.axs.set_xlim(left=x[0])

            if self.xlim_right:
                self.axs.set_xlim(right=self.xlim_right)
            else:
                self.axs.set_xlim(right=x[-1]) 

        else:
            self.axs.set_xlim(self.x_lim)

        if not self.y_lim:
            if isinstance(self.ylim_bottom, (float,int)):
                self.axs.set_ylim(bottom=self.ylim_bottom)
            
            if self.ylim_top:
                self.axs.set_ylim(top=self.ylim_top)
        else:
            self.axs.set_ylim(self.y_lim)      

        # set ticks
        self._set_xticks(self.axs, self.x_ticks)
        self._set_yticks(self.axs, self.y_ticks)
        
        # draw horizontal/vertical lines with ax?line
        self._add_line(ax=self.axs)

        # set tickers & despine
        self._set_y_ticker(self.axs, self.y_dec)
        self._set_x_ticker(self.axs, self.x_dec)
        self._despine(self.axs)

class LazyCorr(Defaults):
    """LazyCorr

    Wrapper around seaborn's regplot. Plot data and a linear regression model fit. In addition to creating the plot, you can also run a regression or correlation using pingouin by setting the corresponding argument to `True`.

    Parameters
    ----------
    x: np.ndarray, list
        First variable to include in regression
    y: np.ndarray, list
        Second variable to include in regression
    color: str, list, optional
        String representing a color, by default "#ccccccc"
    figsize: tuple, optional
        Figure dimensions as per usual matplotlib conventions, by default (8,8)
    axs: <AxesSubplot:>, optional
        Matplotlib axis to store the figure on
    correlation: bool, optional
        Run a correlation between `x` and `y`. The result is stored in `self.correlation_result`
    regression: bool, optional
        Run a regression between `x` and `y`. The result is stored in `self.regression_result`
    scatter_kwargs: dict, optional
        Additional options passed on to the `scatter` function from matplotlib
    stat_kwargs: dict, optional
        Options passed on to pingouin's stats functions

    Example
    ----------
    >>> from linescanning import plotting
    >>> import matplotlib.pyplot as plt
    
    >>> # vanilla version; here, the regression fit has the same color as the dots.
    >>> fig,axs = plt.subplots(figsize=(7,7))
    >>> plotting.LazyCorr(
    >>>     x_data, 
    >>>     y_data, 
    >>>     axs=axs,
    >>>     x_label="add xlabel",
    >>>     y_label="add ylabel")

    >>> # more exotic version: color each dot differently
    >>> from linescanning import utils
    >>> #
    >>> fig,axs = plt.subplots(figsize=(7,7))
    >>> #
    >>> # create color map between red and blue; return as list
    >>> colors = utils.make_between_cm(["r","b], as_list=True, N=len(y_data))
    >>> for ix,val in enumerate(y_data):
    >>>     axs.plot(x_data[ix], val, 'o', color=colors[ix], alpha=0.6)   
    >>> #
    >>> #add the regression fit 
    >>> plotting.LazyCorr(
    >>>     x_data, 
    >>>     y_data, 
    >>>     axs=axs,
    >>>     add_points=False, # turn off points; we've already plotted them
    >>>     x_label="add xlabel",
    >>>     y_label="add ylabel")

    Notes
    ----------
    see documentation of :class:`linescanning.plotting.Defaults()` for formatting options        
    """    

    def __init__(
        self,
        x, 
        y, 
        color: str="#cccccc", 
        figsize: tuple=(7,7),      
        points: bool=True,
        label: str=None,
        scatter_kwargs: dict={},
        stat_kwargs: dict={},
        color_by: Union[list,np.ndarray]=None,
        regression: bool=False,
        correlation: bool=False,
        **kwargs):

        # init default plotter class
        super().__init__(**kwargs)

        self.x              = x
        self.y              = y
        self.color          = color
        self.figsize        = figsize
        self.points         = points
        self.label          = label
        self.scatter_kwargs = scatter_kwargs
        self.stat_kwargs    = stat_kwargs
        self.color_by       = color_by
        self.regression     = regression
        self.correlation    = correlation

        if self.xkcd:
            with plt.xkcd():
                self.plot()
        else:
            self.plot()

        # run quick regression with pingouin
        if self.regression:
            self._run_regression()

        # run quick correlation with pingouin
        if self.correlation:
            self._run_correlation()            

        # save
        self._save_figure(self.save_as)

    def _run_regression(self):
        
        try:
            import pingouin as pg
        except:
            raise ImportError("Could not import pingouin, so this functionality is not available")

        self.regression_result = pg.linear_regression(
            self.x, 
            self.y, 
            **self.stat_kwargs)

    def _run_correlation(self):

        try:
            import pingouin as pg
        except:
            raise ImportError("Could not import pingouin, so this functionality is not available")
        
        # convert to dataframe
        self.data = pd.DataFrame({"x": self.x, "y": self.y})
        self.x = "x"
        self.y = "y"
    
        self.correlation_result = pg.pairwise_corr(
            self.data,
            columns=["x","y"], 
            **self.stat_kwargs)

    def plot(self):

        # set figure axis
        self._set_figure_axs()       

        # c-arguments clashes with "color" argument if you pass it to sns.regplot in "scatter_kws"; hence this solution
        if isinstance(self.color_by, (list, np.ndarray)):
            points = self.axs.scatter(
                self.x,
                self.y, 
                c=self.color_by, 
                **self.scatter_kwargs)
            
            # set colorbar
            self.cbar = plt.colorbar(points)
            if "label" in list(self.scatter_kwargs.keys()):
                self.cbar.set_label(
                    self.scatter_kwargs["label"], 
                    fontsize=self.label_size,
                    fontname=self.fontname)            
            
            # sort out ticks
            self._set_tick_params(self.cbar.ax)
            self._set_spine_width(self.cbar.ax)

            # remove outside edge from colorbar
            self.cbar.ax.set_frame_on(False)

            # set stuff to false/empty for sns.regplot
            self.points = False
            self.scatter_kwargs = {}

        self.kde_color = utils.make_between_cm(self.color,self.color,as_list=True)
        sns.regplot(
            x=self.x, 
            y=self.y, 
            color=self.color, 
            ax=self.axs,
            scatter=self.points,
            label=self.label,
            scatter_kws=self.scatter_kwargs)

        # set labels and titles
        for lbl,func in zip(
            [self.x_label, self.y_label, self.title],
            [self._set_xlabel, self._set_ylabel, self._set_title]):

            if isinstance(lbl, str):
                func(self.axs, lbl)

        # sort out ticks
        self._set_spine_width(self.axs)
        self._set_tick_params(self.axs)

        for lbl,func in zip(
            [self.x_ticks, self.y_ticks],
            [self._set_xticks, self._set_yticks]):

            if isinstance(lbl, list):
                func(self.axs, lbl)

        for lim,func in zip(
            [self.x_lim, self.y_lim], 
            [self._set_xlim, self._set_ylim]):
            if lim:
                func(self.axs, lim)

        # draw horizontal/vertical lines with ax?line
        self._add_line(ax=self.axs)

        # set tickers & despine
        self._set_y_ticker(self.axs, self.y_dec)
        self._set_x_ticker(self.axs, self.x_dec)
        self._despine(self.axs)

class LazyBar():

    """LazyBar

    Wrapper around :func:`seaborn.barplot` to follow the same aesthetics of the other Lazy* functions. It is strongly recommended to use a dataframe for this function to make the formatting somewhat easier, but you can input arrays for `x` and `y`. You can round the edges of the bar using `fancy=True`.

    Parameters
    ----------
    data: pd.DataFrame, optional
        Input dataframe, by default None
    x: str, list, np.ndarray, optional
        Variable for the x-axis, by default None. Can be a column name from `data`, or a list/np.ndarray with labels for input `y`. 
    y: str, list, np.ndarray, optional
        Variable for the y-axis, by default None. Can be a column name from `data`, or a list/np.ndarray. If `x` is not specified, indices from 0 to `y.shape` will be used to construct the input dataframe.
    labels: list, np.ndarray, optional 
        custom labels that can be used when `x` denotes a column name in dataframe `data`. The replacing labels should have the same length as the labels that are being overwritten.
    axs: <AxesSubplot:>, optional
        Subplot axis to put the plot on, by default None
    add_points: bool, optional
        Add the actual datapoints rather than just the bars, by default False. Though default is `False`
    points_color: str, tuple, optional
        Color of the points if you do not have nested categories, by default None
    points_palette: list, sns.palettes._ColorPalette, optional
        Color palette for the points if you have nested categories (e.g., multiple variables per subject so you can color the individual subjects' data points), by default None
    points_cmap: str, optional
        Color map for the points if you did not specify `points_palette`, by default "viridis"
    points_legend: bool, optional
        Add legend of the data points (if you have nested categories), by default False. The functionality of these interchangeable legends (`bar_legend` and `points_legend`) is quite tricky, so user discretion is advised.
    points_alpha: float, optional
        Alpha of the points, by default 1. Sometimes useful to adjust if you have LOADS of data points
    error: str, optional
        Type of error bar to use for the bar, by default "sem". Can be "sem" or "std". Internally, we'll check if there's enough samples to calculate errors from, otherwise `error` will be set to `None`
    fancy: bool, optional
        Flag to round the edges of the bars, by default False. By default, the rounding is scaled by the min/max of the plot, regardless whether `lim` was specified. This ensures equal rounding across inputs. The other `fancy`-arguments below are a bit vague, so leaving them default will ensure nice rounding of the bars
    fancy_rounding: float, optional
        Amount of rounding, by default 0.15
    fancy_pad: float, optional
        Vague variable, by default -0.004
    fancy_aspect: float, optional
        Vague variable, by default None. If None, the rounding is scaled by the min/max of the plot, regardless whether `lim` was specified.
    fancy_denom: int, optional
        Scaling factor for `fancy_aspect`, by default 4 (which works well for data where the max value is ~50). Use higher values (e.g., 6) if your data range is large
    bar_legend: bool, optional
        Legend for the bars, rather than points, by default False. The functionality of these interchangeable legends (`bar_legend` and `points_legend`) is quite tricky, so user discretion is advised.
    strip_kw, dict, optional
        Additional kwargs passed on to seaborn's stripplot. Several factors are being set via regular arguments in the function, such as `dodge`, `palette`, `color`, and `hue`.

    Example
    ----------
    >>> # this figure size works well for plots with 2 bars
    >>> fig,axs = plt.subplots(figsize=(2,8))
    >>> plotting.LazyBar(
    >>>     data=df_wm,
    >>>     x="group",
    >>>     y="t1",
    >>>     sns_ori="v",
    >>>     axs=axs,
    >>>     add_labels=True,
    >>>     palette=[con_color,mdd_color],
    >>>     add_points=True,
    >>>     points_color="k",
    >>>     trim_bottom=True,   
    >>>     sns_offset=4,
    >>>     y_label2="white matter T1 (ms)",
    >>>     lim=[800,1600],
    >>>     fancy=True,
    >>>     fancy_denom=6)

    Notes
    ----------
    see documentation of :class:`linescanning.plotting.Defaults()` for formatting options
    """
    
    def __init__(
        self, 
        data: pd.DataFrame=None,
        x: Union[str,np.ndarray]=None, 
        y: Union[str,np.ndarray]=None, 
        labels: list=None,
        palette: Union[list,sns.palettes._ColorPalette]=None,
        cmap: str="inferno",
        hue: str=None,
        figsize=(4,7),
        add_labels: bool=False,
        lim: list=None,
        ticks: list=None,
        add_points: bool=False,
        points_color: Union[str,tuple]=None,
        points_palette: Union[list,sns.palettes._ColorPalette]=None,
        points_cmap: str="viridis",
        points_legend: bool=False,
        points_alpha: float=1,
        error: str="se",
        fancy: bool=False,
        fancy_rounding: float=0.15,
        fancy_pad: float=-0.004,
        fancy_aspect: float=None,
        fancy_denom: int=4,
        bar_legend: bool=False,
        strip_kw: dict={},
        **kwargs):

        self.data               = data
        self.x                  = x
        self.y                  = y
        self.hue                = hue
        self.labels             = labels
        self.palette            = palette
        self.cmap               = cmap
        self.add_labels         = add_labels
        self.lim                = lim
        self.ticks              = ticks
        self.bar_legend         = bar_legend
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
        self.figsize            = figsize
        self.strip_kw           = strip_kw
        self.kw_defaults = Defaults()

        # avoid that these kwargs are passed down to matplotlib.bar.. Throws errors
        ignore_kwargs = [
            "trim_left",
            "trim_bottom",
            "points_hue",
            "points_alpha",
            "bbox_to_anchor",
            "fancy",
            "fancy_rounding",
            "fancy_pad",
            "fancy_aspect",
            "fancy_denom",
            "font_name",
            "bar_legend",
            "labels"
            "strip_kw",
            "fontname"
        ]

        kw_sns = {}
        for ii in kwargs:
            # filter out non-ls kwargs
            if ii not in self.kw_defaults.ls_kwargs+ignore_kwargs:
                kw_sns[ii] = kwargs[ii]
            else:
                # overwrite ls-kwargs
                if ii in self.kw_defaults.ls_kwargs:
                    if not getattr(self.kw_defaults, ii) == kwargs[ii]:
                        setattr(self.kw_defaults, ii, kwargs[ii])

        self.__dict__.update(**self.kw_defaults.__dict__)
        self.__dict__.update(**kwargs)
        self.kw_defaults.update_rc(self.fontname)

        if not hasattr(self, "bbox_to_anchor"):
            self.bbox_to_anchor = None

        if self.xkcd:
            with plt.xkcd():
                self.plot(**kw_sns)
        else:
            self.plot(**kw_sns)
        
        # save
        self.kw_defaults._save_figure(self.save_as)

    def plot(self, **kw_sns):

        # set figure axis
        self.kw_defaults._set_figure_axs(figsize=self.figsize)   

        # construct dataframe from loose inputs
        if isinstance(self.y, (np.ndarray,list)):
            if isinstance(self.y, list):
                self.y = np.array(self.y)
                
            if not isinstance(self.x, (np.ndarray, list)):
                self.x = np.arange(0,self.y.shape[0])

            self.data = pd.DataFrame({"x": self.x, "y": self.y})
            self.x = "x"
            self.y = "y"
        
        # check if we should reset the index of dataframe
        try:
            self.data = self.data.reset_index()
        except:
            pass

        # check if we got custom labels
        if isinstance(self.labels, (np.ndarray,list)):
            self.data[self.x] = self.labels

        if self.sns_ori == "h":
            xx = self.y
            yy = self.x
            self.trim_bottom = False
            self.trim_left   = True
        elif self.sns_ori == "v":
            xx = self.x 
            yy = self.y
            self.trim_bottom = True
            self.trim_left   = False            
        else:
            raise ValueError(f"sns_ori must be 'v' or 'h', not '{self.sns_ori}'")
        
        if isinstance(self.color, (str,tuple,list)):
            if isinstance(self.color, (str,tuple)):
                self.palette = None
                self.cmap = None
            elif isinstance(self.color, list):
                self.palette = sns.color_palette(palette=self.color)
                self.color = None
        else:
            self.color = None
            if isinstance(self.palette, list):
                self.palette = sns.color_palette(palette=self.palette)

            if not isinstance(self.palette, sns.palettes._ColorPalette):
                # self.palette = sns.color_palette(self.cmap, self.data.shape[0])
                self.palette = self.cmap
        
        self.ff = sns.barplot(
            data=self.data,
            x=xx, 
            y=yy, 
            ax=self.axs, 
            orient=self.sns_ori,
            errorbar=self.error,
            hue=self.hue,
            **dict(
                kw_sns,
                color=self.color,
                palette=self.palette
            ))

        multi_strip = False
        if self.add_points:

            if not hasattr(self, "points_hue"):
                self.points_hue = None
            
            if not self.points_palette:
                self.points_palette = self.points_cmap

            # give priority to given points_color
            if isinstance(self.points_color, (str,tuple)):
                self.points_palette = None
                self.points_hue = None

            
            if isinstance(self.hue, str):

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
                                dodge=False, 
                                palette=[color] * 2,
                                ax=self.ff,
                                **self.strip_kw
                            )
                else:
                    multi_strip = True
                    sns.stripplot(
                        data=self.data, 
                        x=xx, 
                        y=yy, 
                        hue=self.hue,
                        dodge=True, 
                        ax=self.ff,
                        color=self.points_color,
                        palette=self.points_palette,
                        alpha=self.points_alpha,
                        **self.strip_kw
                    )                                
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
                    alpha=self.points_alpha,
                    **self.strip_kw
                )

        # sort out legend
        if self.bar_legend or self.points_legend:
            
            self.add_legend = True

            # filter out handles that correspond to labels
            self.legend_kw = {}
            for key,val in zip(
                ["fontsize","handletextpad","frameon"],
                [self.label_size,self.legend_handletext,False]):
                self.legend_kw[key] = val

            if isinstance(self.bbox_to_anchor, tuple):
                self.legend_kw["bbox_to_anchor"] = self.bbox_to_anchor

            # get handles
            handles,labels = self.ff.get_legend_handles_labels()
            
            # bar legend
            if self.bar_legend:    

                # do some more exotic stuff to disentangle coloring from bars and hue
                if isinstance(self.hue, str):

                    # find categorical handles
                    handles,labels = self.ff.get_legend_handles_labels()
                    # find indices of categorical handles in list
                    cc = self.data[self.hue].values
                    indexes = np.unique(cc, return_index=True)[1]
                    cond = [cc[index] for index in sorted(indexes)]
                    
                    if multi_strip:
                        handles = handles[-len(cond):]
                        labels = labels[-len(cond):]                       

            else:
                if not self.add_points:
                    self.add_legend = False

        else:
            self.add_legend = False

        # fill in legend
        if self.add_legend:
            self.ff.legend(
                handles,
                labels,
                **self.legend_kw
            )
        else:
            self.ff.legend([],[], frameon=False)

        # set tick params
        self.kw_defaults._set_tick_params(self.ff)

        # set spine widths
        self.kw_defaults._set_spine_width(self.ff)

        if not self.add_labels:
            if self.sns_ori == 'h':
                self.ff.set_yticks([])
            elif self.sns_ori == "v":                
                self.ff.set_xticks([])
            else:
                raise ValueError(f"sns_ori must be 'v' or 'h', not '{self.sns_ori}'")
        elif isinstance(self.add_labels,list):
            self.kw_defaults._set_xlabel(self.ff, self.add_labels)

        if isinstance(self.sns_rot, (int,float)):
            if self.sns_ori == 'h':
                self.ff.set_yticklabels(
                    self.ff.get_yticklabels(), 
                    rotation=self.sns_rot,
                    fontname=self.fontname)
            elif self.sns_ori == "v":
                self.ff.set_xticklabels(
                    self.ff.get_xticklabels(), 
                    rotation=self.sns_rot,
                    fontname=self.fontname)
            else:
                raise ValueError(f"sns_ori must be 'v' or 'h', not '{self.sns_ori}'")

        # set limits depending on orientation
        self.kw_defaults._set_bar_lim(self.ff, self.lim)

        # set ticks depending on orientation
        self.kw_defaults._set_bar_ticks(self.ff, self.ticks)

        # from: https://stackoverflow.com/a/61569240
        if self.fancy:
            new_patches = []

            for patch in reversed(self.ff.patches):
                bb = patch.get_bbox()
                color = patch.get_facecolor()

                # max of axis divided by 4 gives nice rounding
                if not isinstance(self.fancy_aspect, (int,float)):
                    if self.sns_ori == "v":
                        y_limiter = patch._axes.get_ylim()[-1]
                        if isinstance(self.lim, list):
                            y_limiter-=self.lim[0]
                        
                        self.fancy_aspect = y_limiter/self.fancy_denom
                    else:
                        x_limiter = patch._axes.get_xlim()[-1]
                        if isinstance(self.lim, list):
                            x_limiter-=self.lim[0]

                        self.fancy_aspect = x_limiter/self.fancy_denom

                
                # make rounding at limit
                if isinstance(self.lim, list):
                    if self.sns_ori == "v":
                        ymin = self.lim[0]
                        xmin = bb.xmin
                        height = bb.height - ymin
                        width = bb.width
                    else:
                        xmin = self.lim[0]
                        ymin = bb.ymin
                        width = bb.width - xmin
                        height = bb.height
                else:
                    xmin = bb.xmin
                    ymin = bb.ymin
                    height = bb.height
                    width = bb.width

                p_bbox = patches.FancyBboxPatch(
                    (xmin, ymin),
                    abs(width), abs(height),
                    boxstyle=f"round,pad={self.fancy_pad},rounding_size={self.fancy_rounding}",
                    ec="none", 
                    fc=color,
                    mutation_aspect=self.fancy_aspect
                )

                patch.remove()
                new_patches.append(p_bbox)

            for patch in new_patches:
                self.ff.add_patch(patch)

        # set xlabel to none of nothing is specified
        if isinstance(self.x, str) and not isinstance(self.x_label, str):
            self.ff.set(xlabel=None)

        if isinstance(self.y, str) and not isinstance(self.y_label, str):
            self.ff.set(ylabel=None)            

        self.kw_defaults._set_xlabel(self.ff, self.x_label)
        self.kw_defaults._set_ylabel(self.ff, self.y_label)

        # set these explicitly; remove left axis is orientation = horizontal | remove bottom axis if orientation is vertical
        if hasattr(self, "trim_left"):
            trim_left = self.trim_left
        else:
            trim_left = False

        if hasattr(self, "trim_bottom"):
            trim_bottom = self.trim_bottom
        else:
            trim_bottom = False

        # draw horizontal/vertical lines with ax?line
        self.kw_defaults._add_line(ax=self.ff)

        # set tickers & despine
        self.kw_defaults._set_y_ticker(self.ff, self.y_dec)
        self.kw_defaults._set_x_ticker(self.ff, self.x_dec)
        self.kw_defaults._despine(
            self.ff,
            left=trim_left,
            bottom=trim_bottom
        )

        # set title
        self.kw_defaults._set_title(self.ff, self.title)


class LazyHist(Defaults):
    """LazyHist

    Wrapper around seaborn's histogram plotter

    Parameters
    ----------
    data: numpy.ndarray
        Input data for histogram
    kde: bool, optional
        Add kernel density plot to histogram with seaborn (https://seaborn.pydata.org/generated/seaborn.kdeplot.html). Default is False
    hist: bool, optional
        Add histogram to plot. Default is True
    fill: bool, optional
        Fill the area below the kde plot. Default is False
    bins: str, optional
        Set bins for histogram; default = "auto"
    kde_kwargs: dict, optional
        Additional arguments passed on the seaborn's `kde_plot`
    hist_kwargs: dict, optional
        Additional arguments passed on to matplotlib's `hist` fuction

    Returns
    ----------
    matplotlib.pyplot plot

    Example
    ----------
    >>> from linescanning import plotting
    >>> import matplotlib.pyplot as plt
    >>> fig,axs = plt.subplots(figsize=(7,7))
    >>> plotting.LazyHist(
    >>>     y_data,
    >>>     axs=axs,
    >>>     kde=True,
    >>>     hist=True,
    >>>     fill=False,
    >>>     y_label2="add y_label",
    >>>     x_label2="add x_label",
    >>>     hist_kwargs={"alpha": 0.4},
    >>>     kde_kwargs={"linewidth": 4}
    >>> )

    Notes
    ----------
    see documentation of :class:`linescanning.plotting.Defaults()` for formatting options
    """

    def __init__(
        self, 
        data, 
        x=None,
        y=None,
        figsize=(7,7),
        kde=False,
        hist=True,
        bins="auto",
        fill=False,
        kde_kwargs={},
        hist_kwargs={},
        color="#cccccc",
        fancy: bool=False,
        fancy_rounding: float=0.15,
        fancy_pad: float=-0.004,
        fancy_aspect: float=None,
        **kwargs):
        
        super().__init__()
        self.__dict__.update(kwargs)
        self.update_rc(self.fontname)

        # read regular arguments
        self.data           = data
        self.x              = x
        self.y              = y
        self.figsize        = figsize
        self.kde            = kde
        self.kde_kwargs     = kde_kwargs
        self.hist_kwargs    = hist_kwargs
        self.hist           = hist
        self.bins           = bins
        self.fill           = fill
        self.color          = color
        self.kwargs         = kwargs
        self.fancy          = fancy
        self.fancy_rounding = fancy_rounding
        self.fancy_pad      = fancy_pad
        self.fancy_aspect   = fancy_aspect        
        # self.__dict__.update(self.kde_kwargs)

        if self.xkcd:
            with plt.xkcd():
                self.plot()
        else:
            self.plot()

        if self.kde:
            self.kde_ = self.return_kde()

        # save
        self._save_figure(self.save_as)

    def return_kde(self):
        return self.ff.get_lines()[0].get_data()
    
    def force_kde_color(self):
        if "color" in list(self.kde_kwargs.keys()):
            color = self.kde_kwargs["color"]
        else:
            color = self.color
        self.ff.get_lines()[0].set_color(color)

    def plot(self):

        # set figure axis
        self._set_figure_axs()   

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
            
            # turn off legend by default
            if not "legend" in list(self.kde_kwargs):
                self.kde_kwargs["legend"] = False

            self.ff = sns.kdeplot(
                data=self.data,
                x=self.x,
                y=self.y,
                ax=self.axs,
                fill=self.fill,
                **self.kde_kwargs
            )

            # the color argument is very unstable for some reason..
            self.force_kde_color()

        # there's no self.ff if kde=False
        if hasattr(self, "ff"):
            self.active_axs = self.ff
        else:
            self.active_axs = self.axs

        # set titles
        self._set_title(self.active_axs, self.title)
        
        # set tick params/axis width
        self._set_tick_params(self.active_axs)
        self._set_spine_width(self.active_axs)

        # set limits
        self._set_xlim(self.active_axs, self.x_lim)
        self._set_ylim(self.active_axs, self.y_lim)
        
        # set ticks
        self._set_xticks(self.active_axs, self.x_ticks)
        self._set_yticks(self.active_axs, self.y_ticks)        

        # set axis labels
        if not isinstance(self.x_label, str):
            self.active_axs.set(xlabel=None)

        if  not isinstance(self.y_label, str):
            self.active_axs.set(ylabel=None)            

        self._set_xlabel(self.active_axs, self.x_label)
        self._set_ylabel(self.active_axs, self.y_label)

        if hasattr(self, "trim_left"):
            trim_left = self.trim_left
        else:
            trim_left = False

        if "trim_bottom" in list(self.kwargs.keys()):
            trim_bottom = self.kwargs["trim_bottom"]
        else:
            trim_bottom = False
        
        self._despine(
            self.active_axs,
            left=trim_left,
            bottom=trim_bottom
        )

        # set title
        self._set_title(self.active_axs, self.title)

def conform_ax_to_obj(
    ax,
    obj=None,
    **lazy_args):

    """conform_ax_to_obj

    Function to conform any plot to the aesthetics of this plotting module. Can be used when a plot is created with functions other than :class:`linescanning.plotting.LazyPlot`, :class:`linescanning.plotting.LazyCorr`, :class:`linescanning.plotting.LazyHist`, or any other function specified in this file. Assumes `ax` is a `matplotlib.axes._subplots.AxesSubplot` object, and `obj` a `linescanning.plotting.Lazy*`-object.

    Parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot
        input axis that needs to be modified
    obj: linescanning.plotting.Lazy*
        linecanning-specified plotting object containing the information with which `ax` will be conformed
    **lazy_args: dict, optional
        other elements defined in :class:`linescanning.plotting.Defaults`, such as `font_size`, `label_size`, or `axis_width`. Overwrites elements in `obj`, if passed

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
    
    # check if Lazy* object was passed on
    if obj == None:
        obj = Defaults()

    # update with lazy_args | do separate
    if len(lazy_args) > 0:
        for key,val in lazy_args.items():
            setattr(obj,key,val)

    # try to read some stuff from the passed axis
    if not isinstance(obj.title, str):
        obj.title = ax.get_title()

    if not isinstance(obj.y_label, str):
        obj.y_label = ax.get_ylabel()

    if not isinstance(obj.x_label, str):
        obj.x_label = ax.get_xlabel()           

    # format labels and titles
    for lbl,func in zip(
        [obj.x_label, obj.y_label, obj.title],
        [obj._set_xlabel, obj._set_ylabel, obj._set_title]):

        if isinstance(lbl, str):
            func(ax, lbl)

    # format ticks
    obj._set_spine_width(ax)
    obj._set_tick_params(ax)

    for lbl,func in zip(
        [obj.x_ticks, obj.y_ticks],
        [obj._set_xticks, obj._set_yticks]):

        if isinstance(lbl, list):
            func(ax, lbl)

    # format limits
    for lim,func in zip(
        [obj.x_lim, obj.y_lim], 
        [obj._set_xlim, obj._set_ylim]):
        if isinstance(lim, (float,int,list)):
            func(ax, lim)

    # draw horizontal/vertical lines with ax?line
    obj._add_line(ax=ax)

    # set tickers & despine
    obj._set_y_ticker(ax, obj.y_dec)
    obj._set_x_ticker(ax, obj.x_dec)
    obj._despine(ax)

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
        cm_nr=5,
        cm_decimal=3,
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
        self.cm_nr = cm_nr
        self.cm_decimal = cm_decimal

        super().__init__()
        self.__dict__.update(kwargs)
        self.update_rc(self.fontname)

        if self.axs == None:
            if isinstance(self.save_as, str):
                self.fig, self.axs = plt.subplots(figsize=self.figsize)
            else:
                self.fig, self.axs = plt.subplots(figsize=self.figsize)
            
        # make colorbase instance
        if isinstance(self.cmap, str):
            self.cmap = mpl.cm.get_cmap(self.cmap, 256)

        # decide ticks
        if not isinstance(self.ticks, (np.ndarray,list)):
            self.ticks = self.colormap_ticks(
                vmin=self.vmin,
                vmax=self.vmax,
                key=self.txt,
                dec=self.cm_decimal,
                nr=self.cm_nr
            )   

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
        self._set_tick_params(self.axs)   

        # turn off frame
        self.axs.set_frame_on(False)

        # save
        self._save_figure(self.save_as)

    @staticmethod
    def colormap_ticks(
        vmin=None, 
        vmax=None, 
        key=None,
        dec=3, 
        nr=5):

        # store colormaps
        if isinstance(key, str):
            if key == "polar" or key == "polar angle" or "polar" in key:
                ticks = [-np.pi,0,np.pi]
            else:
                ticks = list(np.linspace(vmin,vmax, endpoint=True, num=nr))
        else:
            ticks = list(np.linspace(vmin,vmax, endpoint=True, num=nr))
        
        # round ticks
        ticks = [round(ii,dec) for ii in ticks]

        # check if minimum of ticks > minimum of data
        if ticks[0] < vmin:
            ticks[0] = utils.round_decimals_up(vmin, dec)

        # check if maximum of ticks < maximum of data
        if ticks[-1] > vmax:
            ticks[-1] = utils.round_decimals_down(vmax, dec)  

        return ticks
    
    def show(self):
        
        fig = plt.figure()
        new_manager = fig.canvas.manager
        new_manager.canvas.figure = self.fig
        self.fig.set_canvas(new_manager.canvas)

def fig_annot(
    fig, 
    axs=None, 
    y=1.01, 
    x0_corr=0, 
    x_corr=-0.09, 
    fontsize=28,
    **kwargs):

    # get figure letters
    alphabet = list(string.ascii_uppercase)

    if isinstance(axs, list):
        ax_list = axs
    else:
        ax_list = fig.axes
        
    # make annotations
    for ix,ax in enumerate(ax_list):
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        if ix == 0:
            move_frac = x0_corr/bbox.width
        else:
            move_frac = x_corr/bbox.width

        pos = move_frac

        if isinstance(y, list):
            if len(y) != len(ax_list):
                raise ValueError(f"List with y-values must match list with axes. y contains {len(y)} elements, while {len(ax_list)} axes are specified")
            y_pos = y[ix]
        else:
            y_pos = y

        ax.annotate(
            alphabet[ix], 
            (pos,y_pos), 
            fontsize=fontsize, 
            xycoords="axes fraction",
            **kwargs)
