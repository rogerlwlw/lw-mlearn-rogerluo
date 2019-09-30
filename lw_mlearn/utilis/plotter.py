# -*- coding: utf-8 -*-
"""plot_mode

Description
---------------
data visualization module @author: roger

Contain functions
---------------
1. plotter method  
"""
import numpy as np
import pandas as pd
import inspect
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from collections import OrderedDict

from functools import reduce
from seaborn.categorical import *
from pandas.core.dtypes import api
from scipy import interp
from sklearn.metrics import auc, roc_curve

from lw_mlearn.utilis.utilis import get_flat_list, get_kwargs, dict_diff
from lw_mlearn.utilis.docstring import Appender

from scipy.interpolate import interp1d

## import mlens visualization tools 
#from mlens.visualization import (
#        pca_comp_plot, pca_plot, corr_X_y, corrmat, clustered_corrmap, 
#        exp_var_plot
#        )

plt.style.use('seaborn')

plt.rcParams.update({
    'figure.dpi': 120.0,
    'axes.unicode_minus':
    False,
    'font.family': ['sans-serif'],
    'font.sans-serif': [
        'Microsoft JhengHei', # ch  
        'STSong', # ch
        'STXihei', # ch   
        'sans-serif',
        'Arial',
        'Liberation Sans',
        'DejaVu Sans',
        'Bitstream Vera Sans',
        'sans-serif',
        'DFKai-SB',
        'STKaiti',
    ],
    'font.size':
    10.0,
    'font.style':
    'normal',
    'font.weight':
    'semibold',
})


def txt_fontdict(**kwargs):
    '''
    '''
    font = {
        'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 12,
    }
    font.update(**kwargs)
    print(font)
    return font

#def _get_plot_fn(kind):
#    '''return plot function
#    '''
#    if callable(kind):
#        return kind
#    
#    plot_fn = dict(
#            dist=sns.distplot,
#            hist=plt.hist,            
#            )
#    fn = plot_fn.get(kind)
#    if kind is None:
#        return plot_fn.keys()
#    
#    if fn is None:
#        raise ValueError("invalid input for 'kind'")
#    else:        
#        return fn

def _get_snsplot(kind=None):
    '''return plot functions in seaborn catplot module
    '''
    if callable(kind):
        return kind
    lst = inspect.getmembers(sns, inspect.isfunction) 
    fn_dict = dict([i for i in lst if i[0].count('plot')>0])
    if kind is None:
        return fn_dict.keys()
    # Determine the plotting function
    try:
        plot_func = fn_dict[kind]
    except KeyError:
        err = "Plot kind '{}' is not recognized".format(kind)
        raise ValueError(err)
    return plot_func

@Appender(sns.FacetGrid.__doc__, join='\nsee Facetgrid\n')
def plotter_facet(data,  plot_args, subset=None, kind='distplot', 
                  savefig=None, **kwargs):
    '''plot grids of plots using seaborn Facetgrid ::
    
    parameter
    -----
    data : DataFrame
    
        Tidy (“long-form”) dataframe where each column is a variable and each 
        row is an observation.

    subset (dict):
        fitler subset of data by column's categorical values
        eg: {col1 : [str1, str2, ...], ...}
        
    kind:
        callable plot fn or str to call plot api in _get_plot_fn
        
    plot_args (tuple):
        (colname2 as x, colname2 as y ) indexed by DataFrame   
        
    row, col, hue : strings
    
        Variables that define subsets of the data, which will be drawn on 
        separate facets in the grid. See the *_order parameters to control 
        the order of levels of this variable.
    
    col_wrap : int, optional
    
        “Wrap” the column variable at this width, so that the column facets
        span multiple rows. Incompatible with a row facet.
    
    share{x,y} : bool, ‘col’, or ‘row’ optional
    
        If true, the facets will share y axes across columns and/or x axes 
        across rows.
    
    height : scalar, optional
    
        Height (in inches) of each facet. See also: aspect.
    
    aspect : scalar, optional
    
        Aspect ratio of each facet, so that aspect * height gives the width of 
        each facet in inches.
    
    palette : palette name, list, or dict, optional
    
        Colors to use for the different levels of the hue variable. 
        Should be something that can be interpreted by color_palette(), or a dictionary mapping hue levels to matplotlib colors.
    
    {row,col,hue}_order : lists, optional
    
        Order for the levels of the faceting variables. By default, 
        this will be the order that the levels appear in data or, if the variables are pandas categoricals, the category order.

    '''
    if subset is not None:
        data = filter_subset(data, subset)
        
    fn_plot = _get_snsplot(kind)
    # get facet kwds
    facet_kws = get_kwargs(sns.FacetGrid, **kwargs)
    # get fn kwds
    plot_fn_kws = get_kwargs(fn_plot, **kwargs)
    # get other than kwds
    ax_kws = dict_diff(kwargs, facet_kws.keys() | plot_fn_kws.keys())
    # generate grid
    g = sns.FacetGrid(data, **facet_kws)
    # map plot function
    g.map(fn_plot, *plot_args, **plot_fn_kws)
    
    if len(ax_kws) > 0 :
        g.set(**ax_kws)  
    
    g.add_legend()  
    
    if savefig:
        _save_fig(g, savefig)    
    return g

@Appender(sns.catplot.__doc__, join='\nsee catplot\n')
@Appender(sns.violinplot.__doc__, join='\nsee violinplot\n')
def plotter_catplot(data, kind='violin', swarm=False, hline=None,
                    subset=None, **kwargs):
    '''make a distr plot through catplot function ::    

    parameter
    ---------
    data (DataFrame):
        data to generate violin plot through seaborn
    kind (str): 'violin' default
        ['violin', 'swarm', 'box', 'bar', 'count'], see 
    swarm (bool):
        whether to combine a swarmplot, default False
    hline (int):
        add a horizontal base line 
    subset (dict):
        fitler subset of data by column's categorical values
    kwargs:
        other keywords to customize ax and to pass to plot functions
    
    return
    --------    
        g : FacetGrid
            Returns the FacetGrid object with the plot on it for further 
            tweaking.
    '''
    if subset is not None:
        data = filter_subset(data, subset)
        
    # get plot function key words
    fn_kws = dict(
        violin = get_kwargs(sns.violinplot, **kwargs),
        box = get_kwargs(sns.boxplot, **kwargs),
        swarm = get_kwargs(sns.swarmplot, **kwargs),
        bar = get_kwargs(sns.barplot, **kwargs),
        count = get_kwargs(sns.countplot, **kwargs),
        cat = get_kwargs(sns.catplot, **kwargs),
        point = get_kwargs(sns.pointplot, **kwargs),
        factor = get_kwargs(sns.factorplot, **kwargs),
    )
    plot_fn_kws = fn_kws.get(kind) 
    plot_fn_kws.update(fn_kws.get('cat'))
            
    
    if hline is not None:
        plot_fn_kws.update(legend_out=False)
    # plot categorical data   
    g = sns.catplot(data=data, kind=kind, **plot_fn_kws)
    
    if swarm:
        g.map(sns.swarmplot, data=data, ax=g.ax, x=kwargs.get('x'),
              y=kwargs.get('y'), size=2.5, color='k', alpha=0.3)
    if hline is not None:
        g.map(plt.axhline, y=hline, color='red', linestyle='--', 
              label='baseline%s'%hline)
        g._legend_out = True
        g.add_legend()  
    
    ax_kws = dict_diff(kwargs, plot_fn_kws.keys())       
    if 'savefig' in ax_kws:
        ax_kws.pop('savefig')
    if len(ax_kws) > 0 :
        g.set(**ax_kws)
    # save fig to savefig path    
    if kwargs.get('savefig') is not None:
        _save_fig(g, kwargs['savefig'])
    return g

def plotter_auc(fpr,
                tpr,
                ax=None,
                alpha=0.95,
                lw=1.2,
                curve_label=None,
                title=None,
                cm=None):
    '''plot roc_auc curve given fpr, tpr, or list of fpr, tpr
    
    cm:
        color map default 'tab20'
    
    return
    ----
    ax
    '''
    fpr, tpr = get_flat_list(fpr), get_flat_list(tpr)
    if len(fpr) != len(tpr):
        raise ValueError("length of fpr and tpr doesn't match")
    n = len(fpr)
    names = range(n) if curve_label is None else get_flat_list(curve_label)
    if len(names) != n:
        print('n_curve label not match with n_fpr or n_tpr')
        names = range(n)

    # -- plot each line
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    aucs = []
    kss = []
    if cm is None:
        cm = plt.get_cmap('tab20')
    cmlist = [cm(i) for i in np.linspace(0, 1, n)]
    for i in range(n):
        if len(fpr[i]) != len(tpr[i]):
            print("length of {}th fpr and tpr doesn't match".format(i))
            continue
        else:
            auc_score = auc(fpr[i], tpr[i])
            ks_score = max(np.array(tpr[i]) - np.array(fpr[i]))
            aucs.append(auc_score)
            kss.append(ks_score)
            ax.plot(fpr[i],
                    tpr[i],
                    color=cmlist[i],
                    alpha=alpha,
                    lw=lw,
                    label='ROC %r (AUC=%0.2f;KS=%0.2f)' %
                    (names[i], auc_score, ks_score))
    # plot mean tpr line
    if n > 1:
        mean_fpr = np.linspace(0, 1, 100)
        tprs = [interp(mean_fpr, x, y) for x, y in zip(fpr, tpr)]
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr,
                mean_tpr,
                'b-.',
                alpha=1,
                lw=1.5,
                label='Mean ROC(AUC=%0.2f $\pm$ %0.2f)' % (mean_auc, std_auc))
        #plot variance
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr,
                        tprs_lower,
                        tprs_upper,
                        color='grey',
                        alpha=.3,
                        label=r'$\pm$ 1 std. dev.')
    # plot chance line
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance (AUC=0.5)')
    # set property
    if title is None:
        title = 'Receiver operating characteristic'
    plt.setp(ax,
             xlabel='False Positive Rate',
             ylabel='True Positive Rate',
             xlim=[-0.05, 1.05],
             ylim=[-0.05, 1.05],
             title=title)
    plt.legend(loc="lower right", fontsize='medium', bbox_to_anchor=(1, 0))
    plt.tight_layout()
    return ax


def plotter_auc_y(y_pre, y_true, **kwargs):
    '''plot roc_auc curve given y_pre, y_true
    '''
    fpr, tpr, threshhold = roc_curve(y_true, y_pre,
                                     **get_kwargs(roc_curve, **kwargs))
    ax = plotter_auc(fpr, tpr, **get_kwargs(plotter_auc, **kwargs))
    return ax


def plotKS(y_pred, y_true, n, asc):
    '''
    # preds is score: asc=1
    # preds is prob: asc=0
    '''
    pred = y_pred  # 预测值
    bad = y_true  # 取1为bad, 0为good
    ksds = pd.DataFrame({'bad': bad, 'pred': pred})
    ksds['good'] = 1 - ksds.bad

    if asc == 1:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, True])
    elif asc == 0:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, True])
    ksds1.index = range(len(ksds1.pred))
    ksds1['cumsum_good1'] = 1.0 * ksds1.good.cumsum() / sum(ksds1.good)
    ksds1['cumsum_bad1'] = 1.0 * ksds1.bad.cumsum() / sum(ksds1.bad)

    if asc == 1:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, False])
    elif asc == 0:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, False])
    ksds2.index = range(len(ksds2.pred))
    ksds2['cumsum_good2'] = 1.0 * ksds2.good.cumsum() / sum(ksds2.good)
    ksds2['cumsum_bad2'] = 1.0 * ksds2.bad.cumsum() / sum(ksds2.bad)

    # ksds1 ksds2 -> average
    ksds = ksds1[['cumsum_good1', 'cumsum_bad1']]
    ksds['cumsum_good2'] = ksds2['cumsum_good2']
    ksds['cumsum_bad2'] = ksds2['cumsum_bad2']
    ksds['cumsum_good'] = (ksds['cumsum_good1'] + ksds['cumsum_good2']) / 2
    ksds['cumsum_bad'] = (ksds['cumsum_bad1'] + ksds['cumsum_bad2']) / 2

    # ks
    ksds['ks'] = ksds['cumsum_bad'] - ksds['cumsum_good']
    ksds['tile0'] = range(1, len(ksds.ks) + 1)
    ksds['tile'] = 1.0 * ksds['tile0'] / len(ksds['tile0'])

    qe = list(np.arange(0, 1, 1.0 / n))
    qe.append(1)
    qe = qe[1:]

    ks_index = pd.Series(ksds.index)
    ks_index = ks_index.quantile(q=qe)
    ks_index = np.ceil(ks_index).astype(int)
    ks_index = list(ks_index)

    ksds = ksds.loc[ks_index]
    ksds = ksds[['tile', 'cumsum_good', 'cumsum_bad', 'ks']]
    ksds0 = np.array([[0, 0, 0, 0]])
    ksds = np.concatenate([ksds0, ksds], axis=0)
    ksds = pd.DataFrame(ksds,
                        columns=['tile', 'cumsum_good', 'cumsum_bad', 'ks'])

    ks_value = ksds.ks.max()
    ks_pop = ksds.tile[ksds.ks.idxmax()]
    print ('ks_value is ' + str(np.round(ks_value, 4)) + \
           ' at pop = ' + str(np.round(ks_pop, 4)))

    # chart
    plt.plot(ksds.tile,
             ksds.cumsum_good,
             label='cum_good',
             color='blue',
             linestyle='-',
             linewidth=2)

    plt.plot(ksds.tile,
             ksds.cumsum_bad,
             label='cum_bad',
             color='red',
             linestyle='-',
             linewidth=2)

    plt.plot(ksds.tile,
             ksds.ks,
             label='ks',
             color='green',
             linestyle='-',
             linewidth=2)

    plt.axvline(ks_pop, color='gray', linestyle='--')
    plt.axhline(ks_value, color='green', linestyle='--')
    plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_good'],
                color='blue',
                linestyle='--')
    plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_bad'],
                color='red',
                linestyle='--')
    plt.title('KS=%s ' % np.round(ks_value, 4) +
              'at Pop=%s' % np.round(ks_pop, 4),
              fontsize=15)
    plt.xlabel('Percentage')

    return ksds


def plotter_cv_results_(results,
                        train_style='mo-',
                        test_style='go-.',
                        title=None):
    '''plot univariate parameter cross validated results after 
    grid search of model
    
    return
    -----
    ax, or tuple of ax
    '''
    scoring = results.filter(like='mean_train_').columns
    scoring = [i.replace('mean_train_', '') for i in scoring]
    df_param = results.filter(like='param_')
    param_array = df_param.columns
    if len(param_array) > 1:
        print('multi-parameter is encountered ... ')
        print(df_param.apply(lambda x: pd.Series(pd.unique(x))))
    # plot
    n = len(scoring)
    i, j = plt.rcParams['figure.figsize']
    fig, ax = plt.subplots(n, 1, figsize=(i, j + 2.5 * (n // 2)))
    ax = get_flat_list(ax) if n == 1 else ax
    for s, ax0 in zip(scoring, ax):
        df = results[['mean_train_' + s, 'mean_test_' + s, 'std_test_' + s]]
        if len(param_array) == 1:
            df.index = results[param_array[0]]
            xlabel = param_array[0]
            num_param = api.is_numeric_dtype(df.index)
            if not num_param:
                df.index = np.arange(len(df.index))
        else:
            xlabel = ' + '.join([i.split('__')[-1] for i in param_array])

        df.sort_index(inplace=True)
        # plot
        mean = df['mean_test_' + s].values
        std = df.pop('std_test_' + s)
        x = df.index.get_values()
        df.plot.line(style=[train_style, test_style], ax=ax0)
        ax0.fill_between(x,
                         mean - std,
                         mean + std,
                         color='grey',
                         alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
        # annotate
        x_max = df.index[np.argmax(mean)]
        best_score = np.max(mean)
        std = np.mean(std)
        h, l = ax0.get_legend_handles_labels()
        ax0.legend(
            [h[-1]],
            ['score_max= %0.4f $\pm$ %0.2f' % (np.max(mean), np.mean(std))])
        ax0.axvline(x_max, linestyle='--', marker='x', color='y')
        ax0.annotate("%0.4f" % best_score, (x_max, best_score))
        ax0.set_xlim(x.min() - 0.5, x.max() + 0.5)
        plt.setp(ax0, ylabel=s)

    # set title
    ax[0].set_title(title, fontsize=13)
    # use fig legend
    fig.legend(h, ('train', 'test', r'$\pm$ 1 std. dev.'),
               loc='upper right',
               ncol=3,
               bbox_to_anchor=(0.98, 1))
    ax[-1].set_xlabel(xlabel)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    return ax


def plotter_rateVol(df,
                    ax=None,
                    lstyle='k-.o',
                    bar_c='g',
                    ylim=(0, None),
                    ymajor_formatter='percent',
                    xlabel_position='bottom',
                    xlabelrotation=30,
                    anno=False,
                    show_mean=True,
                    **subplot_kw):
    ''' plot rate along with volume
    
    df - 3 cols [D1, rate, denominator]
        --> D1-dimensional label, rate= numerator / denominator
    lstyle
        -->2Dline stype egg '-.ko'
    bar_c
        -->  color of bar chart              
    ylim
        --> left y axis limit
    ymajor_formatter
        -->
    xlabel_position
        --> str, 'top'/'down'/'both'   
    return
    ----
    list of artists drawed in ax
    '''
    L = locals().copy()
    L.pop('df')
    L.pop('ax')

    if ax:
        fig, axe = plt.gcf(), ax
    else:
        fig, axe = plt.subplots(1, 1, **subplot_kw)

    df = df.reset_index(drop=True)
    labels = df.iloc[:, 0]
    rate = df.iloc[:, 1]
    vol = df.iloc[:, 2]
    rate_weighted = np.average(rate, weights=vol)
    print(df, '\n')
    # plot artists
    out = []
    out.append(rate.plot(ax=axe, kind='line', style=lstyle))
    axe_right = axe.twinx()
    out.append(vol.plot(ax=axe_right, kind='bar', alpha=0.4, color=bar_c))
    
    if show_mean:
        axe.axhline(rate_weighted, linestyle='--', color='yellow')

    # set axe attr
    fmt = get_ticks_formatter(ymajor_formatter, decimals=1)
    plt.setp(axe,
             ylabel=rate.name,
             xlabel=labels.name,
             xticklabels=labels,
             ylim=ylim)

    axe.yaxis.set_major_formatter(fmt)
    axe.xaxis.set_label_position(xlabel_position)
    if labels.astype(str).apply(len).max() > 8:
        axe.tick_params('x', labelrotation=xlabelrotation)
    # set axe_right attr
    axe_right.set_ylabel(vol.name)
    axe_right.grid(False)
    if show_mean:
        bbox = dict(boxstyle="round", fc='w', alpha=1)
        axe.annotate('mean rate=%-0.2f%%(vol=%d)' %
                     (rate_weighted * 100, sum(vol)), (0, rate_weighted),
                     (0.01, 0.95),
                     xycoords='data',
                     textcoords='axes fraction',
                     bbox=bbox)
    if anno is True:
        _annotate(rate.index.values, rate.values, axe)
    # get legends
    fig.legend(bbox_to_anchor=(1, 1), ncol=2, fontsize='medium')
    plt.tight_layout(pad=1.08, rect=(0, 0, 1, 0.98))

    return out

def plotter_dist_thresh(s, step=1, thresh=53, subplot_kw=None, savefig=None,
                        **fig_kws):
    '''plot distribution of series and percentage above thresh
    
    s - ndarray or series:
        vector to calculate percentile distribution
    step - integer or float:
        step of percentages to plot cummulative distribution
    thresh - integer:
        threshhold/cutoff of decision
    return
    -----
        quantiles of data
    '''
    s = pd.Series(s)
    q = np.arange(100, step=step) / 100
    q = np.append(q, 1)
    perc = s.quantile(q).drop_duplicates()
    
    x = perc
    y = 1 - perc.index
    y = y.rename('percentage above')
    f = interp1d(x, y, kind='cubic')
    
    fig, axes = plt.subplots(2, 1, sharex=True, subplot_kw=subplot_kw, 
                             **fig_kws)
    # hist plot
    sns.distplot(s.dropna(), ax=axes[1], kde=False,
                 hist_kws={'rwidth' : 0.9})
    # line plot
    sns.lineplot(x, y, ax=axes[0], c='green')
    fmt = get_ticks_formatter('percent', decimals=1)
    axes[0].yaxis.set_major_formatter(fmt)
    if thresh:       
        axes[0].axvline(thresh, linestyle='--', lw=3, color='y', 
            label='%1.0f'%thresh)
        axes[0].annotate(
                ' %1.0f%% above %1.0f score '%(100*f(thresh),thresh),
                xy=(thresh, f(thresh)), xycoords='data',
                xytext=(0.9, 0.9), textcoords='axes fraction',
                ha='right', va='bottom', size=15, 
                arrowprops=dict(
                        arrowstyle="->",
                        shrinkA=0, shrinkB=10,
                        connectionstyle="arc3,rad=.2",
                        lw=3))

    axes[1].set_ylabel('count')
    
    plt.tight_layout()
    
    if savefig:
        _save_fig(None, savefig)    
    
    return perc.reset_index()

def get_ticks_formatter(name, *args, **kwargs):
    ''' return ticks formattor  
    ---
    name - form of ticks formatter
    see matplotlib.ticker        
    '''
    if name == 'percent':
        frm = ticker.PercentFormatter(xmax=1, *args, **kwargs)
    if name == 'func':
        frm = ticker.FuncFormatter(*args, **kwargs)
    if name == 'scalar':
        frm = ticker.ScalarFormatter(*args, **kwargs)

    return frm

def plotter_k_timeseries(time_rate, subplot_kw=None, **fig_kws):
    '''plot time series of rate and volume
    
    time_rate - df:
        pass rate at different nodes
    time_vol - series:
        vol at different nodes  
        
    .. note ::         
        index of df/series is used as xaxis 
    
    '''
    
    time_vol = None
    for name, col in time_rate.iteritems():
        if any(col > 1.0):
            time_vol = time_rate.pop(name)
            break
    if time_vol is not None:
        
        fig, axe = plt.subplots(2, 1, sharex=True,
                                subplot_kw=subplot_kw, **fig_kws)
        # plot line plot for rate data
        sns.lineplot(data=time_rate, ax=axe[0], palette='Set1',
                     markers=True, markersize=6)
        fmt = get_ticks_formatter('percent', decimals=1)
        axe[0].yaxis.set_major_formatter(fmt)
        # plot area plot for volume data
        sns.lineplot(data=time_vol.to_frame(), markers=['o'], markersize=6, 
                     markerfacecoloralt='red', ax=axe[1])
        axe[1].fill_between(time_vol.index, 0, time_vol, alpha=.1 )
    else:
        fig, ax = plt.subplots(subplot_kw=subplot_kw, **fig_kws)
        sns.lineplot(data=time_rate, ax=ax, palette='Set1',
                     markers=True, markersize=6)
        fmt = get_ticks_formatter('percent', decimals=1)
        ax.yaxis.set_major_formatter(fmt)   
        
    fig.autofmt_xdate()
    plt.tight_layout()
    return


def plotter_score_path(df_score, title=None, cm=None, style='-.o'):
    '''
    df_score:
        data frame of scores of metrics
    '''
    # plot
    data = df_score.select_dtypes(include='number')
    n = len(data.columns)
    i, j = plt.rcParams['figure.figsize']
    fig, ax = plt.subplots(n, 1, figsize=(i, j + 2.5 * (n // 2)))
    ax = get_flat_list(ax) if n == 1 else ax
    if cm is None:
        cm = plt.get_cmap('tab10')
    cmlist = [cm(i) for i in np.linspace(0, 1, n)]

    i = 0
    for ax0, col in zip(ax, data.columns):
        s = data[col]
        if api.is_numeric_dtype(s):

            s.plot(ax=ax0, color=cmlist[i], style=style)
            ax0.fill_between(s.index,
                             s - s.std(),
                             s + s.std(),
                             color='grey',
                             alpha=.3,
                             label=r'{} = {}$\pm$ {}'.format(
                                 col, round(s.mean(), 4), round(s.std(), 4)))
            plt.setp(ax0, ylabel=col)
            h, l = ax0.get_legend_handles_labels()
            ax0.legend([h[-1]], [l[-1]])
            i += 1
    ax[0].set_title(title)
    ax[-1].set_xlabel('index')
    plt.tight_layout(rect=(0, 0, 0.98, 0.96))
    return fig


def _annotate(x, y, ax):
    ''' plot annotate
    '''
    if api.is_list_like(x):
        for i, j in zip(x, y):
            ax.annotate(s='%.1f%%' % (100 * j),
                        xy=(i, j),
                        xytext=(10, 10),
                        textcoords='offset pixels')
    return ax


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plotter_contours(ax,
                     clf,
                     x,
                     y,
                     h=0.02,
                     pre_method='predict',
                     pos_label=1,
                     **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    pos_label: index of predicted class
    params: dictionary of params to pass to contourf, optional
    """
    xx, yy = make_meshgrid(x, y, h)
    pre = getattr(clf, pre_method)
    if pre is not None:
        Z = pre(np.c_[xx.ravel(), yy.ravel()])
    if np.ndim(Z) > 1:
        Z = Z[:, pos_label]

    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def filter_subset(data, filter_con, **kwargs):
    '''
    data:
        DataFrame
    filter_con (dict):
        filter condition on data's columns, 
        eg: {col1 : [str1, str2, ...], ...}
        
    '''
    if not isinstance(data, pd.DataFrame):
       raise ValueError("'data' must be DataFrame") 
    gen = (data[k].isin(v) for k, v in filter_con.items())        
    filtered = reduce(lambda x, y: x & y, gen)
    return data.loc[filtered]
    
def _save_fig(fig, file, **kwargs):
    '''save fig object to 'path' if it has 'savefig' method
    '''
    if fig is None:
        fig = plt.gcf()
    
    if hasattr(fig, 'savefig'):
        fig.savefig(file, **kwargs)
    else:
        getattr(fig, 'get_figure')().savefig(file, **kwargs)    
        
def color_reference(keys=None):
    '''
    '''   
    cmaps = OrderedDict()
    
    cmaps['Perceptually Uniform Sequential'] = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']

    cmaps['Sequential'] = [
                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    
    cmaps['Diverging'] = [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
    
    cmaps['Miscellaneous'] = [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
    
    cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
    
    
    nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps.items())
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))


    def plot_color_gradients(cmap_category, cmap_list, nrows):
        fig, axes = plt.subplots(nrows=nrows)
        fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
        axes[0].set_title(cmap_category + ' colormaps', fontsize=14)
    
        for ax, name in zip(axes, cmap_list):
            ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
            pos = list(ax.get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3]/2.
            fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)
    
        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axes:
            ax.set_axis_off()

    if keys is None:
        return cmaps.keys()
    else:       
        for cmap_category, cmap_list in cmaps.items():
            if cmap_category in keys:
                plot_color_gradients(cmap_category, cmap_list, nrows)
        
            plt.show()
    
# BBD plotter
def plotter_k_status(data, savefig=None):
    ''' calculate pass rate at different application-appraisal nodes, and 
    plot data
    
    data - DataFrame (is_passed series, keys as name):
        columns:
            represent application-appraisal nodes name
        values (ndarray or series):
            represent passed applications at different nodes, binary array 
            1 indicate pass, 0 indicate not pass
        egg. (apply, admission, score, amount)
   
    return (df):
        dataframe with columns  [nodes, rate, volume]
        
    '''

    vol = data.sum()
    rate = vol/vol.max()
    
    plot_data = pd.DataFrame({'节点': vol.index, 
                              '百分比' : rate, 
                              '单量' : vol})    
    # plot       
    plotter_rateVol(plot_data, anno=True, show_mean=False, dpi=100, 
                    bar_c=sns.color_palette('Blues_d',n_colors=4)) 
    
    if savefig:
        _save_fig(None, savefig)
        
    return plot_data 

def plotter_timeseries_rate(data, date, freq, agg='sum', div_length=True, savefig=None):
    '''calculate time-series aggregated data for rate and volume
    
    data - DataFrame:
        columns are node labels, values are binary 1 indicate pass,
        0 indicate not pass, egg. 
        [apply_date, adm_pass, score_pass, amount_pass, man_pass]
    
    date - str:
        column label for datetime data
    freq:
        frequency of datetime to aggreagte
    agg:
        aggregate method
    
    return:
        DataFrame resampled by date freq
    '''
    
    date = pd.to_datetime(date)
    data.index = date
    re = data.resample(freq)
    num = getattr(re, agg)()
    
    if div_length:
        num /= re.count()
    
    num['申请量'] = re.count().iloc[:, 0]
    # plot
    plotter_k_timeseries(num)
    if savefig:
        _save_fig(None, savefig)
        
    return num             