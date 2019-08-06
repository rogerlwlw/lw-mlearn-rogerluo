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
import matplotlib.pyplot as plt

import matplotlib.ticker as ticker
from pandas.core.dtypes import api
from scipy import interp
from sklearn.metrics import auc, roc_curve

from .utilis import get_flat_list, get_kwargs

# import mlens visualization tools 
from mlens.visualization import (
        pca_comp_plot, pca_plot, corr_X_y, corrmat, clustered_corrmap, 
        exp_var_plot
        )

plt.style.use('seaborn')
plt.rcParams.update({
    'figure.dpi':
    90.0,
    'axes.unicode_minus':
    False,
    'font.family': ['sans-serif'],
    'font.sans-serif': [
        'Arial',
        'Liberation Sans',
        'DejaVu Sans',
        'Bitstream Vera Sans',
        'sans-serif'
        'Microsoft YaHei',
        'Microsoft JhengHei',
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
                    anno=False):
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
        fig, axe = plt.subplots(1, 1)

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
    axe.axhline(rate_weighted, linestyle='--', color='yellow')

    # set axe attr
    fmt = get_ticks_formatter(ymajor_formatter)
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


def get_ticks_formatter(name, *args, **kwargs):
    ''' return ticks formattor  
    ---
    name - form of ticks formatter
    see matplotlib.ticker        
    '''
    if name == 'percent':
        frm = ticker.PercentFormatter(xmax=1, decimals=2, *args, **kwargs)
    if name == 'func':
        frm = ticker.FuncFormatter(*args, **kwargs)
    if name == 'scalar':
        frm = ticker.ScalarFormatter(*args, **kwargs)
    if name == 'FormatStrFormatter':
        frm = ticker.FormatStrFormatter(*args, **kwargs)
    return frm


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
