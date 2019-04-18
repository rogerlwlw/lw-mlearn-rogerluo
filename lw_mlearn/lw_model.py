# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 14:26:14 2019

@author: roger luo

class
-----

ML_model:
    quantifying predictions of an estimator and perform parameter tuning
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy import interp
from sklearn.utils import validation
from sklearn.base import clone, BaseEstimator, is_classifier, is_regressor
from sklearn.model_selection import _split
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_score, cross_validate)
from sklearn.model_selection import _validation
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline

from lw_mlearn.utilis import get_flat_list, get_kwargs
from lw_mlearn.plotter import plotter_rateVol, plotter_auc, plotter_cv_results_
from lw_mlearn.read_write import Objs_management
from lw_mlearn.lw_preprocess import pipe_main, pipe_grid, _binning
from functools import wraps
from shutil import rmtree


class ML_model(BaseEstimator):
    '''quantifying predictions of an estimator
    
    parameters
    ---
    estimator
        - sklearn estimator or pipeline instance
    path
        - dir to place model and other files, default 'model'
    seed
        - random state seed, 0 default
    pos_label
            - positive label default 1
       
    attributes
    -----
    path_
        - directory to read/dump object from 
    gridcv_results
        - cv_results after running grid_searchcv
    folder
        - read_write object to load/dump datafiles from self.path_
    estimator.bins
        - bin edges of predictions of estimator
    
    method
    ----
    cv_score:
        return cross score of estimator
    cv_validate:
        return cross score of estimator, allowing multi scorers
    grid_searchcv:
        perform grid search of param_grid, update self esimator estimator
    rand_searchcv:
        perform randomized search of param_grid, update self estimator
    fit:
        perform fit of estimator
    predict:
        perform predict of estimator 
        
    plot_auc_test:
        plot auc of test data
    plot_auc_traincv:
        plot auc of train data
    plot_lift:
        plot lift curve of model
    plot_gridcv:
        plot  grid seach cv results of model
    '''

    def __init__(self,
                 estimator=None,
                 path='model',
                 seed=0,
                 verbose=1,
                 pos_label=1):
        '''   
        '''
        self.folder = Objs_management(path)
        self.path_ = self.folder.path_
        self.verbose = verbose
        self.pos_label = pos_label
        self.gridcv_results = None
        self.seed = seed

        if estimator is not None:
            if isinstance(estimator, str):
                self.estimator =  pipe_main(estimator)
            elif hasattr(estimator, '_estimator_type'):
                self.estimator = estimator
            else:
                raise ValueError('invalid estimator input')
        else:
            try:
                gen, _ = self.folder.read_all(suffix='.estimator')
                self.estimator = next(gen)
                print('estimator {} has been read from {}'.format(
                    self.estimator.__class__.__name__, self.path_))
            except Exception as e:
                print(repr(e))
                raise ValueError('no estimator input')

    def _shut_temp_folder(self):
        '''shut temp folder directory
        '''
        if getattr(self.estimator, 'memory') is not None:
            while os.path.exists(self.estimator.memory):
                rmtree(self.estimator.memory, ignore_errors=True)

            print('%s has been removed' % self.estimator.memory)
            self.estimator.memory = None

    def _check_fitted(self, estimator):
        '''check if estimator has been fitted
        '''
        validation.check_is_fitted(
            estimator,
            ['classes_', 'coef_', 'feature_importances_', 'booster', 'tree_'],
            all_or_any=any)

    def _pre_continueous(self, estimator, X):
        '''make continueous predictions
        '''
        classes_ = getattr(estimator, 'classes_')
        if len(classes_) > 2:
            raise ValueError(' estimator should only output binary classes...')

        if hasattr(estimator, 'decision_function'):
            method = getattr(estimator, 'decision_function')
            y_pre = method(X)
        elif hasattr(estimator, 'predict_proba'):
            method = getattr(estimator, 'predict_proba')
            y_pre = method(X)
        else:
            raise ValueError('estimator have no continuous predictions')
        
        if np.ndim(y_pre) > 1: 
                y_pre = y_pre[:, self.pos_label]
        return y_pre


    def plot_auc_test(self,
                      X,
                      y,
                      cv=1,
                      groups=None,
                      title=None,
                      ax=None,
                      save_fig=False):
        '''plot roc_auc curve for given fitted estimator, must have continuous
        predictons (decision_function or predict_proba) to evaluate model by
        roc_auc metrics(iterables of X, y can be passed or X, y 
        can be splited using cv > 1), to assess model fit performance

        X
            -2D array or list of 2D ndarrays
        y
            -binary or list of class labels
        cv 
            -int, cross-validation generator or an iterable
            - if cv>1, generate splits by StratifyKfold method
        title
            - title added to plot header
        return
        --------
        ax, mean-auc, std-auc,
       
        data_splits:
           list of test data set in the form of DataFrame (combined X & y)
        '''
        L = locals().copy()
        L.pop('self')
        estimator = self.estimator
        # split test set by cv
        if cv > 1:
            xs = []
            ys = []
            data_splits = tuple(
                _split_cv(
                    X, y=y, cv=cv, groups=groups, random_state=self.seed))
            for x_set, y_set in data_splits:
                xs.append(x_set[1])
                ys.append(y_set[1])
            L.update({'X': xs, 'y': ys, 'cv': 1})
            return self.plot_auc_test(**L)

        self._check_fitted(estimator)
        X = get_flat_list(X)
        y = get_flat_list(y)
        validation.check_consistent_length(X, y)
        fprs = []
        tprs = []
        aucs = []
        n_sample = 0
        for i in range(len(X)):
            x0 = X[i]
            y0 = y[i]
            y_pre = self._pre_continueous(estimator, x0)
            fpr, tpr, threshhold = roc_curve(y0, y_pre, drop_intermediate=True)
            fprs.append(fpr)
            tprs.append(tpr)
            aucs.append(auc(fpr, tpr))
            n_sample += len(x0)
        # -- plot
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax = plotter_auc(fprs, tprs, ax=ax)

        header = '-'.join([
            _get_estimator_name(estimator), 'testCV',
            '{} samples'.format(n_sample)
        ])
        if isinstance(title, str):
            header = '-'.join([title, header])
        ax.set_title(header)

        data_splits = [
            pd.concat((pd.DataFrame(i) for i in item), axis=1)
            for item in zip(X, y)
        ]

        if save_fig is True:
            if isinstance(title, str):
                plot_name = 'plots/roc_test_' + title + '.pdf'
            else:
                plot_name = 'plots/roc_test.pdf'
            self.folder.write(plt.gcf(), plot_name)
            plt.close()
        return ax, np.mean(aucs), np.std(aucs), data_splits

    def plot_auc_traincv(self,
                         X,
                         y,
                         cv=5,
                         groups=None,
                         title=None,
                         ax=None,
                         save_fig=False,
                         **fit_params):
        '''fit & plot roc_auc of an estimator, must have continuous
        predictons (to assess hyper parameter settings performance)

        X
            -2D array  or DataFrame
        y
            -binary class labels  , 1D array             
        cv 
            -int, cross-validation generator or an iterable
            - if cv>1, generate splits by StratifyKfold method
        title
            - title added to plot header
        fit_params
            -other fit parameters
        return
        ----
        ax, mean_auc, std_auc,
        
        data_splits:
            list of test data set in the form of DataFrame
        '''

        estimator = self.estimator

        clf = clone(estimator)
        tprs = []
        aucs = []
        fpr_ = []
        tpr_ = []
        mean_fpr = np.linspace(0, 1, 100)
        data_splits = tuple(
            _split_cv(X, y=y, cv=cv, groups=groups, random_state=self.seed))
        for x_set, y_set in data_splits:
            clf.fit(x_set[0], y_set[0], **fit_params)
            y_pre = self._pre_continueous(clf, x_set[1])
            fpr, tpr, threshhold = roc_curve(
                y_set[1], y_pre, drop_intermediate=True)
            tprs.append(interp(mean_fpr, fpr, tpr))
            fpr_.append(fpr)
            tpr_.append(tpr)
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        # -- plot
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax = plotter_auc(fpr_, tpr_, ax=ax)

        header = '-'.join([
            _get_estimator_name(estimator), 'trainCV', '{} samples'.format(
                len(y))
        ])
        if isinstance(title, str):
            header = '-'.join([title, header])
        ax.set_title(header)
        if save_fig is True:
            if isinstance(title, str):
                plot_name = 'plots/roc_train_' + title + '.pdf'
            else:
                plot_name = 'plots/roc_train.pdf'
            self.folder.write(plt.gcf(), plot_name)
            plt.close()
        return ax, mean_auc, std_auc, _get_splits_combined(data_splits)

    def plot_lift(self,
                  X,
                  y,
                  q=None,
                  bins=None,
                  max_leaf_nodes=None,
                  use_self_bins=False,
                  labels=False,
                  ax=None,
                  title=None,
                  save_fig=False,
                  **tree_kwargs):
        '''plot list curve of (X, y) data, update self bins
        
            given bins(n equal width) or q( n equal frequency) or 
            max_leaf_nodes cut by tree
        X
            -2D array
        y
            -binary class labels , 1D array

        q
            - number of equal frequency
        bins
            - number of equal width or array of edges
        max_leaf_nodes
            - if not None perform supervised cutting, 
            - number of tree nodes using tree cut
        use_self_bins
            - use self.estimator.bins if true

        .. note::
            -  only 1 of (q, bins, max_leaf_nodes) can be specified

        **tree_kwargs - Decision tree keyswords, egg:
            - min_impurity_decrease=0.001
            - random_state=0
        labels
            - see pd.cut, if False return integer indicator of bins, 
            - if True/None return arrays of labels (or can be passed )       
        title
            - title XXX of plot, output format: 'XXX' + estimator's name
        return
        ----
        ax,  plotted_data;
        '''

        self._check_fitted(self.estimator)
        estimator = self.estimator
        y_pre = self._pre_continueous(estimator, X)

        if use_self_bins is True:
            if self.estimator.bins is not None:
                bins = self.estimator.bins
                q = None
                max_leaf_nodes = None
            else:
                raise ValueError('self bins is None')

        header = _get_estimator_name(estimator) + ' - lift curve'
        if not (title is None):
            header = ' - '.join([title, header])

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax, y_cut, bins, plotted_data = plotter_lift_curve(
            y_pre,
            y_true=y,
            bins=bins,
            q=q,
            header=header,
            max_leaf_nodes=max_leaf_nodes,
            labels=labels,
            ax=ax,
            **tree_kwargs)
        # update self bins
        self.estimator.bins = bins

        if save_fig is True:
            title = 0 if title is None else str(title)
            self.folder.write(plt.gcf(), 'plots/lift_{}.pdf'.format(title))
            plt.close()
        return ax, plotted_data

    def plot_gridcv(self, title=None, save_fig=False):
        '''plot grid seatch cv results
        '''
        header = '-'.join([_get_estimator_name(self.estimator), 'gridcv'])
        if title is None:
            pass
        else:
            header = '-'.join([title, header])
        if self.gridcv_results is None:
            print('no grid cv results')
        else:
            plotter_cv_results_(self.gridcv_results, title=header)

        if save_fig is True:
            if isinstance(title, str):
                plot_name = 'plots/gridcv_' + title + '.pdf'
            else:
                plot_name = 'plots/gridcv.pdf'
            self.folder.write(plt.gcf(), plot_name)
            plt.close()
        
    @wraps(cross_val_score)
    def cv_score(self, X, y, scoring='roc_auc', cv=5, **kwargs):
        '''
        return cross validated score of estimator (see cross_val_score)
        ---------
        '''
        estimator = self.estimator
        L = locals().copy()
        L.pop('self')
        return cross_val_score(**get_kwargs(cross_validate, **L, **kwargs))

    @wraps(cross_validate)
    def cv_validate(self,
                    X,
                    y,
                    scoring='roc_auc',
                    cv=5,
                    return_estimator=False,
                    return_train_score=False,
                    **kwargs):
        '''       
        return cross_validate results of estimator(see cross_validate)
        -----
        cv_results: 
            (as DataFrame, allowing for multi-metrics) in the form of
            'test_xxx', train_xxx' where  'xxx' is scorer name
        '''
        estimator = self.estimator
        L = locals().copy()
        L.pop('self')
        cv_results = cross_validate(
            **get_kwargs(cross_validate, **L, **kwargs))
        return pd.DataFrame(cv_results)

    def test_score(self, X, y, cv, scoring):
        '''return test scores of estimator 
        '''
        # test scores
        data_splits = _split_cv(X, y=y, cv=cv, random_state=self.seed)
        get_scorers = _validation._check_multimetric_scoring
        scorer, _ = get_scorers(self.estimator, scoring=scoring)
        is_multimetric = not callable(scorer)
        scores = []
        for item in data_splits:
            x0 = item[0][1]
            y0 = item[1][1]
            scores.append(
                _validation._score(self.estimator, x0, y0, scorer,
                                   is_multimetric))
        scores = pd.DataFrame(scores).reset_index()
        return scores

    @wraps(GridSearchCV)
    def grid_searchcv(self,
                      X,
                      y,
                      param_grid,
                      scoring='roc_auc',
                      cv=3,
                      refit='roc_auc',
                      return_train_score=True,
                      n_jobs=2,
                      fit_params={},
                      **kwargs):
        '''tune hyper parameters of estimator by searching param_grid
        , update self estimator & grid search results
        
        return
        -----
        cv_results as DataFrame
        '''
        L = locals().copy()
        L.pop('self')
        L.pop('fit_params')
        # --
        estimator = self.estimator
        grid = GridSearchCV(estimator, **get_kwargs(GridSearchCV, **L),
                            **kwargs)
        grid.fit(X, y, **fit_params)
        cv_results = pd.DataFrame(grid.cv_results_)
        self.estimator = grid.best_estimator_
        self.gridcv_results = cv_results
        return cv_results

    @wraps(RandomizedSearchCV)
    def rand_searchcv(self,
                      X,
                      y,
                      param_distributions,
                      scoring='roc_auc',
                      cv=3,
                      refit=None,
                      return_train_score=True,
                      fit_params={},
                      njobs=-1,
                      **kwargs):
        '''tune hyper parameters of estimaotr by randomly searching param_grid
        , update self estimator & grid search results     
        return
        -----
        cv_results as DataFrame
        '''
        L = locals().copy()
        L.pop('self')
        # --
        estimator = self.estimator
        grid = RandomizedSearchCV(estimator,
                                  **get_kwargs(RandomizedSearchCV, **L),
                                  **kwargs)
        grid.fit(X, y, **fit_params)
        cv_results = pd.DataFrame(grid.cv_results_)
        self.set_params(estimator=grid.best_estimator_)
        return cv_results

    def fit(self, X, y, **fit_params):
        '''perform fit of estimator
        '''
        self.estimator.fit(X, y, **fit_params)
        return self

    def predit(self,
               X,
               pre_method='predict_proba',
               pre_level=False,
               pos_label=1,
               **kwargs):
        '''return predictions of estimator
        
        pre_method: str
            sklearn estimator method name: ['predict', predict_proba,
            decision_function]
        pre_level: bool
             if true, output score as integer rankings starting from 0
        pos_label
            index of predicted class
        '''
        estimator = self.estimator
        pre_func = getattr(estimator, pre_method)
        if pre_func is None:
            print('{} has no {} method'.format(
                _get_estimator_name(estimator, pre_method)))
        y_pre = pre_func(X, **kwargs)
        if np.ndim(y_pre) > 1:
            y_pre = y_pre[:, pos_label]
        if pre_level:
            y_pre, bins =  _binning(
                y_pre, bins=self.estimator.bins, labels=False)
        return y_pre

    def run_train(self,
                  train_set,
                  scoring=[
                      'roc_auc', 'average_precision'],
                  cv=5,
                  q=10,
                  bins=None,
                  max_leaf_nodes=None,
                  fit_params={},
                  title=None,
                  save_fig=True,
                  **kwargs):
        '''run training of an esimator and dump performance
        ([plots/spreadsheets]) to self.path_ folder
        
        train_set: 
            2 element tuple, (X, y) of train data
        cv:
           n of cross validation folder, if cv==1, no cross validation        
        fit_params
            -other fit parameters of estimator

        '''
        L = locals().copy()
        L.pop('self')
        # --
        folder = self.folder
        # save train performance
        X = train_set[0]
        y = train_set[1]
        traincv = self.plot_auc_traincv(
            X, y, **get_kwargs(self.plot_auc_traincv, **L), **fit_params)

        self.fit(X, y, **fit_params)
        lift_data = self.plot_lift(X, y, **get_kwargs(self.plot_lift, **L),
                                   **kwargs)

        if self.verbose > 0:
            print('train data & cv_score & cv_splits data are being saved...')
            # save (X, y) data
            title = title if title is not None else 0
            folder.write(train_set, 'data/train{}.data'.format(title))

            cv_score = self.cv_validate(X, y,
                                        **get_kwargs(self.cv_validate, **L),
                                        **kwargs)
            folder.write([lift_data[-1], cv_score],
                         'spreadsheet/train{}.xlsx'.format(title),
                         sheet_name=['liftcurve', 'train_score'])
            folder.write(traincv[-1],
                         'spreadsheet/train_datasplits{}.xlsx'.format(title))

    def run_test(self,
                 test_set,
                 cv=5,
                 q=None,
                 bins=None,
                 max_leaf_nodes=None,
                 use_self_bins=True,
                 scoring=[
                     'roc_auc', 'average_precision'],
                 title=None,
                 save_fig=True,
                 **kwargs):
        '''run test performance of an estimator, dump: [plots/spreadsheet] to 
        self.path_    
        
        test_set:
            2 element tuple (X_test, y_test)
        q
            - n equal frequency for lift curve 
        '''
        L = locals().copy()
        L.pop('self')
        # --
        folder = self.folder
        # test performance
        X_test = test_set[0]
        y_test = test_set[1]
        # plot test auc
        testcv = self.plot_auc_test(
            X_test, y_test, **get_kwargs(self.plot_auc_test, **L, **kwargs))
        # plot lift curve
        test_lift = self.plot_lift(X_test, y_test,
                                   **get_kwargs(self.plot_lift, **L), **kwargs)

        if self.verbose > 0:
            print('test cv_score & cv_splits test data are being saved... ')
            title = title if title is not None else 0
            folder.write(test_set, 'data/test{}.data'.format(title))
            folder.write(testcv[-1],
                         'spreadsheet/test_datasplits{}.xlsx'.format(title))
            # test scores
            scores = self.test_score(X_test, y_test, cv=cv, scoring=scoring)
            folder.write([test_lift[-1], scores],
                         sheet_name=['lift_curve', 'test_score'],
                         file='spreadsheet/test{}.xlsx'.format(title))

    def run_sensitivity(self,
                        train_set,
                        param_grid,
                        refit='roc_auc',
                        scoring=['roc_auc', 'average_precision'],
                        fit_params={},
                        save_fig=True,
                        title=None,
                        **kwargs):
        '''run sensitivity of param_grid space, dump plots/spreadsheets
        
        train_set: 
            2 element tuple, (X, y) of train data
        param_grid:
            parameter grid space
        **kwargs:
            GridSearchCV keywords
        '''
        L = locals().copy()
        L.pop('self')
        #--
        # memory cache
        if isinstance(self.estimator, Pipeline):
            self.estimator.memory = os.path.relpath(
                os.path.join(self.path_, 'tempfolder'))

        folder = self.folder
        X, y = train_set
        param_grid = L.pop('param_grid')
        cv_results = []
        for i, grid in enumerate(get_flat_list(param_grid)):
            self.grid_searchcv(
                X,
                y,
                param_grid=grid,
                **get_kwargs(self.grid_searchcv, **L),
                **kwargs)
            self.plot_gridcv(save_fig=save_fig, title=str(i))
            cv_results.append(self.gridcv_results)
       
        print('sensitivity results & data are being saved... ')
        title = 0 if title is None else str(title)
        folder.write(cv_results, 
                     'spreadsheet/sensitivity_{}.xlsx'.format(title))
        folder.write(train_set, 'data/sensitivity_{}.data'.format(title))

        self.save_estimator()
        self._shut_temp_folder()

    def save_estimator(self):
        '''save current estimator and all construction settings
        '''
        folder = self.folder
        # save esimator
        folder.write(self.estimator,
                     _get_estimator_name(self.estimator) + '.estimator')
        folder.write(self.get_params(),
                     _get_estimator_name(self.estimator) + 'ML_parameters.pkl')
        
        folder.write(self, self.__class__.__name__ + '.model')
    
    def delete_model(self):
        '''delete self.path folder containing model
        '''
        del self.folder.path_

    @property
    def feature_names(self):  #need update
        '''get input feature names of final estimator
        '''
        estimator = self.estimator

        try:
            if isinstance(estimator, Pipeline):
                fn = None
                steps = estimator.steps
                n = len(steps) - 1
                while n > -1:
                    n -= 1
                    tr = steps[n][1]
                    if hasattr(tr, 'get_feature_names'):
                        fn = tr.get_feature_names()
                        if fn is not None: break
                if fn is None:
                    print('estimator has no feature_names attribute')
                    return

                tr = steps[-2][1]
                if hasattr(tr, 'get_support'):
                    fn = pd.Series(fn)[tr.get_support()]

        except Exception as e:
            print(repr(e))

        return fn

    @feature_names.setter
    def feature_names(self, value):

        raise ValueError('feature_names cannot be input')


def _reset_index(*array):
    '''reset_index df or series, return list of *arrays
    '''
    rst = []
    for i in array:
        if isinstance(i, (pd.DataFrame, pd.Series)):
            rst.append(i.reset_index(drop=True))
        else:
            rst.append(i)
    return rst


def _split_cv(*arrays, y=None, groups=None, cv=3, random_state=None):
    '''supervise splitting
    y
        - class label,if None not to stratify
    groups
        - split by groups
    cv
        - number of splits

    return
    ----
    generator of list containing splited arrays,shape = [m*n*k], for 1 fold
    [(0train, 0test), (1train, 1test), ...]

    m - indices of folds [0 : cv-1]
    n - indice of variable/arrays [0 : n_arrays-1]
    k - indice of train(0)/test[1] set [0:1]
    '''

    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    validation.check_consistent_length(*arrays, y, groups)
    arrays = list(arrays)
    
    if cv == 1:
        if y is not None: 
            arrays.append(y)
        return [[(i, i) for i in arrays]]
    # get cross validator
    if y is not None:
        arrays.append(y)
        cv = _split.check_cv(cv, y=y, classifier=True)
    else:
        cv = _split.check_cv(cv, classifier=False)
    # set random state
    if hasattr(cv, 'random_state'):
        cv.random_state = random_state
    # reset_index pandas df or series
    arrays = _reset_index(*arrays)
    arrays = _split.indexable(*arrays)
    # get indexing method
    safe_index = _split.safe_indexing
    train_test = ([
        (safe_index(i, train_index), safe_index(i, test_index)) for i in arrays
    ] for train_index, test_index in cv.split(arrays[0], y, groups))

    return train_test


def _get_estimator_name(estimator):
    '''return estimator's class name
    '''
    if isinstance(estimator, Pipeline):
        estimator = estimator._final_estimator
    if is_classifier(estimator) or is_regressor(estimator):
        return getattr(estimator, '__class__').__name__
    else:
        raise TypeError('esimator is not an sklearn estimator')


def plotter_lift_curve(y_pre,
                       y_true,
                       bins,
                       q,
                       max_leaf_nodes,
                       labels,
                       ax,
                       header,
                       xlabel='xlabel',
                       **kwargs):
    '''return lift curve of y_pre on y_true 
   
    y_pre
        - array_like, value of y to be cut
    y_true
        - true value of y for supervised cutting based on decision tree 
    bins
        - number of equal width or array of edges
    q
        - number of equal frequency              
    max_leaf_nodes
        - number of tree nodes using tree cut
        - if not None use supervised cutting based on decision tree
    **kwargs - Decision tree keyswords, egg:
        - min_impurity_decrease=0.001
        - random_state=0 
    .. note::
        -  only 1 of (q, bins, max_leaf_nodes) can be specified       
    labels
        - see pd.cut, if False return integer indicator of bins, 
        - if True return arrays of labels (or can be passed )
    header
        - title of plot
    xlabel
        - xlabel for xaxis
    '''
    y_cut, bins =  _binning(
        y_pre,
        y_true=y_true,
        bins=bins,
        q=q,
        max_leaf_nodes=max_leaf_nodes,
        labels=labels,
        **kwargs)
    df0 = pd.DataFrame({'y_cut': y_cut, 'y_true': y_true})
    df_gb = df0.groupby('y_cut')
    df1 = pd.DataFrame()
    df1[xlabel] = df_gb.sum().index.values
    df1['rate'] = (df_gb.sum() / df_gb.count()).values
    df1['vol'] = df_gb.count().values
    # plot
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    plotted_data = df1.dropna()
    ax = plotter_rateVol(plotted_data, ax=ax)
    plt.title(header, fontsize=14)
    return ax, y_cut, bins, plotted_data


def _get_splits_combined(xy_splits, ret_type='test'):
    '''return list of combined X y DataFrame for cross validated test set
    '''
    data_splits_test = [
        pd.concat((pd.DataFrame(i[1]) for i in item), axis=1)
        for item in xy_splits
    ]

    data_splits_train = [
        pd.concat((pd.DataFrame(i[0]) for i in item), axis=1)
        for item in xy_splits
    ]

    if ret_type == 'test':
        return data_splits_test
    if ret_type == 'train':
        return data_splits_train
    
if __name__ == '__main__':
    from sklearn.datasets import make_classification
    # Import some data to play with
    X, y = make_classification(1000)
    # test
    estimator = 'GaussianProcessClassifier'
    E = ML_model(estimator, path='../tests_model')
    E.run_train((X, y), q=5, save_fig=True)
    E.run_test((X, y), save_fig=True)
    
    try:
        grid = pipe_grid(estimator, True)
        E.run_sensitivity((X,y), grid, cv=3, save_fig=True)
    except:
        pass
    E.save_estimator()
    E.delete_model()

    
