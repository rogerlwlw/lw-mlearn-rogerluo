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
import copy

from scipy import interp
from sklearn.utils import validation, check_consistent_length
from sklearn.base import BaseEstimator
from sklearn.model_selection import _split
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_score, cross_validate)
from sklearn.model_selection import _validation
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification

from imblearn.pipeline import Pipeline

from functools import wraps
from shutil import rmtree

from .utilis import get_flat_list, get_kwargs
from .plotter import (plotter_auc, plotter_cv_results_, plotter_score_path)
from .read_write import Objs_management
from .lw_preprocess import (pipe_main, pipe_grid, _binning, re_fearturename,
                            plotter_lift_curve, get_custom_scorer)


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
    path
        - directory to read/dump object from 
    gridcv_results
        - cv_results after running grid_searchcv
    folder
        - read_write object to load/dump datafiles from self.path
    bins
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

    def from_config(config):
        ''' return ML_model instance from saved configuration parameters
        '''
        return ML_model(**config)

    def __init__(self,
                 estimator=None,
                 path='model',
                 seed=0,
                 verbose=1,
                 pos_label=1):
        ''' if estimator is None, try to read an '.pipe' estimator from path, 
        and if there's no such '.pipe', use a dummy classifier instead
        '''
        self.path = path
        self.verbose = verbose
        self.pos_label = pos_label
        self.seed = seed
        self.gridcv_results = None

        if estimator is not None:
            if isinstance(estimator, str):
                self.estimator = pipe_main(estimator)
            elif hasattr(estimator, '_estimator_type'):
                self.estimator = estimator
            else:
                raise ValueError('invalid estimator input type: {}'.format(
                    estimator.__class__.__name__))
        else:
            gen, _ = self.folder.read_all(suffix='.pipe')
            if len(gen) > 0:
                self.estimator = gen[0]
                print('estimator {} has been read from {}'.format(
                    self.estimator.__class__.__name__, self.folder.path_))
            else:
                self.estimator = pipe_main('dummy')
                print('no estimator input, use a dummy classifier ... \n')

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

    def _get_dataset(self, suffix):
        '''return list of obj read from 'data' folder given suffix type
        '''
        gen, _ = self.folder.read_all(suffix, path='data')
        if len(gen) == 0:
            raise FileNotFoundError(
                "file with '{}' suffix not found in 'data' folder... \n".
                format(suffix))
        return gen

    def _get_scorer(self, scoring):
        ''' return sklearn scorer
        '''
        scorer = {}
        sk_scoring = []
        custom_scorer = get_custom_scorer()
        for i in get_flat_list(scoring):
            if i in custom_scorer:
                scorer.update({i: custom_scorer[i]})
            else:
                sk_scoring.append(i)
        if len(sk_scoring) > 0:
            s, _ = _validation._check_multimetric_scoring(self.estimator,
                                                          scoring=sk_scoring)
            scorer.update(s)
        return scorer

    @property
    def folder(self):
        return Objs_management(self.path)

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
            - title added to plot header as to indicate (X, y)
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
                _split_cv(X, y=y, cv=cv, groups=groups,
                          random_state=self.seed))
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

        clf = copy.deepcopy(estimator)
        tprs = []
        aucs = []
        fpr_ = []
        tpr_ = []
        mean_fpr = np.linspace(0, 1, 100)
        data_splits = list(
            _split_cv(X, y=y, cv=cv, groups=groups, random_state=self.seed))

        for x_set, y_set in data_splits:
            xx = x_set[0]
            yy = y_set[0]
            clf.fit(xx, yy, **fit_params)
            y_pre = self._pre_continueous(clf, x_set[1])
            fpr, tpr, threshhold = roc_curve(y_set[1],
                                             y_pre,
                                             drop_intermediate=True)
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

        header = '-'.join(
            [_get_estimator_name(clf), 'trainCV', '{} samples'.format(len(y))])
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
            self.folder.write(plt.gcf(), 'plots/lift{}.pdf'.format(title))
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
        scorer = self._get_scorer(scoring)
        return cross_val_score(self.estimator,
                               X=X,
                               y=y,
                               scoring=scorer,
                               cv=cv,
                               **get_kwargs(cross_val_score, **kwargs))

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
        L.pop('scoring')
        scorer = self._get_scorer(scoring)
        # --
        cv_results = cross_validate(scoring=scorer,
                                    **get_kwargs(cross_validate, **L,
                                                 **kwargs))
        return pd.DataFrame(cv_results)

    def test_score(self, X, y, cv, scoring):
        '''return test scores of estimator 
        '''
        # test scores
        data_splits = _split_cv(X, y=y, cv=cv, random_state=self.seed)
        # get_scorers = _validation._check_multimetric_scoring
        # scorer, _ = get_scorers(self.estimator, scoring=scoring)
        # is_multimetric = not callable(scorer)
        scorer = self._get_scorer(scoring)
        is_multimetric = not callable(scorer)
        scores = []
        for item in data_splits:
            x0 = item[0][1]
            y0 = item[1][1]
            scores.append(
                _validation._score(self.estimator, x0, y0, scorer,
                                   is_multimetric))
        scores = pd.DataFrame(scores).reset_index(drop=True)
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
        , update self.estimator & self.gridcv_results
        
        return
        -----
        cv_results as DataFrame
        '''
        L = locals().copy()
        L.pop('self')
        L.pop('fit_params')
        L.pop('scoring')
        scorer = self._get_scorer(scoring)
        # --
        estimator = self.estimator
        grid = GridSearchCV(estimator,
                            scoring=scorer,
                            **get_kwargs(GridSearchCV, **L),
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
                      njobs=2,
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
            y_pre, bins = _binning(y_pre,
                                   bins=self.estimator.bins,
                                   labels=False)
        return y_pre

    def run_train(self,
                  train_set=None,
                  title='Train',
                  scoring=['roc_auc', 'KS'],
                  q=None,
                  bins=None,
                  max_leaf_nodes=None,
                  fit_params={},
                  cv=3,
                  save_fig=True,
                  **kwargs):
        '''
        - run train performance of an estimator; 
        - dump lift curve and ROC curve for train data under self.folder.path_; 
        - optionally dump spreadsheets of calculated data
        
        train_set: 
            2 element tuple, (X, y) of train data
        cv:
           n of cross validation folder, if cv==1, no cross validation        
        fit_params
            -other fit parameters of estimator
            
        return
        ----
        averaged train score

        '''
        L = locals().copy()
        L.pop('self')
        folder = self.folder
        # --
        title = title if title is not None else 0
        if train_set is None:
            train_set = self._get_dataset('.traindata')[0]
        else:
            folder.write(train_set, 'data/0.traindata')

        # trainning
        X = train_set[0]
        y = train_set[1]
        traincv = self.plot_auc_traincv(
            X, y, **get_kwargs(self.plot_auc_traincv, **L), **fit_params)

        self.fit(X, y, **fit_params)
        lift_data = self.plot_lift(X, y, **get_kwargs(self.plot_lift, **L),
                                   **kwargs)

        cv_score = self.cv_validate(X, y, **get_kwargs(self.cv_validate, **L),
                                    **kwargs)
        if self.verbose > 0:
            print('train data & cv_score & cv_splits data are being saved...')
            folder.write([lift_data[-1], cv_score],
                         'spreadsheet/TrainPerfomance{}.xlsx'.format(title),
                         sheet_name=['liftcurve', 'train_score'])
            folder.write(traincv[-1],
                         'spreadsheet/TrainSplits{}.xlsx'.format(title))
        fig = plotter_score_path(cv_score, title='TrainScore_path')
        if save_fig is True:
            folder.write(fig, 'plots/TrainScore_path.pdf')
            plt.close()
        return cv_score.mean()

    def run_test(self,
                 test_set=None,
                 title=None,
                 use_self_bins=True,
                 cv=3,
                 scoring=['roc_auc', 'KS', 'average_precision'],
                 save_fig=True,
                 **kwargs):
        '''
        - run test performance of an estimator; 
        - dump lift curve and ROC curve for test data under self.folder.path_; 
        - optionally dump spreadsheets of calculated data
        
        test_set:
            2 element tuple (X_test, y_test) or list of them
        title:
            title for test_set indicator
        
        return
        ----
            averaged scoring mean
        '''
        L = locals().copy()
        L.pop('self')
        L.pop('title')
        folder = self.folder
        # --

        r = 0
        if test_set is None:
            test_set, title = self._get_dataset('.testdata')[0]
            r -= 1

        test_set_list = get_flat_list(test_set)
        if title is not None:
            title_list = get_flat_list(title)
        else:
            title_list = [str(i) for i in range(len(test_set_list))]
        check_consistent_length(test_set_list, title_list)
        if r == 0:
            folder.write([test_set_list, title_list],
                         'data/{}.testdata'.format(len(title_list)))

        testscore = []
        for i, j in zip(test_set_list, title_list):
            # test performance
            X_test = i[0]
            y_test = i[1]
            # plot test auc
            testcv = self.plot_auc_test(X_test,
                                        y_test,
                                        title=j,
                                        **get_kwargs(self.plot_auc_test, **L,
                                                     **kwargs))
            # plot lift curve
            test_lift = self.plot_lift(X_test,
                                       y_test,
                                       title=j,
                                       **get_kwargs(self.plot_lift, **L),
                                       **kwargs)
            # test scores
            scores = self.test_score(X_test, y_test, cv=cv, scoring=scoring)
            scores['group'] = str(j)
            testscore.append(scores)
            if self.verbose > 0:
                print(
                    'test cv_score & cv_splits test data are being saved... ')
                folder.write(testcv[-1],
                             file='spreadsheet/TestSplits{}.xlsx'.format(j))
                folder.write(
                    [test_lift[-1], scores],
                    sheet_name=['lift_curve', 'test_score'],
                    file='spreadsheet/TestPerfomance{}.xlsx'.format(j))

        testscore_all = pd.concat(testscore, axis=0, ignore_index=True)
        fig = plotter_score_path(testscore_all, title='score_path')
        if save_fig is True:
            folder.write(fig, 'plots/TestScore_path.pdf')
            plt.close()
        if self.verbose > 0 and len(testscore) > 1:
            folder.write(testscore_all, 'spreadsheet/TestPerformanceAll.xlsx')

        return testscore_all[scoring].mean()

    def run_sensitivity(self,
                        train_set=None,
                        title=None,
                        param_grid=-1,
                        refit='roc_auc',
                        scoring=['roc_auc', 'KS'],
                        fit_params={},
                        n_jobs=2,
                        save_fig=True,
                        **kwargs):
        '''
        - run sensitivity of param_grid (if param_grid=-1, use pre-difined); 
        - update self estimator as best estimator, & update self gridcv_results;
        - dump plots/spreadsheets
        
        parmameters
        ----
        train_set: 
            2 element tuple, (X, y) of train data
        param_grid:
            parameter grid space, if -1, use pipe_grid() to return predifined 
            param_grid
        **kwargs:
            GridSearchCV keywords
        '''

        L = locals().copy()
        L.pop('self')
        L.pop('param_grid')
        folder = self.folder
        #--
        if train_set is None:
            train_set = self._get_dataset('.traindata')[0]
        else:
            folder.write(train_set, 'data/0.traindata')

        if param_grid is -1:
            param_grid = []
            for k, v in self.estimator.named_steps.items():
                grid = pipe_grid(k)
                if grid is not None:
                    param_grid.extend(grid)

        if len(param_grid) == 0:
            print('no param_grid found, skip grid search')
            return

        # memory cache
        if isinstance(self.estimator, Pipeline):
            self.estimator.memory = os.path.relpath(
                os.path.join(self.folder.path_, 'tempfolder'))

        X, y = train_set
        cv_results = []
        for i, grid in enumerate(get_flat_list(param_grid)):
            self.grid_searchcv(X,
                               y=y,
                               param_grid=grid,
                               **get_kwargs(self.grid_searchcv, **L),
                               **kwargs)
            self.plot_gridcv(save_fig=save_fig, title=str(i))
            cv_results.append(self.gridcv_results)

        print('sensitivity results are being saved... ')
        title = 0 if title is None else str(title)
        folder.write(cv_results,
                     'spreadsheet/GridcvResults{}.xlsx'.format(title))
        self.save()
        self._shut_temp_folder()

    def run_analysis(self,
                     train_set=None,
                     test_set=None,
                     test_title=None,
                     max_leaf_nodes=None,
                     q=None,
                     bins=None,
                     cv=5,
                     grid_search=False,
                     scoring=['roc_auc', 'KS'],
                     **kwargs):
        '''
        - run self.run_sensitivity(if grid_search=True)
        - run self.run_train 
        - run self.run_test
        - store self trainscore & testscore
        '''
        if grid_search is True:
            self.run_sensitivity(train_set, scoring=scoring)

        self.trainscore = self.run_train(train_set,
                                         cv=cv,
                                         q=q,
                                         bins=bins,
                                         max_leaf_nodes=max_leaf_nodes)
        print('cv score = \n', self.trainscore, '\n')

        try:
            self.testscore = self.run_test(test_set,
                                           title=test_title,
                                           cv=cv,
                                           use_self_bins=True)
            print(self.testscore, '\n')
        except FileNotFoundError:
            print('None test_set data, skip run_test ')
            pass

        self.save()

    def save(self):
        '''save current estimator instance, self instance 
        and self construction settings
        '''
        folder = self.folder
        # save esimator
        folder.write(self.estimator,
                     _get_estimator_name(self.estimator) + '.pipe')

        folder.write(self.get_params(False),
                     self.__class__.__name__ + 'Param.pkl')

        folder.write(
            self, self.__class__.__name__ +
            _get_estimator_name(self.estimator) + '.ml')

    def delete_model(self):
        '''delete self.folder.path_ folder containing model
        '''
        del self.folder.path_

    @property
    def feature_names(self):  #need update
        '''get input feature names of final estimator
        '''
        return re_fearturename(self.estimator)


def train_models(estimator,
                 train_set,
                 test_set,
                 test_title=None,
                 max_leaf_nodes=10,
                 verbose=1,
                 grid_search=True,
                 **kwargs):
    '''run ML_model analysis for given train_set, test_set & estimator
    
    parameter
    ----
    estimator str:
        pipe_main() input str in format of 'xx_xx_xx[_xx]', see pipe_main()
    '''
    model = ML_model(estimator, estimator, verbose=verbose)
    model.run_analysis(train_set,
                       test_set,
                       test_title,
                       max_leaf_nodes,
                       grid_search=grid_search,
                       **kwargs)
    return model


def model_experiments(X=None,
                      y=None,
                      cv=3,
                      scoring=['roc_auc', 'KS'],
                      estimator_lis=None):
    ''' experiment on different piplines
    
    return
    ----
    dataframe 
        - cv scores of each pipeline
    '''
    if estimator_lis is None:
        l = get_default_estimators()
    else:
        l = estimator_lis

    if X is None:
        return set(l)
    else:
        lis = []
        for i in l:
            m = ML_model(estimator=i)
            scores = m.cv_validate(X, y, cv=cv, scoring=scoring).mean()
            scores['pipe'] = i
            lis.append(scores)
        return pd.concat(lis, axis=1, ignore_index=True).T


def _reset_index(*array):
    '''reset_index for df or series, return list of *arrays
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
    if hasattr(estimator, 'classes_'):
        return getattr(estimator, '__class__').__name__
    else:
        raise TypeError('estimator is not an valid sklearn estimator')


def _get_splits_combined(xy_splits, ret_type='test'):
    '''return list of combined X&y DataFrame for cross validated test set
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

def _test(delete=False):
    '''run test of ML_model class methods through train_models functions
    '''
    # --
    if not os.path.exists('tests_train_models'):
        os.makedirs('tests', exist_ok=True)
    os.chdir('tests')    
    X, y = make_classification(300, n_redundant=5, n_features=50)
    l = get_default_estimators()   
    for i in l:
        train_models(i, (X, y), (X, y),
                     max_leaf_nodes=10, scoring=['KS', 'roc_auc'])
    if delete: 
        rmtree('tests')
 
    return
    
def get_default_estimators():
    '''return default list of estimators
    '''
    estimators = [
        # linear models
        'cleanNA_woe_LogisticRegression',
        'cleanNA_woe_frf_LogisticRegression',
        'cleanNA_woe_cleanNN_frf_LogisticRegression',
        'cleanNA_woe_frf20_LogisticRegression',
        'cleanNA_woe_cleanNN_fRFE10log_LogisticRegression',
        'cleanNA_woeq8_cleanNN_frf10_LogisticRegression',
        # default SVM, grid search log/hinge/huber/perceptron
        'cleanMean_woe_frf20_Nys_SGDClassifier',
        'cleanMean_woe_frf20_Nys_cleanNN_SGDClassifier',
        'cleanMean_oht_stdscale_frf20_Nys_SGDClassifier',
        # tree based models
        'clean_oht_XGBClassifier',
        'clean_oht_cleanNN_fxgb_XGBClassifier',
        'clean_oht_cleanNN_inlierForest_fxgb_XGBClassifier',
        'clean_oht_frf_RandomForestClassifier',
        'clean_oht_cleanNN_RandomForestClassifier',
        'clean_oht_fxgb_BalancedRandomForestClassifier',
        'clean_oht_cleanNN_frf_AdaBoostClassifier',
        'clean_oht_fxgb_RUSBoostClassifier',
        'clean_oht_fxgb_cleanNN_GradientBoostingClassifier',
        'clean_oht_fsvm_cleanNN_GradientBoostingClassifier',
        'clean_oht_fxgb_cleanNN_DecisionTreeClassifier',
    ]  
    return estimators