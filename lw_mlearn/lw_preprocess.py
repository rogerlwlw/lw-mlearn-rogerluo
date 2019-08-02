# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:36:20 2018

@author: roger luo
"""
import pandas as pd
import numpy as np
import scipy.stats as stats

from pandas.core.dtypes import api

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (OrdinalEncoder, OneHotEncoder,
                                   PolynomialFeatures, StandardScaler,
                                   MinMaxScaler, RobustScaler, Normalizer,
                                   QuantileTransformer, PowerTransformer,
                                   MaxAbsScaler)
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import (SGDClassifier, LogisticRegression, 
                                  LogisticRegressionCV)
from sklearn.feature_selection import (SelectFromModel,
                                       GenericUnivariateSelect, chi2,
                                       f_classif, mutual_info_classif, RFE)
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              Exponentiation, ConstantKernel)
from sklearn.decomposition import (
    DictionaryLearning, FastICA, IncrementalPCA, KernelPCA,
    MiniBatchDictionaryLearning, MiniBatchSparsePCA, NMF, PCA, SparseCoder,
    SparsePCA, dict_learning, dict_learning_online, fastica,
    non_negative_factorization, randomized_svd, sparse_encode, FactorAnalysis,
    TruncatedSVD, LatentDirichletAllocation)

from sklearn.kernel_approximation import RBFSampler, Nystroem

from sklearn.metrics import roc_curve
from sklearn.metrics import make_scorer
from sklearn.impute import SimpleImputer
from sklearn.utils import validation
from sklearn.utils.testing import all_estimators
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.ensemble import IsolationForest, ExtraTreesClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from xgboost.sklearn import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import (
    RandomUnderSampler,
    TomekLinks,
    NearMiss,
    CondensedNearestNeighbour,
    OneSidedSelection,
    NeighbourhoodCleaningRule,
    EditedNearestNeighbours,
    AllKNN,
    InstanceHardnessThreshold,
)
from imblearn.over_sampling import (
    ADASYN,
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    SVMSMOTE,
    SMOTENC,
)
from imblearn.ensemble import (
    EasyEnsembleClassifier,
    BalancedRandomForestClassifier,
    RUSBoostClassifier,
)

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn import FunctionSampler

from .utilis import (dec_iferror_getargs, get_kwargs)
from .read_write import Path
from .plotter import plt, plotter_rateVol


def index_duplicated(string_list):
    '''index duplicated item in a list to make an unique list
    '''
    d = pd.Series(string_list).duplicated(False)
    n = 0
    new_list = []
    for i, j in zip(string_list, d):
        if j:
            new_list .append(str(n) + i)
        else:
            new_list.append(i)
        n += 1
    return new_list
        
    
    
def pipe_main(pipe=None):
    '''pipeline construction using sklearn estimators, final step support only
    classifiers currently
    
    .. note::
        data flows through a pipeline consisting of steps as below:
            raw data --> clean --> encoding --> scaling --> feature construction 
            --> feature selection --> resampling --> final estimator
            see scikit-learn preprocess & estimators
    parameter
    ----
    pipe - str 
        - in the format of 'xx_xx' of which 'xx' means steps in pipeline,
          default None
    return
    ----
        1) pipeline instance of chosen steps
        2) if pipe is None, a dict indicating possible choice of 'steps'
    '''
    clean = {
        'clean':
        Split_cls(dtype_filter='not_datetime', na1='null', na2=-999),
        'cleanNA':
        Split_cls(dtype_filter='not_datetime', na1=None, na2=None),
        'cleanMean':
        Split_cls(dtype_filter='not_datetime', na1='most_frequent',
                  na2='mean'),
        'cleanMn':
        Split_cls(dtype_filter='not_datetime', na1='missing', na2='mean'),
    }
    #
    encode = {
        'woe': Woe_encoder(max_leaf_nodes=8),
        'woeq8' : Woe_encoder(q=8),
        'woeb5' : Woe_encoder(bins=5),
        'oht': Oht_encoder(),
        'ordi': Ordi_encoder(),
    }

    resample = {

        # over_sampling
        'rover':
        RandomOverSampler(),
        'smote':
        SMOTE(),
        'bsmote':
        BorderlineSMOTE(),
        'adasyn':
        ADASYN(),

        # under sampling controlled methods
        'runder':
        RandomUnderSampler(),
        'nearmiss':
        NearMiss(version=3),
        'pcart':
        InstanceHardnessThreshold(),

        # under sampling cleaning methods
        'tlinks':
        TomekLinks(n_jobs=-1),
        'oside':
        OneSidedSelection(n_jobs=-1),
        'cleanNN':
        NeighbourhoodCleaningRule(n_jobs=-1),
        'enn':
        EditedNearestNeighbours(n_jobs=-1),
        'ann':
        AllKNN(n_jobs=-1),
        'cnn':
        CondensedNearestNeighbour(n_jobs=-1),

        # clean outliers
        'inlierForest':
        FunctionSampler(outlier_rejection,
                        kw_args={'method': 'IsolationForest'}),
        'inlierLocal':
        FunctionSampler(outlier_rejection,
                        kw_args={'method': 'LocalOutlierFactor'}),
        'inlierEllip':
        FunctionSampler(outlier_rejection,
                        kw_args={'method': 'EllipticEnvelope'}),
        'inlierOsvm':
        FunctionSampler(outlier_rejection, kw_args={'method': 'OneClassSVM'}),
        # combine
        'smoteenn':
        SMOTEENN(),
        'smotelink':
        SMOTETomek(),
    }

    scale = {
        'stdscale': StandardScaler(),
        'maxscale': MinMaxScaler(),
        'rscale': RobustScaler(quantile_range=(10, 90)),
        'qauntile': QuantileTransformer(),  # uniform distribution
        'power': PowerTransformer(),  # Gaussian distribution
        'norm': Normalizer(),  # default L2 norm

        # scale sparse data
        'maxabs': MaxAbsScaler(),
        'stdscalesp': StandardScaler(with_mean=False),
    }
    # feature construction
    feature_c = {
        'pca':  PCA(whiten=True),
        'spca': SparsePCA(normalize_components=True, n_jobs=-1),
        'ipca': IncrementalPCA(whiten=True),
        'kpca': KernelPCA(kernel='rbf', n_jobs=-1),
        'poly': PolynomialFeatures(degree=2),
        'rbf' : RBFSampler(random_state=0),
        'Nys' : Nystroem(random_state=0),
        'rtembedding': RandomTreesEmbedding(n_estimators=10),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),
    }
    # select from model
    feature_m = {
        'fwoe':
        SelectFromModel(Woe_encoder(max_leaf_nodes=8)),
        'flog':
        SelectFromModel(LogisticRegression(penalty='l1', solver='saga',
                                           C=1e-2)),
        'fsgd':
        SelectFromModel(SGDClassifier(penalty="l1")),
        'fsvm':
        SelectFromModel(LinearSVC('l1', dual=False, C=1e-2)),
        'fxgb':
        SelectFromModel(XGBClassifier(n_jobs=-1)),
        'frf':
        SelectFromModel(ExtraTreesClassifier(n_estimators=100, max_depth=5)),

        # fixed number of features
        'fxgb20':
        SelectFromModel(XGBClassifier(n_jobs=-1), max_features=20),
        'frf20':
        SelectFromModel(ExtraTreesClassifier(n_estimators=100, max_depth=5),
                        max_features=20),
        'frf10':
        SelectFromModel(ExtraTreesClassifier(n_estimators=100, max_depth=5),
                        max_features=10),
        'flog20':
        SelectFromModel(LogisticRegression(penalty='l1', solver='saga',
                                           C=1e-2),
                        max_features=20),
        'fRFE30log':
        RFE(LogisticRegression(penalty='l1', solver='saga', C=1e-2),
            step=0.3,
            n_features_to_select=30),
        'fRFE20log':
        RFE(LogisticRegression(penalty='l1', solver='saga', C=1e-2),
            step=0.3,
            n_features_to_select=20),
        'fRFE10log':
        RFE(LogisticRegression(penalty='l1', solver='saga', C=1e-2),
            step=0.3,
            n_features_to_select=10)
    }
    # Univariate feature selection
    feature_u = {
        'fchi2':
        GenericUnivariateSelect(chi2, 'percentile', 25),
        'fMutualclf':
        GenericUnivariateSelect(mutual_info_classif, 'percentile', 25),
        'fFclf':
        GenericUnivariateSelect(f_classif, 'percentile', 25),
    }
    # sklearn estimator
    t = all_estimators(type_filter=['classifier'])
    estimator = {}
    for i in t:
        try:
            estimator.update({i[0]: i[1]()})
        except Exception:
            continue

    estimator.update(
        dummy=DummyClassifier(),
        XGBClassifier=XGBClassifier(n_jobs=-1),
        LogisticRegressionCV=LogisticRegressionCV(scoring='roc_auc'),
        EasyEnsembleClassifier=EasyEnsembleClassifier(),
        BalancedRandomForestClassifier=BalancedRandomForestClassifier(),
        RUSBoostClassifier=RUSBoostClassifier(),
        SVC=SVC(C=0.1, gamma='auto'))

    if pipe is None:
        feature_s = {}
        feature_s.update(**feature_m, **feature_u)
        return {
            'clean': clean.keys(),
            'encoding': encode.keys(),
            'resample': resample.keys(),
            'scale': scale.keys(),
            'feature_c': feature_c.keys(),
            'feature_s': feature_s.keys(),
            'classifier': estimator.keys()
        }
    elif isinstance(pipe, str):
        l = pipe.split('_')
        all_keys_dict = {}
        all_keys_dict.update(**clean, **encode, **scale, **feature_c,
                             **feature_m, **feature_u, **estimator, **resample)
        steps = []
        for i, j in zip(l, index_duplicated(l)):
            if all_keys_dict.get(i) is not None:
                steps.append((j, all_keys_dict.get(i)))
            else:
                raise KeyError(
                    "'{}' invalid key for sklearn estimators".format(i))
        return Pipeline(steps)

    else:
        raise ValueError("input pipe must be a string in format 'xx[_xx]'")


def pipe_grid(estimator, pipe_grid=True):
    '''return pre-defined param_grid of given estimator
    
    estimator
        - str or sklearn estimator instance
    pipe_grid
        - bool, if False return param_grid; True return param_grid as embedded
        in pipeline    
    '''
    if isinstance(estimator, str):
        keys = estimator
    else:
        keys = estimator.__class__.__name__

    param_grid = _param_grid(keys)

    if param_grid is None:
        return

    if pipe_grid is True:
        return [{'__'.join([keys, k]): i.get(k)
                 for k in i.keys()} for i in param_grid if api.is_dict_like(i)]
    else:
        return param_grid


def _param_grid(estimator):
    '''    
    estimator:
        str for sklearn estimator's name
    return
    ----
        param_grid dict
    '''

    LogisticRegression = [{'C': np.logspace(-2, 0, 5)}]

    SGDClassifier = [
        {
            'loss': [ 'log', 'hinge', 'perceptron', 'modified_huber']
        },
        {
            'penalty': [ 'l1', 'l2','elasticnet'],
            'alpha' : np.logspace(-5, 0, 3)
        },
    ]
    
    SVC = [{
        'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
    }, {
        'gamma': np.logspace(-5, 5, 5),
    }, {
        'C': np.logspace(-5, 1, 5)
    }]
    
    XGBClassifier = [
        {
            'learning_rate': np.logspace(-3, 0, 5),
            'n_estimators': np.arange(50, 200, 20).astype(int),
        },
        {
            'scale_pos_weight': np.logspace(0, 1.5, 5)
        },
        {
            'max_depth': [2, 3, 4, 5]
        },
        {
            'gamma': np.logspace(-2, 1, 5)
        },
        {
            'reg_alpha': np.logspace(-2, 3, 3),
            'reg_lambda': np.logspace(-2, 3, 10)
        },
        {
            'colsample_bytree': [0.95, 0.9, 0.8, 0.75],
            'subsample': [0.95, 0.9, 0.8, 0.75],
        },
    ]

    AdaBoostClassifier = [
        {
            'learning_rate': np.logspace(-3, 0, 5),
        },
        {
            'n_estimators': np.logspace(1.5, 2.5, 5).astype(int),
        },
    ]

    RandomForestClassifier = [
        {
            'max_depth': range(3, 10),
            'min_samples_leaf': np.logspace(-5, -1, 5),
        },
        {
            'n_estimators': np.logspace(1.5, 2.5, 10).astype(int)
        },
    ]

    GradientBoostingClassifier = [{
        'max_depth': range(2, 5),
        'min_samples_leaf': np.logspace(-5, -1, 5),
    }, {
        'n_estimators':
        np.logspace(1.5, 2.5, 10).astype(int)
    }, {
        'subsample': [1, 0.9, 0.8, 0.75]
    }]

    GaussianProcessClassifier = [{
        'kernel': [ConstantKernel() * RBF(),
                   RationalQuadratic(),
                   Matern()],
    }]

    DecisionTreeClassifier = [{
        'max_depth': range(1, 4, 1),
        'min_samples_leaf': np.logspace(-3, -1, 4),
        'min_impurity_decrease': [1e-3],
    }]

    LabelPropagation = [
        {
            'kernel': ['rbf'],
            'gamma': np.logspace(-5, 1, 5)
        },
        {
            'kernel': ['knn'],
            'n_neighbors': np.logspace(0, 1.2, 5).astype(int)
        },
    ]

    kpca = [{
        'kernel': ['linear', 'sigmoid', 'rbf']
    }, {
        'alpha': np.logspace(0, 2, 5)
    }, {
        'gamma': np.logspace(-5, 0, 5)
    }]

    spca = [{
        'n_components': np.logspace(1.2, 2, 5).astype(int)
    }, {
        'alpha': np.logspace(-1, 3, 5)
    }]  
    
    Nys = [
            {'kernel' : ['linear', 'polynomial', 'sigmoid', 'rbf'],
             'gamma' : np.logspace(-5, 1, 5)}, 
    ]
    
    rbf = [{
            'gamma' : np.logspace(-5, 1, 8),
    }]

    param_grids = locals().copy()
    param_grids.update({
        'RUSBoostClassifier':
        param_grids.get('AdaBoostClassifier'),
        'BalancedRandomForestClassifier':
        param_grids.get('RandomForestClassifier'),
    })

    grid = param_grids.get(estimator)

    if grid is None:
        print("key '{}' not found, param_grid not returned".format(estimator))
    else:
        print("param_grid for '{}' returned as : \n".format(estimator))
        [print(i) for i in grid]
        print('...\n')

    return grid


def outlier_rejection(X=None,
                      y=None,
                      method='IsolationForest',
                      contamination=0.1):
    """This will be our function used to resample our dataset.
    """
    outlier_model = (
        IsolationForest(contamination=contamination),
        LocalOutlierFactor(contamination=contamination),
        OneClassSVM(nu=contamination),
        EllipticEnvelope(contamination=contamination),
    )

    outlier_model = {i.__class__.__name__: i for i in outlier_model}

    if X is None:
        return outlier_model.keys()
    model = outlier_model.get(method)
    if model is None:
        raise ValueError("method '{}' is invalid".format(method))
    y_pred = model.fit_predict(X)
    return X[y_pred == 1], y[y_pred == 1]


class Base_clean():
    '''base cleaner

    attributes
    -----
    out_labels
        - labels for transformed X columns
    input_labels
        - labels for original input X columns
        
    method
    -----
    _fit 
        - to perform before fit method, to store input_labels
    _filter_labels
        - to perform before transform method 
        filter only stored labels (self.input_labels) 
    _check_df
        - convert X to DataFrame, filter duplicated cols, try converting X
        to numeric or datetime or object dtype
    get_feature_names
        - return out_labels
    '''

    def _check_df(self, X):
        '''convert X to DataFrame, drop duplicated cols, try converting X
        to numeric or datetime or object dtype
        
        X
            - data X will be converted as DataFrame        
        return --> cleaned df
        '''
        try:
            X = pd.DataFrame(X)
        except Exception:
            raise ValueError('input must be DataFrame convertible')
        if X.empty:
            raise ValueError('X empty')
        X = to_num_datetime_df(self._drop_duplicated_cols(X))
        return X

    def _filter_labels(self, X):
        '''to perform before transform method 
        '''
        validation.check_is_fitted(self, ['input_labels'])
        # --filter input_labels
        X = self._check_df(X)
        X = X.reindex(columns=getattr(self, 'input_labels'))
        if X.isna().all(None):
            raise ValueError(
                'no X column matchs with transfromer input_labels')
        return X

    def _fit(self, X):
        '''to perform before fit method
        '''
        X = self._check_df(X)
        # -- store input_labels
        self.input_labels = X.columns.tolist()
        return X

    def _drop_duplicated_cols(self, X):
        '''drop duplicated cols 
        '''
        columns = X.columns
        if columns.is_unique:
            return X
        else:
            col_dup = columns[columns.duplicated('first')]
            if getattr(self, 'verbose') > 0:
                print("{} duplicated columns '{}' are dropped\n ".format(
                    len(col_dup), col_dup))
            return X.drop(columns=col_dup)

    def _raise_error(self):
        '''raise error if not fit
        '''
        raise ValueError('X not fitted, perform fit  method first')

    def get_feature_names(self, ):
        '''get out_labels, return input_labels if None 
        '''
        try:
            return getattr(self, 'out_labels', self.input_labels)
        except:
            self._raise_error()


class Split_cls(BaseEstimator, TransformerMixin, Base_clean):
    '''
    - clean(convert to numeric/str & drop na or uid columns); 
    - filter columns of specific dtypes; 
    - store input & output columns; 
    - drop all na/constant/UID columns 
    
    params
    ---- 

    dtype_filter -->  str, default not_datetime
        - num - filter only numeric dtype
        - obj - filter only obj dtype
        - datetime - filter only datetime dtype
        - not_datetime - exclude only datetime dtype
        - all - all dtypes
    na1
        - fill na strategy for categorical data column , default None
    na2
        - fill na stategy for numeric data column, default None
    na_thresh
        - int or float(0.0-1.0) thresh number of null values to drop column
    '''

    def __init__(self,
                 dtype_filter='not_datetime',
                 verbose=0,
                 na1=None,
                 na2=None,
                 na_thresh=1):
        ''' 
        '''
        L = locals().copy()
        L.pop('self')
        self.set_params(**L)

    def fit(self, X, y=None):
        '''fit input_labels & out_labels 
        '''
        X = self._fit(X)

        # drop na columns
        na_col = X.columns[X.apply(lambda x: all(x.isna()))]
        length = len(X)
        if api.is_integer(self.na_thresh):
            thresh = self.na_thresh
        if api.is_float(self.na_thresh):
            thresh = length * self.na_thresh

        X.dropna(axis=1, how='any', thresh=thresh, inplace=True)

        # drop uid cols or too discrete data
        uid_col = []
        for k, col in X.iteritems():
            if api.is_object_dtype(col):
                if len(pd.unique(col)) > 40:
                    X.drop(k, axis=1, inplace=True)
                    uid_col.append(k)
            elif api.is_integer_dtype(col):
                if len(pd.unique(col)) > 0.85 * len(col):
                    X.drop(k, axis=1, inplace=True)
                    uid_col.append(k)

        # drop constant
        const_col = []
        for k, col in X.iteritems():
            if (api.is_numeric_dtype(col) and col.std()<0.01) \
            or len(pd.unique(col))==1:
                X.drop(k, axis=1, inplace=True)
                const_col.append(k)

        # filter dtypes
        options = {
            'not_datetime': X.select_dtypes(exclude='datetime').columns,
            'number': X.select_dtypes(include='number').columns,
            'object': X.select_dtypes(include='object').columns,
            'datetime': X.select_dtypes(include='datetime').columns,
            'all': X.columns
        }

        self.objcols = options.get('object')
        self.numcols = options.get('number')
        self.datetimecols = options.get('datetime')
        self.obj_na = _get_imputer(self.na1)
        self.num_na = _get_imputer(self.na2)

        if self.obj_na is not None and not self.objcols.empty:
            self.obj_na.fit(X.reindex(columns=self.objcols))
        if self.num_na is not None and not self.numcols.empty:
            self.num_na.fit(X.reindex(columns=self.numcols))

        self.out_labels = options.get(self.dtype_filter).tolist()
        # --
        if len(na_col) > 0:
            print('{} ...\n total {} columns are null , have been dropped \n'.
                  format(na_col, len(na_col)))
        if len(uid_col) > 0:
            print(
                '''{} ...\n total {} columns are uid or has too many discrete 
                categories (>40) , have been dropped \n'''.format(
                    uid_col, len(uid_col)))
        if len(const_col) > 0:
            print(
                ''''{} ...\n total {} columns are constant , have been dropped
                \n'''.format(const_col, len(const_col)))

        if self.verbose > 0:
            for k, i in options.items():
                print('data has {} of {} columns'.format(len(i.columns), k))
            if len(na_col) > 0:
                print('null columns:\n {}'.format(list(na_col)))
        return self

    def transform(self, X):
        '''transform X to df of specified filter_dtype
        '''
        X = self._filter_labels(X)
        # --
        if self.obj_na is not None and not self.objcols.empty:
            obj = self.obj_na.transform(X.reindex(columns=self.objcols))
            obj = pd.DataFrame(obj, columns=self.objcols)
        else:
            obj = X.reindex(columns=self.objcols)

        if self.num_na is not None and not self.numcols.empty:
            num = self.num_na.transform(X.reindex(columns=self.numcols))
            num = pd.DataFrame(num, columns=self.numcols)
        else:
            num = X.reindex(columns=self.numcols)

        cols = set(self.out_labels)
        othercols = cols.difference(self.objcols.union(self.numcols))
        date_col = X.reindex(columns=othercols)

        X = pd.concat((i for i in [obj, num, date_col] if not i.empty), axis=1)
        X = X.reindex(columns=self.out_labels)

        return X


def to_num_datetime(col, name='array', thresh=0.80, **kwargs):
    '''convert col to numeric or datetime if possible, otherwise remain
    unchaged 
    
    parameters
    ----
    col --> series, scalar or ndarry will be turned into series type
    
    name --> name of the col series 
    
    thresh --> default 0.8 
        - if more than the thresh percentage of X could be converted, 
          then should commit conversion   
    **kwargs 
    
    - errors - {'ignore', 'raise', 'coerce'}, default --> 'coerce'
        - If 'raise', then invalid parsing will raise an exception
        - If 'coerce', then invalid parsing will be set as NaN
        - If 'ignore', then invalid parsing will return the input
    other pandas to_datetime key words
    
    return
    ----
    converted series or df
    '''
    try:
        col = pd.Series(col)
    except Exception:
        raise Exception('col must be 1-d array/list/tuple/dict/Series')

    if api.is_numeric_dtype(col):
        return col
    if api.is_datetime64_any_dtype(col):
        return col
    if api.is_categorical_dtype(col):
        return col
    if col.count() == 0:
        return col

    is_numeric_convertible = False
    not_null_count = col.count()

    try:
        num = pd.to_numeric(col, errors=kwargs.get('errors', 'coerce'))
        if num.count() / not_null_count >= thresh:
            col = num
            is_numeric_convertible = True
    except:
        pass
    if not is_numeric_convertible:
        params = {'errors': 'coerce', 'infter_datetime_format': True}
        params.update(kwargs)
        try:
            date = pd.to_datetime(col, **params)
            if pd.notnull(date).sum() / not_null_count >= thresh:
                col = date
        except:
            pass
    return col


def to_num_datetime_df(X, thresh=0.8):
    '''convert each column to numeric or datetime if possible, otherwise remain
    unchanged 
    
    thresh --> default 0.8 
        - if more than the thresh percentage of col could be converted, 
          then should commit conversion     
    '''
    try:
        X = pd.DataFrame(X)
    except Exception:
        raise ValueError('X must be df or convertible to df')
    lamf = lambda x: to_num_datetime(x, name=x.name, thresh=0.8)
    rst = X.apply(lamf, axis=0, result_type='reduce')
    return rst


class Woe_encoder(BaseEstimator, TransformerMixin, Base_clean):
    '''to woe_encode feature matrix using auto-binning based on CART tree
    gini impurity/bins or specified by input bin edges = {col : edges},
    calcualte woe & iv of each feature, NaN values will be binned independently
    
    parameters
    ------            
    input_edges={}
        - mannual input cutting edges as 
        {colname : [-inf, point1, point2..., inf]}
    cat_num_lim
        - number of unique vals limit to be treated as continueous feature,
        default 0
    max_leaf_nodes=5
        - max number of bins
    min_samples_leaf=0.01
        - minimum number of samples in leaf node
    min_samples_split=0.01
        - the minimun number of samles required to split a node       
    **tree_params
        - other decision tree keywords
        
    attributes
    -----
    edges 
        - dict={colname : [-inf, point1, point2..., inf]}; 
        - 'fit' method will try to get edges by decision Tree algorithm or
        pandas cut method
    woe_map
        - dict={colname : {category : woe, ...}}
    woe_iv
        - df, woe & iv of all features, concatenated in one df
    feature_importances_ 
        - iv value of each feature
      
    method
    -----
    fit 
        - calculate woe & iv values for each col categories, obtain 
        self edges & woe_map
    transform
        - to get woe encoded feature matrix using self woe_map
    score - not used as estimator method usually to study iv of each feature
        - return (woe_iv, woe_map, iv),  using self edges
    '''

    path_ = Path()

    def __init__(self,
                 input_edges={},
                 cat_num_lim=5,
                 q=None,
                 bins=None,
                 max_leaf_nodes=None,
                 min_samples_leaf=0.01,
                 min_samples_split=0.01,
                 criterion='gini',
                 min_impurity_decrease=1e-5,
                 min_impurity_split=None,
                 random_state=0,
                 splitter='best',
                 verbose=1):

        L = locals().copy()
        L.pop('self')
        self.set_params(**L)

    def _get_binned(self, X):
        '''to get binned matrix using self edges, cols without cutting edges
        will remain unchaged
        '''
        if self.edges is None:
            raise Exception('no bin edges, perform fit first')
        cols = []
        for name, col in X.iteritems():
            if name in self.edges:
                edges = self.edges.get(name)
                col_binned = pd.cut(col,
                                    edges,
                                    retbins=False,
                                    duplicates='drop')
                cols.append(col_binned)
            else:
                cols.append(col)
        return pd.concat(cols, axis=1)

    @property
    def feature_importances_(self):
        '''
        '''
        if hasattr(self, 'feature_iv'):

            print('''IV < 0.02 has been forced to 0 due to
                  low predictive power  value''')
            value = self.feature_iv
            iv = self.feature_iv.where(0.02 < value)
            return iv
        else:
            return

    def fit(self, X, y):
        '''fit X(based on CART Tree) to get cutting edges
        (updated by input edges) and  calculate woe & iv for each cat group
        categorical features will use category as group
        
        parameter
        ----
        X - df
        
        y - class label
        '''
        X = self._fit(X)
        # --
        params = get_kwargs(_woe_binning, **self.get_params())
        params.update(get_kwargs(DecisionTreeClassifier, **self.get_params()))
        self.edges = _woe_binning(X, y, **params)
        self.edges.update(self.input_edges)
        # --
        df_binned = self._get_binned(X)
        self.woe_iv, self.woe_map, self.feature_iv = calc_woe(df_binned, y)
        print(self.woe_iv)
        return self

    def transform(self, X):
        '''to get woe encoded X using self woe_map
        parameters
        ----
        X - df
        
        return
        ----
        df --> X woe encoded value
        '''
        X = self._filter_labels(X)
        # --
        woe_map = self.woe_map
        cols = []
        cols_notcoded = []
        for name, col in X.iteritems():
            if name in woe_map:
                mapper = woe_map.get(name)
                if mapper.get(np.nan) is not None:
                    na = mapper.pop(np.nan)
                    cols.append(col.map(mapper).fillna(na))
                else:
                    cols.append(col.map(mapper).fillna(0))
            else:
                cols_notcoded.append(col.name)

        if cols_notcoded:
            print("{} have not been woe encoded".format(cols_notcoded))

        return pd.concat(cols, axis=1)

    def plot_event_rate(self, save_path=None, suffix='.pdf', dw=0.02, up=0.5):
        '''return iv of each column using self.edges
        '''
        plotter_woeiv_event(self.woe_iv, save_path, suffix, dw, up)


def _tree_univar_bin(arr_x, arr_y, **kwargs):
    '''univariate binning based on binary decision Tree
    
    return
    ----
    ndarray of binning edges
    '''
    validation.check_consistent_length(arr_x, arr_y)
    clf = DecisionTreeClassifier(
        **get_kwargs(DecisionTreeClassifier, **kwargs))
    X = np.array(arr_x).reshape(-1, 1)
    Y = np.array(arr_y).reshape(-1, 1)

    # tree training
    clf.fit(X, Y)
    thresh = clf.tree_.threshold
    feature = clf.tree_.feature
    thresh = np.unique(thresh[(feature >= 0).nonzero()]).round(
        kwargs.get('decimal', 8))
    cut_edges = np.append(np.append(-np.inf, thresh), np.inf)
    return np.unique(cut_edges)


def _mono_cut(Y, X):
    '''return binning edges of X, 
    which increase monotonically with "y" mean value '''
    r = 0
    n = 10
    while np.abs(r) < 1 and n > 2:
        out, bins = pd.qcut(X, n, duplicates='drop', retbins=True)
        d1 = pd.DataFrame({
            "X": X,
            "Y": Y,
            "Bucket": pd.qcut(X, n, duplicates='drop')
        })
        d2 = d1.groupby('Bucket', as_index=False)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    bins[0] = -np.inf
    bins[-1] = np.inf
    return bins


def bin_tree(X,
             y,
             cat_num_lim=0,
             max_leaf_nodes=10,
             min_samples_leaf=0.05,
             random_state=0,
             verbose=0,
             **kwargs):
    '''discrete features based on univariate run of DecisionTree classifier
    (CART tree - gini impurity as criterion, not numeric dtype will be igored,
    unique number of values less than "cat_num_lim" will be ignored)
    
    df_X 
        - df, contain feature matrix, should be numerical dtype
    y 
        - col of class label, binary
    cat_num_lim=10
        - number of unique vals limit to be treated as continueous feature
    max_leaf_nodes=5
        - max number of bins
    min_samples_leaf=0.1
        - minimum number of samples in leaf node
    **kwargs
        - other tree keywords
    
    return
    ----
    bin_edges
        - dict of {'col_name' : bin_edges }
    '''

    bin_edges = {}
    cols = []
    un_split = []
    for name, col in X.iteritems():
        df = pd.DataFrame({'x': col, 'y': y})
        col_notna = df.dropna().x
        y_notna = df.dropna().y
        if (len(pd.unique(col_notna)) > cat_num_lim
                and api.is_numeric_dtype(col_notna)):
            # call _tree_univar_bin
            bin_edges[name] = _tree_univar_bin(
                col_notna,
                y_notna,
                max_leaf_nodes=max_leaf_nodes,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                **get_kwargs(DecisionTreeClassifier, **kwargs))
            if len(bin_edges[name]) < 3:
                un_split.append(name)
        else:
            cols.append(name)

    if verbose > 0:
        msg1 = '''total of {2} unchaged (unique counts less 
               than {1} or categorical dtype) =\n "{0}" 
               '''.format(pd.Index(cols), cat_num_lim, len(cols))
        msg2 = '''total of {1} unsplitable features = \n {0} ...
               '''.format(pd.Index(un_split), len(un_split))
        msg3 = 'total of {} bin_edges obtained \n'.format(len(bin_edges))
        if cols:
            print(msg1)
        if un_split:
            print(msg2)
        if bin_edges:
            print(msg3)

    return bin_edges


def _binning(y_pre=None,
             bins=None,
             q=None,
             max_leaf_nodes=None,
             y_true=None,
             labels=None,
             **kwargs):
    '''supervised binning of y_pre based on y_true if y_true is not None
    
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
        
    return 
    -----
    y_bins:       
         bins of y_pre 
    bins:
         ndarray of bin edges

    '''
    if sum((bins is None, q is None, max_leaf_nodes is None)) != 2:
        raise ValueError(
            'must and only 1 of (q, bins, max_leaf_nodes) can be specified')

    if q is not None:
        bins = np.percentile(y_pre, np.linspace(0, 100, q + 1))
        bins[0] = -np.Inf
        bins[-1] = np.Inf

    if max_leaf_nodes is not None:
        if y_true is None:
            raise ValueError('y_true must be supplied tree cut')
        y_pre0 = pd.DataFrame(y_pre)
        bins_dict = bin_tree(y_pre0,
                             y_true,
                             max_leaf_nodes=max_leaf_nodes,
                             **kwargs)
        bins = list(bins_dict.values())[0]

    if isinstance(bins, int):
        bins = np.linspace(np.min(y_pre), np.max(y_pre), bins + 1)
        bins[0] = -np.inf
        bins[-1] = np.Inf

    if bins is None:
        raise ValueError('no cutting bins supplied')

    if labels is True:
        labels = None
    y_bins, bins = pd.cut(y_pre,
                          bins,
                          duplicates='drop',
                          retbins=True,
                          labels=labels)
    return y_bins, bins


def _woe_binning(X,
                 y,
                 q=None,
                 bins=None,
                 max_leaf_nodes=None,
                 cat_num_lim=0,
                 **kwargs):
    '''use by Woe_encoder to get binning edges
    
    return
    ----
    edges:
        {colname : [-inf, point1, point2..., inf]}
    '''
    bin_edges = {}
    for name, col in X.iteritems():
        df = pd.DataFrame({'x': col, 'y': y})
        col_notna = df.dropna().x
        y_notna = df.dropna().y
        if (len(pd.unique(col_notna)) > cat_num_lim \
            and api.is_numeric_dtype(col_notna)):
            label, bin_edges[name] = _binning(col_notna, bins, q,
                                              max_leaf_nodes, y_notna,
                                              **kwargs)
    return bin_edges


@dec_iferror_getargs
def _single_woe(X, Y, var_name='VAR'):
    '''calculate woe and iv for single binned X feature, with binary Y target
    
    - y=1 event; y=0 non_event 
    - na value in X will be grouped independently
    
    return
    ----
    df, of WOE, IVI and IV_SUM ...
    '''
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X', 'Y']][df1.X.isna()]
    notmiss = df1[['X', 'Y']][df1.X.notna()]
    df2 = notmiss.groupby('X', as_index=True)

    d3 = pd.DataFrame({}, index=[])
    d3["COUNT"] = df2.count().Y
    d3["CATEGORY"] = df2.sum().index
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y

    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({
            "COUNT": [justmiss.count().Y],
            "EVENT": [justmiss.sum().Y],
            "NONEVENT": [justmiss.count().Y - justmiss.sum().Y]
        })
        d3 = pd.concat([d3, d4], axis=0, ignore_index=True, sort=True)

    # add 1 when event or nonevent count equals 0
    dc = d3.copy()
    dc.EVENT.replace(0, 1, True)
    dc.NONEVENT.replace(0, 1, True)
    dc["EVENT_RATE"] = dc.EVENT / dc.COUNT
    dc["NON_EVENT_RATE"] = dc.NONEVENT / dc.COUNT
    dc["DIST_EVENT"] = dc.EVENT / dc.sum().EVENT
    dc["DIST_NON_EVENT"] = dc.NONEVENT / dc.sum().NONEVENT
    # add 1 when event or nonevent count equals 0

    d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
    d3["WOE"] = np.log(dc.DIST_EVENT / dc.DIST_NON_EVENT)
    d3["IV"] = (dc.DIST_EVENT - dc.DIST_NON_EVENT) * np.log(
        dc.DIST_EVENT / dc.DIST_NON_EVENT)

    d3["FEATURE_NAME"] = var_name
    d3 = d3[[
        'FEATURE_NAME', 'CATEGORY', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT',
        'NON_EVENT_RATE', 'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV'
    ]]

    d3['IV_SUM'] = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    return d3


def calc_woe(df_binned, y):
    '''calculate woe and iv 
    
    df_binned
        - binned feature_matrix
    y
        - binary 'y' target   
    
    return
    ----
    df_woe_iv =  [
            'VAR_NAME','CATEGORY', 'COUNT', 'EVENT', 'EVENT_RATE',
            'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT',
            'WOE', 'IV' ]
    
    woe_map = {'colname' : {category : woe}}
    
    iv series
        - colname--> iv 
    '''

    l = []
    woe_map = {}
    iv = []
    var_names = []
    for name, col in df_binned.iteritems():
        col_iv = _single_woe(col, y, name)
        l.append(col_iv)
        woe_map[name] = dict(col_iv[['CATEGORY', 'WOE']].values)
        iv.append(col_iv.IV.sum())
        var_names.append(name)

    # concatenate col_iv
    woe_iv = pd.concat(l, axis=0, ignore_index=True)
    print('---' * 20)
    print('total of {} cols get woe & iv'.format(len(l)))
    print('---' * 20, '\n\n')
    return woe_iv, woe_map, pd.Series(iv, var_names)


class Oht_encoder(BaseEstimator, TransformerMixin, Base_clean):
    ''' 
    - transform categorical features to  one-hot encoded; 
    
    parameters
    -----
    handle_unknown 
        - default 'ignore', for one-hot encoding, unknown feature category
          will be treated as zeros, 'raise' error encountered unknow category
    sparse
        - default False, for one-hot encoding, which will return 2D arrays

    strategy 
        - The imputation strategy."mean"/"median"/"most_frequent"/"constant"
  
    attributes
    ----       
    encoder
        - sklearn transformer instance
    encode_mapper - categories mapper of each column
        - dict egg. {cloname : array(category names)}, 
    '''

    def __init__(self,
                 sparse=False,
                 dtype=np.float64,
                 handle_unknown='ignore',
                 n_values=None,
                 categorical_features=None,
                 categories=None,
                 drop=None):
        '''
        '''
        L = locals().copy()
        L.pop('self')
        self.set_params(**L)

    def _check_categories(self, X):
        '''check if feature category are out of categories scope, treat them as 
        null
        '''
        if len(self.encode_mapper) == 0: return X
        #
        mapper = self.encode_mapper
        isin_cat = X.apply(lambda x: x.isin(mapper.get(x.name, x)) | x.isna(),
                           axis=0)
        out_c = np.ravel(~isin_cat).sum()
        if out_c > 0:
            print('''total of {} element out of categories and 
                  will be treated as np.nan '''.format(out_c))
        return X

    def fit(self, X, y=None):
        '''fit df to get categorical feature using ordinal & one-hot encoder 
        '''
        X = self._fit(X)
        self.obj_cols = X.select_dtypes('object').columns
        self.not_obj = X.columns.difference(self.obj_cols)

        self.encoder = OneHotEncoder(**self.get_params())
        self.encoder.fit(X.reindex(columns=self.obj_cols))

        self.encoder_fnames = self.encoder.get_feature_names(self.obj_cols)
        self.encode_mapper = dict(zip(self.obj_cols, self.encoder.categories_))
        self.out_labels = self.encoder_fnames.tolist() + self.not_obj.tolist()

        return self

    def transform(self, X):
        '''transform categorical columns in X  to dummy variable , other columns 
        remain unchaged
        '''
        X = self._filter_labels(X)
        # --obj cols
        X0 = X.reindex(columns=self.obj_cols)
        X0 = self._check_categories(X0)
        X0 = self.encoder.transform(X0)
        X0 = pd.DataFrame(X0, columns=self.encoder_fnames)
        # --not obj do nothing
        X1 = X.reindex(columns=self.not_obj)
        rst = pd.concat((i for i in [X0, X1] if not i.empty), axis=1)
        rst = rst.reindex(columns=self.out_labels)
        return rst


class Ordi_encoder(BaseEstimator, TransformerMixin, Base_clean):
    ''' 
    - transform categorical features to ordinal encoded; 
    
    parameters
    -----
    handle_unknown 
        - default 'ignore', for one-hot encoding, unknown feature category
          will be treated as zeros, 'raise' error encountered unknow category
    sparse
        - default False, for one-hot encoding, which will return 2D arrays

    strategy 
        - The imputation strategy."mean"/"median"/"most_frequent"/"constant"
  
    attributes
    ----       
    encoder
        - sklearn transformer instance
    encode_mapper - categories mapper of each column
        - dict egg. {cloname : array(category names)}, 
    '''

    def __init__(self, categories='auto', dtype=np.float64):
        '''
        '''
        L = locals().copy()
        L.pop('self')
        self.set_params(**L)

    def _check_categories(self, X):
        '''check if feature category are out of categories scope, treat them as 
        null
        '''
        if len(self.encode_mapper) == 0: return X
        #
        mapper = self.encode_mapper
        isin_cat = X.apply(lambda x: x.isin(mapper.get(x.name, x)) | x.isna(),
                           axis=0)
        out_c = np.ravel(~isin_cat).sum()
        if out_c > 0:
            print('''total of {} element out of categories and 
                  will be treated as np.nan '''.format(out_c))
        X = X.where(isin_cat)
        return X

    def fit(self, X, y=None):
        '''fit df to get categorical feature using ordinal & one-hot encoder 
        '''
        X = self._fit(X)
        self.obj_cols = X.select_dtypes('object').columns
        self.not_obj = X.columns.difference(self.obj_cols)

        self.encoder = OrdinalEncoder(**self.get_params())
        self.encoder.fit(X.reindex(columns=self.obj_cols))
        self.encoder_fnames = self.obj_cols
        self.encode_mapper = dict(zip(self.obj_cols, self.encoder.categories_))
        self.out_labels = self.encoder_fnames.tolist() + self.not_obj.tolist()

        return self

    def transform(self, X):
        '''transform categorical columns in X  to dummy variable , other columns 
        remain unchaged
        '''
        X = self._filter_labels(X)
        # --obj cols
        X0 = X.reindex(columns=self.obj_cols)
        X0 = self._check_categories(X0)
        X0 = self.encoder.transform(X0)
        X0 = pd.DataFrame(X0, columns=self.encoder_fnames)
        # --not obj do nothing
        X1 = X.reindex(columns=self.not_obj)
        rst = pd.concat((i for i in [X0, X1] if not i.empty), axis=1)
        rst = rst.reindex(columns=self.out_labels)
        return rst


#def re_fearturename(estimator):
#    '''return featurenames of an estimator wrapped in a pipeline
#    '''
#    if isinstance(estimator, Pipeline):
#        fn = None
#        su = None
#        steps = estimator.steps
#        n = len(steps) - 1
#        while n > -1:
#            n -= 1
#            tr = steps[n][1]
#            if hasattr(tr, 'get_feature_names'):
#                fn = tr.get_feature_names()
#                if fn is not None: break
#            if hasattr(tr, 'get_support'):
#                su = tr.get_support()
#
#        if fn is None:
#            print('estimator has no feature_names attribute')
#            return
#        if su is not None:
#            fn = pd.Series(fn)[su]
#
#    return fn


def selected_fearturename(estimator):
    '''return featurenames of an estimator wrapped in a pipeline
    '''
    if isinstance(estimator, Pipeline):
        fn = None
        su = None
        steps = estimator.steps
        n = len(steps)
        i = 0
        while i < n:
            tr = steps[i][1]
            if hasattr(tr, 'get_feature_names'):
                fn = pd.Series(tr.get_feature_names())
            if hasattr(tr, 'get_support'):
                su = tr.get_support()
                fn = fn[su]
            i += 1

        if fn is None:
            print('estimator has no feature_names attribute')
            return
    return fn


def _get_imputer(imput):
    '''
    '''
    if api.is_number(imput):
        imputer = SimpleImputer(strategy='constant', fill_value=imput)
    elif imput == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif imput == 'most_frequent':
        imputer = SimpleImputer(strategy='most_frequent')
    elif isinstance(imput, str):
        imputer = SimpleImputer(strategy='constant', fill_value=imput)
    else:
        return

    return imputer


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
    y_cut, bins = _binning(y_pre,
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


def plotter_woeiv_event(woe_iv, save_path=None, suffix='.pdf', dw=0.02,
                        up=1.0):
    '''plot event rate for given woe_iv Dataframe
    see woe_encoder attribute woe_iv
    '''
    n = 0
    for keys, gb in woe_iv.groupby('FEATURE_NAME'):
        if (gb['IV'].sum() > dw) and (gb['IV'].sum() < up):
            plot_data = gb[['CATEGORY', 'EVENT_RATE', 'COUNT']]
            plot_data.columns = [keys, 'EVENT_RATE', 'COUNT']
            plotter_rateVol(plot_data.sort_values(keys))
            if save_path is not None:
                path = '/'.join([save_path, keys + suffix])
                plt.savefig(path, dpi=100, frameon=True)
            n += 1
            print('(%s)-->\n' % n)
            plt.show()
            plt.close()

    return


def ks_score(y_true, y_pred, pos_label=1):
    '''return K-S score of preditions
    '''
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=pos_label)
    ks = (tpr - fpr).max()
    return ks


def psi_score(act, ex):
    '''return psi score
    
    act:
        array of actual ratios
    ex:
        array of expected ratios
        
    return
    ----
    psi score
    '''
    act = np.array(act)
    ex = np.array(ex)
    ex = np.where(ex == 0, 0.0001, ex)
    if len(act) != len(ex):
        raise ValueError("length of 'act' and 'ex' must match")
    delta = act - ex
    ln = np.log(act / ex)
    return np.sum(delta * ln)


def get_custom_scorer():
    ''' return custom scorer dict
    '''
    scorer_dict = {}
    scorer_dict['KS'] = make_scorer(ks_score, needs_threshold=True)

    return scorer_dict
