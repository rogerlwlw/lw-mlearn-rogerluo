# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:36:20 2018

@author: roger luo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from pandas.core.dtypes import api

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OrdinalEncoder, OneHotEncoder)
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.utils import validation

from sklearn_pandas import DataFrameMapper
from xgboost.sklearn import XGBClassifier

from lw_mlearn.utilis import (dec_iferror_getargs, get_kwargs)
from lw_mlearn.plotter import plotter_rateVol
from lw_mlearn.read_write import Path


def pipe_main(pipe_name):
    ''' return pipeline instance by given key names
    
    pipe_name: 
        str        
    transformer:
       ' woe' --> study iv & woe of each feature, plot lift curve of 
       each feature   
    estimator:
        'xgb' --> return xgboost estimator pipeline
        
        'dummy' --> return dummy estimator pipeline
        
    .. note::
        all pipe_estimator starts with clean_oht/clean_ordi transformer which
        cleans data and encodes categorical features
        
    '''
    # --transformer
    oht = Cat_encoder(encode_type='oht')
    ordi = Cat_encoder(encode_type='ordi')
    notdate_dtype = Split_cls()

    # -- to start with
    clean_oht = [('clean', notdate_dtype), ('cat_enc', oht)]
    clean_ordi = [('clean', notdate_dtype), ('cat_enc', ordi)]
    # --feature select from svc model with l1 penalty
    woe = Pipeline([('clean', notdate_dtype), ('woe', Woe_encoder())])
    pca = PCA(n_components='mle')
    l1_s = SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False))
    xgb_s = SelectFromModel(XGBClassifier(n_jobs=-1), threshold=1e-5)

    # --estimator
    dummy = DummyClassifier(random_state=0)
    cart = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.02)
    LRcv = LogisticRegressionCV(cv=5)
    LR = LogisticRegression()
    xgb = XGBClassifier(n_jobs=-1)

    # --pipeline
    pipe_dummy = Pipeline(clean_oht + [('dummy', dummy)])
    
    pipe_xgbs_xgb = Pipeline(clean_oht + [('feature_xgb', xgb_s), ('xgb',
                                                                   xgb)])
    pipe_l1s_xgb = Pipeline(clean_oht + [('feature_l1', l1_s), ('xgb', xgb)])
    
    pipe_xgb = Pipeline(clean_oht + [('xgb', xgb)])
    pipe_cart = Pipeline(clean_oht + [('cart', cart)])

    pipe_LR = Pipeline(clean_oht + [('LR', LR)])
    pipe_l1s_LRcv = Pipeline(clean_oht + [('feature_l1', l1_s), ('LR', LRcv)])

    # --
    pipe = locals().get(pipe_name)
    if pipe is None:
        raise ValueError('no estimator returned')
    return pipe


class Base_clean():
    '''base cleaner

    X
        - data X will be converted as DataFrame
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
    '''filter columns of specific dtypes, store input & output columns  
    
    params
    ---- 

    dtype_filter -->  str, default not_datetime
        - num - filter only numeric dtype
        - obj - filter only obj dtype
        - datetime - filter only datetime dtype
        - not_datetime - exclude only datetime dtype
        - otherwise - all dtypes
    na
        - fill na with 'na' value, -999 default
    ----
    '''

    def __init__(self, dtype_filter='not_datetime', verbose=0):
        ''' '''
        L = locals().copy()
        L.pop('self')
        self.set_params(**L)

    def fit(self, X, y=None):
        '''fit input_labels & out_labels 
        '''
        X = self._fit(X)
        # drop na columns
        na_col = X.columns[X.apply(lambda x: all(x.isna()))]
        X.dropna(axis=1, how='all', inplace=True)
        # --
        options = {
            'not_datetime': X.select_dtypes(exclude='datetime'),
            'number': X.select_dtypes(include='number'),
            'object': X.select_dtypes(include='object'),
            'datetime': X.select_dtypes(include='datetime'),
            'all': X
        }

        self.out_labels = options.get(self.dtype_filter).columns.tolist()
        # --
        if len(na_col) > 0:
            print('{} of columns are null , have been dropped \n'.format(
                len(na_col)))

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
    '''to encode feature matrix using auto-binning based on CART tree
    gini impurity/bins or specified by edges = {col : edges}, calcualte woe &
    iv of each feature
    
    params:               
        input_edges={}
            - mannual input cutting edges as 
            {colname : [-inf, point1, point2..., inf]}
        cat_num_lim=10
            - number of unique vals limit to be treated as continueous feature
        max_leaf_nodes=5
            - max number of bins
        min_samples_leaf=0.05
            - minimum number of samples in leaf node
        min_samples_split=0.08
            - the minimun number of samles required to split a node       
        **tree_params
            - other decision tree keywords
        
    attributes
    ----
    edges 
        - dict={colname : [-inf, point1, point2..., inf]}; 
        - 'fit' method will try to get edges by decision Tree algorithm
    woe_map
        - dict={colname : {category : woe, ...}}
    woe_iv
        - df, woe & iv of all features, concatenated in one df
    feature_importances_ 
        - iv value of each feature
      
    method
    ----
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
                 cat_num_lim=10,
                 input_edges={},
                 q=None,
                 bins=None,
                 max_leaf_nodes=5,
                 min_samples_leaf=0.1,
                 min_samples_split=0.1,
                 criterion='gini',
                 min_impurity_decrease=0.005,
                 min_impurity_split=None,
                 random_state=0,
                 splitter='best',
                 verbose=1):

        L = locals().copy()
        L.pop('self')
        self.set_params(**L)

    def _get_binned(self, X):
        '''to get binned matrix using self edges, 
        cols without cutting edges will remain unchaged
        '''
        if self.edges is None:
            raise Exception('no bin edges, perform fit first')
        cols = []
        for name, col in X.iteritems():
            if name in self.edges:
                edges = self.edges.get(name)
                col_binned = pd.cut(
                    col, edges, retbins=False, duplicates='drop')
                cols.append(col_binned)
            else:
                cols.append(col)
        return pd.concat(cols, axis=1)

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
        self.woe_iv, self.woe_map, self.feature_importances_ = calc_woe(
            df_binned, y)
        print(self.woe_iv[['FEATURE_NAME', 'CATEGORY', 'WOE', 'IV_SUM']], '\n')
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

        def _mapping(x, mapper):
            try:
                if x in mapper: return mapper.get(x)
                if pd.isna(x): return mapper.get(np.nan)
                for k, v in mapper.items():
                    if x in k: return v
            except Exception as e:
                raise Exception
                print(e, x, k, v)

        X = self._filter_labels(X)
        # --
        woe_map = self.woe_map
        cols = []
        cols_notcoded = []

        for name, col in X.iteritems():
            if name in woe_map:
                mapper = woe_map.get(name)
                cols.append(col.apply(_mapping, mapper=mapper))
            else:
                cols_notcoded.append(col.name)

        if cols_notcoded:
            print("{} have not been woe encoded".format(cols_notcoded))
        return pd.concat(cols, axis=1)

    def score(self, X, y):
        '''return iv of each column using self.edges
        '''
        X = self._filter_labels(X)
        # --
        df_binned = self._get_binned(X)
        woe_iv, woe_map, iv_series = calc_woe(df_binned, y)
        return woe_iv, woe_map, iv_series.sort_values()

    def plot_EventRate(self, X=None, y=None, save_path=None, suffix='.pdf'):
        '''plot event rate vs category plus counts/volumes, using self.edges()
        
        suffix
            - most backends support png, pdf, ps, eps and svg.
        '''
        if all([X, y]):
            X = self._filter_labels(X)
            # --
            df_binned = self._get_binned(X)
            woe_iv, woe_map, iv_series = calc_woe(df_binned, y)
        else:
            woe_iv = self.woe_iv

        n = 0
        for keys, gb in woe_iv.groupby('FEATURE_NAME'):
            plot_data = gb[['CATEGORY', 'EVENT_RATE', 'COUNT']]
            plot_data.columns = [keys, 'EVENT_RATE', 'COUNT']
            plotter_rateVol(plot_data.sort_values(keys))
            if save_path:
                self.path_ = save_path
                path = '/'.join([self.path_, keys + suffix])
                plt.savefig(path, dpi=100, frameon=True)
            n += 1
            print('(%s)-->\n' % n)
            yield plt.show()
            plt.close()


@dec_iferror_getargs
def _tree_univar_bin(arr_x, arr_y, **kwargs):
    '''univariate binning based on binary decision Tree
    return
    ----
    ndarray of binning edges
    '''  
    validation.check_consistent_length(arr_x, arr_y)
    clf = DecisionTreeClassifier(**get_kwargs(DecisionTreeClassifier, 
                                              **kwargs))
    X = np.array(arr_x).reshape(-1, 1)
    Y = np.array(arr_y).reshape(-1, 1)

    # tree training
    clf.fit(X, Y)
    thresh = clf.tree_.threshold
    feature = clf.tree_.feature
    thresh = np.unique(thresh[(feature >= 0).nonzero()]).round(
        kwargs.get('decimal', 4))
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
             cat_num_lim=5,
             max_leaf_nodes=10,
             min_samples_leaf=0.05,
             random_state=0,
             verbose=1,
             **kwargs):
    '''discrete features based on univariate run of DecisionTree classifier
    (CART tree - gini impurity as criterion, not numeric dtype will be igored,
    unique number of vals less than "cat_num_lim" will be ignored)
    
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

    print('---' * 20)
    print('begin fit binning_tree...')

    bin_edges = {}
    cols = []
    un_split = []
    for name, col in X.iteritems():
        col_notna = col.dropna()
        y_notna = y[col_notna.index]
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
    msg1 = '''total of {2} unchaged (unique counts less 
              than {1} or categorical dtype) =\n "{0}" 
           '''.format(pd.Index(cols), cat_num_lim, len(cols))
    msg2 = '''total of {1} unsplitable features = \n {0} ...
           '''.format(pd.Index(un_split), len(un_split))
    msg3 = 'total of {} bin_edges obtained'.format(len(bin_edges))
    if verbose > 0:
        if cols:
            print(msg1)
        if un_split:
            print(msg2)
        if bin_edges:
            print(msg3)
    print('complete ...\n')

    return bin_edges

def _binning(y_pre=None,
             bins=None,
             q=None,
             max_leaf_nodes=None,
             y_true=None,
             labels=None,
             **kwargs):
    '''  
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
        bins = np.percentile(y_pre, np.linspace(0, 100, q + 1)).round(4)
        bins[0] = -np.Inf
        bins[-1] = np.Inf

    if max_leaf_nodes is not None:
        if y_true is None:
            raise ValueError('y_true must be supplied tree cut')
        y_pre0 = pd.DataFrame(y_pre)
        bins_dict = bin_tree(
            y_pre0, y_true, max_leaf_nodes=max_leaf_nodes, **kwargs)
        bins = list(bins_dict.values())[0]

    if isinstance(bins, int):
        bins = np.linspace(np.min(y_pre), np.max(y_pre), bins + 1)
        bins[0] = -np.inf
        bins[-1] = np.Inf

    if bins is None:
        raise ValueError('no cutting bins supplied')

    if labels is True:
        labels = None
    y_bins, bins = pd.cut(
        y_pre, bins, duplicates='drop', retbins=True, labels=labels)
    return y_bins, bins

def _woe_binning(X, y, q=None, bins=None, max_leaf_nodes=None,
                 cat_num_lim=5, **kwargs):
    '''use by Woe_encoder
    '''    
    bin_edges = {}
    for name, col in X.iteritems():
        col_notna = col.dropna()
        y_notna = y[col_notna.index]
        if (len(pd.unique(col_notna)) > cat_num_lim
                and api.is_numeric_dtype(col_notna)):
            label, bin_edges[name] = _binning(
                    col_notna, bins, q, max_leaf_nodes, y_notna, **kwargs)
    return bin_edges


@dec_iferror_getargs
def _single_woe(X, Y, var_name='VAR'):
    '''calculate woe and iv for single binned feature X, with binary Y target
    
    - y=1 event; y=0 non_event 
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

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
        d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
        d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
        d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
        d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
        d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(
            d3.DIST_EVENT / d3.DIST_NON_EVENT)

    d3["FEATURE_NAME"] = var_name
    d3 = d3[[
        'FEATURE_NAME', 'CATEGORY', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT',
        'NON_EVENT_RATE', 'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV'
    ]]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3['IV_SUM'] = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    return d3


def calc_woe(df_binned, y):
    '''calculate woe and iv for binned feature_matrix 'df_binned', bins/category 
    will be str dtype (contained as df with a binary 'y' target)
        - woe = 0 if event/non_event = 0
    
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

    print('---' * 20)
    print("begin woe calculation ...")
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
    print('total of {} cols get woe & iv'.format(len(l)))
    print('---' * 20, '\n\n')
    return woe_iv, woe_map, pd.Series(iv, var_names)


class Cat_encoder(BaseEstimator, TransformerMixin, Base_clean):
    ''' transform categorical features to ordinal or one-hot encoded, other 
    columns remain unchanged
    
    parameters
    -----
    handle_unknown 
        - default 'ignore', for one-hot encoding, unknown feature category
          will be treated as zeros, 'raise' error encountered unknow category
    sparse
        - default False, for one-hot encoding, which will return 2D arrays
    na0 
        - str value to fill null in input X, default 'null'
    na1
        - numeric value to fill null
    encode_type
        - 'ordi' ordinal encoding
        - 'oht' one_hot encoding
    strategy 
        - The imputation strategy."mean"/"median"/"most_frequent"/"constant"

    df_out bool default False
        - whether to output as df       
    attributes
    ----       
    encoder
        - sklearn transformer instance
    encode_mapper - categories mapper of each column
        - dict egg. {cloname : array(category names)}, 
    '''

    def __init__(self,
                 handle_unknown='ignore',
                 sparse=False,
                 encode_type='oht',
                 strategy='constant',
                 na0='null',
                 na1=-999,
                 df_out=False):
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
        isin_cat = X.apply(
            lambda x: x.isin(mapper.get(x.name, x)) | x.isna(), axis=0)
        out_c = np.ravel(~isin_cat).sum()
        if out_c > 0:
            print('''total of {} element out of categories and 
                  will be treated as np.nan '''.format(out_c))
        if self.encode_type == 'ordi':
            X = X.where(isin_cat)
        return X

    def fit(self, X, y=None):
        '''fit df to get categorical feature using ordinal & one-hot encoder 
        '''
        X = self._fit(X)
        obj_cols = X.columns[X.dtypes.apply(api.is_object_dtype)].tolist()
        not_obj = X.columns[~X.dtypes.apply(api.is_object_dtype)].tolist()
        # --
        if self.encode_type == 'ordi':
            encoder = OrdinalEncoder()

        if self.encode_type == 'oht':
            param = get_kwargs(OneHotEncoder, **self.get_params())
            encoder = OneHotEncoder(**param)

        imput = SimpleImputer(strategy=self.strategy, fill_value=self.na0)
        imput_n = SimpleImputer(strategy=self.strategy, fill_value=self.na1)
        features = [([i], [imput, encoder]) for i in obj_cols]
        not_obj_features = [([i], imput_n) for i in not_obj]
        features.extend(not_obj_features)

        self.encoder = DataFrameMapper(
            features, default=False, df_out=self.df_out)

        self.encoder.fit_transform(X, y=None)

        self.encode_mapper = {
            i[0][0]: np.ravel(i[1][1].categories_)
            for i in self.encoder.features if i[0][0] in obj_cols
        }
        self.out_labels = self.encoder.transformed_names_
        return self

    def transform(self, X):
        '''transform df to encoded X        
        '''
        X = self._filter_labels(X)
        # --
        X = self._check_categories(X)
        rst = self.encoder.transform(X)
        return rst


class Tree_embedding(BaseEstimator, TransformerMixin, Base_clean):
    '''Transform your features into a higher dimensional, sparse space using 
    'apply'method of decision trees or ensemble of trees. 
    Each sample goes through the decisions of each tree of the ensemble and 
    ends up in one leaf per tree. the indices of leaf node is then 
    one-hot-encoded to get new higher, sparse feature space
    
    parameters
    ----
    estimator
        - estimator instance like decision tree, random forests, GBM
    
    attributes
    ----
    see Base_clean attributes
    '''

    def __init__(self, estimator_=None):
        ''' '''
        if not hasattr(estimator_, 'apply'):
            self._raise_error(0)
        if not hasattr(estimator_, 'fit'):
            self._raise_error(1)
        self.estimator_ = estimator_

    def _raise_error(self, n):
        ''' '''
        if n == 0:
            raise AttributeError("'estimator' has no 'apply' method")
        if n == 1:
            raise AttributeError("'estimator' has no 'fit'  method")

    def fit(self, X, y=None, **fit_params):
        ''' fit estimator and one hot encoder
        '''
        X = self._fit(X)
        # --
        estimator = self.estimator_
        estimator.fit(X, y, **fit_params)
        # --
        X_app = self.estimator_.apply(X)
        Oht = OneHotEncoder(handle_unknown='ignore', sparse=False)
        X_app = self._check_df(X_app)
        Oht.fit(X_app)
        # --
        self.encoder = Oht
        cat_names = Oht.categories_
        self.out_labels = [
            '_'.join([str(i), str(a)]) for i, j in zip(X.columns, cat_names)
            for a in j
        ]
        return self

    def transform(self, X):
        '''transform data as leaf nodes indices '''
        X = self._filter_labels(X)
        # --
        estimator = self.estimator_
        encoder = self.encoder
        X_app = self._check_df(estimator.apply(X))
        X_tr = encoder.transform(X_app)

        return X_tr


if __name__ == '__main__':
    arr_x = np.random.randn(10000)
    arr_z = np.random.choice(['A', 'B', 'C', 'D', 'E'], 10000)
    arr_z = arr_z.astype(object)
    arr_z[100:340] = np.nan
    arr_y = np.random.randint(2, size=10000)
    df = pd.DataFrame({'x': arr_x, 'y': arr_y, 'z': arr_z, 'z2': arr_z})
    y = df.pop('y')
    a = pipe_main('woe')
    a.named_steps.woe.max_leaf_nodes=None
    a.named_steps.woe.bins=10
    print(a.fit(df, y))
