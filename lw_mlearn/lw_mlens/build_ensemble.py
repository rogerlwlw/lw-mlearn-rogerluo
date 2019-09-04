# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:20:44 2019

@author: BBD
"""

import pandas as pd

from sklearn import metrics
from sklearn.utils.testing import all_estimators
from mlens.ensemble import SuperLearner, Subsemble, BlendEnsemble
from mlens.preprocessing import Subset
from lw_mlearn.utilis import docstring
from lw_mlearn.utilis.utilis import get_kwargs, get_sk_estimators
from functools import wraps


def get_score_fn(scoring):
    '''
    scoring: str or callable
    
    return
    ------
    scoring function:
        that accepts an array of true values and an array 
        of predictions: score = f(y_true, y_pred).      
    '''
    if callable(scoring):
        return scoring
    else:
        return getattr(metrics, scoring)

    
@docstring.Appender(SuperLearner.add.__doc__)
@wraps(SuperLearner)
def build_stack(
             # add parameters
             estimators,
             preprocessing,
             proba,              
             propagate_features,                       
             # initiation parameters
             folds=3,
             shuffle=True,
             scorer=None,
             random_state=0,
             raise_on_exception=True,
             n_jobs=-1,
             model_selection=False,
             sample_size=20,
             meta_estimator='LogisticRegression',
             partitions=3,
             partition_estimator=None,
             test_size=0.5,
             ens_type='stack'):
    '''return stack/blend/subsemble ensemble model instance
    
    params 
    -------
    see __doc__ of SuperLearner & SuperLearner.add, already wrapped in
    '''
    if scorer is not None:
        scorer = get_score_fn(scorer) # return scoring functions
    meta_estimator = get_sk_estimators(meta_estimator)
    
    L = locals().copy()
    # --    
    ens_class = {'stack' : SuperLearner, 'subsemble' : Subsemble, 
                 'blend' : BlendEnsemble}[ens_type]
    ens = ens_class(**get_kwargs(ens_class, **L))
    
    ens.add(**get_kwargs(ens.add, **L))
    
    if meta_estimator is not None:
        ens.add_meta(meta_estimator, proba=True)
        
    return ens

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split    
    from lw_mlearn import pipe_main
    from lw_mlearn.lw_model import get_default_estimators
    from sklearn.metrics import roc_auc_score
    from lw_mlearn.lw_preprocess import plotter_lift_curve
    X = pd.read_csv('C:/Users/BBD/myproj/givemecredit/data/cs-training.csv')
    X.pop('Unnamed: 0')
    X.pop('NumberOfTime30-59DaysPastDueNotWorse')
    X.pop('NumberOfTimes90DaysLate')
    X.pop('NumberOfTime60-89DaysPastDueNotWorse')
    y = X.pop('SeriousDlqin2yrs').values
    
    X = pipe_main('cleanNA_woe5').fit_transform(X, y).values    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
    # --
    esti_lst = [pipe_main(i) for i in get_default_estimators('clf')]    
    ens = build_stack(estimators=esti_lst, 
                      preprocessing=None, 
                      proba=True, 
                      propagate_features=range(7), ens_type='stack')
    ens.fit(x_train, y_train)
    y_pre = ens.predict_proba(x_test)
    roc_auc_score(y_test, y_pre[:, 1])
    plotter_lift_curve(y_pre[:,1], y_test, max_leaf_nodes=None, 
                       bins=None, q=20, labels=False, ax=None,
                       header=None)  
    
#    from lw_mlearn import ML_model
#    clf = ML_model('HistGradientBoostingClassifier')     
#    clf.fit(x_train, y_train)
#    clf.plot_lift(x_test, y_test, q=20)
#    clf.test_score(x_test, y_test, cv=3, scoring=['KS', 'roc_auc'])
