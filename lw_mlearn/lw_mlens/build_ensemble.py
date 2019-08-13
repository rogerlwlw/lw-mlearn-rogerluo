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
from lw_mlearn.utilis.utilis import get_kwargs
from functools import wraps


def get_score_fn(scoring):
    '''
    scoring: str or callable
    
    return
    ----
    scoring function that accepts an array of true values and an array 
    of predictions: score = f(y_true, y_pred).      
    '''
    if callable(scoring):
        return scoring
    else:
        return getattr(metrics, scoring)

def get_sk_estimators(clf, type_filter='classifier'):
    '''
    clf (str):
        name of estimators
    '''
    # sklearn estimator
    t = all_estimators(type_filter=['classifier'])
    estimator = {}
    for i in t:
        try:
            estimator.update({i[0]: i[1]()})
        except Exception:
            continue
    return estimator.get(clf)
    
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
             scorer='f1_score',
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
    '''return stack ensemble model instance
    
    params 
    -------
    see __doc__ of SuperLearner & SuperLearner.add, already wrapped in
    '''
    scorer = get_score_fn(scorer) # return scoring functions
    meta_estimator = get_sk_estimators(meta_estimator)
    
    L = locals().copy()
    # --
    ens_model = {'stack' : SuperLearner, 'subsembel' : Subsemble, 
                 'blend' : BlendEnsemble}
    instance = ens_model[ens_type]
    ens = instance(**get_kwargs(instance, **L))
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
    
    X = pipe_main('cleanNA_woe').fit_transform(X, y).values    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # --
    esti_lst = [pipe_main(i) for i in get_default_estimators('clf')]    
    ens = build_stack(estimators=esti_lst, 
                      preprocessing=None, 
                      proba=True, 
                      propagate_features=range(7), ens_type='blend')
    ens.fit(x_train, y_train)
    y_pre = ens.predict_proba(x_test)
    roc_auc_score(y_test, y_pre[:, 1])
    plotter_lift_curve(y_pre[:,1], y_test, max_leaf_nodes=None, 
                       bins=None, q=20, labels=False, ax=None,
                       header=None)   
    
#    m = SuperLearner()
#    m.add(esti_lst, proba=True, propagate_features=range(7))
#    m.add_meta(pipe_main('LogisticRegression'), proba=True)
#    m.fit(x_train, y_train)
#    y_pre = m.predict_proba(x_test)
#    roc_auc_score(y_test, y_pre[:, 1])
