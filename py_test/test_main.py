# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:26:59 2019

@author: rogerluo
"""
import pytest
import numpy as np
from lw_mlearn import pipe_main, ML_model
from sklearn.datasets import make_classification


@pytest.fixture
def data():
    '''test iris data treated as binary target
    '''
    X, y = make_classification(100)
    a = np.random.choice([np.nan, 0, 1, 2], (1, 20))
    X = np.vstack([X, a])
    y = np.append(y, 0)
    return X, y


def _model_run(data, pipe):
    '''single run for one ML_model instance
    '''
    X, y = data
    E = ML_model(pipe, path=pipe)

    E.run_analysis((X, y), (X, y), max_leaf_nodes=5)

    E.delete_model()


@pytest.mark.fast
def test_ml_model(data, pipe='clean_oht_XGBClassifier'):
    ''' test single estimator SVC
    '''
    # test
    check = 0
    try:
        _model_run(data, pipe)
    except Exception as e:
        print(repr(e))
        check -= 1
    assert check == 0


@pytest.mark.pipe
def test_fit_transform(data):
    '''test  fit/fit_transform for all pipelines generated by pipe_main 
    '''
    X, y = data
    items = pipe_main()
    pipe = ['_'.join(['clean_ordi', a])
            for k, v in items.items() for a in v
            if k not in ['clean', 'encoding'] \
            and a not in ['fchi2', 'ComplementNB','MultinomialNB']
    ]

    # test
    check = 0
    n = []
    for i in pipe:
        try:
            test_pipeline = pipe_main(i)
            if hasattr(test_pipeline, 'fit'):
                test_pipeline.fit(X, y=y)
            if hasattr(test_pipeline, 'transform'):
                test_pipeline.transform(X)
            if hasattr(test_pipeline, 'resample'):
                test_pipeline.resample(X, y)
        except Exception as e:
            print(repr(e))
            check -= 1
            n.append(i)
    if check < 0:
        print('{} failed <test_fit_transform> \n'.format(n))
    assert check == 0


@pytest.mark.ALL
def test_ml_model_all(data):
    '''test ML_model for all estimators generated by pipe_main
    '''
    X, y = data
    items = pipe_main()
    pipe = [
        '_'.join(['clean_ordi', a]) for i in items for a in items.get(i)
        if i in ['estimator']
    ]

    # test
    check = 0
    n = []
    for i in pipe:
        try:
            _model_run(data, i)
        except Exception as e:
            print(repr(e))
            check -= 1
            n.append(i)
    if check < 0:
        print('{} failed <test_ml_model_all> \n'.format(n))
    assert check == 0
