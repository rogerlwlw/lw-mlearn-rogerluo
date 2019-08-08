# lw-mlearn

A Python tool that wraps sklearn estimators into pipelines and faciliates workflow 
of:

1) data cleaning:
    - try converting data to numeric dtype; 
    - dropna columns; 
    - drop uid columns;
    - drop constant columns;
    - filter specific dtypes;
2) data encoding: 
    - oridinal/one-hot encoding; 
    - woe encoding; 
    - binning by cart tree/equal frequency/equal width;
3) feature selection:
    - select from model(svc, xgb, cart, random forest); 
    - select from test statisics (chi2, mutual-info, woe);
    - pca/LDA/QDA decomposition
    - RFE
4) resampling (over/under )
5) model training;
6) cross validation, 
7) hyper parameter tuning(grid_search), 
8) performance evaluation (plot multi_metrics/scores)
9) production integration


Main hepler functions:
=============
1) pipe_main:
    return pipeline instance, a chained sequence of transformers and estimators(
    including some pre-difined custom estimators)

2) train_models:
    run ML_model analysis for given train_set, test_set & estimator and dump results
3) model_experiments:
    experiment on different piplines of chained estimators

4) LW_model (Class):
     return a wrapper of pipeline, which implements methods for performance scoring, 
     plotting, hyper parameter tunning, cross validation and model serialization


Contact
=============
If you have any questions or comments about lw_mlearn, please feel free to 
contact me via:
E-mail: coolww@outlook.com
This project is hosted at https://github.com/rogerlwlw/lw-mlearn-rogerluo

