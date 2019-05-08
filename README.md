# lw-mlearn

A Python tool that wraps sklearn estimators into pipelines and faciliates workflow 
of:

1) data cleaning:
    - try converting data to numeric dtype; 
    - dropna columns; 
    - drop uid columns;
    - filter specific dtypes;
2) data encoding: 
    - oridinal/one-hot encoding; 
    - woe encoding; 
    - binning by cart tree/equal frequency/equal width;
3) feature selection:
    - select from model(svc, xgb, cart, random forest); 
    - select from test statisics (chi2, mutual-info, woe);
    - pca decomposition
    - RFE
4) over/under resampling
5) model training;
6) cross validation, 
7) hyper parameter tuning(grid_search), 
8) performance evaluation (multi_metrics)
9) production integration

two main hepler functions:

1) pipe_main:
    return pipeline instance combining transformers and estimators(including some 
    pre-difined custom estimators)
2) LW_model:
     return wrapper of pipeline, which contains methods performming 
     model training, performance scoring, plotting, hyper tunning, 
     cross validation and model serialization

Contact
=============
If you have any questions or comments about lw_mlearn, please feel free to 
contact me via:
E-mail: coolww@outlook.com
This project is hosted at https://github.com/rogerlwlw/lw-mlearn-rogerluo

