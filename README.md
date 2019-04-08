# lw-mlearn

A Python tool that wraps sklearn estimators into pipelines and faciliates workflow 
of:

1) data cleaning:
    - try converting data to numeric dtype; 
    - dropna columns; 
    - drop uid columns;
    - filter specific dtypes;
2) data encoding: 
    - oridinal/one-hot encoding; woe encoding; 
    - binning by cart tree/equal frequency/equal width;
3) feature selection:
    - select from model(svc, xgb, cart,); 
    - select from test statisics (chi2, mutual-info, woe);
    - pca decomposition
    - RFE
4) model training;
5) cross validation, 
6) hyper parameter tuning, 
7) performance evaluation
8) production integration

two main hepler functions:

1) pipe_main:
    return pipeline instance combining transformers and estimators(adding some 
    custom estimators)
2) LW_model:
     return wrapper of estimator or pipeline, which contains methods performming 
     model training, performance scoring, plotting, hyper tunning 
     cross validation and model serialization

Contact
=============
If you have any questions or comments about lw_mlearn, please feel free to 
contact me via:
E-mail: coolww@outlook.com
This project is hosted at https://github.com/rogerlwlw/lw-mlearn-rogerluo

