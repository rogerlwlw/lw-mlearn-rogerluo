# -*- coding: utf-8 -*-
"""
@author: roger
----
1) SQL_engine(): class, to connect data base engine

"""
import os
import pandas as pd
from sqlalchemy.types import Integer, Float, String
from sqlalchemy import create_engine


class SQL_engine():
    '''class: Get data base engine to down/upload data or excute query
        
    Parameters
    ------
    dialect, uname, upwd, host, dbname, port,  DBAPI
    
    dialect : dialect is a database name such as mysql, oracle, postgresql, etc
        --> 'oralce' kuaixin database
        
        --> 'mysql'  antifraud mysql fri database
    
    DBAPI: the name of a DBAPI/driver: 
        --> URL: dialect[+driver]://user:password@host[: port]/dbname[?key=value..]
            "postgresql://scott:tiger@localhost/test"    
    MySQL:    
        MySQL-Python: 'mysqldb'
     
        pymysql : 'pymysql'
      
        MySQL-Connector : 'mysqlconnector'        
    Oracle: 
        cx_Oracle : 'cx_oracle'
    
    Attributes
    -----
    sqlTypes: sqlalchemy types class to map database data types
    
    Method
    -----
    1 getengine(): return engine connection

    2 upload_toDB(): upload df to data base as a table
    
    3 read_df(sql): return data frame,read query data into data frame
    
    4 execute(sql): execute sql query
     
    '''

    def __init__(self, dialect, uname, upwd, host, dbname, port, DBAPI):
        '''
        '''
        self._engine_url = {
            'uname': uname,
            'upwd': upwd,
            'host': host,
            'dbname': dbname,
            'port': port,
            'dialect': dialect,
            'DBAPI': DBAPI
        }

    def getengine(self, **kwargs):

        #Oracle client encoding
        os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
        #        self.sqlTypes = types

        conn_string = '''{dialect}+{DBAPI}://{uname}:{upwd}@{host}:{port}/{dbname}
                          '''.format_map(self._engine_url)

        engine = create_engine(conn_string,
                               encoding=kwargs.get('encoding', 'GBK'),
                               **kwargs)
        print('database %s is called... \n' % (engine))
        return engine

    def upload_toDB(self,
                    df,
                    name=None,
                    if_exists='fail',
                    dtype=None,
                    **kwds_pd):
        ''' 
        Parameters
        ----
        df: 
            data frame, to be uploaded to data base                           
        name: 
            str, table name, default name = 'PYTHON_UPLOAD_TAB',      
        if_exists: 
            {'fail', 'replace', 'append'}, default 'fail'                 
        dtype: 
            dict = {col : sqlTypes}, if None, guess from dtypes of df          
        **kwds_pd: pandas to_sql key words
            index: 
                boolean, default False, not to upload index
            schema :
                string, optional
                Specify the schema (if database flavor supports this). 
                If None, use default schema.        
        '''
        kws = dict(index=False, chunksize=100, con=self.getengine())
        kws.update(**kwds_pd)
        if name is None:
            name = 'PYTHON_UPLOAD_TAB'
        if dtype is None:
            dtype = self.get_map_df_types(df)
        print('begin uploading data...\n')
        df.to_sql(name=name, if_exists=if_exists, dtype=dtype, **kws)
        print("successfullly upload data to table: '%s': ... \n" % name,
              df.head(5), '\n in data base: %s \n' % (self.getengine()))
        return

    def read_df(self, sql):
        '''
        Parameters
        ----
        sql - str 
            - sql query to be executed to get data table        
        Return
        ----
        df - data frame 
        '''
        print('begin reading sql...\n')
        engine = self.getengine()
        df = pd.read_sql_query(sql, engine)
        print('successfully read data: ... \n', df.head(5), '\n',
              'from database: %s \n' % engine)
        return df

    def execute(self, sql):
        self.getengine().execute(sql)
        print("successfully execute SQL:'%s' ..." % sql[:40])
        return

    def get_map_df_types(self, df):
        '''mapper of df dtype to db datatype
        
        return
        ----
        {colname : sqldtype}
        {object : String(length=255), float : Float(6), int : Integer()}
        '''
        dtypedict = {}
        for i, j in zip(df.columns, df.dtypes):
            if "object" in str(j) or "category" in str(j):
                max_length = df[i].apply(lambda x: len(str(x))).max()
                dtypedict.update(
                    {i: String(length=255 * (max_length // 255) + 255)})
            if "float" in str(j):
                dtypedict.update({i: Float(precision=6, asdecimal=True)})
            if "int" in str(j):
                dtypedict.update({i: Integer()})
        return dtypedict
