# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:55:15 2018

@author: roger
"""
import pandas as pd
import inspect

from functools import wraps, reduce


def merge_dfs(df_list, how='inner', **kwargs):
    '''return merged  DataFrames of all in 'df_list'
    '''
    lamf = lambda left, right: pd.merge(left, right, how=how, **kwargs)
    return reduce(lamf, df_list)


def get_flat_list(x):
    '''
    list and flatten object into one dimension list
    
    return
    ----
    one dimension list
    '''
    if isinstance(x, list) and not isinstance(x, (str, bytes)):
        return [a for i in x for a in get_flat_list(i)]
    else:
        return [x]


def default_func(func, new_funcname=None, **kwargs_outer):
    '''return function with default keyword arguments specified in
    kwargs_outer where a new_funcname is given to newly initialized func
    '''
    kwargs_outer = get_kwargs(func, **kwargs_outer)

    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs_outer.update(kwargs)
        return func(*args, **kwargs_outer)

    wrapper.__name__ = new_funcname if new_funcname else func.__name__
    return wrapper


def dict_subset(d, keys):
    '''update dict with a subset of keys
    
    keys --> iterable of keys
    '''
    return {i: d.get(i) for i in keys}


def inverse_dict(d):
    '''return inversed dict {k, val} as  {val, k} 
    '''
    try:
        return {val: k for k, val in d.items()}
    except Exception:
        raise Exception('dict is not 1 to 1 mapping, cannot be inversed')


def get_current_function_name():
    '''get_current_function_name
    '''
    return inspect.stack()[1][3]


def get_kwargs(func, **kwargs):
    '''return subset of **kwargs that are of func arguments
    '''
    func_args = set(inspect.getfullargspec(func).args)
    func_args.intersection_update(kwargs)
    return {i: kwargs[i] for i in func_args}


def dec_iferror_getargs(func):
    ''' catch exceptions when calling func and return arguments input '''

    @wraps(func)
    def wrapper(*args, **kwargs):  #1
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(repr(e))
            print('func name = ', func.__name__)
            print('args =\n ', args, '...')
            print('kwargs =\n ', kwargs, '...\n')
            raise Exception('''Error encountered in func {0} \n'''.format(
                func.__name__))

    return wrapper


if __name__ == '__main__':
    pass
