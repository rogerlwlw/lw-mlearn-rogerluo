# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 13:47:28 2019

@author: roger
"""

import re
from pandas.core.dtypes import api


def get_regex(X, casesensitive=False):
    '''return regex pattern object
    '''
    if api.is_re(X): return X
    if api.is_re_compilable(X): return re.compile(X)
    regex = '|'.join(get_flat_list(X))
    if not casesensitive:
        regex = '(?ix:{0})'.format(regex)
    return re.compile(regex)


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


