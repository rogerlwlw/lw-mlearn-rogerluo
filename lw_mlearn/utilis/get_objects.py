# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:23:48 2019

@author: roger luo
"""

import pkgutil
import os
import numpy as np

def get_modules(path=None, subpkgs=False, prefix=None):
    '''
    path (dir): 
        default current working dir
        path must be None or list of paths to look for modules in
        
    subpkgs (bool): 
        if False only sub-modules, not sub-packages
    prefix:
        prefix used before module names as keyname in return
    filter_type:
        ['module'/'class'/'function']
    return
    ----
    dict of module object under given path
    ''' 
    if path is None:
        path = [os.getcwd()]
    else:
        path = np.array(path)
    
    module = {} 
    if prefix is None:
        prefix = ''
    for finder, name, ispkg in pkgutil.iter_modules(path):
        if not ispkg:
            module[prefix+name] = finder.find_module(name).load_module(name)
        elif subpkgs:
            pkg = finder.find_module(name).load_module(name)
            module.update(get_modules(path=pkg.__path__, subpkgs=True,
                                     prefix=name+'.'))
            
    return module        

def get_class():
    '''
    '''
    
    return

#def get_obj(name_space, filter_type):
#    '''
#    '''
#    is_obj = {'class' : inspect.isclass,
#              'function' : inspect.isfunction,
#             }
#    
#    inspect.getmembers( , is_obj.get(filter_type))
#    }
#    return

def _test():
    '''test functionality of this module
    '''
    return get_modules()
    
if __name__ == '__main__':   
    _test()
