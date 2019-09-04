# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 12:57:54 2019

@author: rogerluo
"""

import pandas as pd 
import numpy as np
from lw_mlearn.utilis.plotter import plotter_violin

if __name__ == '__main__':
    
    data = pd.DataFrame(np.random.randn(1000, 2), columns=list('AB'))

    
    ax = plotter_violin(data, 
                   palette='Paired', 
                   inner='quartile',
                   kind='violin',
                   xlabel='xlabel',
                   ylabel='ylabel',
                   title='indicator distribution中文',
                   )
    from lw_mlearn.utilis import read_write
    ob = read_write.Writer('.')
    ob.write(ax, 'violi.pdf')
