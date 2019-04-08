# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:36:20 2018

@author: roger luo
"""
import pandas as pd
import os
import pickle
import shutil
import matplotlib.pyplot as plt

from sklearn.utils import check_consistent_length
from lw_mlearn.utilis import get_flat_list, get_kwargs


class _Obj():
    pass


class Desc():
    '''base descriptor '''

    def __init__(self):

        self._x = None

    def __get__(self, instance, owner):

        return self._x

    def __delete__(self, instance):

        del self._x


class File(Desc):
    ''' check if file exists 
    '''

    def __set__(self, instance, file):

        if os.path.isfile(file):
            self._x = os.path.abspath(file)
        else:
            raise TypeError("file '{}' does not exist".format(file))

    def __delete__(self, instance):
        '''remove file
        '''
        os.remove(self._x)
        print("file '{}' removed".format(self._x))
        del self._x


class NewFile(File):
    '''create a new file under given path, check for valid path 
    '''

    def __set__(self, instance, file):

        try:
            if os.path.isfile(file):
                os.remove(file)
                print("old file '{}' deleted...\n ".format(file))

            dirs, filename = os.path.split(file)
            if not os.path.exists(dirs) and len(dirs) > 0:
                os.makedirs(dirs, exist_ok=True)
                print("path '{}' created...\n".format(dirs))
            self._x = file
        except Exception as e:
            print(repr(e))
            raise ValueError('invalid path input {}'.format(file))


class Path(Desc):
    '''check for valid path and create new path'''

    def __set__(self, instance, path):
        try:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print("path '{}' created...".format(path))
            self._x = os.path.abspath(path)
        except Exception as e:
            print(repr(e))
            raise TypeError("invalid path input '%s' " % path)

    def __delete__(self, instance):

        for root, dirnames, file in os.walk(self._x, topdown=False):
            for i in file:
                os.remove(os.path.join(root, i))
        shutil.rmtree(self._x, ignore_errors=True)
        print("path '{}' removed...".format(self._x))
        del self._x


class Reader():
    '''read in python objects contained in files, 
    supported suffix of file are
        - ['.xlsx', '.csv', '.pkl', '.txt', '.sql']
    
    method
    ----
    read: 
        return obj read from file
    read_all:
        return generator of read in objs
    '''
    file_ = File()
    path_ = Path()

    def __init__(self, path):
        ''' init path variable 
        '''
        self.path_ = path
        self.obj_collections = _Obj()

    def read(self, file, **kwargs):
        '''return obj from file
        
        supported suffix of file are
        - ['.xlsx', '.csv', '.pkl', '.txt', '.sql']       
        file - str or file object
            - file to read
        '''
        self.file_ = file
        read_api = _rd_apis(self.file_)
        rel_file = os.path.relpath(self.file_)
        try:
            kw = get_kwargs(read_api, **kwargs)
            rst = read_api(file, **kw)
            print("<obj>: '{}' read from '{}\n".format(rst.__class__.__name__,
                                                       rel_file))
            return rst
        except Exception as e:
            print("<failure>: 'file' read failed".format(rel_file))
            print(repr(e), '\n')

    def read_all(self, suffix=None, subfolder=False, **kwargs):
        '''return generator of read in objs, and collect them
        in obj_collection obj as attributes
        '''
        file_dict = _get_files(self.path_, suffix, subfolder)

        for k, v in file_dict.items():
            load = self.read(v, **kwargs)
            if load is not None:
                setattr(self.obj_collections, k.replace('.', '_'), load)

        gen = (self.read(v, **kwargs) for v in file_dict.values()
               if self.read(v, **kwargs) is not None)
        return gen


def _load_pkl(file):
    '''return unpickled obj from 'pkl' file
    '''
    with open(file, 'rb') as f:
        pkl = pickle.Unpickler(f)
        obj = pkl.load()
    return obj


def _read_file(file):
    ''' return 'str' obj from file by calling f.read() method
    '''
    with open(file, 'r') as f:
        obj = f.read()
        print("read 'str' in '{}'  successfully ...".format(file))
    return obj


def _get_files(dirpath, suffix=None, subfolder=False):
    ''' return file dict {filename : file}

    dirpath - str
        - dir_x to traverse
    suffix -->extension name, or list of extension names, egg ['.xlsx', 'csv']
        - to include file extensions, default None, to include all extensions
    subfolder --> bool
        - true to traverse subfolders, False only the given dirpath
    '''
    if subfolder:
        get_dirs = traverse_all_dirs
    else:
        get_dirs = traverse_dir

    rst = {
        k: v
        for k, v in get_dirs(dirpath).items()
        if os.path.splitext(v)[1] in get_flat_list(suffix) or not suffix
    }
    return rst


def _rd_apis(file):
    '''return read api of given suffix of file
    
    api parameters
    ----
    file 
        - file to read obj from
    **kwargs
    '''
    api_collections = {
        '.xlsx': pd.read_excel,
        '.csv': pd.read_csv,
        '.txt': _read_file,
        '.sql': _read_file,
        '.pkl': _load_pkl,
        '.model': _load_pkl
    }

    suffix = os.path.splitext(file)[1]
    rst = api_collections.get(suffix, _load_pkl)
    if rst is None:
        print("'read api' for {} not available... ".format(suffix))
    return rst


class Writer():
    '''write objects into file
    
    method
    -----
    write:
        write obj into file
    '''
    newfile_ = NewFile()
    path_ = Path()

    def __init__(self, path):
        ''' init path variable '''
        self.path_ = path

    def write(self, obj, file, **kwargs):
        '''dump obj into file under self.path_

        file
            - filename + suffix egg 'filename.pkl'
            - supported suffix are [.pkl, .xlsx, .csv, .pdf, .png], 
            see _wr_apis
        
        **kwargs
            - other keys arguments for suffix specified api
        '''
        file = os.path.join(self.path_, file)
        file = os.path.relpath(file)
        self.newfile_ = file
        wr_api = _wr_apis(self.newfile_)
        try:
            wr_api(obj, self.newfile_, **kwargs)
            print("<obj>: '{}' dumped into '{}...\n".format(
                obj.__class__.__name__, file))
        except Exception as e:
            print(repr(e))
            print("<failure>: '{}' written failed ...".format(file))


def _wr_apis(file):
    ''' return write api of given suffix of file
    
    api parameters
    ---
    obj
        - obj to be written
    file
        - file to wirte into
    **kwargs
    '''
    api_collections = {
        '.pkl': _dump_pkl,
        '.model': _dump_pkl,
        '.xlsx': _dump_df_excel,
        '.csv': _dump_df_csv,
        '.pdf': _save_plot,
        '.png': _save_plot
    }

    suffix = os.path.splitext(file)[1]
    rst = api_collections.get(suffix, _dump_pkl)
    if not rst:
        print("write api for '{}' not available... ".format(suffix))
    return rst


def _dump_pkl(obj, file, **kwargs):
    '''
    obj - python objects
    file - file to dump obj into
    '''
    with open(file, 'wb') as f:
        pkl = pickle.Pickler(f)
        pkl.dump(obj)


def _dump_df_excel(obj, file, **kwargs):
    '''dump df to excel
    
    obj: 
        2d array like data
    file:
        str or file obj:        
    '''
    writer = pd.ExcelWriter(file)
    obj = get_flat_list(obj)

    sheet_name = kwargs.get('sheet_name')

    if sheet_name is None:
        sheet_name = ['sheet' + str(i + 1) for i in range(len(obj))]
    else:
        sheet_name = get_flat_list(sheet_name)
        check_consistent_length(obj, sheet_name)

    for data, name in zip(obj, sheet_name):
        try:
            data = pd.DataFrame(data)
            kw = get_kwargs(data.to_excel, **kwargs)
            kw.update({
                'sheet_name': name,
                'index': kwargs.get('index', False)
            })
            data.to_excel(writer, **kw)
        except Exception as e:
            print(repr(e))
            print('data obj is not DataFrame convertible')
            continue
    writer.save()


def _dump_df_csv(obj, file, index=False, **kwargs):
    ''' dump df to csv
    '''
    try:
        data = pd.DataFrame(obj)
        data.to_csv(file, index=index, **get_kwargs(data.to_csv, **kwargs))
    except Exception as e:
        print(repr(e))
        print('data obj is not DataFrame convertible')


def _save_plot(fig, file, **kwargs):
    '''save the figure obj , if fig is None, save the current figure
    '''
    fig.savefig(file, **kwargs)
    plt.show()
    plt.close(fig)


class Objs_management(Reader, Writer):
    def __init__(self, path):
        '''manage read & write of objects from/into file
        '''
        super().__init__(path)

    def _remove_path(self):
        '''remove path and all files within
        '''
        del self.path_


def traverse_dir(rootDir):
    '''traverse files under rootDir not including subfolder
    return
    ----
    dict - {filename : file}
    '''
    file_dict = {}
    for filename in os.listdir(rootDir):
        pathname = os.path.join(rootDir, filename)
        if os.path.isfile(pathname):
            file_dict[filename] = pathname
    return file_dict


def traverse_all_dirs(rootDir):
    '''traverse files under rootDir including subfolders
    return
    ----
    dict - {filename : file}
    '''
    file_dict = dict([file, os.path.join(dirpath, file)]
                     for dirpath, dirnames, filenames in os.walk(rootDir)
                     for file in filenames)
    return file_dict
