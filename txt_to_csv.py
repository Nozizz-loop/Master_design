import os
import pandas as pd
import numpy as np


def get_dir(root):  # 获取根目录下子文件夹名的列表
    for root, dirs, file in os.walk(root):
        return dirs


def get_file(path):     # 获取路径下的文件名称(含文件拓展名)
    flist = []
    for fpath, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                flist.append(file)
    return flist


def get_data(path, file):   # 获取纯数据内容
    datas = pd.read_csv(os.path.join(path, file),
                        names=['YR', 'MO', 'DA', 'HHMM', 'Day', 'Sec', 'Short', 'Long', 'Ratio'],
                        header=None, sep='\s+').loc[19:]
    return datas


def txt_to_csv(file):   # 生成csv文件名
    portion = os.path.splitext(file)
    if portion[1] == '.txt':
        newfile = portion[0] + '.csv'
    return newfile


"""
功能：处理Goes_xray的txt文件，仅保留其中的数据部分，并将其保存为csv文件，存储于子目录中的新建csv文件夹中。
input:
    root   根文件夹
output:
    None    批量生成位于指定目录下csv文件

"""
def to_csv(root):
    dirs = get_dir(root)   # 生成一个列表，获取根目录下的子目录名
    for dir in dirs:
        path = os.path.join(root, dir)  # 生成子目录完整路径
        flist = get_file(path)  # 生成一个列表，获取该子目录下的所有文件名（含文件拓展名）

        for file in flist:
            datas = get_data(path, file)    # 读取文件数据
            newfile = txt_to_csv(file)  # 生成csv文件名
            isExists = os.path.exists(os.path.join(path, 'csv'))    # 再当前目录下生成新文件夹存储csv文件
            if not isExists:
                os.mkdir(os.path.join(path, 'csv'))
            datas.to_csv(os.path.join(path, 'csv', newfile), index=0)



root = r'D:/Goes/Goes_xray'
to_csv(root)
