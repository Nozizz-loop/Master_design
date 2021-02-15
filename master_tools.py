"""
A toolkit may be helpful in my master design
to be completed
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def get_file(path, suffix):
    """
    Get a list containing the filename from a dir

    Arguments:
        path - the path of the directory (without subdirectory)
        suffix - string, the suffix of file;
        (eg: '.csv') (don't miss the ".")

    Returns:
        fl - a list, containing the name of the file in the directory or the subdirectory;
        (eg:"D:/Goes/XXX.csv")
    """

    fl = []
    for fp, dirs, files in os.walk(path):
        for file in files:  # files为文件名列表，不包含路径。其格式为 'xxxx.后缀'
            if os.path.splitext(file)[-1] == suffix:  # 选取含特定后缀的文件
                file = os.path.join(fp, file)    # 拼接路径名与文件名。 file格式为 'path\xxx.后缀'
                fl.append(file)
    return fl


def pick_char(strings, char):
    """
    Pick up the return the string containing the characters 'char'
    Return None if the string don't contain the characters 'char'
    目前仅能提取含特定后缀的字符串，返回对应的字符串列表。
    eg:从XXX._1m.csv与XXX._5m.csv中筛选出XXX1m.csv

    Arguments:
        strings - a list of strings to be picked
        ar - the key word, determine which string should be kept

    Returns:
        strings - the specific string containing the characters 'char'
    """
    results = []
    for f in strings:
        test_text = re.findall('(.*)'+char, f)
        result = ''
        if len(test_text) == 1:
            result = test_text[0] + char
            results.append(result)
    return results


def get_data_1m(file):
    """
    A helper function to get array date from the dataframe object only for X-ray_1m flux data

    :param file: table file. txt or csv.
            The input data from txt or csv, etc.

    :return: dict
            The dictionary of day and X-ray flux.
    """
    xs_data = {}
    if os.path.splitext(file)[-1] == '.txt':
        dates = pd.read_csv(file, names=['YR', 'MO', 'DA', 'HHMM', 'Day', 'Sec', 'Short', 'Long', 'Ratio'], header=None, sep='\s+')
        L = len(dates.loc[19:, 'Short'])
        x_short = dates.loc[19:, 'Short'].apply(float).values.reshape(L, 1)
        x_long = dates.loc[19:, 'Long'].apply(float).values.reshape(L, 1)
        day = str(dates.loc[19, 'YR']) + str(dates.loc[19, 'MO']) + str(dates.loc[19, 'DA'])

        xs_data = {'x_short': x_short,
                   'x_long': x_long,
                    'day': day}

    elif os.path.splitext(file)[-1] == '.csv':
        dates = pd.read_csv(file, names=['YR', 'MO', 'DA', 'HHMM', 'Day', 'Sec', 'Short', 'Long', 'Ratio'], header=None)
        L = len(dates.loc[1:, 'Short'])
        x_short = dates.loc[1:, 'Short'].apply(float).values.reshape(L, 1)
        x_long = dates.loc[1:, 'Long'].apply(float).values.reshape(L, 1)
        day = str(dates.loc[1, 'YR'] + str(dates.loc[1, 'MO']) + str(dates.loc[1, 'DA']))

        xs_data = {'x_short': x_short,
                   'x_long': x_long,
                   'day': day}

    return xs_data


