import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import matplotlib as mpl
import os


# 读取txt文件，获得dataframe对象，各记录项设置为names
# 数据清洗，将txt中的str类型Series数据或csv文件中数据变为float类型数组;
# 转化Series中的Object对象为float类型，并将Series转化为数组array,重置为(1440,1)
def get_data(file):
    """
    A helper function to get array date from the dataframe object

    :param file: table file. txt or csv.
            The input data from txt or csv, etc.

    :return: dict
            The dictionary of day and X-ray flux.
    """
    xs_data = {}
    if os.path.splitext(file)[-1] == '.txt':
        dates = pd.read_csv(file, names=['YR', 'MO', 'DA', 'HHMM', 'Day', 'Sec', 'Short', 'Long', 'Ratio'], header=None, sep='\s+')
        x_short = dates.loc[19:, 'Short'].apply(float).values.reshape(1440, 1)
        x_long = dates.loc[19:, 'Long'].apply(float).values.reshape(1440, 1)
        day = str(dates.loc[19, 'YR']) + str(dates.loc[19, 'MO']) + str(dates.loc[19, 'DA'])

        xs_data = {'x_short': x_short,
                   'x_long': x_long,
                    'day': day}

    elif os.path.splitext(file)[-1] == '.csv':
        dates = pd.read_csv(file, names=['YR', 'MO', 'DA', 'HHMM', 'Day', 'Sec', 'Short', 'Long', 'Ratio'], header=None)
        x_short = dates.loc[1:, 'Short'].apply(float).values.reshape(1440, 1)
        x_long = dates.loc[1:, 'Long'].apply(float).values.reshape(1440, 1)
        day = str(dates.loc[1, 'YR'] + str(dates.loc[1, 'MO']) + str(dates.loc[1, 'DA']))

        xs_data = {'x_short': x_short,
                   'x_long': x_long,
                   'day': day}

    return xs_data


# 绘制X_ray_short的曲线图
def my_plotter_xf(ax, x, data, param_dict):
    """
    A helper function to make plot gragh

    :param ax: Axes
            The axes to draw to
    :param x: array
            The x data
    :param data: array
            The y data
    :param param_dict: dict
            Dictionary of kwargs to pass to ax.plot

    :return:
            out:list
            list of artists added
    """

    out = ax.plot(x, data, **param_dict)
    ax.set_yscale('log')
    ax.set_ylim(pow(10, -9), pow(10, -3))
    ax.set_xlim(0, len(data))
    ax.set_ylabel('5 - 40 AI')
    ax.set_xlabel('Time (mins) ')
    ax.set_title('Goes XR 1-Day Plot')
    ax.text(30, pow(10, -3.3), xs_data['day'], fontsize=7)
    ax.grid()
    ax.legend()
    fig.tight_layout()
    return out


# # 获得X-Flux-short的array类型数据
# xs_data = get_data(r'D:\Goes\Goes_xray\2011\csv\20111230_Gp_xr_1m.csv')
# # 获取X-Flux的数据点个数
# x = np.arange(len(xs_data['x_short']))


# # 绘图
# mpl.rcParams['path.simplify_threshold'] = 0.0
# fig, ax = plt.subplots(figsize=(5, 4))
# my_plotter_xf(ax, x, xs_data['x_short'], {'color': 'b', 'linewidth': '0.8', 'label': '0.5-4 A Primary'})
# my_plotter_xf(ax, x, xs_data['x_long'], {'color': 'r', 'linewidth': '0.8', 'label': '1-8 A Primary'})
# plt.show()

#遍历path中的csv文件，并存进flist列表
## 若文件夹中存在多个文件夹，可先用get_dir返回子文件夹列表
def get_dir(root):  # 获取根目录下子文件夹名的列表
    for root, dirs, file in os.walk(root):
        return dirs

## 用get_file提取单个文件夹的文件名列表。
def get_file(path):     # 获取路径下的文件名称(含文件拓展名)
    flist = []
    for fpath, dirs, files in os.walk(path):
        for file in files:  #files为文件名列表，不包含路径。其格式为 'xxxx.后缀'
            if os.path.splitext(file)[-1] == '.csv':  #选取含特定后缀的文件
                file = os.path.join(path, file)    #拼接路径名与文件名。 file格式为 'path\xxx.后缀'
                flist.append(file)
    return flist


flist = get_file(r'D:\Goes\Goes_xray\2011\csv')

for f in flist:
    print(f)


#获取flist中的1m文件。    ：可尝试用正则表达式
def get_1m_data(flist):
    pass


#遍历flist中的数据并成像

# xs_data = get_data(f)
# x = np.arange(len(xs_data['x_short']))
# mpl.rcParams['path.simplify_threshold'] = 0.0
# fig, ax = plt.subplots(figsize=(5, 4))
# my_plotter_xf(ax, x, xs_data['x_short'], {'color': 'b', 'linewidth': '0.8', 'label': '0.5-4 A Primary'})
# my_plotter_xf(ax, x, xs_data['x_long'], {'color': 'r', 'linewidth': '0.8', 'label': '1-8 A Primary'})
# plt.show()