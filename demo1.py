import os
import master_tools
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# flist = master_tools.get_file(r'D:\Goes\Goes_xray\2011\csv', '.csv')
# for f in flist:
#     print(f)

# def get_dir(root):
#     for root, dirs, file in os.walk(root):
#         print(root)
#         for item in file:
#             path = os.path.join(root, item)
#             print(path)

fl = master_tools.get_file(r'D:\Goes\Goes_xray\2012', '.csv')

fl_1m = master_tools.pick_char(fl, '1m.csv')

L = len(fl_1m)

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

for f in fl_1m:
    xs_data = master_tools.get_data_1m(f)
    x = np.arange(len(xs_data['x_short']))

    mpl.rcParams['path.simplify_threshold'] = 0.0
    fig, ax = plt.subplots(figsize=(5, 4))
    my_plotter_xf(ax, x, xs_data['x_short'], {'color': 'b', 'linewidth': '0.8', 'label': '0.5-4 A '})
    my_plotter_xf(ax, x, xs_data['x_long'], {'color': 'r', 'linewidth': '0.8', 'label': '1-8 A '})
    plt.show()

