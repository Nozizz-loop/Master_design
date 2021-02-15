import os
from data_reform import new_1

# Traverse the file dir and get a list containing the filename


def file_list(file_dir):
    flist = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.cos':
                flist.append(file)
    return flist

# choose the inputfile dir


in_dir = 'D:/Jupyter/practice_cos'

# choose the outputfile dir

out_dir = 'D:/Jupyter/practice_fit'

flist = file_list(in_dir)

for file in flist:
    print(file)

for file in flist:
    new_1.cos_to_fit(file, in_dir, out_dir)
