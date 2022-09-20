import netCDF4
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

######## Histogram of variables ##########
variables = ['TMQ', 'U850', 'V850', 'UBOT',
             'VBOT', 'QREFHT', 'PS', 'PSL',
             'T200', 'T500', 'PRECT', 'TS',
             'TREFHT', 'Z1000', 'Z200', 'ZBOT']
PATH = Path('/Users/choimindong/src/Climate/data/train')
files = sorted(list(PATH.iterdir()))
for i, var in enumerate(variables):
    whole_train = np.zeros((185, 768, 1152))
    for j, p in enumerate(files):
        with netCDF4.Dataset(p) as f:
            whole_train[j] += f.variables[var][:].squeeze()
    fig = plt.figure(figsize=(10, 12))
    plt.hist(whole_train.flatten(), bins=100)
    plt.title(str(variables[i]))
    plt.show()

plt.hist(whole_train.flatten(), bins=100)
plt.title(str(variables[i]))
plt.show()
