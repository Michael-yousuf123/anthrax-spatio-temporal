import numpy as np
import pandas as pd 
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
sns.set(style='darkgrid', context='notebook', color_codes=True)
plt.style.use('ggplot')
large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
def timesplots(x, title = None):
    fig, axes = plt.subplots(figsize = (18, 12), dpi = 80)
    x.plot(ax = axes)
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)
    plt.show()
