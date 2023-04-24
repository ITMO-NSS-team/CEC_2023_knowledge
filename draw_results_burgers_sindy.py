import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


path = 'data/'
df0 = pd.read_csv(f'{path}df0_sindy.csv')
df1 = pd.read_csv(f'{path}df1_sindy.csv')
df2 = pd.read_csv(f'{path}df2_sindy.csv')
df3 = pd.read_csv(f'{path}df3_sindy.csv')
df4 = pd.read_csv(f'{path}df4_sindy.csv')
df_sindy = pd.read_csv(f'{path}time_burg.csv')
df_ls = [df0, df1, df2, df3, df4]

title = "burg_mae_sindy"
feature = 'MAE'
columns = [f'Basic algorithm', f'Fixed initial distr.', f'Altered initial distr.',
           f'Highly altered distr.', f'Uniform initial distr.']
new_df = pd.DataFrame(columns=columns)
for i in range(len(columns)):
    col_values = df_ls[i][feature]
    col_values = np.round(col_values, 4)
    new_df[columns[i]] = col_values

matrix = new_df.values
new_df_text = pd.DataFrame(columns=columns)
ls_column = []
for i in range(len(columns)):
    ls_column = []
    for elem in matrix[:, i]:
        ls_column.append(str(elem))
    new_df_text[columns[i]] = ls_column

new_df.to_csv(f'{path}{title}.csv', index=False)


feature = 'time'
columns = [f'basic_algorithm', f'fixed_initial_distr', f'biased_distr',
           f'highly_biased_distr', f'uniform_distr', 'PySINDy']
new_df = pd.DataFrame(columns=columns)
for i in range(len(columns)-1):
    new_df[columns[i]] = df_ls[i][feature]
new_df['PySINDy'] = df_sindy['time']


ratio_k = 4
fig, (ax1, ax2) = plt.subplots(figsize=(16, 8), ncols=1, nrows=2, sharex=True,
                                        gridspec_kw={'hspace': 0.07, 'height_ratios': [ratio_k, 1]})
sns.boxplot(data=new_df[columns], orient="v", showfliers=False, ax=ax1)
sns.boxplot(data=new_df[columns], orient="v", showfliers=False, ax=ax2)
ax1.set_ylim(24.2, 31.7)
ax2.set_ylim(0.022, 0.028)

d = .005
kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - ratio_k*d, 1 + ratio_k*d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - ratio_k*d, 1 + ratio_k*d), **kwargs)

ax1.tick_params(axis='both', which='major', labelsize=24)
ax1.tick_params(axis='both', which='minor', labelsize=24)

ax2.tick_params(axis='both', which='major', labelsize=24)
ax2.tick_params(axis='both', which='minor', labelsize=24)
ax2.tick_params(axis='x', labelrotation=25, bottom=True)

ax1.grid()
ax2.grid()
fig.text(0.065, 0.5, "time, s", va="center", rotation="vertical", fontsize=24)

ax1.xaxis.tick_bottom()
plt.subplots_adjust(bottom=0.189, top=1.)
plt.show()
