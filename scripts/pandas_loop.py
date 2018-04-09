
#%% Import modules
import pandas             as pd
import numpy              as np
import seaborn            as sns
import matplotlib.pyplot  as plt
import os
import tabulate
import probscale 

# os.chdir('./optimisation_shortest_path')

#%% Load data to pandas data frame
df = pd.read_csv('./output/data_dm.csv')
list(df['NPC'])


#%% QQ Plot
features = df.columns.values.tolist()
fig, axes = plt.subplots(4,4, figsize=(20,20), sharex=True)
axes = axes.flatten()


for ax, attribute in zip(axes, features):
    probscale.probplot(
        df[attribute].tolist(), ax=ax, bestfit=True, estimate_ci=True,
        line_kws={'label': 'BF Line', 'color': 'b'},
        scatter_kws={'label': 'Observations'},
        problabel='Probability (%)'
    )

    ax.legend(loc='lower right')
    ax.set_ylim(bottom=-2, top=4)
    sns.despine(fig)
