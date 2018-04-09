import pandas             as pd
import numpy              as np
import seaborn            as sns
import matplotlib.pyplot  as plt
import os
import tabulate


os.chdir('./optimisation_shortest_path')
print(os.getcwd())

df = pd.read_csv('./output/data_dm.csv')

for x in df.iterrows():
    print(x)

pass