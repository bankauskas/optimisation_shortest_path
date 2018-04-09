import os
import pandas                   as pd
import matplotlib.pyplot        as plt
import numpy                    as np
import itertools

from sklearn                    import cluster
from sklearn                    import mixture
from collections                import defaultdict
from sklearn.metrics.cluster    import normalized_mutual_info_score
from sklearn.metrics.cluster    import adjusted_rand_score

from sklearn.metrics.pairwise   import euclidean_distances
from sklearn.metrics.pairwise   import cosine_distances
from sklearn.metrics.pairwise   import manhattan_distances

from sklearn                    import metrics
from sklearn.preprocessing      import StandardScaler
from sklearn.model_selection    import GridSearchCV

__path__ = os.path.dirname(os.path.realpath(__file__))
os.chdir(__path__)

def split_grid(parameters_grid):
    """Input: {x:[y]}, Output: [{x:y}]"""
    return [list_of_toople_to_dic([*zip(parameters_grid.keys(), values)]) 
            for values 
            in [*itertools.product(*parameters_grid.values())]]        
        
def list_of_toople_to_dic(values):
    """Input: [(x,y)], Output: {x:y}"""
    return {key:value for key, value in values}

def add_metrics(labels, values):
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    return {
        'Estimated number of clusters' : n_clusters_,
        'Silhouette Coefficient' : metrics.silhouette_score(df.values, labels) if n_clusters_ != 0 else -1
    }

df = pd.read_csv('../output/data_nm.csv').set_index('Product')

kClusters = 8
idx = list(df.index.values)
results = {'Product': idx}

parameters_grid = dict(
    eps=np.arange(0.1, 1.1, 0.1),
    min_samples = np.arange(1, 21, 1)
) 

metrics_results_grid = {}

for parameters in split_grid(parameters_grid):

    model = cluster.DBSCAN(n_jobs=-1, **parameters).fit(df.values)
    metrics_results_grid.setdefault('Parameters', []).append(parameters)
    metrics_results = add_metrics(model.labels_, df.values)

    for metric in metrics_results:
        metrics_results_grid.setdefault(metric, [])\
        .append(metrics_results[metric])


x=1