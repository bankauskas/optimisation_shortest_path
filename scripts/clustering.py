import numpy as np
import os
import pandas as pd
import scipy as sp

from TSP_solver.algorithms.TwoOpt import * 

from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import DBSCAN

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_distances


def transformLabel(function):
    """
    >> input:   [0, 0, 1, 0] as list
    (This Is clustering results for components of furniture.)
    ---------------------------------------------------------------------------
    >> output:  [0, 0, 1, 1] as list
    (Wee change some cluster because its can't be difference in sesame 
    furniture.)
    ---------------------------------------------------------------------------
    *   this function requires global variables <bom> , <idx> as dict
        <model_name> as string, <parameters_name> as string
    """
    def wrappedItem(*args, **kwargs):
        start = time.clock()
        # Create dictionary {idx:label}      
        clusters = pd.DataFrame({
            'child': idx, 
            'cluster': function(*args, **kwargs)
            })

        # group by parent and clusters
        df = bom.merge(clusters, how='inner', on='child')\
            .groupby(['parent', 'cluster'], sort=False)['cluster']\
            .apply(lambda x: (x >= 0).sum())\
            .reset_index(name='counts') 

        # find main cluster for parrent
        imax = df\
            .groupby(['parent'], sort=False)['counts']\
            .transform(max) == df['counts']
 
        # filter main cluster and join with BOM
        df = df.loc[imax[imax == True].index.values]\
            .merge(bom, how='inner', on='parent')

        # return results to main sequence 
        df = clusters.merge(df, how='inner', on='child')

        # find main cluster for child
        imax = df\
            .groupby(['child'], sort=False)['counts']\
            .transform(max) == df['counts']

        # filter main cluster
        df = df.loc[imax[imax == True].index.values]
        df = df.drop_duplicates(subset=['child'], keep='first')    

        global model_name 
        model_name = args[0].__class__.__name__

        global parameters_name
        parameters_name = str(args[1])\
                .replace(': ', '_')\
                .replace("'", "")\
                .replace("{", "")\
                .replace("}", "")

        directory = '../output/models/{}'.format(model_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        subDirectory = directory + '/{}'.format(parameters_name)

        if not os.path.exists(subDirectory):
            os.makedirs(subDirectory)

        time_finish = time.clock() - start

        ClusteringResults['Model'].append(model_name)
        ClusteringResults['Parameters'].append(parameters_name)
        ClusteringResults['Clustering time'].append(time_finish)

        # save clusterisation results
        df.drop(['parent', 'counts'], axis=1)\
            .set_index('child')\
            .to_csv(subDirectory + '/labels.csv')
     
        return clusters.merge(df, how='left', on='child')['cluster_y'].values
    return wrappedItem


@ transformLabel
def model(model_, options):
    """
    >> input:   model = any clustering model;
                options = {x:y}
    ---------------------------------------------------------------------------
    >> output:  [0, 0, 1, 2] as list
    ---------------------------------------------------------------------------
    * this function requires global variable <values> 
    """ 
    return model_.set_params(**options).fit(values).labels_


def silhouetteScore(labels, **parameters):
    """
    The best value is 1 and the worst value is -1. Values near 0 indicate 
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.

    >> input:   [0, 0, 1, 2] as list
    ---------------------------------------------------------------------------
    >> output:  1.2 as int
    ---------------------------------------------------------------------------
    * this function requires global variable <values> 
    """
    n_clusters = n_clusters_(labels, **{})
    ClusteringResults['N clusters'].append(n_clusters)

    x = silhouette_score(values, labels) if int(n_clusters) > 1 else -1
    ClusteringResults['Silhouette Score'].append(x)
    return x


def n_clusters_(labels, **parameters):
    """
    Function returns number of clusters unique elements in list.

    >> input:   [0, 0, 1, 2] as list
    ---------------------------------------------------------------------------
    >> output:  3 as int
    ---------------------------------------------------------------------------
    """   
    return len(np.unique(labels))


def scoring(metrics, labels):
    """
    Function returns list of calculated metrics.

    >> input:   metrics = [(<function>, {parameters})] as list
                labels = [0, 0, 1, 2] as list
    ---------------------------------------------------------------------------
    >> output:  [1] as list
    """
    return [function(labels, **parameters) for function, parameters in metrics]


def gridSearch(model_, param_grid, metrics):

    return {str(params): scoring(metrics, model(model_, params)) for params in ParameterGrid(param_grid)}


def Tree(labels):
    """
    Create dictionary of labels where item is list of idx for this labels
    >> input:   labels = [0, 1, 0, 1] as list;
    ---------------------------------------------------------------------------
    >> output:  {label:[idx]} as dic
    ---------------------------------------------------------------------------
    * this function requires global variable <idx> 
    """ 
    dTree = {}
    for cluster, name in zip(labels, idx):
        dTree.setdefault(cluster, []).append(name)

    return dTree


def ShortestPath(labels, function):
    """
    This loop for each cluster to calculate shortest path
    >> input:   labels = [0, 1, 0, 1] as list;
    ---------------------------------------------------------------------------
    >> output:  Saves results of branch and bound algorithm;
    """ 

    tree = Tree(labels)
    results = {}
    for cluster in tree:
        subData = data.loc[tree[cluster]]
        subIdx = subData.index.values
        matrix = euclidean_distances(subData.values)
        results_ = function(matrix)

        results.setdefault('name', []).append(results_[0])
        results.setdefault('cost', []).append(results_[1])  
        results.setdefault('path', []).append(results_[2])  
        results.setdefault('duration', []).append(results_[3])  
        results.setdefault('path_idx', []).append(subIdx)         

    directory = '../output/models/{}/{}/{}'\
        .format(model_name, parameters_name, function.__name__)

    if not os.path.exists(directory):
            os.makedirs(directory)

    pd.DataFrame(results).to_csv(directory + '/results.csv')
 
    return True


def xfrange(start, stop, step, ndigits):
    """
    Generator to make float range
    >> input:   start = 0.0 as float;
                stop = 1.0 as float;
                step = 0.1 as float;
    ---------------------------------------------------------------------------
    >> output:  [0.0, 0.1, 0,2 ...]
    """
    i = 0
    while round(start + i * step, ndigits) < stop:
        yield round(start + i * step, ndigits)
        i += 1      

__path__ = os.path.dirname(os.path.realpath(__file__))
os.chdir(__path__)

data = pd.read_csv('../output/mungy/data2.csv').set_index('child')
bom = pd.read_csv('../input/data_BOM.csv', usecols=['parent', 'child'])

values = data.values
idx = data.index.values

param_grid = dict(
    eps=[*xfrange(0.1, 1.1, 0.2, 1)],
    min_samples=[*xfrange(1, 21, 2, 1)]
)

DBSCAN_model = DBSCAN(n_jobs=-1)

metrics = [
    (n_clusters_, {}),
    (silhouetteScore, {}), 
    (ShortestPath, {'function': two_opt})
]

model_name = ''
parameters_name = ''

ClusteringResults = {
    'Model':[],
    'Parameters':[],
    'Clustering time':[],
    'N clusters':[],
    'Silhouette Score':[]}

dbscan = gridSearch(DBSCAN_model, param_grid, metrics)


x = pd.DataFrame(ClusteringResults)
pd.DataFrame(ClusteringResults).to_csv('../output/models/results.csv')

dbscan.to_csv('../output/metrics/{}__{}.csv'.format('dbscan', '[n_clusters, gridSearch]'))
