import numpy                    as np
import os
import pandas                   as pd


from sklearn.metrics            import silhouette_score
from sklearn.model_selection    import ParameterGrid
from sklearn.cluster            import DBSCAN
from BranchNBound import *


def transformLabel(function):
    """
    >> input:   [0, 0, 1, 0] as list
    (This Is clustering results for components of furniture.)
    ---------------------------------------------------------------------------
    >> output:  [0, 0, 1, 1] as list
    (Wee change some cluster because its can't be difference in sesame 
    furniture.)
    ---------------------------------------------------------------------------
    * this function requires global variables <bom> , <idx>
    """
    def wrappedItem(*args, **kwargs):
              
        clusters = pd.DataFrame({
            'child': idx, 
            'cluster': function(*args, **kwargs)
            })

        df = bom.merge(clusters, how='inner', on='child')\
            .groupby(['parent', 'cluster'], sort = False)['cluster']\
            .apply(lambda x: (x>=0).sum())\
            .reset_index(name='counts')  
  
        imax = df\
            .groupby(['parent'], sort = False)['counts']\
            .transform(max) == df['counts']
            
        df = df[imax].merge(bom, how='inner', on='parent')
        #df.to_csv('../output/data/labels/{0}__{1}.csv'.format(1,2))
        
        return clusters.merge(df, how='left', on='child').values
    return wrappedItem


@ transformLabel
def model(model, options):
    """
    >> input:   model = any clustering model;
                options = {x:y}
    ---------------------------------------------------------------------------
    >> output:  [0, 0, 1, 2] as list
    ---------------------------------------------------------------------------
    * this function requires global variable <values> 
    """ 
    return model.set_params(**options).fit(values).labels_


def silhouetteScore(label):
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
    n_clusters_ = len(np.unique(label))
    return silhouette_score(values, label) if n_clusters_ != 0 else -1


def n_clusters_(label):
    """
    Function returns number of clusters unique elements in list.

    >> input:   [0, 0, 1, 2] as list
    ---------------------------------------------------------------------------
    >> output:  3 as int
    """
    return len(np.unique(label))


def scoring(metrics, label):
    """
    Function returns list of calculated metrics.

    >> input:   metrics = [<function>] as list
                labels = [0, 0, 1, 2] as list
    ---------------------------------------------------------------------------
    >> output:  [1] as list
    """
    return [measure(label) for measure in metrics]


def gridSearch(model, param_grid, metrics):
    return {
            params: scoring(metrics, model(model, params)) 
            for params in ParameterGrid(param_grid)
        }


def Tree(labels):
    """
    Create dictionary of labels where item is list of idx for this labels
    >> input:   labels = [0, 1, 0, 1] as list;
    ---------------------------------------------------------------------------
    >> output:  {label:[idx]} as dic
    ---------------------------------------------------------------------------
    * this function requires global variable <idx> 
    """ 
    d={}
    return {d.setdefault(cluster,[]).append(name) for cluster, name in zip(idx,labels)}


def BranchNBound_(labels):
    """
    This loop for each cluster to callculate sho
    >> input:   labels = [0, 1, 0, 1] as list;
    ---------------------------------------------------------------------------
    >> output:  Saves results of branch and bound algorithm;
    """ 
    tree =  Tree(labels)
    results = []
    for cluster in tree:
        subData = data[tree[cluster]]
        subValues = subData.values
        subIdx = subData.index.values
        results.append(BranchNBound(subValues) + (subIdx,))






__path__ = os.path.dirname(os.path.realpath(__file__))
os.chdir(__path__)

data = pd.read_csv('../output/data_nm.csv').set_index('Product')
bom = pd.read_csv('../input/data_bom.csv', usecols=['parent', 'child'])

values = data.values
idx = data.index.values

param_grid = ParameterGrid(dict(
    eps=np.arange(0.1, 1.1, 0.2),
    min_samples = np.arange(1, 21, 2)
))

DBSCAN_model = DBSCAN(n_jobs=-1)

metrics = [n_clusters_, BranchNBound_]

dbscan = gridSearch(DBSCAN_model, param_grid, metrics)
dbscan.to_csv('../output/metrics/{}__{}.csv'.format('dbscan', '[n_clusters, gridSearch]'))
