import numpy                    as np
import os
import pandas                   as pd

from sklearn.metrics            import silhouette_score
from sklearn.model_selection    import ParameterGrid
from sklearn.cluster            import DBSCAN


import numpy                    as np
import os
import pandas                   as pd

from sklearn.metrics            import silhouette_score
from sklearn.model_selection    import ParameterGrid
from sklearn.cluster            import DBSCAN


def transform_label(function):
    """
    >> input = [0, 0, 1, 0]
    (This Is clustering results for components of furniture.)
    ---------------------------------------------------------------------------
    >> output = [0, 0, 1, 1]
    (Wee change some cluster because its can't be difference in sesame 
    furniture.)
    ---------------------------------------------------------------------------
    * this function requires global variables <bom> , <idx>
    """
    def wrapped_item(*args, **kwargs):
              
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
        
        return clusters.merge(df, how='left', on='child').values
    return wrapped_item


@ transform_label
def model(model, options):
    """
    >> input:   model = any clustering model;
                options = {x:y}
    ---------------------------------------------------------------------------
    >> output = [0, 0, 1, 2]
    ---------------------------------------------------------------------------
    * this function requires global variable <values> 
    """ 
    return model.set_params(**options).fit(values).labels_


# def silhoute_score(label):
#     n_clusters_ = len(np.unique(label))
#     return silhouette_score(values, label) if n_clusters_ != 0 else -1
__file__ = 'Clusterization.ipynb'
__path__ = os.path.dirname(os.path.realpath(__file__))
os.chdir(__path__)

data = pd.read_csv('../output/data_nm.csv').set_index('Product')
bom = pd.read_csv('../input/data_bom.csv', usecols=['parent', 'child'])

values = data.values
idx = data.index.values


md = DBSCAN(n_jobs=-1)
op = {'eps':1, 'min_samples': 8}
x = model(md, op)
z = 1




def n_clusters_(label):
    return len(np.unique(label))


def scoring(metrics, label):
    return [measure(label, values) for measure in metrics]


def grid_search(model, param_grid, metrics):
    return {
            params: scoring(metrics, model(model, params)) 
            for params in ParameterGrid(param_grid)
        }


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

metrics = [n_clusters_]

x = grid_search(DBSCAN_model, param_grid, metrics)
