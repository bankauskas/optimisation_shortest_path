import os
# import pyodbc 
import pandas             as pd
import numpy              as np

__path__ = os.path.dirname(os.path.realpath(__file__))
os.chdir(__path__)

data = pd.read_csv(
    'input/WED_setup_data.csv', sep=';', header=0)

schema = pd.read_csv('input/WED_setup_data_headers.csv', 
    sep=';', 
    header=0,
    encoding = "ISO-8859-1", 
    engine='python')

def set_dummies(data, dummies):
    for dummy in dummies:
        df_dummies = pd.get_dummies(data[dummy], prefix=dummy)
        
    return pd.concat([data, df_dummies], axis=1)

drop = schema.loc[schema['KEEP/DROP']=='drop']['ID'].to_list() 
continuous = schema.loc[(schema['DATA_TYPE']=='continuous') & (schema['KEEP/DROP']!='drop')]['ID'].to_list()  
id = ['id']
categorical = schema.loc[(schema['DATA_TYPE']=='categorical') & (schema['KEEP/DROP']!='drop')]['ID'].to_list()  
binary = schema.loc[(schema['DATA_TYPE']=='binary') & (schema['KEEP/DROP']!='drop')]['ID'].to_list()  


data[continuous] = data[continuous].astype(np.float64)
data[categorical] = data[categorical].astype(np.object)
data[binary] = data[binary].astype(np.uint8)
data[id] = data[id].astype(np.object)





# data = set_dummies(data, categorical)
# data = data.drop(drop+categorical, axis=1)

# data.to_csv('../output/mungy/data1.csv')


# data[continuous] = data[continuous].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

# data.to_csv('../output/mungy/data2.csv', index = False)