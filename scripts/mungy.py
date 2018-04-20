import os
import pyodbc 
import pandas             as pd
import numpy              as np

__path__ = os.path.dirname(os.path.realpath(__file__))
os.chdir(__path__)

conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=Z:\Swedwood KR\Department Production\Technologist\BOM\DB\Inventories_v3_be.accdb;'
    r'UID=user;'
    r'PWD=;'
    )
cnxn = pyodbc.connect(conn_str)

sql = """   
     SELECT 
        ITEMCHAR.WC as WorkCenter,
        ITEMCHAR.ITEM_ID as ProductAGR,
        ITEMCHAR.ITEM_NAME as ProductName,
        ITEMCHAR.NPC_MAX as NPC, 
        ITEMCHAR.QTY_BLANK_1 as Parts,
        ITEMCHAR.MARSRUTAS as Path,
        ITEMCHAR.WIDTH_1 as Width,
        ITEMCHAR.LENGHT_1 as Length,      
        ITEMCHAR.Liftas_Past as LiftPushSpeed,
        ITEMCHAR.Liftas_nuos_veikimas as LiftRegularSpeed,
        ITEMCHAR.pas_stal_iejimas as RT1EntranceSpeed,
        ITEMCHAR.pas_stal_isejimas as RT1ExitSpeed,     
        ITEMCHAR.[2_pas_stal_iejimas] as RT2EntranceSpeed,
        ITEMCHAR.[2_pas_stal_isejimas] as RT2ExitSpeed,
        ITEMCHAR.[1_krast_stak_pastuma] as EM1PushSpeed,
        ITEMCHAR.[1_krast_stak_Ciklas] as EM1PushCycle,
        ITEMCHAR.[2_krast_stak_pastuma] as EM2PushSpeed, 
        ITEMCHAR.[2_krast_stak_Ciklas] as EM2PushCycle,  
        ITEMCHAR.[1_krast_stak_Kumstelio_sk] as Position,        
        
        LEFT(ITEM_ID, inStr(ITEM_ID, '-')+2) +
            COLLORS.COLINX +
            RIGHT(ITEM_ID, LEN(ITEM_ID)-(inStr(ITEM_ID, '-')+3)) as child,
            
        LEFT(ITEM_ID, inStr(ITEM_ID, '-')-1) as BR 
        
    FROM ITEMCHAR, COLLORS
    WHERE 
        ITEMCHAR.COLOR_INDEX = COLLORS.UNKNOWN AND 
        WC in (12010) AND
        ITEMCHAR.archivuotas = False
        
    """
TechData = pd.read_sql(sql,cnxn)

scalaDB_conn_str = (
    r'DSN=scalaDB;'
    r'UID=jakmar-adm;'
    r'Trusted_Connection=Yes;'
    r'APP=Jupyter Notebook;'
    r'WSID=LTKAZ-NB0034;'
    r'DATABASE=scalaSW'
    )
cnxn = pyodbc.connect(scalaDB_conn_str)

sql = """   
    SELECT DISTINCT
        SC01.SC01001 as 'child',
        MP63.MP63007 as 'wc'
    FROM 
        "scalaSW"."dbo"."SC01SW00" SC01,
        "scalaSW"."dbo"."MP63SW00" MP63
    WHERE 
        MP63.MP63001 like 'M' AND
        MP63.MP63002 = SC01.SC01001 AND
        (SC01.SC01091 = 0 OR SC01.SC01091 = '0') AND
        MP63.MP63007 = 12010 AND
        MP63.MP63004 like '010'
"""

Path = pd.read_sql(sql,cnxn)

BOM = pd.read_csv('../input/data_BOM.csv', usecols=['child']).drop_duplicates('child')

data = BOM.merge(Path, how='inner', on = 'child')\
    .merge(TechData, how='left', on = 'child')
    
print('Data:{}, Path:{}, BOM:{}'.format(len(data), len(Path), len(BOM)))

def set_chipboard(row):
    if row['child'][-1] == 'A': return 1
    else: return 0


# set color names by symbols in ProductID
def set_color(row):
    if row['child'][6] == 'W': return 'white'
    elif row['child'][6] == 'B': return 'black'
    elif row['child'][6] == 'O': return 'oak'
    elif row['child'][6] == 'C': return 'chary'
    elif row['child'][6] == 'X': return 'none'
    else: return ''
     

# set path main or not
def set_path(row):
    if row['Path'] == 1: return 1
    else: return 0

data = data\
    .assign(Chipboard=data.apply(set_chipboard, axis=1))\
    .assign(Color=data.apply(set_color, axis=1))\
    .assign(Path=data.apply(set_path, axis=1))

data.to_csv('../output/mungy/data0.csv')

def set_dummies(data, dummies):
    for dummy in dummies:
        df_dummies = pd.get_dummies(data[dummy], prefix=dummy)
        
    return pd.concat([data, df_dummies], axis=1)

continuous = ['NPC', 'Width', 'Length', 'LiftPushSpeed', 'LiftRegularSpeed', 
              'RT1EntranceSpeed', 'RT1ExitSpeed', 'RT2EntranceSpeed', 'RT2ExitSpeed',
              'EM1PushSpeed', 'EM1PushCycle', 'EM2PushSpeed', 'EM2PushCycle']
id = ['child']
categorical = ['Position','BR', 'Color', 'Parts', 'Path']
binary = ['Chipboard']
drop = ['ProductAGR', 'ProductName', 'WorkCenter', 'wc']


data[continuous] = data[continuous].astype(np.float64)
data[categorical] = data[categorical].astype(np.object)
data[binary] = data[binary].astype(np.uint8)
data[id] = data[id].astype(np.object)

data = set_dummies(data, categorical)
data = data.drop(drop+categorical, axis=1)


data.to_csv('../output/mungy/data1.csv')


data[continuous] = data[continuous].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

data.to_csv('../output/mungy/data2.csv', index = False)