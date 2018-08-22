import pandas as pd
import numpy as np

def get_per_attributes(attributes_file):
    attributes = pd.read_csv(attributes_file, index_col = 0)
    attributes = attributes.values[:, 1:]  # (230, 30)
    attributes = np.transpose(attributes)  # (30, 230)
    print("attributes.shape: ", attributes.shape)
        
    return attributes

def classfy(attri_pres):
    types = []
    
    for attri_pre in attri_pres:
        ty = max(attri_pre[200:])  
        types.append('ZJL'+str(ty))
    return types

def test_classfy(attri_pre, attris):
    res = []
    for i in range(len(attris)):
        dis = sum([(attri_pre[0][j] - attris[i][j])**2 for j in range(30)])
        res.append(dis)
   
    return 'ZJL'+str(res.index(min(res)) + 211)


