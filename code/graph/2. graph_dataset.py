#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
import pandas as pd
import seaborn as sns
from utils import highlight_print, reduce_memory

import torch
torch.manual_seed(42)
import networkx as nx
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


# In[ ]:


levels = [
    [],                        # Level 1: Total
    ['state_id'],              # Level 2: State
    ['store_id'],              # Level 3: Store
    ['cat_id'],                # Level 4: Category
    ['dept_id'],               # Level 5: Department
    ['state_id', 'cat_id'],    # Level 6: State-Category
    ['state_id', 'dept_id'],   # Level 7: State-Department
    ['store_id', 'cat_id'],    # Level 8: Store-Category
    ['store_id', 'dept_id'],   # Level 9: Store-Department
    ['item_id'],               # Level 10: Item
    ['item_id', 'state_id'],   # Level 11: Item-State
    ['item_id', 'store_id']    # Level 12: Item-Store
]

node_list = []
level_to_node = {}
node_to_idx = {}
level_to_idx = {}

feature_list = [] # node x time x feature
target_list = [] # node x time

for level_idx, level in enumerate(levels, start=1):
    highlight_print(f"Preparing dataset for level {level_idx}")
    datasets = {'train': {}, 'valid': {}, 'test': {}}

    agg_df = pd.read_csv(f'../../data/preprocessed/agg_df_level_{level_idx}.csv')
    calendar_df = pd.read_csv('../../data/preprocessed/calendar_df.csv')

    agg_df = reduce_memory(agg_df)
    calendar_df = reduce_memory(calendar_df)

    start_date = pd.to_datetime('2011-01-29')
    valid_start_date = pd.to_datetime('2016-03-28')
    agg_df['d'] = agg_df['d'].apply(lambda x: int(x.split('_')[1]) - 1)
    agg_df['d'] = start_date + pd.to_timedelta(agg_df['d'], unit='D')
    calendar_df['d'] = calendar_df['d'].apply(lambda x: int(x.split('_')[1]) - 1)
    calendar_df['d'] = start_date + pd.to_timedelta(calendar_df['d'], unit='D')
    
    if len(level) == 0:
        agg_df.insert(1, 'id', 'total')
    elif len(level) == 1: 
        agg_df.insert(1, 'id', agg_df[level[0]])
        del agg_df[level[0]]
    elif len(level) > 1:
        agg_df.insert(1, 'id', agg_df[level[0]] + '_' + agg_df[level[1]])
        del agg_df[level[0]]
        del agg_df[level[1]]

    nodes = agg_df['id'].unique()
    level_to_node[level_idx] = nodes
    
    nodes_df = agg_df.merge(calendar_df, on="d", how="left")
    for node in nodes:
        feature = nodes_df[nodes_df['id'] == node].drop(columns=['level', 'id', 'd']).values[:-9] # time x feature
        feature_list.append(np.array(feature))
        target = nodes_df[nodes_df['id'] == node]['sales_sum'].values[:-9] # time
        target_list.append(np.array(target))

for level, nodes in level_to_node.items(): 
    level_to_idx[level] = []
    for node in nodes:
        node_to_idx[node] = len(node_list)
        node_list.append(node)
        level_to_idx[level].append(node_to_idx[node])


# In[ ]:


save_dir = '../../data'
os.makedirs(save_dir, exist_ok=True)

# # save
# file_dict = {
#     'node_list': node_list,
#     'level_to_node': level_to_node,
#     'node_to_idx': node_to_idx,
#     'level_to_idx': level_to_idx,
#     'feature_list': feature_list,
#     'target_list': target_list,
# }
# for file_name, data in file_dict.items():
#     file_path = os.path.join(save_dir, f"{file_name}.pkl")
#     with open(file_path, 'wb') as f:
#         pickle.dump(data, f)

# load
for filename in os.listdir(save_dir):
    if filename.endswith('.pkl'):
        var_name = filename[:-4]
        with open(os.path.join(save_dir, filename), 'rb') as f:
            globals()[var_name] = pickle.load(f)


# In[ ]:


# A3TGCN

window_feature_list = [] # num window x node x feature x window size time
window_target_list = [] # num window x node x window size time

window_size = 28
num_window = feature_list[0].shape[0] // window_size

for window_idx in range(num_window):
    window_feature = [] # node x window size time x feature
    for node_idx in range(len(feature_list)):
        feature = feature_list[node_idx][window_idx*window_size:(window_idx+1)*window_size] # window size time x feature
        window_feature.append(feature)
    window_feature = np.array(window_feature)
    window_feature = np.transpose(window_feature, (0, 2, 1)) # node x feature x window_size time
    window_feature_list.append(window_feature)
    
    window_target = [] # node x window_size
    for node_idx in range(len(target_list)):
        target = target_list[node_idx][(window_idx+1)*window_size:(window_idx+2)*window_size]  # window size time
        window_target.append(target)
    window_target = np.array(window_target)
    window_target_list.append(window_target)


# In[ ]:


save_dir = '../../data/graph/A3TGCN'
os.makedirs(save_dir, exist_ok=True)

# # save
# file_dict = {
#     'windowed_feature_list': windowed_feature_list,
#     'windowed_target_list': windowed_target_list
# }
# for file_name, data in file_dict.items():
#     file_path = os.path.join(save_dir, f"{file_name}.pkl")
#     with open(file_path, 'wb') as f:
#         pickle.dump(data, f)

# load
for filename in os.listdir(save_dir):
    if filename.endswith('.pkl'):
        var_name = filename[:-4]
        with open(os.path.join(save_dir, filename), 'rb') as f:
            globals()[var_name] = pickle.load(f)


# In[ ]:


print(window_target_list[0].shape)
print(window_feature_list[0].shape)


# In[ ]:


# else
# 노드 수, feature

total_feature_list = [] # total time x node x feature
total_target_list = [] # total time x node

for window_idx in range(num_window):
    window_feature = [] # node x window_size x feature
    for node_idx in range(len(feature_list)):
        feature = feature_list[node_idx][window_idx*window_size:(window_idx+1)*window_size] # window size x feature
        window_feature.append(feature)
    window_feature = np.array(window_feature)
    window_feature = np.transpose(window_feature, (0, 2, 1)) # node x feature x window_size
    windowed_feature_list.append(window_feature)
    
    window_target = [] # node x window_size
    for node_idx in range(len(target_list)):
        target = target_list[node_idx][(window_idx+1)*window_size:(window_idx+2)*window_size]  # window size
        window_target.append(target)
    window_target = np.array(window_target)
    windowed_target_list.append(window_target)


# In[ ]:


save_dir = '../../data/graph/else'
os.makedirs(save_dir, exist_ok=True)

# # save
# file_dict = {
#     'windowed_feature_list': windowed_feature_list,
#     'windowed_target_list': windowed_target_list
# }
# for file_name, data in file_dict.items():
#     file_path = os.path.join(save_dir, f"{file_name}.pkl")
#     with open(file_path, 'wb') as f:
#         pickle.dump(data, f)

# load
for filename in os.listdir(save_dir):
    if filename.endswith('.pkl'):
        var_name = filename[:-4]
        with open(os.path.join(save_dir, filename), 'rb') as f:
            globals()[var_name] = pickle.load(f)


# In[ ]:


print(windowed_target_list[0].shape)
print(windowed_feature_list[0].shape)


# In[ ]:


# NetworkX 그래프
G = nx.DiGraph()

# 노드 추가
for node, idx in node_to_idx.items():
    G.add_node(idx, name=node)

# 엣지 추가
# 지리적 계층
if 1 in level_to_idx and 2 in level_to_idx:  # Total <-> State
    total_idx = level_to_idx[1][0]
    for state_idx in level_to_idx[2]:
        G.add_edge(total_idx, state_idx, type='geo_hierarchy', direction='down')
        G.add_edge(state_idx, total_idx, type='geo_hierarchy', direction='up')

if 2 in level_to_idx and 3 in level_to_idx:  # State <-> Store
    for state_idx in level_to_idx[2]:
        state_name = node_list[state_idx]
        for store_idx in level_to_idx[3]:
            store_name = node_list[store_idx]
            if store_name.startswith(state_name):
                G.add_edge(state_idx, store_idx, type='geo_hierarchy', direction='down')
                G.add_edge(store_idx, state_idx, type='geo_hierarchy', direction='up')

# 상품 계층
if 1 in level_to_idx and 4 in level_to_idx:  # Total <-> Category
    total_idx = level_to_idx[1][0]
    for cat_idx in level_to_idx[4]:
        G.add_edge(total_idx, cat_idx, type='prod_hierarchy', direction='down')
        G.add_edge(cat_idx, total_idx, type='prod_hierarchy', direction='up')

if 4 in level_to_idx and 5 in level_to_idx:  # Category <-> Department
    for cat_idx in level_to_idx[4]:
        cat_name = node_list[cat_idx]
        for dept_idx in level_to_idx[5]:
            dept_name = node_list[dept_idx]
            if dept_name.startswith(cat_name):
                G.add_edge(cat_idx, dept_idx, type='prod_hierarchy', direction='down')
                G.add_edge(dept_idx, cat_idx, type='prod_hierarchy', direction='up')

if 5 in level_to_idx and 10 in level_to_idx:  # Department <-> Item
    for dept_idx in level_to_idx[5]:
        dept_name = node_list[dept_idx]
        for item_idx in level_to_idx[10]:
            item_name = node_list[item_idx]
            if dept_name.startswith(item_name):
                G.add_edge(dept_idx, item_idx, type='prod_hierarchy', direction='down')
                G.add_edge(item_idx, dept_idx, type='prod_hierarchy', direction='up')

# 결합 계층
if 2 in level_to_idx and 6 in level_to_idx: # State <-> State x Category
    for state_idx in level_to_idx[2]:
        state_name = node_list[state_idx]
        for state_cat_idx in level_to_idx[6]:
            state_cat_name = node_list[state_cat_idx]
            if state_cat_name.startswith(state_name):
                G.add_edge(state_idx, state_cat_idx, type='agg_hierarchy', direction='down')
                G.add_edge(state_cat_idx, state_idx, type='agg_hierarchy', direction='up')

if 4 in level_to_idx and 6 in level_to_idx: # Category <-> State x Category
    for cat_idx in level_to_idx[4]:
        cat_name = node_list[cat_idx]
        for state_cat_idx in level_to_idx[6]:
            state_cat_name = node_list[state_cat_idx]
            if state_cat_name.endswith(cat_name):
                G.add_edge(cat_idx, state_cat_idx, type='agg_hierarchy', direction='down')
                G.add_edge(state_cat_idx, cat_idx, type='agg_hierarchy', direction='up')

if 2 in level_to_idx and 7 in level_to_idx: # State <-> State x Department
    for state_idx in level_to_idx[2]:
        state_name = node_list[state_idx]
        for state_dept_idx in level_to_idx[7]:
            state_dept_name = node_list[state_dept_idx]
            if state_dept_name.startswith(state_name):
                G.add_edge(state_idx, state_dept_idx, type='agg_hierarchy', direction='down')
                G.add_edge(state_dept_idx, state_idx, type='agg_hierarchy', direction='up')

if 5 in level_to_idx and 7 in level_to_idx: # Department <-> State x Department
    for dept_idx in level_to_idx[5]:
        dept_name = node_list[dept_idx]
        for state_dept_idx in level_to_idx[7]:
            state_dept_name = node_list[state_dept_idx]
            if state_dept_name.endswith(dept_name):
                G.add_edge(dept_idx, state_dept_idx, type='agg_hierarchy', direction='down')
                G.add_edge(state_dept_idx, dept_idx, type='agg_hierarchy', direction='up')

if 2 in level_to_idx and 11 in level_to_idx: # State <-> State x Item
    for state_idx in level_to_idx[2]:
        state_name = node_list[state_idx]
        for item_state_idx in level_to_idx[11]:
            item_state_name = node_list[item_state_idx]
            if item_state_name.endswith(state_name):
                G.add_edge(state_idx, item_state_idx, type='agg_hierarchy', direction='down')
                G.add_edge(item_state_idx, state_idx, type='agg_hierarchy', direction='up')

if 10 in level_to_idx and 11 in level_to_idx: # Item <-> State x Item
    for item_idx in level_to_idx[10]:
        item_name = node_list[item_idx]
        for item_state_idx in level_to_idx[11]:
            item_state_name = node_list[item_state_idx]
            if item_state_name.startswith(item_name):
                G.add_edge(item_idx, item_state_idx, type='agg_hierarchy', direction='down')
                G.add_edge(item_state_idx, item_idx, type='agg_hierarchy', direction='up')

if 3 in level_to_idx and 8 in level_to_idx: # Store <-> Store x Category
    for store_idx in level_to_idx[3]:
        store_name = node_list[store_idx]
        for store_cat_idx in level_to_idx[8]:
            store_cat_name = node_list[store_cat_idx]
            if store_cat_name.startswith(store_name):
                G.add_edge(store_idx, store_cat_idx, type='agg_hierarchy', direction='down')
                G.add_edge(store_cat_idx, store_idx, type='agg_hierarchy', direction='up')

if 4 in level_to_idx and 8 in level_to_idx: # Category <-> Store x Category
    for cat_idx in level_to_idx[4]:
        cat_name = node_list[cat_idx]
        for store_cat_idx in level_to_idx[8]:
            store_cat_name = node_list[store_cat_idx]
            if store_cat_name.endswith(cat_name):
                G.add_edge(cat_idx, store_cat_idx, type='agg_hierarchy', direction='down')
                G.add_edge(store_cat_idx, cat_idx, type='agg_hierarchy', direction='up')

if 3 in level_to_idx and 9 in level_to_idx: # Store <-> Store x Department
    for store_idx in level_to_idx[3]:
        store_name = node_list[store_idx]
        for store_dept_idx in level_to_idx[9]:
            store_dept_name = node_list[store_dept_idx]
            if store_dept_name.startswith(store_name):
                G.add_edge(store_idx, store_dept_idx, type='agg_hierarchy', direction='down')
                G.add_edge(store_dept_idx, store_idx, type='agg_hierarchy', direction='up')

if 5 in level_to_idx and 9 in level_to_idx: # Department <-> Store x Department
    for dept_idx in level_to_idx[5]:
        dept_name = node_list[dept_idx]
        for store_dept_idx in level_to_idx[7]:
            store_dept_name = node_list[store_dept_idx]
            if store_dept_name.endswith(dept_name):
                G.add_edge(dept_idx, store_dept_idx, type='agg_hierarchy', direction='down')
                G.add_edge(store_dept_idx, dept_idx, type='agg_hierarchy', direction='up')

if 3 in level_to_idx and 12 in level_to_idx: # Store <-> Store x Item
    for store_idx in level_to_idx[3]:
        store_name = node_list[store_idx]
        for item_store_idx in level_to_idx[12]:
            item_store_name = node_list[item_store_idx]
            if item_store_name.endswith(store_name):
                G.add_edge(store_idx, item_store_idx, type='agg_hierarchy', direction='down')
                G.add_edge(item_store_idx, store_idx, type='agg_hierarchy', direction='up')

if 10 in level_to_idx and 12 in level_to_idx: # Item <-> Store x Item
    for item_idx in level_to_idx[10]:
        item_name = node_list[item_idx]
        for item_store_idx in level_to_idx[12]:
            item_store_name = node_list[item_store_idx]
            if item_store_name.startswith(item_name):
                G.add_edge(item_idx, item_store_idx, type='agg_hierarchy', direction='down')
                G.add_edge(item_store_idx, item_idx, type='agg_hierarchy', direction='up')

# 크로스 계층
if 6 in level_to_idx and 7 in level_to_idx: # State x Category <-> State x Department
    for state_cat_idx in level_to_idx[6]:
        state_cat_name = node_list[state_cat_idx]
        for state_dept_idx in level_to_idx[7]:
            state_dept_name = node_list[state_dept_idx]
            if state_dept_name.startswith(state_cat_name):
                G.add_edge(state_cat_idx, state_dept_idx, type='cross_hierarchy', direction='down')
                G.add_edge(state_dept_idx, state_cat_idx, type='cross_hierarchy', direction='up')

if 7 in level_to_idx and 11 in level_to_idx: # State x Department <-> State x Item
    for state_dept_idx in level_to_idx[7]:
        state_dept_name = node_list[state_dept_idx]
        state_dept_parts = state_dept_name.split('_')
        state = state_dept_parts[0]
        dept = '_'.join(state_dept_parts[1:])
        for state_item_idx in level_to_idx[11]:
            state_item_name = node_list[state_item_idx]
            state_item_parts = state_item_name.split('_')
            item_dept = state_item_parts[0] + '_' + state_item_parts[1]
            item_state = state_item_parts[-1]
            if dept == item_dept and state == item_state:
                G.add_edge(state_dept_idx, state_item_idx, type='cross_hierarchy', direction='down')
                G.add_edge(state_item_idx, state_dept_idx, type='cross_hierarchy', direction='up')

if 8 in level_to_idx and 9 in level_to_idx: # Store x Category <-> Store x Department
    for store_cat_idx in level_to_idx[8]:
        store_cat_name = node_list[store_cat_idx]
        for store_dept_idx in level_to_idx[9]:
            store_dept_name = node_list[store_dept_idx]
            if store_dept_name.startswith(store_cat_name):
                G.add_edge(store_cat_idx, store_dept_idx, type='cross_hierarchy', direction='down')
                G.add_edge(store_dept_idx, store_cat_idx, type='cross_hierarchy', direction='up')

if 9 in level_to_idx and 12 in level_to_idx: # Store x Department <-> Store x Item
    for store_dept_idx in level_to_idx[9]:  
        store_dept_name = node_list[store_dept_idx]  
        store_dept_parts = store_dept_name.split('_')
        store = store_dept_parts[0] + '_' + store_dept_parts[1]     
        dept = store_dept_parts[2] + '_' + store_dept_parts[3]     
        for item_store_idx in level_to_idx[12]: 
            item_store_name = node_list[item_store_idx]
            item_store_parts = item_store_name.split('_')
            item_dept = item_store_parts[0] + '_' + item_store_parts[1]
            item_store = item_store_parts[3] + '_' + item_store_parts[4]    
            if dept == item_dept and store == item_store:
                G.add_edge(store_dept_idx, item_store_idx, type='cross_hierarchy', direction='down')
                G.add_edge(item_store_idx, store_dept_idx, type='cross_hierarchy', direction='up')

if 6 in level_to_idx and 8 in level_to_idx: # State x Category <-> Store x Category
    for state_cat_idx in level_to_idx[6]:
        state_cat_name = node_list[state_cat_idx]
        state_cat_parts = state_cat_name.split('_')
        state = state_cat_parts[0]
        category = state_cat_parts[1]
        for store_cat_idx in level_to_idx[7]:
            store_cat_name = node_list[store_cat_idx]
            store_cat_parts = store_cat_name.split('_')
            store_state = store_cat_parts[0]
            store_category = store_cat_parts[2]
            if state == store_state and category == store_category:
                G.add_edge(state_cat_idx, store_cat_idx, type='cross_hierarchy', direction='down')
                G.add_edge(store_cat_idx, state_cat_idx, type='cross_hierarchy', direction='up')

if 7 in level_to_idx and 9 in level_to_idx: # State x Department <-> Store x Department
    for state_dept_idx in level_to_idx[7]:
        state_dept_name = node_list[state_dept_idx]
        state_dept_parts = state_dept_name.split('_')
        state = state_dept_parts[0]          
        dept = state_dept_parts[1] + '_' + state_dept_parts[2]
        for store_dept_idx in level_to_idx[9]:
            store_dept_name = node_list[store_dept_idx]
            store_dept_parts = store_dept_name.split('_')
            store_state = store_dept_parts[0]     
            store_dept = store_dept_parts[2] + '_' + store_dept_parts[3]
            if state == store_state and dept == store_dept:
                G.add_edge(state_dept_idx, store_dept_idx, type='cross_hierarchy', direction='down')
                G.add_edge(store_dept_idx, state_dept_idx, type='cross_hierarchy', direction='up')

if 11 in level_to_idx and 12 in level_to_idx: # Item x State <-> Item x Store
    for item_state_idx in level_to_idx[11]:
        item_state_name = node_list[item_state_idx]
        for item_store_idx in level_to_idx[12]: 
            item_store_name = node_list[item_store_idx]
            if item_store_name.startswith(item_state_name):
                G.add_edge(item_state_idx, item_store_idx, type='cross_hierarchy', direction='down')
                G.add_edge(item_store_idx, item_state_idx, type='cross_hierarchy', direction='up')


# In[ ]:


save_dir = '../../data/graph'
os.makedirs(save_dir, exist_ok=True)

# # save
# file_path = os.path.join(save_dir, f"graph.pkl")
# with open(file_path, 'wb') as f:
#     pickle.dump(G, f)

# load
file_path = os.path.join(save_dir, f"graph.pkl")
with open(file_path, 'rb') as f:
    G = pickle.load(f)


# In[ ]:


edge_list = list(G.edges())
edge_index = np.array([[u, v] for u, v in edge_list]).T
edge_index = torch.tensor(edge_index, dtype=torch.long)

dataset = StaticGraphTemporalSignal(
    edge_index=edge_index,
    edge_weight=None,
    features=windowed_feature_list,
    targets=windowed_target_list
)


# In[ ]:


save_dir = '../../dataset/graph'
os.makedirs(save_dir, exist_ok=True)

# save
file_path = os.path.join(save_dir, f"dataset.pkl")
with open(file_path, 'wb') as f:
    pickle.dump(dataset, f)

