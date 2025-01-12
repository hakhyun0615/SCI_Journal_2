import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName

'''
id (item_id + store_id)
item_id
dept_id: ['HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2', 'FOODS_1', 'FOODS_2', 'FOODS_3']
cat_id: ['HOBBIES', 'HOUSEHOLD', 'FOODS']
store_id: ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
state_id" ['CA', 'TX', 'WI']
'''

# Convert weekly price information into daily price information for each item.
def convert_price_file(data_path):
    # load data
    calendar = pd.read_csv(f'{data_path}/calendar.csv') # week, day unit
    sales_train_evaluation = pd.read_csv(f'{data_path}/sales_train_evaluation.csv')
    sell_prices = pd.read_csv(f'{data_path}/sell_prices.csv') # week unit (weekly prices)

    # drop last 28 days
    calendar = calendar.iloc[:1941, :]
    sell_prices = sell_prices[sell_prices.wm_yr_wk <= 11617]

    # assign price for all days
    week_and_day = calendar[['wm_yr_wk', 'd']]

    price_all_days_items = pd.merge(week_and_day, sell_prices, on=['wm_yr_wk'], how='left') # join on week number
    price_all_days_items = price_all_days_items.drop(['wm_yr_wk'], axis=1)

    # convert days to column
    price_all_items = price_all_days_items.pivot_table(values='sell_price', index=['store_id', 'item_id'], columns='d') 
    price_all_items.reset_index(drop=False, inplace=True)

    # reorder column
    price_all_items = price_all_items.reindex(['store_id','item_id'] + ['d_%d' % x for x in range(1,1941+1)], axis=1) 

    sales_keys = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales_keys_pd = sales_train_evaluation[sales_keys]

    # join with sales data
    price_converted = pd.merge(sales_keys_pd, price_all_items, on=['store_id','item_id'], how='left')

    return price_converted

def make_features(data_path):
    # convert price file
    converted_price = convert_price_file(data_path)

    # First we need to convert the provided M5 data into a format that is readable by GluonTS. 
    calendar = pd.read_csv(f'{data_path}/calendar.csv')
    sales_train_evaluation = pd.read_csv(f'{data_path}/sales_train_evaluation.csv')

    # drop last 28 days
    calendar = calendar.iloc[:1941, :]
    
    # target_value
    train_df = sales_train_evaluation.drop(["id","item_id","dept_id","cat_id","store_id","state_id"], axis=1)  # d_1 ~ d_1941
    target_values = train_df.values # (item: 30490, day: 1941)
    
    ##########################################################
    # FEAT_DYNAMIC_CAT
    
    # Event type
    event_type_to_idx = {"nan":0, "Cultural":1, "National":2, "Religious":3, "Sporting":4}
    event_type1 = np.array([event_type_to_idx[str(x)] for x in calendar['event_type_1'].values])
    event_type2 = np.array([event_type_to_idx[str(x)] for x in calendar['event_type_2'].values])

    # Event name
    event_name_to_idx = {'nan':0, 'Chanukah End':1, 'Christmas':2, 'Cinco De Mayo':3, 'ColumbusDay':4, 'Easter':5,
                        'Eid al-Fitr':6, 'EidAlAdha':7, "Father's day":8, 'Halloween':9, 'IndependenceDay':10, 'LaborDay':11,
                        'LentStart':12, 'LentWeek2':13, 'MartinLutherKingDay':14, 'MemorialDay':15, "Mother's day":16, 'NBAFinalsEnd':17,
                        'NBAFinalsStart':18, 'NewYear':19, 'OrthodoxChristmas':20, 'OrthodoxEaster':21, 'Pesach End':22, 'PresidentsDay':23,
                        'Purim End':24, 'Ramadan starts':25, 'StPatricksDay':26, 'SuperBowl':27, 'Thanksgiving':28, 'ValentinesDay':29, 'VeteransDay':30}

    event_name1 = np.array([event_name_to_idx[str(x)] for x in calendar['event_name_1'].values])
    event_name2 = np.array([event_name_to_idx[str(x)] for x in calendar['event_name_2'].values])

    event_features = np.stack([event_type1, event_type2, event_name1, event_name2])
    dynamic_cat = np.array([event_features] * len(sales_train_evaluation)) # (item(list): 30490, event type/name: 4, day: 1941)

    ##########################################################
    # FEAT_DYNAMIC_REAL        
    # SNAP_CA, TX, WI
    snap_features = calendar[['snap_CA', 'snap_TX', 'snap_WI']]
    snap_features = snap_features.values.T
    snap_features_expand = np.array([snap_features] * len(sales_train_evaluation)) # (item(list): 30490, snap type: 3, day: 1941)

    # sell_prices
    price_feature = converted_price.drop(["id","item_id","dept_id","cat_id","store_id","state_id"], axis=1).values # (item: 30490, day: 1941)

    # normalized sell prices per each item
    price_mean_per_item = np.nanmean(price_feature, axis=1, keepdims=True)
    price_std_per_item = np.nanstd(price_feature, axis=1, keepdims=True)
    normalized_price_per_item = (price_feature - price_mean_per_item) / (price_std_per_item + 1e-6)
    
    # normalized sell prices per day within the same dept
    numeric_cols = converted_price.select_dtypes(include=['float64']).columns
    dept_groups = converted_price[numeric_cols].groupby(converted_price['dept_id'])
    price_mean_per_dept = dept_groups.transform(np.nanmean)
    price_std_per_dept = dept_groups.transform(np.nanstd)
    normalized_price_per_group_pd = (converted_price[price_mean_per_dept.columns] - price_mean_per_dept) / (price_std_per_dept + 1e-6)
    normalized_price_per_group = normalized_price_per_group_pd.values
 
    price_feature = np.nan_to_num(price_feature) # nan -> 0
    normalized_price_per_item = np.nan_to_num(normalized_price_per_item)
    normalized_price_per_group = np.nan_to_num(normalized_price_per_group)

    all_price_features = np.stack([price_feature, normalized_price_per_item, normalized_price_per_group], axis=1) # (item(list): 30490, price type: 3, day: 1941)
    dynamic_real = np.concatenate([snap_features_expand, all_price_features], axis=1) # (item(list): 30490, price type: 6, day: 1941)
    
    ##########################################################
    # FEAT_STATIC_CAT
    # We then go on to build static features (features which are constant and series-specific). 
    # Here, we make use of all categorical features that are provided to us as part of the M5 data.
    state_ids = sales_train_evaluation["state_id"].astype('category').cat.codes.values # (item: 30490, )
    state_ids_un , _ = np.unique(state_ids, return_counts=True)

    store_ids = sales_train_evaluation["store_id"].astype('category').cat.codes.values
    store_ids_un , _ = np.unique(store_ids, return_counts=True)

    cat_ids = sales_train_evaluation["cat_id"].astype('category').cat.codes.values
    cat_ids_un , _ = np.unique(cat_ids, return_counts=True)

    dept_ids = sales_train_evaluation["dept_id"].astype('category').cat.codes.values
    dept_ids_un , _ = np.unique(dept_ids, return_counts=True)

    item_ids = sales_train_evaluation["item_id"].astype('category').cat.codes.values
    item_ids_un , _ = np.unique(item_ids, return_counts=True)

    stat_cat_list = [item_ids, dept_ids, cat_ids, store_ids, state_ids]

    stat_cat = np.concatenate(stat_cat_list) # (152450, )
    stat_cat = stat_cat.reshape(len(stat_cat_list), len(item_ids)).T # (item: 30490, stat: 5)

    stat_cat_cardinalities = [len(item_ids_un), len(dept_ids_un), len(cat_ids_un), len(store_ids_un), len(state_ids_un)] # [3049, 7, 3, 10, 3]

    return target_values, dynamic_real, dynamic_cat, stat_cat, stat_cat_cardinalities

def make_dataset(data_path):    

    ##########################################################
    # make features
    # target_values: (item: 30490, day: 1941)
    # dynamic_real: (item(list): 30490, price type: 6, day: 1941)
    # dynamic_cat: (item(list): 30490, event type/name: 4, day: 1941)
    # stat_cat: (item: 30490, stat: 5)
    # stat_cat_cardinalities: [3049, 7, 3, 10, 3]
    target_values, dynamic_real, dynamic_cat, stat_cat, stat_cat_cardinalities = make_features(data_path)

    # start_dates: (item: 30490, )
    start_dates = [pd.Timestamp("2011-01-29", freq='1D') for _ in range(target_values.shape[0])]

    ##########################################################
    #  1                   ~               1914      ~     1942
    #                    train               |      test
    idx_train_end = 1914
    idx_test_end = 1942

    ### Train Set
    train_set = [
        {
            FieldName.TARGET: target[..., :idx_train_end],
            FieldName.START: start,
            FieldName.FEAT_DYNAMIC_REAL: fdr[..., :idx_train_end],
            FieldName.FEAT_DYNAMIC_CAT: fdc[..., :idx_train_end],
            FieldName.FEAT_STATIC_CAT: fsc
        }
        for (target, start, fdr, fdc, fsc) in zip(target_values, 
                                                    start_dates, 
                                                    dynamic_real, 
                                                    dynamic_cat,  
                                                    stat_cat)
    ]
    train_ds = ListDataset(train_set, freq="D", shuffle=False)

    ### Test Set
    test_set = [
        {
            FieldName.TARGET: target[..., :idx_test_end],
            FieldName.START: start,
            FieldName.FEAT_DYNAMIC_REAL: fdr[..., :idx_test_end],
            FieldName.FEAT_DYNAMIC_CAT: fdc[..., :idx_test_end],
            FieldName.FEAT_STATIC_CAT: fsc
        }
        for (target, start, fdr, fdc, fdp, fsc) in zip(target_values,
                                                        start_dates,
                                                        dynamic_real,
                                                        dynamic_cat,
                                                        stat_cat)
    ]
    test_ds = ListDataset(test_set, freq="D", shuffle=False)

    return train_ds, test_ds, stat_cat_cardinalities