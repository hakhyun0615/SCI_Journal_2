# fold: cmd k -> cmd 0

import pickle

# sell_prices의 각 행에서 NaN이 끝나고 처음으로 숫자인 column을 찾기
def create_first_sales_column_dict(df, save_result=False):
    first_sales_column_dict = {}
    
    for idx in df.index:
        row = df.iloc[idx]
        # state_id와 item_id로 키 튜플 생성
        key = (row['state_id'], row['item_id'])
        
        # NaN 값들의 위치를 찾습니다
        nan_mask = row.isna()
        
        if not nan_mask.any():
            # NaN이 없는 경우 첫 번째 데이터 컬럼('d_1')을 저장
            first_sales_column_dict[key] = 'd_1'
        else:
            # 마지막 NaN의 위치를 찾습니다
            last_nan_idx = nan_mask[::-1].idxmax()
            # 마지막 NaN의 위치 이후의 첫 번째 숫자가 있는 컬럼을 찾습니다
            last_nan_position = row.index.get_loc(last_nan_idx)
            first_number_col = row.index[last_nan_position + 1]
            first_sales_column_dict[key] = first_number_col
    
    if save_result:
        with open('../data/preprocessed/first_sales_column_dict.pkl', 'wb') as f:
            pickle.dump(first_sales_column_dict, f)
            
    return first_sales_column_dict