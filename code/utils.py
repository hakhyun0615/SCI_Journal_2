import os
import logging
from pathlib import Path

import matplotlib.pyplot as plt

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State

import numpy as np

import pandas as pd
from scipy.signal import detrend, butter, filtfilt
from statsmodels.tsa.seasonal import STL

################## logging

def get_log_path(log_dir, log_comment='temp', trial='t0', mkdir=True):    
    if log_comment=='':
        log_comment='temp'
    
    base_path = os.path.join('logs', log_dir, log_comment)
    trial_path = trial

    full_log_path = f"{base_path}/{trial_path}"

    path = Path(full_log_path)    
    if mkdir==True:
        if not path.exists():
            path.mkdir(parents=True)

    return full_log_path, base_path, trial_path

def set_logger(text_log_path, text_log_file = 'log.txt', level = logging.INFO):
    logger = logging.getLogger("mofl")
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s%(name)18s%(levelname)10s\t%(message)s')

    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(formatter)
    logger.addHandler(stream_hander)

    log_file = f"{text_log_path}/{text_log_file}"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False

    logger.info(f"Logging to {log_file}...")

################## plot

def plot_with_matplotlib_1(aggregated_sales):

    plt.figure(figsize=(15, 8))

    for idx in aggregated_sales.index:
        plt.plot(
            [int(col.split('_')[-1]) for col in aggregated_sales.columns], 
            aggregated_sales.loc[idx].values, 
            '-', label=str(idx)
        )
    
    plt.xlabel('Day Number')
    plt.ylabel('Items Sold')
    plt.legend(loc='best')
    
    return aggregated_sales

def plot_with_matplotlib_2(filtered_and_grouped_sales):
    plt.figure(figsize=(15, 8))

    for group, sales in filtered_and_grouped_sales.items():
        indices = list(sales.keys())
        values = list(sales.values())
        plt.plot(indices, values, '-', label=f"{group}")

    plt.xlabel('Day Number')
    plt.ylabel('Items Sold')
    plt.legend(loc='best')
    plt.show()

def plot_with_matplotlib_3(time_series, group_label, visualize):
    if visualize:
        plt.figure(figsize=(15, 8))

        plt.plot(time_series.index, time_series.values, '-', label=f'{group_label}')

        plt.xlabel('Date')
        plt.ylabel('Items Sold')
        plt.legend(loc='upper left')

        plt.tight_layout()
        plt.show()

def plot_with_plotly(dataframe):

    app = Dash(__name__)

    app.layout = html.Div([
        dcc.Dropdown(
            id='column-selector',
            options=[
                {'label': col, 'value': col}
                for col in ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
            ],
            multi=True, 
            placeholder='Select a column'
        ),
        html.Div([
            dcc.Dropdown(
                id='value-selector',
                multi=True,  
                placeholder='Select grouped values'
            ),
            dcc.Checklist(
                id='select-all',
                options=[{'label': 'Select All', 'value': 'all'}],
                inline=True
            ),
        ]),
        
        dcc.Graph(id='line-chart', config={'displayModeBar': True}),
    ])

    @app.callback(
        [Output('value-selector', 'options'),
         Output('value-selector', 'value')],
        [Input('column-selector', 'value'),
         Input('select-all', 'value')],
        State('value-selector', 'options')
    )
    def update_value_dropdown(selected_column, select_all, current_options):
        if not selected_column:
            return [], []
        
        grouped = dataframe.groupby(selected_column).size().reset_index()
        grouped['group'] = grouped[selected_column].astype(str).agg('_'.join, axis=1)
    
        options = [{'label': group, 'value': group} for group in grouped['group'].unique()]

        if select_all and 'all' in select_all:
            return options, [option['value'] for option in options]

        return options, []

    @app.callback(
        Output('line-chart', 'figure'),
        [Input('column-selector', 'value'),
         Input('value-selector', 'value')]
    )
    def update_chart(selected_column, selected_values):
        if not selected_column or not selected_values:
            return go.Figure()

        grouped = dataframe.groupby(selected_column).sum().reset_index()
        grouped['group'] = grouped[selected_column].astype(str).agg('_'.join, axis=1)
        filtered_dataframe = grouped[grouped['group'].isin(selected_values)]

        fig = go.Figure()
        for _, row in filtered_dataframe.iterrows():
            ts_columns = [col for col in dataframe.columns if col.startswith('d_')]
            fig.add_trace(
                go.Scatter(
                    x=[int(col.split('_')[-1]) for col in ts_columns],
                    y=row[ts_columns],
                    mode='lines',
                    name=row['group']
                )
            )

        fig.update_layout(xaxis_title="Days", yaxis_title="Items Sold")
        return fig

    app.run_server(debug=True)

################## filter and group

# 출시 이후의 판매량만 남기는 함수
def filter_sales_after_launch(row):
    all_sales = row[5:].values # d_1 ~ d_1941
    non_zero_indices = np.where(all_sales != 0)[0] 
    first_non_zero_index = non_zero_indices[0] # d_?
    trimmed_sales = {i: all_sales[i] for i in range(first_non_zero_index, len(all_sales))}
    return trimmed_sales

# 그룹화 및 최소 키값 확인
def group_sales(group):
    # 그룹의 모든 딕셔너리에서 키들을 합치기
    all_keys = set().union(*group["sales_after_launch"]) 
    
    # 그룹의 모든 딕셔너리에서 값을 합산
    combined_sales = {}
    for sales_dict in group["sales_after_launch"]:
        for key in all_keys:
            combined_sales[key] = combined_sales.get(key, 0) + sales_dict.get(key, 0)
    
    return combined_sales

def filter_sales_after_launch_and_group_sales(sales, columns):
    sales = sales.copy()
    sales['sales_after_launch'] = sales.apply(lambda row: filter_sales_after_launch(row), axis=1)

    print(len(sales)) ###
    sales['sales_after_launch_length'] = sales['sales_after_launch'].apply(lambda x: len(x)) 
    sales = sales[sales['sales_after_launch_length'] >= (1941 - 600)]
    print(len(sales))

    grouped_sales = sales.groupby(columns).apply(group_sales)

    return grouped_sales

################## analyze dominant period

# 밴드패스 필터
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# 푸리에 변환을 통한 주기 추정
def estimate_dominant_period_with_fourier_transform(time_series, group_label, visualize):
    filtered_time_series = bandpass_filter(time_series, lowcut=0.01, highcut=0.3, fs=1)
    fft = np.fft.fft(filtered_time_series) # detrend(time_series)
    frequencies = np.fft.fftfreq(len(time_series), d=1) 
    amplitudes = np.abs(fft)                            

    positive_frequencies = frequencies[frequencies > 0]
    positive_amplitudes = amplitudes[frequencies > 0]
    dominant_frequency = positive_frequencies[np.argmax(positive_amplitudes)]
    dominant_period = 1 / dominant_frequency

    # 시각화
    if visualize:
        plt.figure(figsize=(15, 8))
        plt.plot(positive_frequencies, positive_amplitudes, label=group_label)
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.tight_layout()
        plt.show()

    print(f"{group_label}'s Dominant Frequency: {dominant_frequency} 1/days")
    print(f"{group_label}'s Dominant Period: {dominant_period} days")

    return dominant_period

# STL 분해를 통한 계절성 강도 계산
def calculate_seasonal_strength_with_stl(time_series, dominant_period, group_label, visualize):
    
    stl = STL(
        endog = time_series, # 시계열 데이터
        period = dominant_period, # 계절 주기
        seasonal = dominant_period, # 계절성 창 크기
        robust = True # 이상치 처리
    )
    result = stl.fit()
    
    # STL 분해 결과: 트렌드, 계절성, 잔여 성분
    trend = result.trend
    seasonal = result.seasonal
    remainder = result.resid

    # STL 분해 시각화
    if visualize:
        plt.figure(figsize=(15, 8))
        plt.subplot(4, 1, 1)
        plt.plot(time_series, label=f'{group_label}')
        plt.legend(loc='upper left')

        plt.subplot(4, 1, 2)
        plt.plot(trend, label='Trend')
        plt.legend(loc='upper left')

        plt.subplot(4, 1, 3)
        plt.plot(seasonal, label='Seasonal')
        plt.legend(loc='upper left')

        plt.subplot(4, 1, 4)
        plt.plot(remainder, label='Residual')
        plt.legend(loc='upper left')

        plt.tight_layout()
        plt.show()

    # 분산 계산
    var_resid = np.var(remainder, ddof=1)  # 잔여 성분의 분산
    var_seasonal_remainder = np.var(seasonal + remainder, ddof=1)  # (계절성 + 잔여 성분)의 분산

    # 계절성 강도 계산
    F_S = max(0, 1 - var_resid / var_seasonal_remainder)

    print(f"{group_label}'s Seasonal Strength (0~1): {F_S}")
    
    return result, F_S

# 각 그룹에 대해 계절성 분석
def process_each_group(group_label, group_values, visualize=True):
    dates = pd.date_range(end="2016-05-22", periods=len(group_values), freq="D")
    time_series = pd.Series(group_values, index=dates)

    print(len(time_series)) ###
    time_series = time_series[-(1941 - 300):] 
    print(len(time_series))

    plot_with_matplotlib_3(time_series, group_label, visualize)

    dominant_period = estimate_dominant_period_with_fourier_transform(time_series, group_label, visualize)
    rounded_dominant_period = int(round(dominant_period)) # 반올림
    if rounded_dominant_period % 2 == 0:
        rounded_dominant_period += 1 # 홀수로 만들기
    stl_result, seasonal_strength = calculate_seasonal_strength_with_stl(time_series, rounded_dominant_period, group_label, visualize)

    return dominant_period, seasonal_strength