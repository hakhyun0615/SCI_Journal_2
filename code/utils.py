# fold: cmd k -> cmd 0

import pickle
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.special import expit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

import seaborn as sns
import matplotlib.pyplot as plt

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

def highpass_filter(sale_values, cutoff, fs=1, order=5):
    # Nyquist 주파수 계산 (샘플링 주파수의 절반)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_sale_values = filtfilt(b, a, sale_values)
    return filtered_sale_values

def estimate_dominant_periods_with_fourier(sale, start_col, save_plot=False):
    # start_col 이후의 데이터만 선택
    cols = [col for col in sale.index if col.startswith('d_') and int(col.split('_')[1]) >= int(start_col.split('_')[1])]
    sale_values = sale[cols].values.astype(float)

    # High-pass 필터 적용
    filtered_sale_values = highpass_filter(sale_values, cutoff=0.01)
    
    # FFT 수행 및 파워 스펙트럼 계산
    fft = np.fft.fft(filtered_sale_values)
    power_spectrum = np.abs(fft) ** 2
    frequencies = np.fft.fftfreq(len(sale_values), d=1)

    # 양의 주파수만 선택
    positive_mask = frequencies > 0
    positive_frequencies = frequencies[positive_mask]
    positive_power = power_spectrum[positive_mask]

    # 진폭을 내림차순으로 정렬
    sorted_indices = np.argsort(positive_power)[::-1]
    sorted_frequencies = positive_frequencies[sorted_indices]
    sorted_power = positive_power[sorted_indices]

    # 기본 주파수와 주기 세기 계산
    fundamental_frequency = sorted_frequencies[0]  # 가장 높은 파워의 주파수
    fundamental_period = 1 / fundamental_frequency
    period_strength = sorted_power[0] / np.sum(positive_power)  # 기본 주파수 파워 비율

    # 파워 스펙트럼 시각화 및 time_series 추가
    if save_plot:
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))

        # 첫 번째 서브플롯: time_series
        axs[0].plot(sale_values, label=f"Time Series - {sale['state_id']}, {sale['item_id']}")
        axs[0].set_title('Time Series')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Value')
        axs[0].legend()

        # 두 번째 서브플롯: 파워 스펙트럼
        axs[1].plot(positive_frequencies, positive_power, label=f"Power Spectrum - {sale['state_id']}, {sale['item_id']}")
        axs[1].axvline(fundamental_frequency, label='Fundamental Frequency', color='r', linestyle='-', alpha=0.8)
        axs[1].set_title('Power Spectrum')
        axs[1].set_xlabel('Frequency')
        axs[1].set_ylabel('Power')
        axs[1].legend()
        axs[1].set_xticks([fundamental_frequency])
        axs[1].set_xticklabels([f"{fundamental_frequency:.2f}"], color='red', alpha=0.8)

        # 플롯 저장
        plt.tight_layout()
        plt.savefig(f"../data/fourier/plot/{sale['state_id']}, {sale['item_id']}.png")
        plt.close()

    return fundamental_period, period_strength

def normalize_period_strength(fourier_results, save_plot=False):
    # Period Strength 데이터 추출
    period_strengths = np.array([result['period_strength'] for result in fourier_results.values()])
    
    # Min-Max 정규화
    scaler = StandardScaler()
    normalized_period_strengths = scaler.fit_transform(period_strengths.reshape(-1, 1)).flatten()
    normalized_period_strengths = expit(normalized_period_strengths)  # Sigmoid 적용
    
    # Normalized 값 추가 후 기존 데이터 제거
    for i, key in enumerate(fourier_results.keys()):
        fourier_results[key]['normalized_period_strength'] = normalized_period_strengths[i]
        # 기존 period_strength 제거
        del fourier_results[key]['period_strength']

    # KDE Plot 시각화
    if save_plot:
        sns.kdeplot(normalized_period_strengths, color='#1f77b4', fill=True)
        plt.xlabel('Normalized Period Strength')
        plt.ylabel('Density')
        plt.savefig(f"../data/fourier/plot/normalized_period_strengths.png")
        plt.close()

    return fourier_results

def analyze_period_with_fourier(sales, first_sales_column_dict, save_result=False, save_plot=False):
    # Fundamental Period, Period Strength
    fourier_results = {}

    for idx in sales.index:
        sale = sales.iloc[idx]
        key = (sale['state_id'], sale['item_id'])

        # 시작 컬럼 가져오기
        start_col = first_sales_column_dict.get(key)

        # Fourier 분석 수행
        fundamental_period, period_strength = estimate_dominant_periods_with_fourier(sale, start_col, save_plot=save_plot)

        # 결과 저장
        fourier_results[key] = {
            'fundamental_period': round(fundamental_period),
            'period_strength': period_strength,
        }

    # Period Strength 정규화
    fourier_results = normalize_period_strength(fourier_results, save_plot=save_plot)

    # 결과 저장 (Pickle 파일)
    if save_result:
        with open('../data/fourier/results.pkl', 'wb') as f:
            pickle.dump(fourier_results, f)

    return fourier_results

def detrend_with_loess(sales, first_sales_column_dict, span, save_result=False, save_plot=False):

    detrended_sales = sales.copy()

    for idx, sale in sales.iterrows():
        # 행별 시작 컬럼 가져오기
        start_col = first_sales_column_dict.get(tuple(sale[:2]))  # 첫 두 열이 key로 사용됨 (예: (CA, FOODS_2_001))

        # 시작 컬럼 이후 데이터만 선택
        start_index = sales.columns.get_loc(start_col)
        sale_values = sale.iloc[start_index:].values

        # LOESS 적용
        x = np.arange(len(sale_values))
        trend = lowess(sale_values, x, frac=span, return_sorted=False)
        detrended_sale_values = sale_values - trend

        # 결과를 데이터프레임에 저장
        detrended_sales.iloc[idx, start_index:] = detrended_sale_values

        # 시각화
        if save_plot:
            fig, axs = plt.subplots(2, 1, figsize=(15, 10))

            # 원본 데이터와 트렌드 시각화
            axs[0].plot(x, sale_values, label="Original Sales")
            axs[0].plot(x, trend, label="LOESS Trend", linestyle="-", color="red", alpha=0.8)
            axs[0].legend()
            axs[0].set_title(f"Original Sales with LOESS Trend: {sale['state_id']}, {sale['item_id']}")

            # 트렌드 제거된 데이터 시각화
            axs[1].plot(x, detrended_sale_values, label="Detrended Sales", color="green")
            axs[1].legend()
            axs[1].set_title(f"Detrended Sales: {sale['state_id']}, {sale['item_id']}")

            plt.tight_layout()
            plt.savefig(f"../data/loess/plot/{sale['state_id']}_{sale['item_id']}.png")
            plt.close()

    if save_result:
        detrended_sales.to_csv("../data/loess/detrended_sales.csv", index=False)

    return detrended_sales

def calculate_sell_price_changes_with_log_differencing(sell_prices, first_sales_column_dict, save_result=False, save_plot=False):

    # 데이터 복사
    log_differenced_sell_prices = sell_prices.copy()

    # 행별로 시작 컬럼부터 차분 수행
    for idx, sell_price in sell_prices.iterrows():
        # 시작 컬럼 가져오기
        start_col = first_sales_column_dict.get(tuple(sell_price[:2]))
        
        # 시작 컬럼 이후 데이터 선택
        start_index = sell_prices.columns.get_loc(start_col)
        sell_price_values = np.array(sell_price.iloc[start_index:].values, dtype=np.float64)

        # 로그 변환 및 차분 계산
        logged_sell_price_values = np.log(sell_price_values + 1e-9)  # 로그 계산 시 0 방지
        log_differenced_sell_price_values = np.diff(logged_sell_price_values, prepend=logged_sell_price_values[0])

        # 결과 저장
        log_differenced_sell_prices.iloc[idx, start_index:] = log_differenced_sell_price_values

        # 시각화
        if save_plot:
            fig, axs = plt.subplots(2, 1, figsize=(15, 10))

            axs[0].plot(sell_price_values, label=f"Original Sell Prices: {sell_price['state_id']}, {sell_price['item_id']}")
            axs[0].set_xlabel("Time")
            axs[0].set_ylabel("Sell Prices")
            axs[0].legend()

            axs[1].plot(log_differenced_sell_price_values, label=f"Log-Differenced Sell Prices: {sell_price['state_id']}, {sell_price['item_id']}")
            axs[1].set_xlabel("Time")
            axs[1].set_ylabel("Log-Differenced Sell Prices")
            axs[1].legend()

            plt.tight_layout()
            plt.savefig(f"../data/log_differencing/plot/{sell_price['state_id']}_{sell_price['item_id']}.png")
            plt.close()

    if save_result:
        log_differenced_sell_prices.to_csv("../data/log_differencing/log_differenced_sell_prices.csv", index=False)

    return log_differenced_sell_prices