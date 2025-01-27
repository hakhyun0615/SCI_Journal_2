# fold: cmd k -> cmd 0

import pickle
import numpy as np
from scipy.signal import butter, filtfilt, freqz
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

def estimate_periods_with_fourier(sale, start_col, num_periods, save_plot=False):
    # start_col 이후의 데이터만 선택
    cols = [col for col in sale.index if col.startswith('d_') and int(col.split('_')[1]) >= int(start_col.split('_')[1])]
    sale_values = sale[cols].values.astype(float)

    # 원본 신호의 FFT
    original_fft = np.fft.fft(sale_values)
    frequencies = np.fft.fftfreq(len(sale_values), d=1)
    original_magnitudes = np.abs(original_fft)

    # High-pass 필터 파라미터 계산 및 필터 적용
    cutoff, order = 0.01, 10
    nyquist = 0.5
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    w, h = freqz(b, a)  # 필터의 주파수 응답 계산
    filtered_sale_values = filtfilt(b, a, sale_values)  # 직접 필터 적용

    # 필터링된 신호의 FFT
    fft = np.fft.fft(filtered_sale_values)
    magnitudes = np.abs(fft)

    # 양의 주파수만 선택
    positive_mask = frequencies > 0
    positive_frequencies = frequencies[positive_mask]
    positive_magnitudes = magnitudes[positive_mask]

    # 진폭을 내림차순으로 정렬
    sorted_indices = np.argsort(positive_magnitudes)[::-1]
    
    # 진폭 순으로 주파수 정렬 후 주기와 세기 계산
    positive_frequencies = positive_frequencies[sorted_indices]
    positive_periods = 1 / positive_frequencies
    positive_strength = positive_magnitudes[sorted_indices] / np.max(positive_magnitudes)
    

    selected_indices = []  # 원래 positive_periods에서 선택된 인덱스를 저장
    selected_periods = []  # 선택된 고유한 정수 주기
    selected_strengths  = []  # 선택된 주기의 세기
    unique_periods = set()

    for i, period in enumerate(positive_periods):
        int_period = int(period)
        if int_period not in unique_periods and int_period < 100:
            unique_periods.add(int_period) # 고유한 주기 값 추가
            selected_periods.append(int_period) # 정수 주기를 저장
            selected_strengths.append(positive_strength[i])  # 해당 주기의 세기 저장
            selected_indices.append(sorted_indices[i])  # 원래 인덱스를 저장
        
        if len(selected_periods) >= num_periods:
            break

    # 배열로 변환
    selected_periods = np.array(selected_periods)
    selected_strengths = np.array(selected_strengths)
    selected_indices = np.array(selected_indices)

    if save_plot:
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        
        # 첫 번째 서브플롯: FFT magnitude와 필터 응답
        # Original FFT
        freq_range = frequencies[:len(frequencies)//2]
        axs[0].plot(freq_range, original_magnitudes[:len(frequencies)//2] / np.max(original_magnitudes), 
                label="Original FFT (normalized)", alpha=0.7)
        # Filter response
        w_hz = w * nyquist / np.pi  # Convert normalized frequency to Hz
        axs[0].plot(w_hz, np.abs(h), 'r--', label='Filter Response', alpha=0.7)
        axs[0].set_title('Original Frequency Domain & Filter Response')
        axs[0].set_xlabel('Frequency [Hz]')
        axs[0].set_ylabel('Normalized Magnitude')
        axs[0].set_xscale('log')  # 로그 스케일로 변경
        axs[0].set_xlim([0.001, 0.5])  # x축 범위 설정
        axs[0].legend()

        # 두 번째 서브플롯: Filtered FFT
        axs[1] = plt.subplot(313)
        axs[1].plot(freq_range, magnitudes[:len(frequencies)//2] / np.max(magnitudes), 
                label="Filtered FFT (normalized)", alpha=0.7)
        axs[1].set_title('Filtered Frequency Domain')
        axs[1].set_xlabel('Frequency [Hz]')
        axs[1].set_ylabel('Normalized Magnitude')
        axs[1].set_xscale('log')  # 로그 스케일로 변경
        axs[1].set_xlim([0.001, 0.5])  # x축 범위 설정
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(f"../data/fourier/plot/{sale['state_id']}, {sale['item_id']}.png")
        plt.close()

    # 복원
    snr = 0
        
    reconstruct_fft = np.zeros_like(fft)
    
    reconstruct_fft[selected_indices] = fft[selected_indices]
    reconstruct_fft[-np.array(selected_indices)] = fft[-np.array(selected_indices)]  # 음의 주파수 성분도 추가

    # 복원 신호 생성
    reconstructed_values = np.fft.ifft(reconstruct_fft).real

    # SNR 계산
    original_energy = np.sum(np.abs(filtered_sale_values) ** 2)
    error_energy = np.sum(np.abs(filtered_sale_values - reconstructed_values) ** 2)
    snr = 10 * np.log10(original_energy / error_energy)

    if save_plot:
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        
        axs[0].plot(sale_values, label=f"Original - {sale['state_id']}, {sale['item_id']}")
        axs[0].plot(filtered_sale_values, label="Filtered Signal", linestyle="dashed", alpha=0.7)
        axs[0].plot(reconstructed_values, label="Reconstructed Signal", linestyle="dotted", alpha=0.7)
        axs[0].set_title('Time Series')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Value')
        axs[0].legend()

        plt.tight_layout()
        plt.savefig(f"../data/fourier/inverse_plot/{sale['state_id']}, {sale['item_id']}.png")
        plt.close()

    return selected_periods, selected_strengths, snr


def analyze_period_with_fourier(sales, first_sales_column_dict, num_periods, save_result=False, save_plot=False):
    # Fundamental Period, Period Strength
    fourier_results = {}

    for idx in sales.index:
        sale = sales.iloc[idx]
        key = (sale['state_id'], sale['item_id'])

        # 시작 컬럼 가져오기
        start_col = first_sales_column_dict.get(key)

        # Fourier 분석 수행
        selected_periods, selected_strengths, snr = estimate_periods_with_fourier(sale, start_col, num_periods, save_plot=save_plot)

        # 결과 저장
        fourier_results[key] = {
            'selected_periods': selected_periods,
            'selected_strengths': selected_strengths,
            'snr': snr,
        }

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