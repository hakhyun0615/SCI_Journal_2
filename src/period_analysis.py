def estimate_periods_with_fourier(sale, start_col, save_plot=False):
    # ...existing code...
    
    # 주기 계산 (이미 진폭순으로 정렬된 상태)
    raw_periods = [int(round(1/f)) for f in sorted_frequencies if 1/f < len(sale_values)//2]
    
    # 단순화된 주기 병합
    merged_periods = []
    for period in raw_periods:
        # 유사한 주기가 이미 있는지 확인
        if not any(abs(period - p) <= max(1, p * 0.05) for p in merged_periods):
            merged_periods.append(period)

    # ...existing code...
