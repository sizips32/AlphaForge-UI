"""
pytest 설정 및 공통 픽스처
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def sample_stock_data():
    """테스트용 샘플 주가 데이터"""
    np.random.seed(42)
    
    # 날짜 범위 생성
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    # 주말 제외
    business_days = date_range[date_range.weekday < 5]
    
    # 종목 코드
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    data_list = []
    
    for ticker in tickers:
        # 각 종목별로 가격 시뮬레이션
        initial_price = np.random.uniform(100, 300)
        prices = [initial_price]
        
        for i in range(1, len(business_days)):
            # 랜덤 워크로 가격 생성
            change = np.random.normal(0, 0.02) * prices[-1]
            new_price = max(prices[-1] + change, 10)  # 최소 10달러
            prices.append(new_price)
        
        # OHLCV 데이터 생성 (올바른 OHLC 관계 보장)
        for i, (date, close) in enumerate(zip(business_days, prices)):
            # OHLC 관계를 올바르게 생성
            open_price = close * np.random.uniform(0.95, 1.05)
            high = max(open_price, close) * np.random.uniform(1.0, 1.03)
            low = min(open_price, close) * np.random.uniform(0.97, 1.0)
            
            # OHLC 관계 검증 및 수정
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = np.random.randint(1000000, 10000000)
            
            data_list.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Ticker': ticker,
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close, 2),
                'Volume': volume
            })
    
    return pd.DataFrame(data_list)

@pytest.fixture
def processed_data(sample_stock_data):
    """전처리된 데이터"""
    from utils.data_processor import DataProcessor
    processor = DataProcessor()
    return processor.process_data(sample_stock_data)

@pytest.fixture
def factor_settings():
    """팩터 마이닝 설정"""
    return {
        'factor_types': ['Momentum', 'Value', 'Quality'],
        'factor_pool_size': 5,
        'min_ic': 0.02,
        'min_icir': 0.5
    }

@pytest.fixture
def backtest_settings():
    """백테스팅 설정"""
    return {
        'start_date': '2020-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 100000,
        'transaction_cost': 0.001,
        'max_position_size': 0.1
    }