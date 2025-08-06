"""
데이터 처리 모듈 테스트
"""

import pytest
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor

class TestDataProcessor:
    """데이터 처리 클래스 테스트"""
    
    def test_init(self):
        """초기화 테스트"""
        processor = DataProcessor()
        assert processor.processed_data is None
        assert processor.features == {}
    
    def test_process_data_success(self, sample_stock_data):
        """데이터 처리 성공 테스트"""
        processor = DataProcessor()
        result = processor.process_data(sample_stock_data)
        
        assert result is not None
        assert not result.empty
        assert 'Date' in result.columns
        assert 'Ticker' in result.columns
        assert 'Close' in result.columns
        
        # 날짜가 datetime 타입으로 변환되었는지 확인
        assert pd.api.types.is_datetime64_any_dtype(result['Date'])
    
    def test_process_data_empty_input(self):
        """빈 데이터 입력 테스트"""
        processor = DataProcessor()
        
        with pytest.raises(Exception) as exc_info:
            processor.process_data(pd.DataFrame())
        
        assert "입력 데이터가 비어있습니다" in str(exc_info.value)
    
    def test_process_data_missing_columns(self):
        """필수 컬럼 누락 테스트"""
        processor = DataProcessor()
        
        # Close 컬럼이 없는 데이터
        invalid_data = pd.DataFrame({
            'Date': ['2023-01-01'],
            'Ticker': ['AAPL']
        })
        
        with pytest.raises(Exception) as exc_info:
            processor.process_data(invalid_data)
        
        assert "필수 컬럼이 누락되었습니다" in str(exc_info.value)
    
    def test_clean_data(self, sample_stock_data):
        """데이터 정리 테스트"""
        processor = DataProcessor()
        
        # 중복 데이터 추가
        duplicated_data = pd.concat([sample_stock_data, sample_stock_data.head(10)])
        
        cleaned = processor._clean_data(duplicated_data)
        
        assert len(cleaned) < len(duplicated_data)  # 중복 제거됨
        assert not cleaned.duplicated().any()  # 중복 없음
    
    def test_process_dates(self, sample_stock_data):
        """날짜 처리 테스트"""
        processor = DataProcessor()
        result = processor._process_dates(sample_stock_data)
        
        assert pd.api.types.is_datetime64_any_dtype(result['Date'])
    
    def test_calculate_technical_indicators(self, processed_data):
        """기술적 지표 계산 테스트"""
        processor = DataProcessor()
        
        # 처리된 데이터에 기술적 지표가 포함되어 있는지 확인
        technical_columns = [col for col in processed_data.columns if any(
            indicator in col.lower() for indicator in ['sma', 'ema', 'rsi', 'bb', 'macd']
        )]
        
        assert len(technical_columns) > 0  # 기술적 지표가 계산되었는지 확인
    
    def test_data_integrity(self, processed_data):
        """데이터 무결성 테스트"""
        # NaN 값 확인
        critical_columns = ['Date', 'Ticker', 'Close']
        for col in critical_columns:
            assert not processed_data[col].isna().any(), f"{col} 컬럼에 NaN 값 존재"
        
        # 가격 데이터가 양수인지 확인
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in processed_data.columns:
                assert (processed_data[col] > 0).all(), f"{col} 컬럼에 음수 값 존재"
    
    def test_date_sorting(self, processed_data):
        """날짜 정렬 테스트"""
        for ticker in processed_data['Ticker'].unique():
            ticker_data = processed_data[processed_data['Ticker'] == ticker]
            dates = ticker_data['Date'].values
            
            # 날짜가 정렬되어 있는지 확인
            assert (dates[:-1] <= dates[1:]).all(), f"{ticker} 종목 데이터가 날짜순으로 정렬되지 않음"