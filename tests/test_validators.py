"""
데이터 검증 모듈 테스트
"""

import pytest
import pandas as pd
import numpy as np
from utils.validators import DataValidator

class TestDataValidator:
    """데이터 검증 클래스 테스트"""
    
    def test_init(self):
        """초기화 테스트"""
        validator = DataValidator()
        assert validator is not None
    
    def test_validate_data_success(self, sample_stock_data):
        """정상 데이터 검증 테스트"""
        validator = DataValidator()
        is_valid, issues = validator.validate_data(sample_stock_data)
        
        assert is_valid == True
        assert isinstance(issues, list)
    
    def test_validate_data_empty(self):
        """빈 데이터 검증 테스트"""
        validator = DataValidator()
        is_valid, issues = validator.validate_data(pd.DataFrame())
        
        assert is_valid == False
        assert "데이터가 비어있습니다" in str(issues)
    
    def test_validate_required_columns(self, sample_stock_data):
        """필수 컬럼 검증 테스트"""
        validator = DataValidator()
        
        # Close 컬럼 제거
        invalid_data = sample_stock_data.drop(columns=['Close'])
        is_valid, issues = validator.validate_required_columns(invalid_data)
        
        assert is_valid == False
        assert any("Close" in issue for issue in issues)
    
    def test_validate_date_format(self):
        """날짜 형식 검증 테스트"""
        validator = DataValidator()
        
        # 올바른 날짜 형식
        valid_data = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02'],
            'Ticker': ['AAPL', 'AAPL'],
            'Close': [150.0, 151.0]
        })
        is_valid, issues = validator.validate_date_format(valid_data)
        assert is_valid == True
        
        # 잘못된 날짜 형식
        invalid_data = pd.DataFrame({
            'Date': ['01/01/2023', '01/02/2023'],
            'Ticker': ['AAPL', 'AAPL'],
            'Close': [150.0, 151.0]
        })
        is_valid, issues = validator.validate_date_format(invalid_data)
        assert is_valid == False
    
    def test_validate_price_data(self):
        """가격 데이터 검증 테스트"""
        validator = DataValidator()
        
        # 올바른 가격 데이터
        valid_data = pd.DataFrame({
            'Date': ['2023-01-01'],
            'Ticker': ['AAPL'],
            'Close': [150.0],
            'Open': [149.0],
            'High': [152.0],
            'Low': [148.0]
        })
        is_valid, issues = validator.validate_price_data(valid_data)
        assert is_valid == True
        
        # 음수 가격 데이터
        invalid_data = pd.DataFrame({
            'Date': ['2023-01-01'],
            'Ticker': ['AAPL'],
            'Close': [-150.0],
            'Open': [149.0],
            'High': [152.0],
            'Low': [148.0]
        })
        is_valid, issues = validator.validate_price_data(invalid_data)
        assert is_valid == False
        assert any("음수 가격" in issue for issue in issues)
    
    def test_validate_missing_data(self, sample_stock_data):
        """결측치 검증 테스트"""
        validator = DataValidator()
        
        # 결측치 추가
        data_with_missing = sample_stock_data.copy()
        data_with_missing.loc[:10, 'Close'] = np.nan
        
        is_valid, issues = validator.validate_missing_data(data_with_missing)
        
        # 결측치가 많으면 실패
        if len(data_with_missing) > 0:
            missing_ratio = data_with_missing['Close'].isna().sum() / len(data_with_missing)
            if missing_ratio > 0.05:  # 5% 이상
                assert is_valid == False
    
    def test_validate_data_consistency(self, sample_stock_data):
        """데이터 일관성 검증 테스트"""
        validator = DataValidator()
        
        # OHLC 데이터 일관성 검증
        if all(col in sample_stock_data.columns for col in ['Open', 'High', 'Low', 'Close']):
            is_valid, issues = validator.validate_data_consistency(sample_stock_data)
            
            # High >= Low 조건 위반하는 데이터 생성
            invalid_data = sample_stock_data.copy()
            invalid_data.loc[0, 'High'] = 100
            invalid_data.loc[0, 'Low'] = 150  # Low > High
            
            is_valid, issues = validator.validate_data_consistency(invalid_data)
            assert is_valid == False
    
    def test_validate_ticker_data(self, sample_stock_data):
        """종목 데이터 검증 테스트"""
        validator = DataValidator()
        
        # 각 종목별로 충분한 데이터가 있는지 확인
        is_valid, issues = validator.validate_ticker_data(sample_stock_data)
        
        # 종목별 최소 데이터 수 확인
        ticker_counts = sample_stock_data['Ticker'].value_counts()
        min_count = ticker_counts.min()
        
        if min_count < 100:  # 최소 100일 데이터
            assert is_valid == False
        else:
            assert is_valid == True
    
    def test_get_data_quality_score(self, sample_stock_data):
        """데이터 품질 점수 테스트"""
        validator = DataValidator()
        score = validator.get_data_quality_score(sample_stock_data)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
    
    def test_comprehensive_validation(self, sample_stock_data):
        """종합 검증 테스트"""
        validator = DataValidator()
        
        result = validator.comprehensive_validation(sample_stock_data)
        
        assert isinstance(result, dict)
        assert 'is_valid' in result
        assert 'issues' in result
        assert 'quality_score' in result
        assert 'recommendations' in result
        
        assert isinstance(result['is_valid'], bool)
        assert isinstance(result['issues'], list)
        assert isinstance(result['quality_score'], float)
        assert isinstance(result['recommendations'], list)