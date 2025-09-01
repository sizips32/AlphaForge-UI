"""
팩터 마이닝 모듈 테스트
"""

import pytest
import pandas as pd
import numpy as np
from utils.factor_miner import FactorMiner

class TestFactorMiner:
    """팩터 마이닝 클래스 테스트"""
    
    def test_init(self, factor_settings):
        """초기화 테스트"""
        miner = FactorMiner(factor_settings)
        assert miner.settings == factor_settings
        assert miner.scaler is not None
    
    def test_generate_basic_factors_success(self, processed_data, factor_settings):
        """기본 팩터 생성 성공 테스트"""
        miner = FactorMiner(factor_settings)
        factors = miner.generate_basic_factors(processed_data)
        
        assert factors is not None
        assert len(factors) > 0
        assert len(factors) <= factor_settings['factor_pool_size']
    
    def test_generate_basic_factors_empty_data(self, factor_settings):
        """빈 데이터 입력 테스트"""
        miner = FactorMiner(factor_settings)
        
        with pytest.raises(Exception) as exc_info:
            miner.generate_basic_factors(pd.DataFrame())
        
        assert "입력 데이터가 비어있습니다" in str(exc_info.value)
    
    def test_generate_basic_factors_missing_close(self, factor_settings):
        """Close 컬럼 누락 테스트"""
        miner = FactorMiner(factor_settings)
        
        invalid_data = pd.DataFrame({
            'Date': ['2023-01-01'],
            'Ticker': ['AAPL']
        })
        
        with pytest.raises(Exception) as exc_info:
            miner.generate_basic_factors(invalid_data)
        
        assert "Close 컬럼이 없습니다" in str(exc_info.value)
    
    def test_generate_momentum_factors(self, processed_data, factor_settings):
        """모멘텀 팩터 생성 테스트"""
        miner = FactorMiner(factor_settings)
        momentum_factors = miner._generate_momentum_factors(processed_data)
        
        assert len(momentum_factors) > 0
        
        # 각 팩터가 딕셔너리 형태인지 확인
        for factor in momentum_factors:
            assert isinstance(factor, dict)
            assert 'name' in factor
            assert 'formula' in factor
            assert 'values' in factor
    
    def test_generate_value_factors(self, processed_data, factor_settings):
        """밸류 팩터 생성 테스트"""
        miner = FactorMiner(factor_settings)
        value_factors = miner._generate_value_factors(processed_data)
        
        assert len(value_factors) > 0
        
        for factor in value_factors:
            assert isinstance(factor, dict)
            assert 'name' in factor
            assert 'formula' in factor
            assert 'values' in factor
    
    def test_calculate_factor_performance(self, processed_data, factor_settings):
        """팩터 성과 계산 테스트"""
        miner = FactorMiner(factor_settings)
        
        # 단순한 모멘텀 팩터 생성
        factor_values = processed_data.groupby('Ticker', observed=False)['Close'].pct_change(20).fillna(0)
        
        ic, icir, hit_rate = miner._calculate_factor_performance(
            factor_values.values, 
            processed_data['Close'].pct_change().fillna(0).values
        )
        
        assert isinstance(ic, float)
        assert isinstance(icir, float) 
        assert isinstance(hit_rate, float)
        assert 0 <= hit_rate <= 1
    
    def test_filter_factors_by_performance(self, processed_data, factor_settings):
        """성과 기준 팩터 필터링 테스트"""
        miner = FactorMiner(factor_settings)
        basic_factors = miner.generate_basic_factors(processed_data)
        
        # 성과 계산
        for factor in basic_factors:
            factor_values = factor['values']
            returns = processed_data['Close'].pct_change().fillna(0).values
            
            ic, icir, hit_rate = miner._calculate_factor_performance(factor_values, returns)
            factor['ic'] = ic
            factor['icir'] = icir
            factor['hit_rate'] = hit_rate
        
        filtered_factors = miner._filter_factors_by_performance(basic_factors)
        
        assert len(filtered_factors) <= len(basic_factors)
        
        # 필터링된 팩터들이 최소 기준을 만족하는지 확인
        for factor in filtered_factors:
            if 'ic' in factor and 'icir' in factor:
                # 최소 기준을 만족하거나, 기준이 너무 엄격한 경우 완화됨
                assert factor['ic'] >= factor_settings['min_ic'] * 0.5  # 50% 완화
    
    def test_factor_values_validity(self, processed_data, factor_settings):
        """팩터 값 유효성 테스트"""
        miner = FactorMiner(factor_settings)
        factors = miner.generate_basic_factors(processed_data)
        
        for factor in factors:
            values = factor['values']
            
            # 값이 존재하는지 확인
            assert len(values) > 0
            
            # 모든 값이 NaN이 아닌지 확인 (일부 NaN은 허용)
            non_nan_count = np.sum(~np.isnan(values))
            assert non_nan_count > len(values) * 0.5  # 최소 50% 이상이 유효한 값
            
            # 무한대 값이 없는지 확인
            assert not np.isinf(values).any()
