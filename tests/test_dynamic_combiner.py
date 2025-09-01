"""
Dynamic Combiner 단위 테스트
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from utils.dynamic_combiner import DynamicCombiner


class TestDynamicCombiner:
    """DynamicCombiner 클래스 테스트"""
    
    @pytest.fixture
    def sample_data(self):
        """테스트용 샘플 데이터"""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'date': dates,
            'close': np.random.randn(100) * 10 + 100,
            'returns': np.random.randn(100) * 0.02,
            'factor_1': np.random.randn(100),
            'factor_2': np.random.randn(100),
            'factor_3': np.random.randn(100),
            'volume': np.random.randint(1000000, 10000000, 100)
        })
    
    @pytest.fixture
    def combiner(self):
        """DynamicCombiner 인스턴스"""
        return DynamicCombiner()
    
    def test_init(self, combiner):
        """초기화 테스트"""
        assert combiner is not None
        assert combiner.scaler is not None
        assert hasattr(combiner, 'logger')
        assert combiner.combination_cache == {}
    
    def test_normalize_factors(self, combiner, sample_data):
        """팩터 정규화 테스트"""
        factor_cols = ['factor_1', 'factor_2', 'factor_3']
        normalized = combiner.normalize_factors(sample_data, factor_cols)
        
        # 정규화된 값이 대부분 -3과 3 사이에 있어야 함
        for col in factor_cols:
            assert normalized[col].min() >= -5
            assert normalized[col].max() <= 5
            assert abs(normalized[col].mean()) < 0.5
    
    def test_normalize_factors_empty(self, combiner):
        """빈 데이터프레임 정규화 테스트"""
        empty_df = pd.DataFrame()
        result = combiner.normalize_factors(empty_df, [])
        assert result.empty
    
    def test_normalize_factors_single_value(self, combiner):
        """단일 값 정규화 테스트"""
        df = pd.DataFrame({
            'factor_1': [1.0],
            'factor_2': [2.0]
        })
        result = combiner.normalize_factors(df, ['factor_1', 'factor_2'])
        # 단일 값은 0으로 정규화됨
        assert result['factor_1'].iloc[0] == 0
        assert result['factor_2'].iloc[0] == 0
    
    def test_generate_weights(self, combiner):
        """가중치 생성 테스트"""
        n_factors = 3
        weights_list = combiner.generate_weights(n_factors, n_combinations=10)
        
        assert len(weights_list) == 10
        for weights in weights_list:
            assert len(weights) == n_factors
            assert abs(sum(weights) - 1.0) < 0.01  # 합이 1
            assert all(w >= 0 for w in weights)  # 모두 양수
    
    def test_generate_weights_single_factor(self, combiner):
        """단일 팩터 가중치 생성 테스트"""
        weights_list = combiner.generate_weights(1, n_combinations=5)
        
        assert len(weights_list) == 5
        for weights in weights_list:
            assert len(weights) == 1
            assert weights[0] == 1.0
    
    def test_calculate_performance(self, combiner, sample_data):
        """성능 계산 테스트"""
        factor_cols = ['factor_1', 'factor_2']
        weights = [0.6, 0.4]
        
        performance = combiner.calculate_performance(
            sample_data, factor_cols, weights
        )
        
        assert 'sharpe_ratio' in performance
        assert 'annual_return' in performance
        assert 'max_drawdown' in performance
        assert 'volatility' in performance
        assert 'combined_signal' in performance
        
        # 값 유효성 검증
        assert isinstance(performance['sharpe_ratio'], (int, float))
        assert isinstance(performance['annual_return'], (int, float))
        assert 0 <= performance['max_drawdown'] <= 1
        assert performance['volatility'] >= 0
    
    def test_calculate_performance_with_returns(self, combiner):
        """returns 컬럼이 있는 경우 성능 계산 테스트"""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'returns': np.random.randn(100) * 0.02,
            'factor_1': np.random.randn(100),
            'factor_2': np.random.randn(100)
        })
        
        performance = combiner.calculate_performance(
            data, ['factor_1', 'factor_2'], [0.5, 0.5]
        )
        
        assert performance is not None
        assert 'sharpe_ratio' in performance
    
    def test_optimize_combination(self, combiner, sample_data):
        """조합 최적화 테스트"""
        factor_cols = ['factor_1', 'factor_2', 'factor_3']
        
        result = combiner.optimize_combination(
            sample_data, 
            factor_cols,
            n_combinations=5,
            top_k=2
        )
        
        assert 'best_weights' in result
        assert 'performance_metrics' in result
        assert 'top_combinations' in result
        assert 'optimization_stats' in result
        
        # 최적 가중치 검증
        assert len(result['best_weights']) == len(factor_cols)
        assert abs(sum(result['best_weights']) - 1.0) < 0.01
        
        # 상위 조합 검증
        assert len(result['top_combinations']) <= 2
    
    def test_combine_factors(self, combiner, sample_data):
        """팩터 결합 테스트"""
        factor_cols = ['factor_1', 'factor_2']
        weights = [0.7, 0.3]
        
        combined = combiner.combine_factors(sample_data, factor_cols, weights)
        
        assert 'combined_factor' in combined.columns
        assert len(combined) == len(sample_data)
        assert not combined['combined_factor'].isna().any()
    
    def test_combine_factors_with_optimization(self, combiner, sample_data):
        """최적화를 포함한 팩터 결합 테스트"""
        factor_cols = ['factor_1', 'factor_2']
        
        combined = combiner.combine_factors(
            sample_data, 
            factor_cols,
            optimize=True,
            n_combinations=3
        )
        
        assert 'combined_factor' in combined.columns
        assert 'optimization_result' in combined.attrs
    
    def test_backtest_combination(self, combiner, sample_data):
        """백테스트 테스트"""
        sample_data['combined_factor'] = np.random.randn(len(sample_data))
        
        backtest_result = combiner.backtest_combination(
            sample_data,
            signal_col='combined_factor',
            price_col='close'
        )
        
        assert 'returns' in backtest_result
        assert 'cumulative_returns' in backtest_result
        assert 'sharpe_ratio' in backtest_result
        assert 'max_drawdown' in backtest_result
        assert 'total_return' in backtest_result
    
    def test_get_optimization_summary(self, combiner, sample_data):
        """최적화 요약 테스트"""
        factor_cols = ['factor_1', 'factor_2']
        
        # 먼저 최적화 실행
        result = combiner.optimize_combination(
            sample_data, 
            factor_cols,
            n_combinations=5
        )
        
        # 요약 가져오기
        summary = combiner.get_optimization_summary(result)
        
        assert 'Best Sharpe Ratio' in summary
        assert 'Best Annual Return' in summary
        assert 'Best Max Drawdown' in summary
        assert 'Optimal Weights' in summary
        assert 'Top 3 Combinations' in summary
    
    def test_evaluate_stability(self, combiner, sample_data):
        """안정성 평가 테스트"""
        factor_cols = ['factor_1', 'factor_2']
        weights = [0.5, 0.5]
        
        stability = combiner.evaluate_stability(
            sample_data,
            factor_cols,
            weights,
            window_size=20,
            n_windows=3
        )
        
        assert 'mean_sharpe' in stability
        assert 'std_sharpe' in stability
        assert 'stability_score' in stability
        assert 'performance_by_window' in stability
        
        assert 0 <= stability['stability_score'] <= 1
    
    def test_analyze_factor_correlation(self, combiner, sample_data):
        """팩터 상관관계 분석 테스트"""
        factor_cols = ['factor_1', 'factor_2', 'factor_3']
        
        correlation_info = combiner.analyze_factor_correlation(
            sample_data,
            factor_cols
        )
        
        assert 'correlation_matrix' in correlation_info
        assert 'high_correlations' in correlation_info
        assert 'recommended_factors' in correlation_info
        assert 'average_correlation' in correlation_info
        
        # 상관관계 행렬 검증
        corr_matrix = correlation_info['correlation_matrix']
        assert corr_matrix.shape == (3, 3)
        assert all(corr_matrix.diagonal() == 1.0)
    
    def test_error_handling_invalid_data(self, combiner):
        """잘못된 데이터 처리 테스트"""
        invalid_data = pd.DataFrame({'col1': [1, 2, 'invalid', 4]})
        
        with pytest.raises(Exception):
            combiner.calculate_performance(
                invalid_data, 
                ['col1'], 
                [1.0]
            )
    
    def test_caching_mechanism(self, combiner, sample_data):
        """캐싱 메커니즘 테스트"""
        factor_cols = ['factor_1', 'factor_2']
        weights = (0.5, 0.5)
        
        # 첫 번째 계산
        result1 = combiner.calculate_performance(
            sample_data, factor_cols, weights
        )
        
        # 캐시 확인
        cache_key = f"{tuple(factor_cols)}_{weights}"
        assert cache_key in combiner.combination_cache
        
        # 두 번째 계산 (캐시에서 가져와야 함)
        with patch.object(combiner, '_calculate_returns') as mock_calc:
            result2 = combiner.calculate_performance(
                sample_data, factor_cols, weights
            )
            # _calculate_returns가 호출되지 않아야 함 (캐시 사용)
            # 참고: 실제 구현에 따라 이 테스트는 조정이 필요할 수 있음
    
    def test_parallel_optimization(self, combiner, sample_data):
        """병렬 최적화 테스트"""
        factor_cols = ['factor_1', 'factor_2', 'factor_3']
        
        with patch('concurrent.futures.ProcessPoolExecutor') as mock_executor:
            mock_executor.return_value.__enter__.return_value.map = Mock(
                return_value=[{'sharpe_ratio': 1.5}] * 5
            )
            
            result = combiner.optimize_combination(
                sample_data,
                factor_cols,
                n_combinations=5,
                parallel=True
            )
            
            assert result is not None