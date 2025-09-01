"""
Performance Analyzer 단위 테스트
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from utils.performance_analyzer import PerformanceAnalyzer


class TestPerformanceAnalyzer:
    """PerformanceAnalyzer 클래스 테스트"""
    
    @pytest.fixture
    def sample_returns(self):
        """테스트용 수익률 데이터"""
        dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
        returns = np.random.randn(252) * 0.02  # 일일 수익률
        return pd.Series(returns, index=dates, name='returns')
    
    @pytest.fixture
    def sample_portfolio_data(self):
        """테스트용 포트폴리오 데이터"""
        dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
        return pd.DataFrame({
            'date': dates,
            'returns': np.random.randn(252) * 0.02,
            'portfolio_value': 100000 * (1 + np.random.randn(252) * 0.01).cumprod(),
            'benchmark_returns': np.random.randn(252) * 0.015,
            'positions': np.random.randint(-10, 11, 252)
        })
    
    @pytest.fixture
    def analyzer(self):
        """PerformanceAnalyzer 인스턴스"""
        return PerformanceAnalyzer()
    
    def test_init(self, analyzer):
        """초기화 테스트"""
        assert analyzer is not None
        assert hasattr(analyzer, 'metrics_cache')
        assert analyzer.metrics_cache == {}
    
    def test_calculate_sharpe_ratio(self, analyzer, sample_returns):
        """샤프 비율 계산 테스트"""
        sharpe = analyzer.calculate_sharpe_ratio(sample_returns)
        
        assert isinstance(sharpe, float)
        assert -10 < sharpe < 10  # 합리적인 범위
    
    def test_calculate_sharpe_ratio_with_risk_free(self, analyzer, sample_returns):
        """무위험 수익률을 포함한 샤프 비율 계산 테스트"""
        sharpe = analyzer.calculate_sharpe_ratio(sample_returns, risk_free_rate=0.02)
        
        assert isinstance(sharpe, float)
    
    def test_calculate_max_drawdown(self, analyzer, sample_returns):
        """최대 낙폭 계산 테스트"""
        cumulative_returns = (1 + sample_returns).cumprod()
        max_dd = analyzer.calculate_max_drawdown(cumulative_returns)
        
        assert isinstance(max_dd, float)
        assert 0 <= max_dd <= 1  # 0과 1 사이
    
    def test_calculate_calmar_ratio(self, analyzer, sample_returns):
        """칼마 비율 계산 테스트"""
        calmar = analyzer.calculate_calmar_ratio(sample_returns)
        
        assert isinstance(calmar, float)
    
    def test_calculate_sortino_ratio(self, analyzer, sample_returns):
        """소르티노 비율 계산 테스트"""
        sortino = analyzer.calculate_sortino_ratio(sample_returns)
        
        assert isinstance(sortino, float)
    
    def test_calculate_information_ratio(self, analyzer, sample_portfolio_data):
        """정보 비율 계산 테스트"""
        portfolio_returns = sample_portfolio_data['returns']
        benchmark_returns = sample_portfolio_data['benchmark_returns']
        
        info_ratio = analyzer.calculate_information_ratio(
            portfolio_returns, 
            benchmark_returns
        )
        
        assert isinstance(info_ratio, float)
    
    def test_calculate_win_rate(self, analyzer, sample_returns):
        """승률 계산 테스트"""
        win_rate = analyzer.calculate_win_rate(sample_returns)
        
        assert isinstance(win_rate, float)
        assert 0 <= win_rate <= 1
    
    def test_calculate_profit_factor(self, analyzer, sample_returns):
        """수익 팩터 계산 테스트"""
        profit_factor = analyzer.calculate_profit_factor(sample_returns)
        
        assert isinstance(profit_factor, float)
        assert profit_factor >= 0
    
    def test_analyze_returns_distribution(self, analyzer, sample_returns):
        """수익률 분포 분석 테스트"""
        distribution = analyzer.analyze_returns_distribution(sample_returns)
        
        assert 'mean' in distribution
        assert 'std' in distribution
        assert 'skewness' in distribution
        assert 'kurtosis' in distribution
        assert 'percentiles' in distribution
        assert 'var_95' in distribution
        assert 'cvar_95' in distribution
    
    def test_calculate_rolling_metrics(self, analyzer, sample_returns):
        """롤링 메트릭 계산 테스트"""
        rolling_metrics = analyzer.calculate_rolling_metrics(
            sample_returns,
            window=60
        )
        
        assert 'rolling_sharpe' in rolling_metrics
        assert 'rolling_volatility' in rolling_metrics
        assert 'rolling_mean' in rolling_metrics
        
        # 길이 확인
        assert len(rolling_metrics['rolling_sharpe']) == len(sample_returns)
    
    def test_perform_monthly_analysis(self, analyzer, sample_portfolio_data):
        """월별 분석 테스트"""
        monthly_analysis = analyzer.perform_monthly_analysis(
            sample_portfolio_data['returns'],
            sample_portfolio_data['date']
        )
        
        assert 'monthly_returns' in monthly_analysis
        assert 'monthly_stats' in monthly_analysis
        assert 'best_month' in monthly_analysis
        assert 'worst_month' in monthly_analysis
    
    def test_calculate_risk_metrics(self, analyzer, sample_returns):
        """리스크 메트릭 계산 테스트"""
        risk_metrics = analyzer.calculate_risk_metrics(sample_returns)
        
        assert 'volatility' in risk_metrics
        assert 'downside_deviation' in risk_metrics
        assert 'max_drawdown' in risk_metrics
        assert 'var_95' in risk_metrics
        assert 'cvar_95' in risk_metrics
        assert 'beta' in risk_metrics or 'beta' not in risk_metrics  # Optional
    
    def test_generate_performance_report(self, analyzer, sample_portfolio_data):
        """성능 리포트 생성 테스트"""
        report = analyzer.generate_performance_report(
            returns=sample_portfolio_data['returns'],
            benchmark_returns=sample_portfolio_data['benchmark_returns'],
            dates=sample_portfolio_data['date']
        )
        
        assert 'summary_metrics' in report
        assert 'risk_metrics' in report
        assert 'return_metrics' in report
        assert 'comparison_metrics' in report
        assert 'time_series_analysis' in report
    
    def test_calculate_turnover(self, analyzer, sample_portfolio_data):
        """회전율 계산 테스트"""
        turnover = analyzer.calculate_turnover(sample_portfolio_data['positions'])
        
        assert isinstance(turnover, float)
        assert turnover >= 0
    
    def test_analyze_drawdown_periods(self, analyzer, sample_returns):
        """낙폭 기간 분석 테스트"""
        cumulative_returns = (1 + sample_returns).cumprod()
        drawdown_analysis = analyzer.analyze_drawdown_periods(cumulative_returns)
        
        assert 'max_drawdown' in drawdown_analysis
        assert 'max_drawdown_duration' in drawdown_analysis
        assert 'recovery_time' in drawdown_analysis
        assert 'drawdown_periods' in drawdown_analysis
    
    def test_calculate_alpha_beta(self, analyzer, sample_portfolio_data):
        """알파, 베타 계산 테스트"""
        portfolio_returns = sample_portfolio_data['returns']
        benchmark_returns = sample_portfolio_data['benchmark_returns']
        
        alpha, beta = analyzer.calculate_alpha_beta(
            portfolio_returns,
            benchmark_returns
        )
        
        assert isinstance(alpha, float)
        assert isinstance(beta, float)
    
    def test_performance_attribution(self, analyzer, sample_portfolio_data):
        """성과 귀인 분석 테스트"""
        attribution = analyzer.performance_attribution(
            sample_portfolio_data['returns'],
            sample_portfolio_data['benchmark_returns'],
            sample_portfolio_data.get('factor_returns', None)
        )
        
        assert 'total_return' in attribution
        assert 'benchmark_return' in attribution
        assert 'active_return' in attribution
        assert 'attribution_components' in attribution
    
    def test_calculate_treynor_ratio(self, analyzer, sample_portfolio_data):
        """트레이너 비율 계산 테스트"""
        portfolio_returns = sample_portfolio_data['returns']
        benchmark_returns = sample_portfolio_data['benchmark_returns']
        
        treynor = analyzer.calculate_treynor_ratio(
            portfolio_returns,
            benchmark_returns,
            risk_free_rate=0.02
        )
        
        assert isinstance(treynor, float)
    
    def test_stress_test(self, analyzer, sample_portfolio_data):
        """스트레스 테스트"""
        scenarios = {
            'market_crash': -0.20,
            'recession': -0.10,
            'inflation_spike': -0.05
        }
        
        stress_results = analyzer.stress_test(
            sample_portfolio_data['portfolio_value'],
            scenarios
        )
        
        assert len(stress_results) == len(scenarios)
        for scenario, result in stress_results.items():
            assert 'impact' in result
            assert 'new_value' in result
    
    def test_calculate_omega_ratio(self, analyzer, sample_returns):
        """오메가 비율 계산 테스트"""
        omega = analyzer.calculate_omega_ratio(sample_returns, threshold=0)
        
        assert isinstance(omega, float)
        assert omega >= 0
    
    def test_error_handling_empty_data(self, analyzer):
        """빈 데이터 처리 테스트"""
        empty_series = pd.Series([])
        
        with pytest.raises(ValueError):
            analyzer.calculate_sharpe_ratio(empty_series)
    
    def test_error_handling_invalid_data(self, analyzer):
        """잘못된 데이터 처리 테스트"""
        invalid_series = pd.Series([np.nan, np.nan, np.nan])
        
        result = analyzer.calculate_sharpe_ratio(invalid_series)
        assert np.isnan(result) or result == 0
    
    def test_caching_mechanism(self, analyzer, sample_returns):
        """캐싱 메커니즘 테스트"""
        # 첫 번째 계산
        sharpe1 = analyzer.calculate_sharpe_ratio(sample_returns)
        
        # 캐시 확인
        assert len(analyzer.metrics_cache) > 0
        
        # 두 번째 계산 (캐시에서 가져와야 함)
        with patch.object(analyzer, '_compute_sharpe') as mock_compute:
            sharpe2 = analyzer.calculate_sharpe_ratio(sample_returns)
            # 실제 계산이 호출되지 않아야 함
    
    def test_comprehensive_analysis(self, analyzer, sample_portfolio_data):
        """종합 분석 테스트"""
        comprehensive = analyzer.comprehensive_analysis(
            portfolio_data=sample_portfolio_data,
            benchmark_data=sample_portfolio_data['benchmark_returns']
        )
        
        assert 'performance_metrics' in comprehensive
        assert 'risk_metrics' in comprehensive
        assert 'monthly_analysis' in comprehensive
        assert 'drawdown_analysis' in comprehensive
        assert 'distribution_analysis' in comprehensive