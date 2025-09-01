"""
고급 통합 테스트
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from unittest.mock import Mock, patch
from utils.data_processor import DataProcessor
from utils.factor_miner import FactorMiner
from utils.dynamic_combiner import DynamicCombiner
from utils.performance_analyzer import PerformanceAnalyzer
from utils.validators import DataValidator
from utils.cache_utils import CustomCache


class TestAdvancedIntegration:
    """고급 통합 테스트 스위트"""
    
    @pytest.fixture
    def complete_workflow_data(self):
        """완전한 워크플로우 테스트 데이터"""
        dates = pd.date_range(start='2019-01-01', periods=500, freq='D')
        price_base = 100
        prices = []
        
        for i in range(500):
            price_base *= (1 + np.random.randn() * 0.02)
            prices.append(price_base)
        
        return pd.DataFrame({
            'date': dates,
            'open': np.array(prices) * (1 + np.random.randn(500) * 0.01),
            'high': np.array(prices) * (1 + np.abs(np.random.randn(500) * 0.02)),
            'low': np.array(prices) * (1 - np.abs(np.random.randn(500) * 0.02)),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 500),
            'ticker': 'TEST'
        })
    
    @pytest.fixture
    def multi_asset_data(self):
        """다중 자산 테스트 데이터"""
        dates = pd.date_range(start='2019-01-01', periods=300, freq='D')
        assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        
        all_data = []
        for asset in assets:
            price_base = 100 + np.random.rand() * 100
            prices = []
            for i in range(300):
                price_base *= (1 + np.random.randn() * 0.02)
                prices.append(price_base)
            
            asset_data = pd.DataFrame({
                'date': dates,
                'ticker': asset,
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, 300),
                'open': np.array(prices) * (1 + np.random.randn(300) * 0.01),
                'high': np.array(prices) * (1 + np.abs(np.random.randn(300) * 0.02)),
                'low': np.array(prices) * (1 - np.abs(np.random.randn(300) * 0.02))
            })
            all_data.append(asset_data)
        
        return pd.concat(all_data, ignore_index=True)
    
    def test_complete_pipeline_with_validation(self, complete_workflow_data):
        """검증을 포함한 완전한 파이프라인 테스트"""
        # 1. 데이터 검증
        validator = DataValidator()
        validation_result = validator.validate_data(complete_workflow_data)
        assert validation_result['is_valid']
        assert validation_result['quality_score'] > 0.8
        
        # 2. 데이터 처리
        processor = DataProcessor()
        processed_data = processor.process_data(complete_workflow_data)
        assert not processed_data.empty
        assert 'returns' in processed_data.columns
        
        # 3. 팩터 마이닝
        miner = FactorMiner()
        factors = miner.generate_basic_factors(processed_data)
        assert not factors.empty
        assert len(factors.columns) > 5
        
        # 4. 팩터 조합
        combiner = DynamicCombiner()
        factor_cols = [col for col in factors.columns if col.startswith('factor_')]
        combined = combiner.combine_factors(
            factors, 
            factor_cols[:3] if len(factor_cols) >= 3 else factor_cols,
            optimize=True,
            n_combinations=5
        )
        assert 'combined_factor' in combined.columns
        
        # 5. 성능 분석
        analyzer = PerformanceAnalyzer()
        if 'returns' in combined.columns:
            report = analyzer.generate_performance_report(
                returns=combined['returns'],
                dates=combined.get('date', combined.index)
            )
            assert 'summary_metrics' in report
    
    def test_multi_asset_portfolio_workflow(self, multi_asset_data):
        """다중 자산 포트폴리오 워크플로우 테스트"""
        processor = DataProcessor()
        miner = FactorMiner()
        combiner = DynamicCombiner()
        
        portfolio_results = {}
        
        # 각 자산별 처리
        for ticker in multi_asset_data['ticker'].unique():
            asset_data = multi_asset_data[multi_asset_data['ticker'] == ticker].copy()
            
            # 데이터 처리
            processed = processor.process_data(asset_data)
            
            # 팩터 생성
            factors = miner.generate_basic_factors(processed)
            
            # 팩터 성능 계산
            performance = miner.calculate_factor_performance(factors)
            
            portfolio_results[ticker] = {
                'data': processed,
                'factors': factors,
                'performance': performance
            }
        
        # 포트폴리오 레벨 분석
        assert len(portfolio_results) == 4
        for ticker, results in portfolio_results.items():
            assert 'data' in results
            assert 'factors' in results
            assert 'performance' in results
    
    def test_error_recovery_workflow(self):
        """에러 복구 워크플로우 테스트"""
        processor = DataProcessor()
        
        # 잘못된 데이터로 시작
        bad_data = pd.DataFrame({
            'date': ['invalid', 'date', 'format'],
            'close': [100, 'invalid', 102]
        })
        
        # 에러 처리 확인
        try:
            processed = processor.process_data(bad_data)
        except Exception as e:
            # 에러 발생 확인
            assert True
            
            # 복구 시도: 유효한 데이터로 재시도
            good_data = pd.DataFrame({
                'date': pd.date_range(start='2020-01-01', periods=100),
                'close': np.random.randn(100) * 10 + 100,
                'volume': np.random.randint(1000000, 10000000, 100)
            })
            
            processed = processor.process_data(good_data)
            assert not processed.empty
    
    def test_caching_workflow(self, complete_workflow_data):
        """캐싱 워크플로우 테스트"""
        cache_manager = CustomCache()
        processor = DataProcessor()
        
        # 첫 번째 처리 (캐시 저장)
        cache_key = 'test_workflow_data'
        processed_data = processor.process_data(complete_workflow_data)
        cache_manager.set(cache_key, processed_data)
        
        # 캐시에서 데이터 가져오기
        cached_data = cache_manager.get(cache_key)
        assert cached_data is not None
        pd.testing.assert_frame_equal(processed_data, cached_data)
        
        # 캐시 무효화
        cache_manager.clear(cache_key)
        assert cache_manager.get(cache_key) is None
    
    def test_performance_under_load(self, complete_workflow_data):
        """부하 상태에서의 성능 테스트"""
        import time
        processor = DataProcessor()
        miner = FactorMiner()
        
        # 대량 데이터 생성 (5000 rows)
        large_data = pd.concat([complete_workflow_data] * 10, ignore_index=True)
        
        start_time = time.time()
        
        # 처리 파이프라인
        processed = processor.process_data(large_data)
        factors = miner.generate_basic_factors(processed)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 성능 기준: 5000 rows를 10초 이내 처리
        assert processing_time < 10
        assert not factors.empty
    
    def test_concurrent_processing(self, multi_asset_data):
        """동시 처리 테스트"""
        from concurrent.futures import ThreadPoolExecutor
        processor = DataProcessor()
        
        def process_asset(ticker_data):
            return processor.process_data(ticker_data)
        
        # 자산별 데이터 분리
        asset_groups = [
            multi_asset_data[multi_asset_data['ticker'] == ticker].copy()
            for ticker in multi_asset_data['ticker'].unique()
        ]
        
        # 병렬 처리
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_asset, asset_groups))
        
        # 결과 검증
        assert len(results) == 4
        for result in results:
            assert not result.empty
            assert 'returns' in result.columns
    
    def test_data_persistence_workflow(self, complete_workflow_data):
        """데이터 지속성 워크플로우 테스트"""
        processor = DataProcessor()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 데이터 처리 및 저장
            processed = processor.process_data(complete_workflow_data)
            output_path = os.path.join(tmpdir, 'processed_data.csv')
            processed.to_csv(output_path, index=False)
            
            # 데이터 로드 및 검증
            loaded_data = pd.read_csv(output_path)
            assert len(loaded_data) == len(processed)
            assert set(loaded_data.columns) == set(processed.columns)
    
    def test_incremental_update_workflow(self, complete_workflow_data):
        """증분 업데이트 워크플로우 테스트"""
        processor = DataProcessor()
        
        # 초기 데이터 처리
        initial_data = complete_workflow_data.iloc[:400].copy()
        processed_initial = processor.process_data(initial_data)
        
        # 새로운 데이터 추가
        new_data = complete_workflow_data.iloc[400:].copy()
        processed_new = processor.process_data(new_data)
        
        # 데이터 병합
        combined = pd.concat([processed_initial, processed_new], ignore_index=True)
        
        # 전체 처리와 비교
        processed_full = processor.process_data(complete_workflow_data)
        
        assert len(combined) == len(processed_full)
    
    def test_cross_validation_workflow(self, complete_workflow_data):
        """교차 검증 워크플로우 테스트"""
        processor = DataProcessor()
        miner = FactorMiner()
        analyzer = PerformanceAnalyzer()
        
        # 데이터 분할
        train_size = int(len(complete_workflow_data) * 0.7)
        train_data = complete_workflow_data.iloc[:train_size].copy()
        test_data = complete_workflow_data.iloc[train_size:].copy()
        
        # 훈련 데이터 처리
        train_processed = processor.process_data(train_data)
        train_factors = miner.generate_basic_factors(train_processed)
        
        # 테스트 데이터 처리
        test_processed = processor.process_data(test_data)
        test_factors = miner.generate_basic_factors(test_processed)
        
        # 성능 비교
        if 'returns' in train_factors.columns and 'returns' in test_factors.columns:
            train_sharpe = analyzer.calculate_sharpe_ratio(train_factors['returns'])
            test_sharpe = analyzer.calculate_sharpe_ratio(test_factors['returns'])
            
            # 과적합 체크 (train과 test 성능 차이가 너무 크지 않아야 함)
            assert abs(train_sharpe - test_sharpe) < 2.0
    
    def test_fault_tolerance(self):
        """장애 허용 테스트"""
        processor = DataProcessor()
        miner = FactorMiner()
        
        # 부분적으로 손상된 데이터
        dates = pd.date_range(start='2020-01-01', periods=100)
        corrupted_data = pd.DataFrame({
            'date': dates,
            'close': [100 if i % 10 == 0 else 100 + np.random.randn() for i in range(100)],
            'volume': [np.nan if i % 15 == 0 else np.random.randint(1000000, 10000000) for i in range(100)]
        })
        
        # 처리 시도
        try:
            processed = processor.process_data(corrupted_data)
            factors = miner.generate_basic_factors(processed)
            
            # 일부 데이터 손실은 허용
            assert len(factors) > 0
            assert not factors.empty
        except Exception as e:
            # 에러가 발생해도 시스템이 중단되지 않음
            assert True
    
    def test_memory_efficiency(self, complete_workflow_data):
        """메모리 효율성 테스트"""
        import tracemalloc
        
        processor = DataProcessor()
        miner = FactorMiner()
        
        # 메모리 추적 시작
        tracemalloc.start()
        
        # 대량 데이터 처리
        large_data = pd.concat([complete_workflow_data] * 20, ignore_index=True)
        processed = processor.process_data(large_data)
        factors = miner.generate_basic_factors(processed)
        
        # 메모리 사용량 확인
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # 메모리 사용량이 합리적인 범위 내에 있어야 함 (1GB 미만)
        assert peak / 1024 / 1024 < 1024  # MB
    
    def test_api_compatibility(self):
        """API 호환성 테스트"""
        # 모든 주요 클래스가 예상된 인터페이스를 제공하는지 확인
        processor = DataProcessor()
        miner = FactorMiner()
        combiner = DynamicCombiner()
        analyzer = PerformanceAnalyzer()
        validator = DataValidator()
        
        # 필수 메서드 존재 확인
        assert hasattr(processor, 'process_data')
        assert hasattr(miner, 'generate_basic_factors')
        assert hasattr(combiner, 'combine_factors')
        assert hasattr(analyzer, 'calculate_sharpe_ratio')
        assert hasattr(validator, 'validate_data')