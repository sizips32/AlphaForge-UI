#!/usr/bin/env python3
"""
AlphaForge-UI 전체 워크플로우 테스트 스크립트
데이터 생성 → 팩터 마이닝 → 동적 결합 → 백테스팅 → 리포트 생성의 전체 과정을 테스트합니다.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_complete_workflow():
    """전체 워크플로우를 테스트합니다."""
    print("🚀 AlphaForge-UI 전체 워크플로우 테스트 시작")
    print("=" * 60)
    
    try:
        # 1. 데이터 생성 테스트
        print("\n📊 1단계: 샘플 데이터 생성")
        from pages.data_management import create_sample_data
        sample_data = create_sample_data()
        print(f"✅ 샘플 데이터 생성 완료: {len(sample_data):,}행, {sample_data['Ticker'].nunique()}종목")
        
        # 2. 데이터 처리 테스트
        print("\n🔧 2단계: 데이터 처리")
        from utils.data_processor import DataProcessor
        processor = DataProcessor()
        processed_data = processor.process_data(sample_data)
        print(f"✅ 데이터 처리 완료: {len(processed_data):,}행")
        
        # 3. 팩터 마이닝 테스트
        print("\n🧠 3단계: 팩터 마이닝")
        from utils.factor_miner import FactorMiner
        
        # 팩터 마이닝 설정
        mining_settings = {
            'factor_types': ['Momentum', 'Value', 'Quality'],
            'factor_pool_size': 10,
            'min_ic': 0.02,
            'min_icir': 0.5,
            'hidden_layers': 3,
            'neurons_per_layer': 128,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 64,
            'dropout_rate': 0.2
        }
        
        miner = FactorMiner(mining_settings)
        
        # 기본 팩터 생성
        basic_factors = miner.generate_basic_factors(processed_data)
        print(f"✅ 기본 팩터 생성 완료: {len(basic_factors)}개")
        
        # AI 팩터 생성
        ai_factors = miner.generate_ai_factors(processed_data, basic_factors)
        print(f"✅ AI 팩터 생성 완료: {len(ai_factors)}개")
        
        # 마이닝 결과 구성
        mining_results = {
            'factors': basic_factors + ai_factors,
            'basic_factors': basic_factors,
            'ai_factors': ai_factors,
            'settings': mining_settings
        }
        print(f"✅ 팩터 마이닝 완료: {len(mining_results['factors'])}개 팩터 생성")
        
        # 4. 동적 결합 테스트
        print("\n⚖️ 4단계: 동적 결합")
        from utils.dynamic_combiner import DynamicCombiner
        
        combination_settings = {
            'combination_method': '동적 가중치',
            'top_factors': 5,
            'rebalancing_freq': '월간',
            'min_ic_threshold': 0.02,
            'min_icir_threshold': 0.5,
            'weight_decay': 0.1,
            'optimization_target': '샤프 비율',
            'max_weight': 0.3,
            'volatility_limit': 0.2,
            'max_drawdown_limit': 0.15
        }
        
        combiner = DynamicCombiner(combination_settings)
        
        combination_results = combiner.create_mega_alpha(
            processed_data, 
            mining_results['factors']
        )
        print(f"✅ 동적 결합 완료: 메가-알파 생성됨")
        
        # 5. 백테스팅 테스트
        print("\n📊 5단계: 백테스팅")
        from utils.backtester import Backtester
        
        backtest_settings = {
            'initial_capital': 100000,
            'portfolio_size': 20,
            'rebalancing_freq': '월간',
            'transaction_cost': 0.001,
            'max_position_size': 0.1,
            'stop_loss': 0.05,
            'take_profit': 0.15,
            'risk_free_rate': 0.02
        }
        
        backtester = Backtester(backtest_settings)
        
        # 포트폴리오 구성
        portfolio = backtester.construct_portfolio(
            processed_data,
            combination_results['values']
        )
        
        # 백테스팅 실행
        backtest_results = backtester.run_backtest(processed_data, portfolio)
        
        # 성과 지표 계산
        performance_metrics = backtester.calculate_performance_metrics(backtest_results)
        backtest_results['performance'] = performance_metrics
        print(f"✅ 백테스팅 완료: 수익률 {backtest_results['performance']['total_return']:.2%}")
        
        # 6. 성과 분석 테스트
        print("\n📈 6단계: 성과 분석")
        from utils.performance_analyzer import PerformanceAnalyzer
        analyzer = PerformanceAnalyzer()
        
        # 백테스팅 결과에서 성과 지표 추출
        performance_metrics = backtest_results['performance']
        print(f"✅ 성과 분석 완료: 샤프 비율 {performance_metrics.get('sharpe_ratio', 0):.2f}")
        
        # 7. 결과 요약
        print("\n" + "=" * 60)
        print("🎉 전체 워크플로우 테스트 완료!")
        print("=" * 60)
        
        print(f"📊 데이터: {len(processed_data):,}행, {processed_data['Ticker'].nunique()}종목")
        print(f"🧠 팩터: {len(mining_results['factors'])}개")
        print(f"⚖️ 메가-알파: 생성됨")
        print(f"📈 수익률: {backtest_results['performance']['total_return']:.2%}")
        print(f"📊 샤프 비율: {performance_metrics['sharpe_ratio']:.2f}")
        print(f"📉 최대 낙폭: {performance_metrics['max_drawdown']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """개별 컴포넌트들을 테스트합니다."""
    print("\n🔍 개별 컴포넌트 테스트")
    print("-" * 40)
    
    # 데이터 검증 테스트
    try:
        from utils.validators import DataValidator
        validator = DataValidator()
        print("✅ DataValidator 로드 성공")
    except Exception as e:
        print(f"❌ DataValidator 로드 실패: {e}")
    
    # 야후 파이낸스 다운로더 테스트
    try:
        from utils.yahoo_finance_downloader import YahooFinanceDownloader
        downloader = YahooFinanceDownloader()
        print("✅ YahooFinanceDownloader 로드 성공")
    except Exception as e:
        print(f"❌ YahooFinanceDownloader 로드 실패: {e}")
    
    # 설정 테스트
    try:
        from utils.config import UI_SETTINGS, DATA_VALIDATION
        print("✅ 설정 로드 성공")
    except Exception as e:
        print(f"❌ 설정 로드 실패: {e}")

if __name__ == "__main__":
    print("AlphaForge-UI 워크플로우 테스트")
    print("=" * 60)
    
    # 개별 컴포넌트 테스트
    test_individual_components()
    
    # 전체 워크플로우 테스트
    success = test_complete_workflow()
    
    if success:
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
        print("AlphaForge-UI가 정상적으로 작동합니다.")
    else:
        print("\n⚠️ 일부 테스트에서 문제가 발생했습니다.")
        print("로그를 확인하여 문제를 해결하세요.") 
