"""
백테스팅 페이지
포트폴리오 성과 분석 및 리스크 지표를 제공합니다.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.backtester import Backtester
from utils.performance_analyzer import PerformanceAnalyzer

def show_page():
    """백테스팅 페이지를 표시합니다."""
    st.title("📊 백테스팅")
    st.markdown("포트폴리오 성과 분석 및 리스크 지표를 제공합니다.")
    
    # 데이터 확인
    if 'combination_results' not in st.session_state:
        st.error("❌ 먼저 동적 결합을 완료해주세요.")
        st.info("⚖️ 동적 결합 페이지에서 메가-알파를 생성하세요.")
        return
    
    # 탭 인터페이스
    tab1, tab2, tab3 = st.tabs(["⚙️ 설정", "🔄 실행", "📊 결과"])
    
    with tab1:
        show_settings_tab()
    
    with tab2:
        show_execution_tab()
    
    with tab3:
        show_results_tab()

def show_settings_tab():
    """설정 탭을 표시합니다."""
    st.subheader("⚙️ 백테스팅 설정")
    
    # 기본 설정
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 백테스팅 기간")
        
        # 시작 날짜
        start_date = st.date_input(
            "시작 날짜",
            value=pd.to_datetime("2020-01-01").date(),
            help="백테스팅 시작 날짜"
        )
        
        # 종료 날짜
        end_date = st.date_input(
            "종료 날짜",
            value=pd.to_datetime("2024-12-31").date(),
            help="백테스팅 종료 날짜"
        )
        
        # 초기 자본
        initial_capital = st.number_input(
            "초기 자본",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=1000,
            help="백테스팅 초기 자본금"
        )
    
    with col2:
        st.markdown("### 🎯 포트폴리오 설정")
        
        # 포트폴리오 크기
        portfolio_size = st.slider(
            "포트폴리오 크기",
            min_value=10,
            max_value=100,
            value=50,
            help="포함할 종목 수"
        )
        
        # 리밸런싱 주기
        rebalancing_freq = st.selectbox(
            "리밸런싱 주기",
            options=["일간", "주간", "월간", "분기"],
            index=2,
            help="포트폴리오 리밸런싱 주기"
        )
        
        # 거래 비용
        transaction_cost = st.number_input(
            "거래 비용 (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.2f",
            help="거래당 비용 (수수료 + 슬리피지)"
        )
    
    # 고급 설정
    with st.expander("🔧 고급 설정"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🛡️ 리스크 관리")
            
            # 최대 포지션 크기
            max_position_size = st.slider(
                "최대 포지션 크기 (%)",
                min_value=1,
                max_value=20,
                value=5,
                help="단일 종목 최대 비중"
            )
            
            # 스톱로스
            stop_loss = st.number_input(
                "스톱로스 (%)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=1.0,
                format="%.1f",
                help="손실 제한 비율"
            )
        
        with col2:
            st.markdown("#### 📊 벤치마크 설정")
            
            # 벤치마크 선택
            benchmark = st.selectbox(
                "벤치마크",
                options=["SPY", "QQQ", "IWM", "EFA", "AGG", "사용자 정의"],
                index=0,
                help="성과 비교 기준"
            )
            
            # 무위험 수익률
            risk_free_rate = st.number_input(
                "무위험 수익률 (%)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.1,
                format="%.1f",
                help="연간 무위험 수익률"
            )
    
    # 설정 저장
    if st.button("💾 설정 저장", use_container_width=True):
        settings = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'portfolio_size': portfolio_size,
            'rebalancing_freq': rebalancing_freq,
            'transaction_cost': transaction_cost / 100,
            'max_position_size': max_position_size / 100,
            'stop_loss': stop_loss / 100,
            'benchmark': benchmark,
            'risk_free_rate': risk_free_rate / 100
        }
        
        st.session_state['backtest_settings'] = settings
        st.success("✅ 설정이 저장되었습니다!")

def show_execution_tab():
    """실행 탭을 표시합니다."""
    st.subheader("🔄 백테스팅 실행")
    
    # 설정 확인
    if 'backtest_settings' not in st.session_state:
        st.warning("⚠️ 먼저 설정 탭에서 백테스팅 설정을 완료해주세요.")
        return
    
    settings = st.session_state['backtest_settings']
    
    # 설정 요약
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 백테스팅 설정")
        st.write(f"**기간**: {settings['start_date']} ~ {settings['end_date']}")
        st.write(f"**초기 자본**: ${settings['initial_capital']:,}")
        st.write(f"**포트폴리오 크기**: {settings['portfolio_size']}종목")
        st.write(f"**리밸런싱 주기**: {settings['rebalancing_freq']}")
    
    with col2:
        st.markdown("### 🎯 리스크 설정")
        st.write(f"**거래 비용**: {settings['transaction_cost']:.1%}")
        st.write(f"**최대 포지션**: {settings['max_position_size']:.1%}")
        st.write(f"**스톱로스**: {settings['stop_loss']:.1%}")
        st.write(f"**벤치마크**: {settings['benchmark']}")
    
    # 실행 버튼
    if st.button("🔄 백테스팅 시작", use_container_width=True, type="primary"):
        run_backtesting(settings)

def run_backtesting(settings):
    """백테스팅을 실행합니다."""
    # 진행 상황 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. 데이터 준비 (10%)
        status_text.text("📊 데이터 준비 중...")
        combination_results = st.session_state['combination_results']
        mega_alpha = combination_results['mega_alpha']
        data = st.session_state['processed_data']
        progress_bar.progress(10)
        
        # 2. 백테스터 초기화 (20%)
        status_text.text("🔄 백테스터 초기화 중...")
        backtester = Backtester(settings)
        progress_bar.progress(20)
        
        # 3. 포트폴리오 구성 (40%)
        status_text.text("📊 포트폴리오 구성 중...")
        portfolio = backtester.construct_portfolio(data, mega_alpha)
        progress_bar.progress(40)
        
        # 4. 백테스팅 실행 (70%)
        status_text.text("🔄 백테스팅 실행 중...")
        backtest_results = backtester.run_backtest(data, portfolio)
        progress_bar.progress(70)
        
        # 5. 성과 분석 (100%)
        status_text.text("📊 성과 분석 중...")
        analyzer = PerformanceAnalyzer()
        performance_results = analyzer.analyze_backtest_results(backtest_results, settings)
        progress_bar.progress(100)
        
        # 결과 저장
        st.session_state['backtest_results'] = {
            'portfolio': portfolio,
            'backtest_results': backtest_results,
            'performance': performance_results,
            'settings': settings
        }
        
        st.success("🎉 백테스팅이 성공적으로 완료되었습니다!")
        st.info("📊 결과 탭에서 상세한 분석 결과를 확인하세요.")
        
    except Exception as e:
        st.error(f"❌ 백테스팅 중 오류가 발생했습니다: {str(e)}")
        progress_bar.progress(0)
        status_text.text("")

def show_results_tab():
    """결과 탭을 표시합니다."""
    st.subheader("📊 백테스팅 결과")
    
    if 'backtest_results' not in st.session_state:
        st.info("📋 아직 백테스팅이 실행되지 않았습니다.")
        st.info("🔄 실행 탭에서 백테스팅을 시작하세요.")
        return
    
    results = st.session_state['backtest_results']
    performance = results['performance']
    
    # 결과 요약
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 수익률", f"{performance.get('total_return', 0):.1%}")
    
    with col2:
        st.metric("연간 수익률", f"{performance.get('annual_return', 0):.1%}")
    
    with col3:
        st.metric("샤프 비율", f"{performance.get('sharpe_ratio', 0):.2f}")
    
    with col4:
        st.metric("최대 낙폭", f"{performance.get('max_drawdown', 0):.1%}")
    
    # 성과 비교
    st.markdown("### 📈 성과 비교")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 누적 수익률 차트
        if 'cumulative_returns' in performance:
            fig_returns = px.line(
                x=performance['cumulative_returns'].index,
                y=performance['cumulative_returns'].values,
                title="누적 수익률",
                labels={'x': '날짜', 'y': '누적 수익률'}
            )
            fig_returns.update_layout(height=400)
            st.plotly_chart(fig_returns, use_container_width=True, key="backtest_returns_chart")
    
    with col2:
        # 벤치마크 대비 성과
        if 'benchmark_comparison' in performance:
            benchmark_data = performance['benchmark_comparison']
            fig_benchmark = px.line(
                benchmark_data,
                title="벤치마크 대비 성과",
                labels={'x': '날짜', 'y': '누적 수익률'}
            )
            fig_benchmark.update_layout(height=400)
            st.plotly_chart(fig_benchmark, use_container_width=True, key="backtest_benchmark_chart")
    
    # 리스크 분석
    st.markdown("### 🛡️ 리스크 분석")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("변동성", f"{performance.get('volatility', 0):.1%}")
        st.metric("VaR (95%)", f"{performance.get('var_95', 0):.1%}")
        st.metric("CVaR (95%)", f"{performance.get('cvar_95', 0):.1%}")
    
    with col2:
        st.metric("베타", f"{performance.get('beta', 0):.2f}")
        st.metric("알파", f"{performance.get('alpha', 0):.1%}")
        st.metric("정보 비율", f"{performance.get('information_ratio', 0):.2f}")
    
    with col3:
        st.metric("승률", f"{performance.get('win_rate', 0):.1%}")
        st.metric("평균 수익", f"{performance.get('avg_return', 0):.1%}")
        st.metric("평균 손실", f"{performance.get('avg_loss', 0):.1%}")
    
    # 월별 수익률 분석
    st.markdown("### 📅 월별 수익률 분석")
    
    if 'monthly_returns' in performance:
        monthly_returns = performance['monthly_returns']
        
        # 히트맵
        fig_heatmap = px.imshow(
            monthly_returns,
            title="월별 수익률 히트맵",
            color_continuous_scale='RdYlGn',
            aspect='auto'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True, key="backtest_heatmap_chart")
    
    # 낙폭 분석
    st.markdown("### 📉 낙폭 분석")
    
    if 'drawdown' in performance:
        drawdown_data = performance['drawdown']
        
        fig_drawdown = px.line(
            x=drawdown_data.index,
            y=drawdown_data.values,
            title="낙폭 추이",
            labels={'x': '날짜', 'y': '낙폭 (%)'}
        )
        fig_drawdown.update_layout(height=400)
        fig_drawdown.update_traces(line_color='red')
        st.plotly_chart(fig_drawdown, use_container_width=True, key="backtest_drawdown_chart")
    
    # 거래 통계
    st.markdown("### 💰 거래 통계")
    
    if 'trade_statistics' in performance:
        trade_stats = performance['trade_statistics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 거래 수", f"{trade_stats.get('total_trades', 0):,}")
        
        with col2:
            st.metric("승률", f"{trade_stats.get('win_rate', 0):.1%}")
        
        with col3:
            st.metric("평균 보유 기간", f"{trade_stats.get('avg_holding_period', 0):.1f}일")
        
        with col4:
            st.metric("거래 비용", f"${trade_stats.get('total_cost', 0):,.0f}")
    
    # 결과 다운로드
    st.markdown("### 💾 결과 다운로드")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 백테스팅 결과 CSV 다운로드
        if 'backtest_results' in results:
            backtest_df = pd.DataFrame(results['backtest_results'])
            csv = backtest_df.to_csv(index=False)
            st.download_button(
                label="📊 백테스팅 결과 다운로드 (CSV)",
                data=csv,
                file_name="backtest_results.csv",
                mime="text/csv"
            )
    
    with col2:
        # 성과 지표 다운로드
        performance_df = pd.DataFrame([performance])
        csv_performance = performance_df.to_csv(index=False)
        st.download_button(
            label="📈 성과 지표 다운로드 (CSV)",
            data=csv_performance,
            file_name="performance_metrics.csv",
            mime="text/csv"
        )
    
    with col3:
        # 포트폴리오 구성 다운로드
        if 'portfolio' in results:
            portfolio_df = pd.DataFrame(results['portfolio'])
            csv_portfolio = portfolio_df.to_csv(index=False)
            st.download_button(
                label="📋 포트폴리오 구성 다운로드 (CSV)",
                data=csv_portfolio,
                file_name="portfolio_composition.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    show_page() 
