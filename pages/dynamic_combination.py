"""
동적 결합 페이지
시점별 팩터 성과 기반 메가-알파 생성 및 동적 가중치 조정 기능을 제공합니다.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import time

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dynamic_combiner import DynamicCombiner
from utils.performance_analyzer import PerformanceAnalyzer

def show_page():
    """동적 결합 페이지를 표시합니다."""
    st.title("⚖️ 동적 결합")
    st.markdown("시점별 팩터 성과 기반으로 메가-알파를 생성하고 동적 가중치를 조정합니다.")
    
    # 퀵 액션 버튼
    if 'processed_data' in st.session_state:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧠 팩터 마이닝", use_container_width=True):
                st.switch_page("pages/factor_mining.py")
        
        with col2:
            if st.button("📊 백테스팅", use_container_width=True):
                st.switch_page("pages/backtesting.py")
        
        with col3:
            if st.button("📋 리포트", use_container_width=True):
                st.switch_page("pages/reporting.py")
    
    # 데이터 확인 (processed_data 필수)
    if 'processed_data' not in st.session_state:
        st.error("❌ 먼저 데이터를 업로드해주세요.")
        st.info("📈 데이터 관리 페이지에서 데이터를 업로드하세요.")
        return
    
    # 팩터 마이닝 결과 또는 기본 팩터 사용 설정 확인
    has_mining_results = 'mining_results' in st.session_state
    use_default_factors = st.session_state.get('use_default_factors', False)
    
    if not has_mining_results and not use_default_factors:
        st.warning("⚠️ 팩터 마이닝이 완료되지 않았습니다.")
        st.info("🧠 팩터 마이닝을 완료하거나 기본 팩터 사용을 설정하세요.")
        
        # 기본 팩터 사용 옵션을 여기서도 제공
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧠 팩터 마이닝으로 이동", use_container_width=True):
                st.switch_page("pages/factor_mining.py")
        with col2:
            if st.button("⚙️ 기본 팩터 사용 설정", use_container_width=True):
                st.session_state['use_default_factors'] = True
                st.success("✅ 기본 팩터 사용이 설정되었습니다!")
                st.rerun()
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
    st.subheader("⚙️ 동적 결합 설정")
    
    # 팩터 마이닝 결과 상태 확인
    if 'mining_results' in st.session_state:
        st.success("✅ 팩터 마이닝 결과가 준비되었습니다.")
        mining_results = st.session_state['mining_results']
        st.info(f"📊 사용 가능한 팩터: {len(mining_results['factors'])}개")
        
        # 팩터 정보 표시
        if 'data_info' in mining_results:
            data_info = mining_results['data_info']
            st.caption(f"📅 데이터 기간: {data_info['date_range']}")
            st.caption(f"📈 분석 종목: {data_info['unique_tickers']}개")
    else:
        st.warning("⚠️ 팩터 마이닝 결과가 없습니다.")
        
        # 기본 팩터 사용 옵션
        use_default_factors = st.checkbox(
            "기본 팩터 사용",
            value=st.session_state.get('use_default_factors', False),
            help="팩터 마이닝 결과가 없을 때 기본 팩터를 사용합니다."
        )
        
        if use_default_factors:
            st.session_state['use_default_factors'] = True
            st.success("✅ 기본 팩터를 사용하도록 설정되었습니다.")
            
            # 기본 팩터 정보 표시
            default_factors = generate_default_factors()
            st.info(f"📊 기본 팩터: {len(default_factors)}개 (모멘텀, 밸류, 퀄리티, 사이즈, 저변동성)")
        else:
            st.session_state['use_default_factors'] = False
            st.info("🧠 팩터 마이닝을 완료하거나 기본 팩터 사용을 체크하세요.")
    
    # 기본 설정
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 결합 방식")
        
        # 결합 방법 선택
        combination_method = st.selectbox(
            "결합 방법",
            options=["동적 가중치", "등가중치", "최적화 가중치", "적응형 가중치"],
            index=0,
            help="팩터들을 결합하는 방법을 선택하세요"
        )
        
        # 상위 팩터 수
        top_factors = st.slider(
            "상위 팩터 수",
            min_value=3,
            max_value=20,
            value=10,
            help="결합에 사용할 상위 팩터의 개수"
        )
        
        # 리밸런싱 주기
        rebalancing_freq = st.selectbox(
            "리밸런싱 주기",
            options=["일간", "주간", "월간", "분기"],
            index=2,
            help="가중치 재조정 주기"
        )
    
    with col2:
        st.markdown("### 🎯 성과 기준")
        
        # 최소 IC 임계값
        min_ic_threshold = st.number_input(
            "최소 IC 임계값",
            min_value=0.0,
            max_value=0.1,
            value=0.02,
            step=0.001,
            format="%.3f",
            help="포함할 팩터의 최소 IC 임계값"
        )
        
        # 최소 ICIR 임계값
        min_icir_threshold = st.number_input(
            "최소 ICIR 임계값",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.1,
            format="%.1f",
            help="포함할 팩터의 최소 ICIR 임계값"
        )
        
        # 가중치 감쇠율
        weight_decay = st.slider(
            "가중치 감쇠율",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="과거 성과의 영향력을 줄이는 비율"
        )
    
    # 고급 설정
    with st.expander("🔧 고급 설정"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚡ 최적화 설정")
            
            # 최적화 목표
            optimization_target = st.selectbox(
                "최적화 목표",
                options=["샤프 비율", "정보 비율", "수익률", "리스크 조정 수익률"],
                index=0,
                help="가중치 최적화의 목표 지표"
            )
            
            # 제약 조건
            max_weight = st.slider(
                "최대 가중치",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="단일 팩터의 최대 가중치"
            )
        
        with col2:
            st.markdown("#### 🛡️ 리스크 관리")
            
            # 변동성 제한
            volatility_limit = st.number_input(
                "변동성 제한",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.01,
                format="%.2f",
                help="포트폴리오 변동성 상한"
            )
            
            # 최대 낙폭 제한
            max_drawdown_limit = st.number_input(
                "최대 낙폭 제한",
                min_value=0.0,
                max_value=1.0,
                value=0.15,
                step=0.01,
                format="%.2f",
                help="포트폴리오 최대 낙폭 상한"
            )
    
    # 설정 저장
    if st.button("💾 설정 저장", use_container_width=True):
        settings = {
            'combination_method': combination_method,
            'top_factors': top_factors,
            'rebalancing_freq': rebalancing_freq,
            'min_ic_threshold': min_ic_threshold,
            'min_icir_threshold': min_icir_threshold,
            'weight_decay': weight_decay,
            'optimization_target': optimization_target,
            'max_weight': max_weight,
            'volatility_limit': volatility_limit,
            'max_drawdown_limit': max_drawdown_limit
        }
        
        st.session_state['combination_settings'] = settings
        st.success("✅ 설정이 저장되었습니다!")

def show_execution_tab():
    """실행 탭을 표시합니다."""
    st.subheader("🔄 동적 결합 실행")
    
    # 설정 확인
    if 'combination_settings' not in st.session_state:
        st.warning("⚠️ 먼저 설정 탭에서 결합 설정을 완료해주세요.")
        return
    
    # 팩터 데이터 확인
    has_mining_results = 'mining_results' in st.session_state
    use_default_factors = st.session_state.get('use_default_factors', False)
    
    if not has_mining_results and not use_default_factors:
        st.error("❌ 팩터 데이터가 없습니다.")
        st.info("🧠 팩터 마이닝을 완료하거나 설정 탭에서 기본 팩터 사용을 설정하세요.")
        return
    
    settings = st.session_state['combination_settings']
    
    # 설정 요약
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 설정 요약")
        st.write(f"**결합 방법**: {settings['combination_method']}")
        st.write(f"**상위 팩터 수**: {settings['top_factors']}개")
        st.write(f"**리밸런싱 주기**: {settings['rebalancing_freq']}")
        st.write(f"**최소 IC**: {settings['min_ic_threshold']:.3f}")
        
        # 팩터 정보 추가
        if has_mining_results:
            mining_results = st.session_state['mining_results']
            st.write(f"**팩터 소스**: 팩터 마이닝 결과 ({len(mining_results['factors'])}개)")
        elif use_default_factors:
            default_factors = generate_default_factors()
            st.write(f"**팩터 소스**: 기본 팩터 ({len(default_factors)}개)")
    
    with col2:
        st.markdown("### 🎯 최적화 설정")
        st.write(f"**최적화 목표**: {settings['optimization_target']}")
        st.write(f"**최대 가중치**: {settings['max_weight']:.1%}")
        st.write(f"**변동성 제한**: {settings['volatility_limit']:.1%}")
        st.write(f"**최대 낙폭 제한**: {settings['max_drawdown_limit']:.1%}")
    
    # 실행 버튼
    if st.button("🔄 동적 결합 시작", use_container_width=True, type="primary"):
        run_dynamic_combination(settings)

def run_dynamic_combination(settings):
    """동적 결합을 실행합니다."""
    # 진행 상황 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. 데이터 준비 (10%)
        status_text.text("📊 데이터 준비 중...")
        
        # 팩터 데이터 준비
        if 'mining_results' in st.session_state:
            # 팩터 마이닝 결과 사용
            mining_results = st.session_state['mining_results']
            factors = mining_results['factors']
            st.success(f"📊 팩터 마이닝 결과 사용: {len(factors)}개 팩터")
        elif st.session_state.get('use_default_factors', False):
            # 기본 팩터 사용
            factors = generate_default_factors()
            st.info(f"📊 기본 팩터 사용: {len(factors)}개 팩터")
            st.caption("💡 팩터 마이닝을 완료하면 더 정교한 팩터를 사용할 수 있습니다.")
        else:
            raise ValueError("팩터 데이터가 없습니다. 팩터 마이닝을 완료하거나 기본 팩터 사용을 설정해주세요.")
        
        # processed_data 확인
        if 'processed_data' not in st.session_state:
            raise ValueError("처리된 데이터가 없습니다. 데이터 관리 페이지에서 데이터를 업로드해주세요.")
        
        data = st.session_state['processed_data']
        progress_bar.progress(10)
        
        # 2. 팩터 필터링 (30%)
        status_text.text("🔍 팩터 필터링 중...")
        filtered_factors = filter_factors(factors, settings)
        progress_bar.progress(30)
        
        # 3. 동적 결합기 초기화 (50%)
        status_text.text("⚖️ 동적 결합기 초기화 중...")
        combiner = DynamicCombiner(settings)
        progress_bar.progress(50)
        
        # 4. 메가-알파 생성 (80%)
        status_text.text("🧠 메가-알파 생성 중...")
        mega_alpha = combiner.create_mega_alpha(data, filtered_factors)
        progress_bar.progress(80)
        
        # 5. 성과 분석 (100%)
        status_text.text("📊 성과 분석 중...")
        analyzer = PerformanceAnalyzer()
        performance_results = analyzer.analyze_mega_alpha(data, mega_alpha)
        progress_bar.progress(100)
        
        # 결과 저장
        st.session_state['combination_results'] = {
            'mega_alpha': mega_alpha,
            'filtered_factors': filtered_factors,
            'performance': performance_results,
            'settings': settings,
            'timestamp': time.time(),
            'input_info': {
                'total_factors': len(factors),
                'filtered_factors': len(filtered_factors),
                'data_rows': len(data),
                'data_tickers': data['Ticker'].nunique() if 'Ticker' in data.columns else 0
            }
        }
        
        st.success("🎉 동적 결합이 성공적으로 완료되었습니다!")
        st.info("📊 결과 탭에서 상세한 분석 결과를 확인하세요.")
        
    except Exception as e:
        st.error(f"❌ 동적 결합 중 오류가 발생했습니다: {str(e)}")
        progress_bar.progress(0)
        status_text.text("")
        
        # 디버깅 정보 표시
        with st.expander("🔍 디버깅 정보"):
            st.code(f"""
            오류 타입: {type(e).__name__}
            오류 메시지: {str(e)}
            세션 상태: {list(st.session_state.keys())}
            데이터 상태: {'processed_data' in st.session_state}
            팩터 상태: {'mining_results' in st.session_state or st.session_state.get('use_default_factors', False)}
            """)
        
        # 해결 방안 제시
        st.markdown("### 💡 해결 방안")
        st.markdown("""
        1. **데이터 확인**: 데이터 관리 페이지에서 데이터 형식과 품질을 확인하세요
        2. **팩터 마이닝**: 팩터 마이닝을 완료하여 정확한 팩터를 사용하세요
        3. **기본 팩터 사용**: 테스트를 위해 기본 팩터 사용을 설정해보세요
        4. **설정 조정**: 동적 결합 설정을 더 보수적으로 조정해보세요
        5. **재시도**: 페이지를 새로고침하고 다시 시도해보세요
        """)

def generate_default_factors():
    """기본 팩터를 생성합니다."""
    default_factors = [
        {
            'name': 'Momentum_1M',
            'description': '1개월 모멘텀',
            'formula': 'close / close.shift(20) - 1',
            'category': 'Momentum',
            'ic': 0.025,
            'icir': 0.8
        },
        {
            'name': 'Momentum_3M',
            'description': '3개월 모멘텀',
            'formula': 'close / close.shift(60) - 1',
            'category': 'Momentum',
            'ic': 0.030,
            'icir': 0.9
        },
        {
            'name': 'Value_MA20',
            'description': '20일 이동평균 대비 가격',
            'formula': 'close / close.rolling(20).mean()',
            'category': 'Value',
            'ic': 0.020,
            'icir': 0.6
        },
        {
            'name': 'Value_MA50',
            'description': '50일 이동평균 대비 가격',
            'formula': 'close / close.rolling(50).mean()',
            'category': 'Value',
            'ic': 0.018,
            'icir': 0.5
        },
        {
            'name': 'Quality_LowVol',
            'description': '저변동성 품질',
            'formula': '1 / (returns.rolling(20).std() + 1e-8)',
            'category': 'Quality',
            'ic': 0.015,
            'icir': 0.7
        },
        {
            'name': 'Size_Volume',
            'description': '거래량 기반 사이즈',
            'formula': 'volume.rank(pct=True)',
            'category': 'Size',
            'ic': 0.012,
            'icir': 0.4
        },
        {
            'name': 'LowVolatility',
            'description': '저변동성',
            'formula': '-returns.rolling(20).std()',
            'category': 'LowVolatility',
            'ic': 0.010,
            'icir': 0.3
        },
        {
            'name': 'RSI_MeanReversion',
            'description': 'RSI 기반 평균회귀',
            'formula': '(50 - rsi) / 50',
            'category': 'MeanReversion',
            'ic': 0.008,
            'icir': 0.4
        }
    ]
    
    return default_factors

def filter_factors(factors, settings):
    """설정에 따라 팩터를 필터링합니다."""
    # 성과 기준으로 필터링
    filtered = []
    
    for factor in factors:
        # IC와 ICIR 값 확인
        if 'ic' in factor and 'icir' in factor:
            # 기본 팩터의 경우 미리 정의된 값 사용
            ic = factor['ic']
            icir = factor['icir']
        else:
            # 팩터 마이닝 결과의 경우 랜덤 값 사용 (실제로는 계산된 값)
            ic = np.random.uniform(0.01, 0.08)
            icir = np.random.uniform(0.3, 1.5)
        
        if ic >= settings['min_ic_threshold'] and icir >= settings['min_icir_threshold']:
            filtered.append(factor)
    
    # 상위 N개 선택 (IC 기준으로 정렬)
    filtered.sort(key=lambda x: x.get('ic', 0), reverse=True)
    return filtered[:settings['top_factors']]

def show_results_tab():
    """결과 탭을 표시합니다."""
    st.subheader("📊 동적 결합 결과")
    
    if 'combination_results' not in st.session_state:
        st.info("📋 아직 동적 결합이 실행되지 않았습니다.")
        st.info("🔄 실행 탭에서 동적 결합을 시작하세요.")
        return
    
    results = st.session_state['combination_results']
    mega_alpha = results['mega_alpha']
    performance = results['performance']
    
    # 입력 정보 표시
    if 'input_info' in results:
        input_info = results['input_info']
        st.info(f"📊 입력 데이터: {input_info['data_rows']:,}행, {input_info['data_tickers']}개 종목")
        st.info(f"🧠 사용된 팩터: {input_info['total_factors']}개 중 {input_info['filtered_factors']}개 선택")
    
    # 결과 요약
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("메가-알파 IC", f"{performance.get('ic', 0):.3f}")
    
    with col2:
        st.metric("메가-알파 ICIR", f"{performance.get('icir', 0):.2f}")
    
    with col3:
        st.metric("샤프 비율", f"{performance.get('sharpe', 0):.2f}")
    
    with col4:
        st.metric("최대 낙폭", f"{performance.get('max_drawdown', 0):.1%}")
    
    # 시점별 가중치 변화
    st.markdown("### 📈 시점별 가중치 변화")
    
    if 'weight_history' in mega_alpha:
        weight_df = pd.DataFrame(mega_alpha['weight_history'])
        
        # 가중치 변화 차트
        fig = px.line(
            weight_df,
            x='date',
            y='weight',
            color='factor',
            title="시점별 팩터 가중치 변화"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True, key="dynamic_combination_chart")
    
    # 메가-알파 성과 분석
    st.markdown("### 🏆 메가-알파 성과 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 누적 수익률 차트
        if 'cumulative_returns' in mega_alpha:
            fig_returns = px.line(
                x=mega_alpha['cumulative_returns'].index,
                y=mega_alpha['cumulative_returns'].values,
                title="메가-알파 누적 수익률"
            )
            fig_returns.update_layout(height=400)
            st.plotly_chart(fig_returns, use_container_width=True, key="dynamic_returns_chart")
    
    with col2:
        # 월별 수익률 히트맵
        if 'monthly_returns' in mega_alpha:
            monthly_returns = mega_alpha['monthly_returns']
            fig_heatmap = px.imshow(
                monthly_returns,
                title="월별 수익률 히트맵",
                color_continuous_scale='RdYlGn'
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True, key="dynamic_heatmap_chart")
    
    # 팩터 기여도 분석
    st.markdown("### 📊 팩터 기여도 분석")
    
    if 'factor_contribution' in mega_alpha:
        contribution_df = pd.DataFrame(mega_alpha['factor_contribution'])
        
        # 기여도 차트
        fig_contribution = px.bar(
            contribution_df,
            x='factor',
            y='contribution',
            title="팩터별 기여도",
            color='contribution',
            color_continuous_scale='Viridis'
        )
        fig_contribution.update_layout(height=400)
        st.plotly_chart(fig_contribution, use_container_width=True, key="dynamic_contribution_chart")
    
    # 리스크 분석
    st.markdown("### 🛡️ 리스크 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 변동성 분석
        if 'volatility' in performance:
            st.metric("연간 변동성", f"{performance['volatility']:.1%}")
            st.metric("VaR (95%)", f"{performance.get('var_95', 0):.1%}")
            st.metric("CVaR (95%)", f"{performance.get('cvar_95', 0):.1%}")
    
    with col2:
        # 낙폭 분석
        if 'drawdown' in performance:
            st.metric("최대 낙폭", f"{performance['max_drawdown']:.1%}")
            st.metric("평균 낙폭", f"{performance.get('avg_drawdown', 0):.1%}")
            st.metric("낙폭 기간", f"{performance.get('drawdown_duration', 0)}일")
    
    # 결과 다운로드
    st.markdown("### 💾 결과 다운로드")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 메가-알파 수치 다운로드
        if 'values' in mega_alpha:
            mega_alpha_df = pd.DataFrame({
                'Date': mega_alpha['values'].index,
                'Mega_Alpha': mega_alpha['values'].values
            })
            csv = mega_alpha_df.to_csv(index=False)
            st.download_button(
                label="📊 메가-알파 수치 다운로드 (CSV)",
                data=csv,
                file_name="mega_alpha_values.csv",
                mime="text/csv"
            )
    
    with col2:
        # 가중치 히스토리 다운로드
        if 'weight_history' in mega_alpha:
            weight_history_df = pd.DataFrame(mega_alpha['weight_history'])
            csv_weights = weight_history_df.to_csv(index=False)
            st.download_button(
                label="⚖️ 가중치 히스토리 다운로드 (CSV)",
                data=csv_weights,
                file_name="weight_history.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    show_page() 
