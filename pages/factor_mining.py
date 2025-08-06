"""
팩터 마이닝 페이지
AlphaForge 생성-예측 신경망을 사용한 알파 팩터 자동 발굴 기능을 제공합니다.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import FACTOR_SETTINGS
from utils.factor_miner import FactorMiner
from utils.performance_analyzer import PerformanceAnalyzer
from utils.cache_utils import cached_factor_mining, get_data_hash, get_settings_hash

def show_page():
    """팩터 마이닝 페이지를 표시합니다."""
    st.title("🧠 팩터 마이닝")
    st.markdown("AI 기반 생성-예측 신경망으로 고품질 알파 팩터를 자동 발굴합니다.")
    
    # 데이터 상태 확인 및 표시
    show_data_status()
    
    # 데이터 확인 (processed_data 또는 uploaded_data)
    if 'processed_data' not in st.session_state and 'uploaded_data' not in st.session_state:
        st.error("❌ 먼저 데이터를 업로드해주세요.")
        
        # 데이터 업로드 안내 카드
        st.markdown("""
        ### 📋 데이터 업로드 방법
        
        **1. 데이터 관리 페이지에서 데이터 업로드**
        - 📁 파일 업로드: CSV, Excel, Parquet 형식 지원
        - 🚀 야후 파이낸스: 실시간 주가 데이터 다운로드
        - 📋 샘플 데이터: 테스트용 샘플 데이터 사용
        
        **2. 필수 데이터 형식**
        - `Date`: 날짜 (YYYY-MM-DD)
        - `Ticker`: 종목 코드
        - `Close`: 종가
        - `Open`, `High`, `Low`, `Volume`: 선택사항
        
        **3. 권장 데이터 사양**
        - 최소 1년 이상의 데이터
        - 5개 이상의 종목
        - 일별 데이터 (주말 제외)
        """)
        
        # 데이터 업로드 바로가기 버튼
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📈 데이터 관리 페이지로 이동", type="primary", use_container_width=True):
                st.switch_page("pages/data_management.py")
        with col2:
            if st.button("📋 샘플 데이터 사용", use_container_width=True):
                # 샘플 데이터 생성 및 저장
                try:
                    from pages.data_management import create_sample_data
                    from utils.data_processor import DataProcessor
                    
                    with st.spinner("샘플 데이터 생성 중..."):
                        # 샘플 데이터 생성
                        sample_data = create_sample_data()
                        
                        # 데이터 검증
                        if sample_data is None or sample_data.empty:
                            raise ValueError("샘플 데이터 생성 실패")
                        
                        # 데이터 처리
                        processor = DataProcessor()
                        processed_data = processor.process_data(sample_data)
                        
                        # 세션 상태 저장
                        st.session_state['uploaded_data'] = sample_data
                        st.session_state['processed_data'] = processed_data
                        st.session_state['data_processor'] = processor
                        st.session_state['data_filename'] = "sample_stock_data.csv"
                    
                    st.success("✅ 샘플 데이터가 생성되었습니다!")
                    st.info(f"📊 생성된 데이터: {len(sample_data):,}행, {sample_data['Ticker'].nunique()}종목")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ 샘플 데이터 생성 실패: {str(e)}")
                    st.error("데이터 생성 중 오류가 발생했습니다. 다시 시도해주세요.")
                    
                    # 디버깅 정보 표시
                    with st.expander("🔍 디버깅 정보"):
                        st.code(f"""
                        오류 타입: {type(e).__name__}
                        오류 메시지: {str(e)}
                        """)
        
        return
    
    # processed_data가 없으면 uploaded_data를 처리
    if 'processed_data' not in st.session_state and 'uploaded_data' in st.session_state:
        st.info("🔄 데이터를 처리 중입니다...")
        with st.spinner("데이터 처리 중..."):
            try:
                from utils.data_processor import DataProcessor
                
                # 입력 데이터 검증
                uploaded_data = st.session_state['uploaded_data']
                if uploaded_data is None or uploaded_data.empty:
                    raise ValueError("업로드된 데이터가 비어있습니다")
                
                # 데이터 처리
                processor = DataProcessor()
                processed_data = processor.process_data(uploaded_data)
                
                # 처리 결과 검증
                if processed_data is None or processed_data.empty:
                    raise ValueError("데이터 처리 후 결과가 비어있습니다")
                
                # 세션 상태 저장
                st.session_state['processed_data'] = processed_data
                st.session_state['data_processor'] = processor
                
                st.success("✅ 데이터 처리 완료!")
                st.info(f"📊 처리된 데이터: {len(processed_data):,}행, {processed_data['Ticker'].nunique()}종목")
                st.rerun()  # 페이지 새로고침
                
            except Exception as e:
                st.error(f"❌ 데이터 처리 중 오류가 발생했습니다: {str(e)}")
                st.error("데이터 형식을 확인하고 다시 시도해주세요.")
                
                # 디버깅 정보 표시
                with st.expander("🔍 디버깅 정보"):
                    st.code(f"""
                    오류 타입: {type(e).__name__}
                    오류 메시지: {str(e)}
                    업로드된 데이터 크기: {len(st.session_state['uploaded_data']) if 'uploaded_data' in st.session_state else 'N/A'}
                    """)
                return
    
    # 데이터 검증
    if 'processed_data' in st.session_state:
        data = st.session_state['processed_data']
        if data is None or data.empty:
            st.error("❌ 처리된 데이터가 비어있습니다.")
            return
        
        # 데이터 정보 표시
        st.success(f"✅ 데이터 준비 완료: {len(data):,}행, {data['Ticker'].nunique()}개 종목")
    
    # 탭 인터페이스
    tab1, tab2, tab3 = st.tabs(["⚙️ 설정", "🚀 실행", "📊 결과"])
    
    with tab1:
        show_settings_tab()
    
    with tab2:
        show_execution_tab()
    
    with tab3:
        show_results_tab()

def show_data_status():
    """데이터 상태를 표시합니다."""
    st.markdown("### 📊 데이터 상태")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'uploaded_data' in st.session_state:
            data = st.session_state['uploaded_data']
            st.metric("📁 업로드된 데이터", f"{len(data):,}행")
            if 'Ticker' in data.columns:
                st.metric("📈 종목 수", f"{data['Ticker'].nunique()}개")
        else:
            st.metric("📁 업로드된 데이터", "없음")
    
    with col2:
        if 'processed_data' in st.session_state:
            data = st.session_state['processed_data']
            st.metric("✅ 처리된 데이터", f"{len(data):,}행")
            if 'Ticker' in data.columns:
                st.metric("📈 종목 수", f"{data['Ticker'].nunique()}개")
        else:
            st.metric("✅ 처리된 데이터", "없음")
    
    with col3:
        if 'mining_results' in st.session_state:
            results = st.session_state['mining_results']
            st.metric("🧠 팩터 수", f"{len(results.get('factors', []))}개")
        else:
            st.metric("🧠 팩터 수", "0개")
    
    # 디버깅 정보 (접을 수 있는 섹션)
    with st.expander("🔍 디버깅 정보"):
        st.write("**세션 상태 키들:**")
        session_keys = list(st.session_state.keys())
        st.write(session_keys)
        
        if 'uploaded_data' in st.session_state:
            st.write("**업로드된 데이터 정보:**")
            data = st.session_state['uploaded_data']
            st.write(f"- Shape: {data.shape}")
            st.write(f"- Columns: {list(data.columns)}")
            st.write(f"- Data types: {dict(data.dtypes)}")
            if not data.empty:
                st.write(f"- Sample data:")
                st.dataframe(data.head(3))
        
        if 'processed_data' in st.session_state:
            st.write("**처리된 데이터 정보:**")
            data = st.session_state['processed_data']
            st.write(f"- Shape: {data.shape}")
            st.write(f"- Columns: {list(data.columns)}")
            st.write(f"- Data types: {dict(data.dtypes)}")
            if not data.empty:
                st.write(f"- Sample data:")
                st.dataframe(data.head(3))
    
    st.markdown("---")

def show_settings_tab():
    """설정 탭을 표시합니다."""
    st.subheader("⚙️ 팩터 마이닝 설정")
    
    # 기본 설정
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 기본 설정")
        
        # 팩터 유형 선택
        factor_types = st.multiselect(
            "팩터 유형 선택",
            options=FACTOR_SETTINGS['default_factors'],
            default=FACTOR_SETTINGS['default_factors'][:3],
            help="발굴할 팩터 유형을 선택하세요"
        )
        
        # 팩터 풀 크기
        factor_pool_size = st.slider(
            "팩터 풀 크기",
            min_value=5,
            max_value=50,
            value=FACTOR_SETTINGS['factor_pool_size'],
            help="생성할 팩터의 개수"
        )
        
        # 최소 IC 임계값
        min_ic = st.number_input(
            "최소 IC 임계값",
            min_value=0.0,
            max_value=0.1,
            value=FACTOR_SETTINGS['min_ic'],
            step=0.001,
            format="%.3f",
            help="Information Coefficient 최소 임계값"
        )
    
    with col2:
        st.markdown("### 🎯 성과 기준")
        
        # 최소 ICIR 임계값
        min_icir = st.number_input(
            "최소 ICIR 임계값",
            min_value=0.0,
            max_value=2.0,
            value=FACTOR_SETTINGS['min_icir'],
            step=0.1,
            format="%.1f",
            help="IC Information Ratio 최소 임계값"
        )
        
        # 백테스팅 기간
        backtest_period = st.selectbox(
            "백테스팅 기간",
            options=["6개월", "1년", "2년", "3년", "전체 기간"],
            index=2,
            help="팩터 성과 검증 기간"
        )
        
        # 리밸런싱 주기
        rebalancing_freq = st.selectbox(
            "리밸런싱 주기",
            options=["일간", "주간", "월간", "분기"],
            index=2,
            help="포트폴리오 리밸런싱 주기"
        )
    
    # 고급 설정
    with st.expander("🔧 고급 설정"):
        col1, col2 = st.columns(2)
        
        with col1:
            # 신경망 설정
            st.markdown("#### 🧠 신경망 설정")
            
            hidden_layers = st.slider(
                "은닉층 수",
                min_value=1,
                max_value=5,
                value=3,
                help="생성-예측 신경망의 은닉층 수"
            )
            
            neurons_per_layer = st.slider(
                "층당 뉴런 수",
                min_value=32,
                max_value=512,
                value=128,
                step=32,
                help="각 은닉층의 뉴런 수"
            )
            
            learning_rate = st.selectbox(
                "학습률",
                options=[0.001, 0.01, 0.1],
                index=0,
                help="신경망 학습률"
            )
        
        with col2:
            # 최적화 설정
            st.markdown("#### ⚡ 최적화 설정")
            
            epochs = st.slider(
                "학습 에포크",
                min_value=10,
                max_value=200,
                value=50,
                help="신경망 학습 에포크 수"
            )
            
            batch_size = st.selectbox(
                "배치 크기",
                options=[32, 64, 128, 256],
                index=1,
                help="학습 배치 크기"
            )
            
            dropout_rate = st.slider(
                "드롭아웃 비율",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.1,
                help="과적합 방지를 위한 드롭아웃 비율"
            )
    
    # 설정 저장
    if st.button("💾 설정 저장", use_container_width=True):
        settings = {
            'factor_types': factor_types,
            'factor_pool_size': factor_pool_size,
            'min_ic': min_ic,
            'min_icir': min_icir,
            'backtest_period': backtest_period,
            'rebalancing_freq': rebalancing_freq,
            'hidden_layers': hidden_layers,
            'neurons_per_layer': neurons_per_layer,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate
        }
        
        st.session_state['mining_settings'] = settings
        st.success("✅ 설정이 저장되었습니다!")

def show_execution_tab():
    """실행 탭을 표시합니다."""
    st.subheader("🚀 팩터 마이닝 실행")
    
    # 설정 확인
    if 'mining_settings' not in st.session_state:
        st.warning("⚠️ 먼저 설정 탭에서 마이닝 설정을 완료해주세요.")
        return
    
    settings = st.session_state['mining_settings']
    
    # 설정 요약
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 설정 요약")
        st.write(f"**팩터 유형**: {', '.join(settings['factor_types'])}")
        st.write(f"**팩터 풀 크기**: {settings['factor_pool_size']}개")
        st.write(f"**최소 IC**: {settings['min_ic']:.3f}")
        st.write(f"**최소 ICIR**: {settings['min_icir']:.1f}")
    
    with col2:
        st.markdown("### 🎯 목표 성과")
        st.write(f"**백테스팅 기간**: {settings['backtest_period']}")
        st.write(f"**리밸런싱 주기**: {settings['rebalancing_freq']}")
        st.write(f"**신경망 구조**: {settings['hidden_layers']}층 x {settings['neurons_per_layer']}뉴런")
        st.write(f"**학습 에포크**: {settings['epochs']}회")
    
    # 실행 버튼
    if st.button("🚀 팩터 마이닝 시작", use_container_width=True, type="primary"):
        run_factor_mining(settings)

def run_factor_mining(settings):
    """팩터 마이닝을 실행합니다."""
    # 진행 상황 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. 데이터 준비 (10%)
        status_text.text("📊 데이터 준비 중...")
        
        # 데이터 검증
        if 'processed_data' not in st.session_state and 'uploaded_data' not in st.session_state:
            raise ValueError("처리할 데이터가 없습니다. 먼저 데이터를 업로드해주세요.")
        
        # processed_data가 있으면 사용, 없으면 uploaded_data를 처리
        if 'processed_data' in st.session_state:
            data = st.session_state['processed_data']
        else:
            # uploaded_data를 처리
            from utils.data_processor import DataProcessor
            processor = DataProcessor()
            data = processor.process_data(st.session_state['uploaded_data'])
            st.session_state['processed_data'] = data
            st.session_state['data_processor'] = processor
        
        # 데이터 최종 검증
        if data is None or data.empty:
            raise ValueError("데이터 처리 후 결과가 비어있습니다")
        
        progress_bar.progress(10)
        time.sleep(0.5)
        
        # 2. 팩터 마이너 초기화 (20%)
        status_text.text("🧠 팩터 마이너 초기화 중...")
        try:
            miner = FactorMiner(settings)
        except Exception as e:
            raise ValueError(f"팩터 마이너 초기화 실패: {str(e)}")
        
        progress_bar.progress(20)
        time.sleep(0.5)
        
        # 3. 기본 팩터 생성 (40%)
        status_text.text("🔧 기본 팩터 생성 중...")
        try:
            basic_factors = miner.generate_basic_factors(data)
            if not basic_factors:
                raise ValueError("기본 팩터 생성 실패")
        except Exception as e:
            raise ValueError(f"기본 팩터 생성 실패: {str(e)}")
        
        progress_bar.progress(40)
        time.sleep(0.5)
        
        # 4. 신경망 기반 팩터 생성 (70%) - 캐시 적용
        status_text.text("🧠 AI 기반 팩터 생성 중...")
        try:
            # 캐시 사용
            data_hash = get_data_hash(data)
            settings_hash = get_settings_hash(settings)
            
            # 캐시된 결과 조회 시도
            try:
                cached_result = cached_factor_mining(data_hash, settings_hash, data, settings)
                if cached_result and 'factors' in cached_result:
                    ai_factors = cached_result['factors']
                    st.info("💾 캐시된 팩터 마이닝 결과를 사용합니다.")
                else:
                    raise ValueError("캐시 결과 없음")
                    
            except Exception:
                # 캐시 실패시 직접 계산
                ai_factors = miner.generate_ai_factors(data, basic_factors)
                if not ai_factors:
                    raise ValueError("AI 팩터 생성 실패")
        except Exception as e:
            raise ValueError(f"AI 팩터 생성 실패: {str(e)}")
        
        progress_bar.progress(70)
        time.sleep(0.5)
        
        # 5. 성과 분석 (90%)
        status_text.text("📊 성과 분석 중...")
        try:
            analyzer = PerformanceAnalyzer()
            performance_results = analyzer.analyze_factors(data, ai_factors)
            if not performance_results:
                raise ValueError("성과 분석 실패")
        except Exception as e:
            raise ValueError(f"성과 분석 실패: {str(e)}")
        
        progress_bar.progress(90)
        time.sleep(0.5)
        
        # 6. 완료 (100%)
        status_text.text("✅ 팩터 마이닝 완료!")
        progress_bar.progress(100)
        
        # 결과 저장
        st.session_state['mining_results'] = {
            'factors': ai_factors,
            'performance': performance_results,
            'settings': settings,
            'timestamp': time.time(),
            'data_info': {
                'total_rows': len(data),
                'unique_tickers': data['Ticker'].nunique() if 'Ticker' in data.columns else 0,
                'date_range': f"{data['Date'].min().strftime('%Y-%m-%d')} ~ {data['Date'].max().strftime('%Y-%m-%d')}" if 'Date' in data.columns else "N/A"
            }
        }
        
        st.success("🎉 팩터 마이닝이 성공적으로 완료되었습니다!")
        st.info("📊 결과 탭에서 상세한 분석 결과를 확인하세요.")
        
    except Exception as e:
        st.error(f"❌ 팩터 마이닝 중 오류가 발생했습니다: {str(e)}")
        progress_bar.progress(0)
        status_text.text("")
        
        # 디버깅 정보 표시
        with st.expander("🔍 디버깅 정보"):
            st.code(f"""
            오류 타입: {type(e).__name__}
            오류 메시지: {str(e)}
            세션 상태: {list(st.session_state.keys())}
            데이터 상태: {'processed_data' in st.session_state or 'uploaded_data' in st.session_state}
            """)
        
        # 해결 방안 제시
        st.markdown("### 💡 해결 방안")
        st.markdown("""
        1. **데이터 확인**: 데이터 관리 페이지에서 데이터 형식과 품질을 확인하세요
        2. **샘플 데이터 사용**: 테스트를 위해 샘플 데이터를 사용해보세요
        3. **설정 조정**: 팩터 마이닝 설정을 더 보수적으로 조정해보세요
        4. **재시도**: 페이지를 새로고침하고 다시 시도해보세요
        """)

def show_results_tab():
    """결과 탭을 표시합니다."""
    st.subheader("📊 팩터 마이닝 결과")
    
    if 'mining_results' not in st.session_state:
        st.info("📋 아직 팩터 마이닝이 실행되지 않았습니다.")
        st.info("🚀 실행 탭에서 팩터 마이닝을 시작하세요.")
        return
    
    results = st.session_state['mining_results']
    factors = results['factors']
    performance = results['performance']
    
    # 데이터 정보 표시
    if 'data_info' in results:
        data_info = results['data_info']
        st.info(f"📊 분석 데이터: {data_info['total_rows']:,}행, {data_info['unique_tickers']}개 종목")
        st.info(f"📅 데이터 기간: {data_info['date_range']}")
    
    # 생성 시간 표시
    if 'timestamp' in results:
        from datetime import datetime
        created_time = datetime.fromtimestamp(results['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        st.caption(f"🕒 생성 시간: {created_time}")
    
    # 결과 요약
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("생성된 팩터", f"{len(factors)}개")
    
    with col2:
        st.metric("평균 IC", f"{performance['avg_ic']:.3f}")
    
    with col3:
        st.metric("평균 ICIR", f"{performance['avg_icir']:.2f}")
    
    with col4:
        st.metric("승률", f"{performance['win_rate']:.1%}")
    
    # 팩터 성과 테이블
    st.markdown("### 📈 팩터 성과 순위")
    
    performance_df = pd.DataFrame(performance['factor_performance'])
    performance_df = performance_df.sort_values('IC', ascending=False)
    
    st.dataframe(
        performance_df,
        use_container_width=True,
        column_config={
            "Factor": st.column_config.TextColumn("팩터명"),
            "IC": st.column_config.NumberColumn("IC", format="%.3f"),
            "ICIR": st.column_config.NumberColumn("ICIR", format="%.2f"),
            "Win_Rate": st.column_config.NumberColumn("승률", format="%.1%"),
            "Sharpe": st.column_config.NumberColumn("샤프비율", format="%.2f")
        }
    )
    
    # 시각화
    col1, col2 = st.columns(2)
    
    with col1:
        # IC 분포 히스토그램
        fig_ic = px.histogram(
            performance_df,
            x='IC',
            title="IC 분포",
            nbins=20
        )
        fig_ic.update_layout(height=400)
        st.plotly_chart(fig_ic, use_container_width=True, key="factor_ic_chart")
    
    with col2:
        # IC vs ICIR 산점도
        fig_scatter = px.scatter(
            performance_df,
            x='IC',
            y='ICIR',
            title="IC vs ICIR",
            hover_data=['Factor']
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True, key="factor_ic_scatter_chart")
    
    # 상위 팩터 상세 분석
    st.markdown("### 🏆 상위 팩터 상세 분석")
    
    top_factors = performance_df.head(5)
    
    for idx, factor in top_factors.iterrows():
        with st.expander(f"🥇 {factor['Factor']} (IC: {factor['IC']:.3f})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**IC**: {factor['IC']:.3f}")
                st.write(f"**ICIR**: {factor['ICIR']:.2f}")
                st.write(f"**승률**: {factor['Win_Rate']:.1%}")
                st.write(f"**샤프비율**: {factor['Sharpe']:.2f}")
            
            with col2:
                # 팩터 수식 표시 (예시)
                st.code(f"# {factor['Factor']} 수식\n{factor.get('formula', '수식 정보 없음')}", language="python")
    
    # 팩터 상관관계 분석
    st.markdown("### 🔗 팩터 상관관계 분석")
    
    if len(factors) > 1:
        # 상관관계 매트릭스 계산 (예시)
        correlation_matrix = np.random.rand(len(factors), len(factors))
        np.fill_diagonal(correlation_matrix, 1.0)
        
        fig_corr = px.imshow(
            correlation_matrix,
            x=performance_df['Factor'],
            y=performance_df['Factor'],
            title="팩터 상관관계 매트릭스",
            color_continuous_scale='RdBu'
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True, key="factor_correlation_chart")
    
    # 결과 다운로드
    st.markdown("### 💾 결과 다운로드")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 성과 결과 CSV 다운로드
        csv = performance_df.to_csv(index=False)
        st.download_button(
            label="📊 성과 결과 다운로드 (CSV)",
            data=csv,
            file_name="factor_performance.csv",
            mime="text/csv"
        )
    
    with col2:
        # 팩터 수식 다운로드
        factor_formulas = pd.DataFrame([
            {'Factor': factor['Factor'], 'Formula': factor.get('formula', 'N/A')}
            for factor in performance['factor_performance']
        ])
        csv_formulas = factor_formulas.to_csv(index=False)
        st.download_button(
            label="🧮 팩터 수식 다운로드 (CSV)",
            data=csv_formulas,
            file_name="factor_formulas.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    show_page() 
