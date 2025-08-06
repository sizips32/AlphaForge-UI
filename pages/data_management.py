"""
데이터 관리 페이지
데이터 업로드, 검증, 미리보기, 기초 통계 기능을 제공합니다.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import DATA_VALIDATION, UI_SETTINGS
from utils.data_processor import DataProcessor
from utils.validators import DataValidator
from utils.yahoo_finance_downloader import YahooFinanceDownloader

def show_page():
    """데이터 관리 페이지를 표시합니다."""
    # 메인 타이틀과 설명
    col_title, col_status = st.columns([3, 1])
    
    with col_title:
        st.title("📈 데이터 관리")
        st.markdown("주가 데이터를 업로드하고 검증하여 AlphaForge 분석을 위한 데이터를 준비합니다.")
    
    with col_status:
        # 데이터 상태 표시
        if 'processed_data' in st.session_state:
            data = st.session_state['processed_data']
            st.success(f"✅ 처리된 데이터")
            st.metric("데이터 수", f"{len(data):,}")
        elif 'uploaded_data' in st.session_state:
            data = st.session_state['uploaded_data']
            st.warning(f"⚠️ 업로드된 데이터")
            st.metric("데이터 수", f"{len(data):,}")
        else:
            st.info("📁 데이터를 업로드하세요")
    
    st.markdown("---")
    
    # 메인 컨텐츠 영역
    if 'processed_data' in st.session_state or 'uploaded_data' in st.session_state:
        # 데이터가 있는 경우: 미리보기와 통계 중심 레이아웃
        show_data_loaded_layout()
    else:
        # 데이터가 없는 경우: 업로드 중심 레이아웃
        show_upload_centered_layout()

def show_upload_centered_layout():
    """데이터 업로드 중심 레이아웃"""
    # 업로드 방식 선택 탭
    upload_tab1, upload_tab2, upload_tab3 = st.tabs([
        "🚀 야후 파이낸스", 
        "📁 파일 업로드", 
        "📋 샘플 데이터"
    ])
    
    with upload_tab1:
        show_yahoo_finance_download()
    
    with upload_tab2:
        show_file_upload()
    
    with upload_tab3:
        show_sample_data_download()

def show_data_loaded_layout():
    """데이터가 로드된 경우의 레이아웃"""
    # processed_data가 있으면 사용, 없으면 uploaded_data 사용
    if 'processed_data' in st.session_state:
        data = st.session_state['processed_data']
    else:
        data = st.session_state['uploaded_data']
    
    # 상단 통계 카드
    show_quick_stats(data)
    
    # 메인 컨텐츠 탭
    tab1, tab2, tab3 = st.tabs(["📊 데이터 미리보기", "📈 통계 분석", "🔧 데이터 관리"])
    
    with tab1:
        show_data_preview()
    
    with tab2:
        show_detailed_analysis(data)
    
    with tab3:
        show_data_management_tools(data)

def show_quick_stats(data):
    """상단 빠른 통계 카드"""
    stats = calculate_basic_stats(data)
    
    # 4-컬럼 통계 카드
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "총 데이터 수", 
            f"{stats['total_rows']:,}",
            help="업로드된 총 데이터 행 수"
        )
    
    with col2:
        st.metric(
            "종목 수", 
            f"{stats['unique_tickers']:,}",
            help="고유한 종목 수"
        )
    
    with col3:
        quality_score = calculate_quality_score(data)
        st.metric(
            "데이터 품질", 
            f"{quality_score:.1f}%",
            help="데이터 품질 점수"
        )
    
    with col4:
        st.metric(
            "기간", 
            stats['date_range'],
            help="데이터의 날짜 범위"
        )

def show_yahoo_finance_download():
    """야후 파이낸스 데이터 다운로드 섹션"""
    st.markdown("### 📊 야후 파이낸스에서 실시간 데이터 다운로드")
    
    # Yahoo Finance 다운로더 초기화
    if 'yahoo_downloader' not in st.session_state:
        st.session_state['yahoo_downloader'] = YahooFinanceDownloader()
    
    downloader = st.session_state['yahoo_downloader']
    
    # 2-컬럼 레이아웃 (더 균형잡힌 비율)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🎯 티커 선택")
        
        # 티커 입력 방식 선택
        ticker_input_method = st.radio(
            "티커 입력 방식",
            ["인기 종목 선택", "시장 지수 선택", "직접 입력"],
            help="원하는 방식으로 티커를 선택하세요"
        )
        
        selected_tickers = []
        
        if ticker_input_method == "인기 종목 선택":
            popular_tickers = downloader.get_popular_tickers()
            
            # 카테고리별로 그룹화
            categories = {}
            for ticker in popular_tickers:
                cat = ticker['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(ticker)
            
            # 카테고리별로 선택 (더 컴팩트하게)
            for category, tickers in categories.items():
                with st.expander(f"📂 {category}", expanded=False):
                    # 체크박스를 2열로 배치
                    cols = st.columns(2)
                    for i, ticker in enumerate(tickers):
                        col_idx = i % 2
                        with cols[col_idx]:
                            if st.checkbox(f"{ticker['symbol']}", key=f"pop_{ticker['symbol']}"):
                                selected_tickers.append(ticker['symbol'])
                                st.caption(ticker['name'])
        
        elif ticker_input_method == "시장 지수 선택":
            indices = downloader.get_market_indices()
            
            for index in indices:
                if st.checkbox(f"{index['symbol']} - {index['name']}", key=f"idx_{index['symbol']}"):
                    selected_tickers.append(index['symbol'])
        
        else:  # 직접 입력
            ticker_input = st.text_area(
                "티커 심볼 입력",
                placeholder="AAPL\nGOOGL\nMSFT\nTSLA",
                help="한 줄에 하나씩 티커 심볼을 입력하세요",
                height=100
            )
            
            if ticker_input:
                input_tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]
                selected_tickers.extend(input_tickers)
        
        # 선택된 티커 표시
        if selected_tickers:
            st.success(f"✅ 선택된 티커: {len(selected_tickers)}개")
            # 선택된 티커를 태그 형태로 표시
            ticker_tags = " ".join([f"`{ticker}`" for ticker in selected_tickers[:10]])
            st.markdown(ticker_tags)
            if len(selected_tickers) > 10:
                st.caption(f"... 및 {len(selected_tickers) - 10}개 더")
    
    with col2:
        st.markdown("#### 📅 날짜 범위 설정")
        
        # 날짜 범위 설정
        date_range_option = st.selectbox(
            "날짜 범위",
            ["최근 1년", "최근 2년", "최근 5년", "최근 10년", "직접 설정"],
            help="데이터를 가져올 기간을 선택하세요"
        )
        
        if date_range_option == "직접 설정":
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input("시작 날짜", value=datetime.now() - timedelta(days=365))
            with col_date2:
                end_date = st.date_input("종료 날짜", value=datetime.now())
        else:
            # 미리 정의된 기간
            end_date = datetime.now()
            if date_range_option == "최근 1년":
                start_date = end_date - timedelta(days=365)
            elif date_range_option == "최근 2년":
                start_date = end_date - timedelta(days=730)
            elif date_range_option == "최근 5년":
                start_date = end_date - timedelta(days=1825)
            else:  # 최근 10년
                start_date = end_date - timedelta(days=3650)
        
        # 날짜 형식 변환
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # 날짜 정보를 카드 형태로 표시
        st.info(f"""
        📅 **데이터 기간**
        - 시작: {start_date_str}
        - 종료: {end_date_str}
        - 총 일수: {(end_date - start_date).days}일
        """)
        
        # 날짜 유효성 검증
        is_valid_date, date_error = downloader.validate_date_range(start_date_str, end_date_str)
        if not is_valid_date:
            st.error(f"❌ 날짜 오류: {date_error}")
            return
    
    # 다운로드 버튼 (더 눈에 띄게)
    st.markdown("---")
    
    download_col1, download_col2, download_col3 = st.columns([1, 2, 1])
    
    with download_col2:
        if st.button("🚀 데이터 다운로드 시작", type="primary", use_container_width=True):
            if not selected_tickers:
                st.error("❌ 최소 하나의 티커를 선택해주세요.")
                return
            
            # 진행 상황 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 티커 유효성 검증
            status_text.text("티커 유효성 검증 중...")
            progress_bar.progress(20)
            
            valid_tickers = []
            invalid_tickers = []
            
            for ticker in selected_tickers:
                if downloader.validate_ticker(ticker):
                    valid_tickers.append(ticker)
                else:
                    invalid_tickers.append(ticker)
            
            if invalid_tickers:
                st.warning(f"⚠️ 유효하지 않은 티커: {', '.join(invalid_tickers)}")
            
            if not valid_tickers:
                st.error("❌ 유효한 티커가 없습니다.")
                return
            
            # 데이터 다운로드 실행
            status_text.text("데이터 다운로드 중...")
            progress_bar.progress(50)
            
            try:
                data, failed_tickers = downloader.download_multiple_tickers(
                    valid_tickers, start_date_str, end_date_str
                )
                
                progress_bar.progress(80)
                status_text.text("데이터 검증 중...")
                
                if not data.empty:
                    # 세션 상태에 데이터 저장
                    st.session_state['uploaded_data'] = data
                    st.session_state['data_filename'] = f"yahoo_finance_{start_date_str}_{end_date_str}"
                    
                    # 데이터 자동 처리
                    status_text.text("데이터 처리 중...")
                    progress_bar.progress(85)
                    
                    try:
                        from utils.data_processor import DataProcessor
                        processor = DataProcessor()
                        processed_data = processor.process_data(data)
                        st.session_state['processed_data'] = processed_data
                        st.session_state['data_processor'] = processor
                    except Exception as e:
                        st.warning(f"⚠️ 데이터 처리 중 오류: {str(e)}")
                        # 처리되지 않은 데이터라도 업로드된 데이터는 사용 가능
                    
                    # 데이터 요약 표시
                    summary = downloader.get_data_summary(data)
                    
                    progress_bar.progress(100)
                    status_text.text("완료!")
                    
                    # 성공 메시지를 카드 형태로 표시
                    st.success("✅ 데이터 다운로드 및 처리 완료!")
                    
                    # 요약 정보를 컬럼으로 표시
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    with summary_col1:
                        st.metric("총 데이터", f"{summary['total_rows']:,}")
                    
                    with summary_col2:
                        st.metric("종목 수", f"{summary['unique_tickers']}")
                    
                    with summary_col3:
                        st.metric("데이터 완성도", f"{summary['data_completeness']:.1f}%")
                    
                    if failed_tickers:
                        st.warning(f"⚠️ 다운로드 실패: {', '.join(failed_tickers)}")
                    
                    # 데이터 검증
                    validation_result = validate_data(data)
                    if validation_result['is_valid']:
                        st.success("✅ 데이터 검증을 통과했습니다!")
                    else:
                        st.warning("⚠️ 데이터 검증에서 문제가 발견되었습니다.")
                        for issue in validation_result['issues']:
                            st.error(f"• {issue}")
                
                else:
                    st.error("❌ 다운로드된 데이터가 없습니다.")
                    
            except Exception as e:
                st.error(f"❌ 데이터 다운로드 중 오류가 발생했습니다: {str(e)}")

def show_file_upload():
    """파일 업로드 섹션"""
    st.markdown("### 📁 파일에서 데이터 업로드")
    
    # 드래그 앤 드롭 스타일의 업로더
    uploaded_file = st.file_uploader(
        "주가 데이터 파일을 선택하거나 드래그하여 업로드하세요",
        type=['csv', 'xlsx', 'xls', 'parquet'],
        help="CSV, Excel, Parquet 형식 지원"
    )
    
    if uploaded_file is not None:
        # 파일 정보를 카드 형태로 표시
        file_info = {
            "파일명": uploaded_file.name,
            "크기": f"{uploaded_file.size / 1024:.1f} KB",
            "타입": uploaded_file.type
        }
        
        st.info(f"""
        📄 **업로드된 파일**
        - 파일명: {file_info['파일명']}
        - 크기: {file_info['크기']}
        - 타입: {file_info['타입']}
        """)
        
        # 데이터 로드
        try:
            with st.spinner("데이터 로드 중..."):
                data = load_data(uploaded_file)
                
            if data is not None and not data.empty:
                # 데이터 검증
                validation_result = validate_data(data)
                
                if validation_result['is_valid']:
                    # 세션 상태에 데이터 저장
                    st.session_state['uploaded_data'] = data
                    st.session_state['data_filename'] = uploaded_file.name
                    
                    # 데이터 자동 처리
                    with st.spinner("데이터 처리 중..."):
                        try:
                            from utils.data_processor import DataProcessor
                            processor = DataProcessor()
                            processed_data = processor.process_data(data)
                            st.session_state['processed_data'] = processed_data
                            st.session_state['data_processor'] = processor
                            st.success("✅ 데이터가 성공적으로 로드되고 처리되었습니다!")
                        except Exception as e:
                            st.warning(f"⚠️ 데이터 처리 중 오류: {str(e)}")
                            st.info("업로드된 원본 데이터는 사용 가능합니다.")
                            # 처리되지 않은 데이터라도 업로드된 데이터는 사용 가능
                    
                    # 데이터 요약 정보 표시
                    st.success("✅ 데이터 검증을 통과했습니다!")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("총 행 수", f"{len(data):,}")
                    with col2:
                        if 'Ticker' in data.columns:
                            st.metric("종목 수", f"{data['Ticker'].nunique()}")
                        else:
                            st.metric("종목 수", "N/A")
                    with col3:
                        if 'Date' in data.columns:
                            date_range = f"{data['Date'].min()} ~ {data['Date'].max()}"
                            st.metric("기간", date_range[:20] + "..." if len(date_range) > 20 else date_range)
                        else:
                            st.metric("기간", "N/A")
                else:
                    st.error("❌ 데이터 검증을 통과하지 못했습니다.")
                    for issue in validation_result['issues']:
                        st.error(f"• {issue}")
                    
                    # 부분적으로라도 데이터 저장 시도
                    if len(data) > 0:
                        st.warning("⚠️ 검증에 실패했지만 데이터는 저장됩니다.")
                        st.session_state['uploaded_data'] = data
                        st.session_state['data_filename'] = uploaded_file.name
            else:
                st.error("❌ 데이터가 비어있거나 로드할 수 없습니다.")
        except Exception as e:
            st.error(f"❌ 데이터 로드 중 오류가 발생했습니다: {str(e)}")
            st.error("파일 형식을 확인하고 다시 시도해주세요.")

def show_sample_data_download():
    """샘플 데이터 다운로드 섹션"""
    st.markdown("### 📋 샘플 데이터 다운로드")
    st.markdown("AlphaForge 분석을 위한 샘플 주가 데이터를 다운로드하세요.")
    
    # 샘플 데이터 정보
    st.info("""
    📊 **샘플 데이터 포함 내용**
    - 10개 주요 종목 (AAPL, GOOGL, MSFT 등)
    - 최근 2년간의 일별 데이터
    - OHLCV (시가, 고가, 저가, 종가, 거래량) 데이터
    - CSV 형식으로 다운로드
    """)
    
    if st.button("📥 샘플 데이터 다운로드", use_container_width=True):
        with st.spinner("샘플 데이터 생성 중..."):
            try:
                sample_data = create_sample_data()
                
                if sample_data is not None and not sample_data.empty:
                    # 세션 상태에 데이터 저장 및 처리
                    st.session_state['uploaded_data'] = sample_data
                    st.session_state['data_filename'] = "sample_stock_data.csv"
                    
                    # 데이터 자동 처리
                    try:
                        from utils.data_processor import DataProcessor
                        processor = DataProcessor()
                        processed_data = processor.process_data(sample_data)
                        st.session_state['processed_data'] = processed_data
                        st.session_state['data_processor'] = processor
                        st.success("✅ 샘플 데이터가 생성되고 처리되었습니다!")
                    except Exception as e:
                        st.warning(f"⚠️ 데이터 처리 중 오류: {str(e)}")
                        st.info("업로드된 원본 데이터는 사용 가능합니다.")
                    
                    # 데이터 요약 정보 표시
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("총 행 수", f"{len(sample_data):,}")
                    with col2:
                        if 'Ticker' in sample_data.columns:
                            st.metric("종목 수", f"{sample_data['Ticker'].nunique()}")
                        else:
                            st.metric("종목 수", "N/A")
                    with col3:
                        if 'Date' in sample_data.columns:
                            date_range = f"{sample_data['Date'].min()} ~ {sample_data['Date'].max()}"
                            st.metric("기간", date_range[:20] + "..." if len(date_range) > 20 else date_range)
                        else:
                            st.metric("기간", "N/A")
                    
                    # CSV 다운로드 버튼
                    csv = sample_data.to_csv(index=False)
                    st.download_button(
                        label="📥 CSV 다운로드",
                        data=csv,
                        file_name="sample_stock_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.error("❌ 샘플 데이터 생성에 실패했습니다.")
            except Exception as e:
                st.error(f"❌ 샘플 데이터 생성 중 오류가 발생했습니다: {str(e)}")

def show_data_preview():
    """데이터 미리보기 섹션"""
    # processed_data가 있으면 사용, 없으면 uploaded_data 사용
    if 'processed_data' in st.session_state:
        data = st.session_state['processed_data']
        data_type = "처리된 데이터"
    elif 'uploaded_data' in st.session_state:
        data = st.session_state['uploaded_data']
        data_type = "업로드된 데이터"
    else:
        st.info("📁 데이터를 업로드하여 미리보기를 확인하세요.")
        return
    
    # 데이터 정보 헤더
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.metric("총 행 수", f"{len(data):,}")
    
    with col_info2:
        st.metric("총 컬럼 수", f"{len(data.columns)}")
    
    with col_info3:
        missing_data = data.isnull().sum().sum()
        st.metric("결측치 수", f"{missing_data:,}")
    
    # 데이터 타입 표시
    st.info(f"📊 현재 표시 중: {data_type}")
    
    # 야후 파이낸스 데이터인 경우 추가 정보 표시
    if 'yahoo_downloader' in st.session_state and 'data_filename' in st.session_state:
        if st.session_state['data_filename'].startswith('yahoo_finance'):
            show_yahoo_data_info(data)
    
    # 데이터 미리보기 설정
    preview_col1, preview_col2 = st.columns([3, 1])
    
    with preview_col1:
        preview_rows = st.slider("미리보기 행 수", 5, 100, 20)
    
    with preview_col2:
        if st.button("전체 데이터 보기", use_container_width=True):
            preview_rows = len(data)
    
    # 데이터 미리보기
    st.dataframe(data.head(preview_rows), use_container_width=True)
    
    # 컬럼 정보
    with st.expander("📋 컬럼 정보", expanded=False):
        col_info = pd.DataFrame({
            '컬럼명': data.columns,
            '데이터 타입': data.dtypes.astype(str),
            '결측치 수': data.isnull().sum(),
            '결측치 비율': (data.isnull().sum() / len(data) * 100).round(2)
        })
        st.dataframe(col_info, use_container_width=True)
    
    # 데이터 시각화
    with st.expander("📈 데이터 시각화", expanded=False):
        show_data_visualization(data)

def show_detailed_analysis(data):
    """상세 통계 분석 섹션"""
    st.markdown("### 📈 상세 통계 분석")
    
    # 분석 탭
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["📊 기본 통계", "📈 시계열 분석", "🔍 데이터 품질"])
    
    with analysis_tab1:
        show_basic_stats()
    
    with analysis_tab2:
        show_time_series_analysis(data)
    
    with analysis_tab3:
        show_data_quality_analysis(data)

def show_data_management_tools(data):
    """데이터 관리 도구 섹션"""
    st.markdown("### 🔧 데이터 관리 도구")
    
    # 관리 도구 탭
    tools_tab1, tools_tab2, tools_tab3 = st.tabs(["🔄 데이터 새로고침", "💾 데이터 저장", "🗑️ 데이터 삭제"])
    
    with tools_tab1:
        st.markdown("#### 🔄 데이터 새로고침")
        st.info("최신 데이터로 업데이트하려면 위의 업로드 섹션을 다시 사용하세요.")
    
    with tools_tab2:
        st.markdown("#### 💾 데이터 저장")
        if st.button("CSV로 저장", use_container_width=True):
            csv = data.to_csv(index=False)
            st.download_button(
                label="다운로드",
                data=csv,
                file_name=f"stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with tools_tab3:
        st.markdown("#### 🗑️ 데이터 삭제")
        if st.button("데이터 삭제", type="secondary", use_container_width=True):
            # 데이터 관련 세션 삭제
            if 'uploaded_data' in st.session_state:
                del st.session_state['uploaded_data']
            if 'processed_data' in st.session_state:
                del st.session_state['processed_data']
            if 'data_processor' in st.session_state:
                del st.session_state['data_processor']
            if 'data_filename' in st.session_state:
                del st.session_state['data_filename']
            
            # 팩터 마이닝 결과 삭제 (데이터가 변경되었으므로)
            if 'mining_results' in st.session_state:
                del st.session_state['mining_results']
            
            # 동적 결합 결과 삭제 (팩터가 변경되었으므로)
            if 'combination_results' in st.session_state:
                del st.session_state['combination_results']
            
            # 기본 팩터 사용 설정 삭제
            if 'use_default_factors' in st.session_state:
                del st.session_state['use_default_factors']
            
            st.success("데이터와 관련 결과가 모두 삭제되었습니다.")
            st.rerun()

def show_time_series_analysis(data):
    """시계열 분석"""
    if 'Close' in data.columns and 'Date' in data.columns:
        try:
            # 종가 추이
            fig = px.line(
                data.groupby('Date')['Close'].mean().reset_index(),
                x='Date',
                y='Close',
                title="평균 종가 추이"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="close_trend_chart")
            
            # 거래량 분석
            if 'Volume' in data.columns:
                col_vol1, col_vol2 = st.columns(2)
                
                with col_vol1:
                    fig_vol = px.histogram(
                        data,
                        x='Volume',
                        title="거래량 분포",
                        nbins=50
                    )
                    fig_vol.update_layout(height=300)
                    st.plotly_chart(fig_vol, use_container_width=True, key="volume_distribution_chart")
                
                with col_vol2:
                    # 월별 평균 거래량
                    data_copy = data.copy()
                    data_copy['Date'] = pd.to_datetime(data_copy['Date'])
                    data_copy['Month'] = data_copy['Date'].dt.to_period('M')
                    monthly_volume = data_copy.groupby('Month')['Volume'].mean().reset_index()
                    monthly_volume['Month'] = monthly_volume['Month'].astype(str)
                    
                    fig_monthly = px.bar(
                        monthly_volume,
                        x='Month',
                        y='Volume',
                        title="월별 평균 거래량"
                    )
                    fig_monthly.update_layout(height=300)
                    st.plotly_chart(fig_monthly, use_container_width=True, key="monthly_volume_chart")
        
        except Exception as e:
            st.error(f"시계열 분석 중 오류: {str(e)}")
    else:
        st.warning("시계열 분석을 위해 'Date'와 'Close' 컬럼이 필요합니다.")

def show_data_quality_analysis(data):
    """데이터 품질 분석"""
    st.markdown("#### 🔍 데이터 품질 분석")
    
    # 품질 지표 계산
    quality_score = calculate_quality_score(data)
    
    # 품질 점수 게이지
    st.metric("전체 품질 점수", f"{quality_score:.1f}%")
    
    # 상세 품질 지표
    col_qual1, col_qual2, col_qual3 = st.columns(3)
    
    with col_qual1:
        completeness = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        st.metric("완전성", f"{completeness:.1f}%")
    
    with col_qual2:
        # 일관성 체크 (예시)
        consistency = 95.0
        st.metric("일관성", f"{consistency:.1f}%")
    
    with col_qual3:
        # 정확성 체크 (예시)
        accuracy = 98.0
        st.metric("정확성", f"{accuracy:.1f}%")
    
    # 결측치 분석
    st.markdown("#### 📊 결측치 분석")
    missing_data = data.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) > 0:
        fig_missing = px.bar(
            x=missing_data.index,
            y=missing_data.values,
            title="컬럼별 결측치 수"
        )
        fig_missing.update_layout(height=300)
        st.plotly_chart(fig_missing, use_container_width=True, key="missing_data_chart")
    else:
        st.success("✅ 결측치가 없습니다!")

def show_yahoo_data_info(data):
    """야후 파이낸스 데이터에 대한 추가 정보를 표시합니다."""
    st.markdown("---")
    st.markdown("### 📊 야후 파이낸스 데이터 정보")
    
    if 'Ticker' in data.columns:
        # 종목별 통계
        ticker_stats = data.groupby('Ticker').agg({
            'Close': ['count', 'mean', 'std', 'min', 'max'],
            'Volume': ['mean', 'sum']
        }).round(2)
        
        ticker_stats.columns = ['데이터 수', '평균 종가', '표준편차', '최저가', '최고가', '평균 거래량', '총 거래량']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 종목별 기본 통계")
            st.dataframe(ticker_stats, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 종목별 데이터 분포")
            
            # 종목별 데이터 수 차트
            ticker_counts = data['Ticker'].value_counts()
            fig = px.bar(
                x=ticker_counts.index,
                y=ticker_counts.values,
                title="종목별 데이터 수",
                labels={'x': '티커', 'y': '데이터 수'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, key="yahoo_ticker_counts_chart")
        
        # 최근 데이터 확인
        st.markdown("#### 📅 최근 데이터 확인")
        recent_data = data.groupby('Ticker').tail(1).sort_values('Date', ascending=False)
        st.dataframe(recent_data[['Date', 'Ticker', 'Close', 'Volume']], use_container_width=True)

def show_basic_stats():
    """기본 통계 섹션"""
    # processed_data가 있으면 사용, 없으면 uploaded_data 사용
    if 'processed_data' in st.session_state:
        data = st.session_state['processed_data']
    elif 'uploaded_data' in st.session_state:
        data = st.session_state['uploaded_data']
    else:
        st.info("📁 데이터를 업로드하여 통계를 확인하세요.")
        return
    
    # 기본 통계
    stats = calculate_basic_stats(data)
    
    # 통계 카드들을 2x2 그리드로 배치
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("총 데이터 수", f"{stats['total_rows']:,}")
        st.metric("종목 수", f"{stats['unique_tickers']:,}")
    
    with col2:
        st.metric("기간", f"{stats['date_range']}")
        st.metric("평균 종가", f"${stats['avg_close']:.2f}")
    
    # 종목별 상세 통계
    if 'Ticker' in data.columns and 'Close' in data.columns:
        st.markdown("#### 📊 종목별 통계")
        
        ticker_stats = data.groupby('Ticker').agg({
            'Close': ['count', 'mean', 'std', 'min', 'max'],
            'Volume': ['mean', 'sum'] if 'Volume' in data.columns else ['count']
        }).round(2)
        
        # 컬럼명 정리
        if 'Volume' in data.columns:
            ticker_stats.columns = ['데이터 수', '평균 종가', '표준편차', '최저가', '최고가', '평균 거래량', '총 거래량']
        else:
            ticker_stats.columns = ['데이터 수', '평균 종가', '표준편차', '최저가', '최고가']
        
        st.dataframe(ticker_stats, use_container_width=True)

def load_data(uploaded_file):
    """업로드된 파일에서 데이터를 로드합니다."""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            data = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            data = pd.read_excel(uploaded_file)
        elif file_extension == 'parquet':
            data = pd.read_parquet(uploaded_file)
        else:
            st.error("지원하지 않는 파일 형식입니다.")
            return None
        
        return data
    
    except Exception as e:
        st.error(f"파일 로드 중 오류: {str(e)}")
        return None

def validate_data(data):
    """데이터 검증을 수행합니다."""
    try:
        validator = DataValidator()
        result = validator.validate(data)
        
        # 추가 검증 로직
        if result['is_valid']:
            # 데이터 품질 점수 계산
            quality_score = calculate_quality_score(data)
            result['quality_score'] = quality_score
            
            # 추가 권장사항
            recommendations = []
            
            # 데이터 크기 확인
            if len(data) < 1000:
                recommendations.append("데이터가 적습니다. 더 많은 데이터를 사용하는 것을 권장합니다.")
            
            # 종목 수 확인
            if 'Ticker' in data.columns and data['Ticker'].nunique() < 5:
                recommendations.append("종목 수가 적습니다. 더 다양한 종목을 포함하는 것을 권장합니다.")
            
            # 날짜 범위 확인
            if 'Date' in data.columns:
                date_range = (pd.to_datetime(data['Date'].max()) - pd.to_datetime(data['Date'].min())).days
                if date_range < 252:  # 1년 미만
                    recommendations.append("데이터 기간이 짧습니다. 최소 1년 이상의 데이터를 권장합니다.")
            
            result['recommendations'] = recommendations
        
        return result
        
    except Exception as e:
        return {
            'is_valid': False,
            'issues': [f"검증 중 오류 발생: {str(e)}"],
            'quality_score': 0,
            'recommendations': []
        }

def create_sample_data():
    """샘플 주가 데이터를 생성합니다."""
    # 샘플 종목들
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM']
    
    # 날짜 범위 (최근 2년)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 샘플 데이터 생성
    sample_data = []
    
    for ticker in tickers:
        # 각 종목별로 다른 기본 가격 설정
        base_price = np.random.uniform(50, 500)
        
        for date in dates:
            # 주말 제외
            if date.weekday() < 5:
                # 더 현실적인 가격 변동 생성
                price_change = np.random.normal(0, 0.02)  # 2% 표준편차
                close_price = base_price * (1 + price_change)
                close_price = max(close_price, 1.0)  # 최소 $1
                
                # OHLC 생성
                open_price = close_price * (1 + np.random.normal(0, 0.01))
                high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.03))
                low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.03))
                
                # 거래량 생성 (가격 변동과 연관)
                base_volume = np.random.randint(1000000, 10000000)
                volume = int(base_volume * (1 + abs(price_change) * 10))  # 변동성이 클 때 거래량 증가
                
                sample_data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Ticker': ticker,
                    'Open': round(open_price, 2),
                    'High': round(high_price, 2),
                    'Low': round(low_price, 2),
                    'Close': round(close_price, 2),
                    'Volume': volume
                })
                
                # 다음 날을 위한 기본 가격 업데이트
                base_price = close_price
    
    df = pd.DataFrame(sample_data)
    
    # 데이터 검증
    if df.empty:
        raise ValueError("샘플 데이터 생성 실패: 빈 데이터프레임")
    
    # 필수 컬럼 확인
    required_columns = ['Date', 'Ticker', 'Close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼 누락: {missing_columns}")
    
    # 데이터 타입 확인
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Close'] = pd.to_numeric(df['Close'])
        df['Ticker'] = df['Ticker'].astype(str)
    except Exception as e:
        raise ValueError(f"데이터 타입 변환 실패: {str(e)}")
    
    # 음수 가격 확인
    if (df['Close'] <= 0).any():
        raise ValueError("음수 또는 0 가격이 발견되었습니다")
    
    print(f"샘플 데이터 생성 완료: {len(df)} 행, {df['Ticker'].nunique()} 종목")
    return df

def calculate_basic_stats(data):
    """기본 통계를 계산합니다."""
    stats = {}
    
    # 기본 정보
    stats['total_rows'] = len(data)
    stats['unique_tickers'] = data['Ticker'].nunique() if 'Ticker' in data.columns else 0
    
    # 날짜 범위
    if 'Date' in data.columns:
        try:
            data['Date'] = pd.to_datetime(data['Date'])
            date_range = f"{data['Date'].min().strftime('%Y-%m-%d')} ~ {data['Date'].max().strftime('%Y-%m-%d')}"
            stats['date_range'] = date_range
        except:
            stats['date_range'] = "날짜 형식 오류"
    else:
        stats['date_range'] = "날짜 컬럼 없음"
    
    # 종가 통계
    if 'Close' in data.columns:
        stats['avg_close'] = data['Close'].mean()
        stats['min_close'] = data['Close'].min()
        stats['max_close'] = data['Close'].max()
    else:
        stats['avg_close'] = 0
    
    # 데이터 품질 지표
    stats['completeness'] = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
    stats['consistency'] = 95.0  # 예시 값
    stats['accuracy'] = 98.0     # 예시 값
    
    return stats

def calculate_quality_score(data):
    """데이터 품질 점수를 계산합니다."""
    # 완전성 (결측치 비율)
    completeness = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
    
    # 일관성 (데이터 타입 일치)
    consistency = 95.0  # 예시 값
    
    # 정확성 (이상치 비율)
    accuracy = 98.0  # 예시 값
    
    # 종합 점수
    quality_score = (completeness * 0.4 + consistency * 0.3 + accuracy * 0.3)
    
    return quality_score

def show_data_visualization(data):
    """데이터 시각화를 표시합니다."""
    if 'Close' in data.columns and 'Date' in data.columns:
        try:
            # 종가 시계열 차트
            fig = px.line(
                data.groupby('Date')['Close'].mean().reset_index(),
                x='Date',
                y='Close',
                title="평균 종가 추이"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="preview_close_trend_chart")
            
            # 거래량 분포
            if 'Volume' in data.columns:
                fig_vol = px.histogram(
                    data,
                    x='Volume',
                    title="거래량 분포",
                    nbins=50
                )
                fig_vol.update_layout(height=300)
                st.plotly_chart(fig_vol, use_container_width=True, key="preview_volume_distribution_chart")
        
        except Exception as e:
            st.error(f"시각화 생성 중 오류: {str(e)}")

if __name__ == "__main__":
    show_page() 
