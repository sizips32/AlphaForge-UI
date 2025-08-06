"""
설정 페이지
시스템 설정, 사용자 설정, 데이터베이스 설정 등을 관리합니다.
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import UI_SETTINGS, DATA_VALIDATION

def show_page():
    """설정 페이지를 표시합니다."""
    st.title("⚙️ 설정")
    st.markdown("AlphaForge-UI의 시스템 설정을 관리합니다.")
    
    # 탭으로 설정 분류
    settings_tab1, settings_tab2, settings_tab3, settings_tab4 = st.tabs([
        "🔧 시스템 설정", "👤 사용자 설정", "📊 데이터 설정", "🔍 고급 설정"
    ])
    
    with settings_tab1:
        show_system_settings()
    
    with settings_tab2:
        show_user_settings()
    
    with settings_tab3:
        show_data_settings()
    
    with settings_tab4:
        show_advanced_settings()

def show_system_settings():
    """시스템 설정 섹션"""
    st.markdown("### 🔧 시스템 설정")
    
    # 앱 기본 설정
    st.subheader("📱 앱 기본 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 페이지 레이아웃
        layout_option = st.selectbox(
            "페이지 레이아웃",
            ["wide", "centered"],
            index=0,
            help="앱의 레이아웃 모드를 선택하세요"
        )
        
        # 사이드바 상태
        sidebar_state = st.selectbox(
            "사이드바 기본 상태",
            ["expanded", "collapsed"],
            index=0,
            help="앱 시작 시 사이드바 상태를 설정하세요"
        )
    
    with col2:
        # 테마 설정
        theme_option = st.selectbox(
            "테마",
            ["light", "dark"],
            index=0,
            help="앱의 테마를 선택하세요"
        )
        
        # 언어 설정
        language_option = st.selectbox(
            "언어",
            ["한국어", "English"],
            index=0,
            help="앱의 언어를 선택하세요"
        )
    
    # 성능 설정
    st.subheader("⚡ 성능 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 캐시 설정
        cache_enabled = st.checkbox(
            "데이터 캐시 활성화",
            value=True,
            help="데이터 로딩 성능 향상을 위해 캐시를 사용합니다"
        )
        
        if cache_enabled:
            cache_ttl = st.slider(
                "캐시 유효 시간 (분)",
                min_value=5,
                max_value=60,
                value=30,
                help="캐시된 데이터의 유효 시간을 설정하세요"
            )
    
    with col2:
        # 병렬 처리
        parallel_processing = st.checkbox(
            "병렬 처리 활성화",
            value=True,
            help="대용량 데이터 처리 시 병렬 처리를 사용합니다"
        )
        
        if parallel_processing:
            max_workers = st.slider(
                "최대 워커 수",
                min_value=2,
                max_value=8,
                value=4,
                help="병렬 처리에 사용할 최대 워커 수를 설정하세요"
            )
    
    # 저장 버튼
    if st.button("💾 시스템 설정 저장", type="primary"):
        save_system_settings({
            'layout': layout_option,
            'sidebar_state': sidebar_state,
            'theme': theme_option,
            'language': language_option,
            'cache_enabled': cache_enabled,
            'cache_ttl': cache_ttl if cache_enabled else 30,
            'parallel_processing': parallel_processing,
            'max_workers': max_workers if parallel_processing else 4
        })
        st.success("✅ 시스템 설정이 저장되었습니다!")

def show_user_settings():
    """사용자 설정 섹션"""
    st.markdown("### 👤 사용자 설정")
    
    # 사용자 정보
    st.subheader("👤 사용자 정보")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_name = st.text_input(
            "사용자명",
            value=st.session_state.get('user_name', ''),
            help="사용자 이름을 입력하세요"
        )
        
        user_email = st.text_input(
            "이메일",
            value=st.session_state.get('user_email', ''),
            help="이메일 주소를 입력하세요"
        )
    
    with col2:
        user_role = st.selectbox(
            "사용자 역할",
            ["분석가", "개발자", "관리자", "일반 사용자"],
            index=0,
            help="사용자 역할을 선택하세요"
        )
        
        user_organization = st.text_input(
            "소속",
            value=st.session_state.get('user_organization', ''),
            help="소속 기관을 입력하세요"
        )
    
    # 알림 설정
    st.subheader("🔔 알림 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        email_notifications = st.checkbox(
            "이메일 알림",
            value=st.session_state.get('email_notifications', False),
            help="분석 완료 시 이메일 알림을 받습니다"
        )
        
        browser_notifications = st.checkbox(
            "브라우저 알림",
            value=st.session_state.get('browser_notifications', True),
            help="브라우저 알림을 활성화합니다"
        )
    
    with col2:
        analysis_notifications = st.checkbox(
            "분석 완료 알림",
            value=st.session_state.get('analysis_notifications', True),
            help="팩터 마이닝 완료 시 알림을 받습니다"
        )
        
        error_notifications = st.checkbox(
            "오류 알림",
            value=st.session_state.get('error_notifications', True),
            help="오류 발생 시 알림을 받습니다"
        )
    
    # 저장 버튼
    if st.button("💾 사용자 설정 저장", type="primary"):
        save_user_settings({
            'user_name': user_name,
            'user_email': user_email,
            'user_role': user_role,
            'user_organization': user_organization,
            'email_notifications': email_notifications,
            'browser_notifications': browser_notifications,
            'analysis_notifications': analysis_notifications,
            'error_notifications': error_notifications
        })
        st.success("✅ 사용자 설정이 저장되었습니다!")

def show_data_settings():
    """데이터 설정 섹션"""
    st.markdown("### 📊 데이터 설정")
    
    # 데이터 소스 설정
    st.subheader("📡 데이터 소스 설정")
    
    # 야후 파이낸스 설정
    st.markdown("#### 📈 야후 파이낸스 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        yahoo_timeout = st.slider(
            "API 타임아웃 (초)",
            min_value=10,
            max_value=60,
            value=30,
            help="야후 파이낸스 API 호출 타임아웃을 설정하세요"
        )
        
        yahoo_retry_count = st.slider(
            "재시도 횟수",
            min_value=1,
            max_value=5,
            value=3,
            help="API 호출 실패 시 재시도 횟수를 설정하세요"
        )
    
    with col2:
        yahoo_delay = st.slider(
            "요청 간격 (초)",
            min_value=0.1,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="API 요청 간 간격을 설정하세요"
        )
        
        max_tickers = st.slider(
            "최대 티커 수",
            min_value=10,
            max_value=100,
            value=50,
            help="한 번에 다운로드할 최대 티커 수를 설정하세요"
        )
    
    # 데이터 처리 설정
    st.subheader("🔄 데이터 처리 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_clean_data = st.checkbox(
            "자동 데이터 정리",
            value=True,
            help="업로드된 데이터를 자동으로 정리합니다"
        )
        
        remove_outliers = st.checkbox(
            "이상치 제거",
            value=True,
            help="통계적 이상치를 자동으로 제거합니다"
        )
    
    with col2:
        fill_missing_data = st.selectbox(
            "결측치 처리 방법",
            ["제거", "전진 채우기", "후진 채우기", "평균값", "중앙값"],
            index=0,
            help="결측치 처리 방법을 선택하세요"
        )
        
        data_validation = st.checkbox(
            "데이터 검증 활성화",
            value=True,
            help="업로드된 데이터의 유효성을 검증합니다"
        )
    
    # 저장 버튼
    if st.button("💾 데이터 설정 저장", type="primary"):
        save_data_settings({
            'yahoo_timeout': yahoo_timeout,
            'yahoo_retry_count': yahoo_retry_count,
            'yahoo_delay': yahoo_delay,
            'max_tickers': max_tickers,
            'auto_clean_data': auto_clean_data,
            'remove_outliers': remove_outliers,
            'fill_missing_data': fill_missing_data,
            'data_validation': data_validation
        })
        st.success("✅ 데이터 설정이 저장되었습니다!")

def show_advanced_settings():
    """고급 설정 섹션"""
    st.markdown("### 🔍 고급 설정")
    
    # 개발자 설정
    st.subheader("👨‍💻 개발자 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        debug_mode = st.checkbox(
            "디버그 모드",
            value=False,
            help="디버그 정보를 표시합니다"
        )
        
        verbose_logging = st.checkbox(
            "상세 로깅",
            value=False,
            help="상세한 로그를 출력합니다"
        )
    
    with col2:
        log_level = st.selectbox(
            "로그 레벨",
            ["INFO", "DEBUG", "WARNING", "ERROR"],
            index=0,
            help="로그 출력 레벨을 설정하세요"
        )
        
        save_logs = st.checkbox(
            "로그 파일 저장",
            value=True,
            help="로그를 파일로 저장합니다"
        )
    
    # 성능 최적화
    st.subheader("⚡ 성능 최적화")
    
    col1, col2 = st.columns(2)
    
    with col1:
        memory_limit = st.slider(
            "메모리 제한 (GB)",
            min_value=1,
            max_value=16,
            value=4,
            help="앱이 사용할 최대 메모리를 설정하세요"
        )
        
        chunk_size = st.slider(
            "청크 크기",
            min_value=1000,
            max_value=10000,
            value=5000,
            step=1000,
            help="대용량 데이터 처리 시 청크 크기를 설정하세요"
        )
    
    with col2:
        max_file_size = st.slider(
            "최대 파일 크기 (MB)",
            min_value=10,
            max_value=1000,
            value=100,
            help="업로드 가능한 최대 파일 크기를 설정하세요"
        )
        
        compression_enabled = st.checkbox(
            "데이터 압축",
            value=True,
            help="데이터 저장 시 압축을 사용합니다"
        )
    
    # 위험 설정
    st.subheader("⚠️ 위험 설정")
    
    with st.expander("🚨 위험한 설정 (주의 필요)"):
        st.warning("⚠️ 이 설정들은 시스템에 영향을 줄 수 있습니다. 신중하게 변경하세요.")
        
        reset_all_data = st.button(
            "🗑️ 모든 데이터 초기화",
            type="secondary",
            help="모든 저장된 데이터를 삭제합니다 (복구 불가)"
        )
        
        if reset_all_data:
            if st.checkbox("정말로 모든 데이터를 삭제하시겠습니까?"):
                clear_all_data()
                st.success("✅ 모든 데이터가 초기화되었습니다.")
    
    # 저장 버튼
    if st.button("💾 고급 설정 저장", type="primary"):
        save_advanced_settings({
            'debug_mode': debug_mode,
            'verbose_logging': verbose_logging,
            'log_level': log_level,
            'save_logs': save_logs,
            'memory_limit': memory_limit,
            'chunk_size': chunk_size,
            'max_file_size': max_file_size,
            'compression_enabled': compression_enabled
        })
        st.success("✅ 고급 설정이 저장되었습니다!")

def save_system_settings(settings):
    """시스템 설정을 저장합니다."""
    try:
        # 세션 상태에 저장
        st.session_state['system_settings'] = settings
        
        # 파일로 저장 (선택사항)
        settings_file = "data/system_settings.json"
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)
        
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        st.error(f"설정 저장 중 오류가 발생했습니다: {str(e)}")

def save_user_settings(settings):
    """사용자 설정을 저장합니다."""
    try:
        # 세션 상태에 저장
        for key, value in settings.items():
            st.session_state[key] = value
        
        # 파일로 저장
        settings_file = "data/user_settings.json"
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)
        
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        st.error(f"설정 저장 중 오류가 발생했습니다: {str(e)}")

def save_data_settings(settings):
    """데이터 설정을 저장합니다."""
    try:
        # 세션 상태에 저장
        st.session_state['data_settings'] = settings
        
        # 파일로 저장
        settings_file = "data/data_settings.json"
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)
        
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        st.error(f"설정 저장 중 오류가 발생했습니다: {str(e)}")

def save_advanced_settings(settings):
    """고급 설정을 저장합니다."""
    try:
        # 세션 상태에 저장
        st.session_state['advanced_settings'] = settings
        
        # 파일로 저장
        settings_file = "data/advanced_settings.json"
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)
        
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        st.error(f"설정 저장 중 오류가 발생했습니다: {str(e)}")

def clear_all_data():
    """모든 데이터를 초기화합니다."""
    try:
        # 세션 상태 초기화
        keys_to_clear = [
            'uploaded_data', 'data_filename', 'yahoo_downloader',
            'system_settings', 'data_settings', 'advanced_settings'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # 사용자 설정은 유지 (선택사항)
        # st.session_state.clear()
        
        # 데이터 파일 삭제 (선택사항)
        data_dir = "data"
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.csv') or file.endswith('.json'):
                    os.remove(os.path.join(data_dir, file))
                    
    except Exception as e:
        st.error(f"데이터 초기화 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    show_page() 
