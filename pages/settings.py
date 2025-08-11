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
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import UI_SETTINGS, DATA_VALIDATION
from utils.env_manager import env_manager
from utils.cache_utils import clear_all_caches, get_cache_info
from utils.logger import show_log_viewer, analyze_logs

def show_page():
    """설정 페이지를 표시합니다."""
    st.title("⚙️ 설정")
    st.markdown("AlphaForge-UI의 시스템 설정을 관리합니다.")
    
    # 탭으로 설정 분류
    settings_tab1, settings_tab2, settings_tab3, settings_tab4, settings_tab5, settings_tab6, settings_tab7 = st.tabs([
        "🔧 시스템 설정", "👤 사용자 설정", "📊 데이터 설정", "🔍 고급 설정", "🔑 환경변수", "📋 로그", "⚡ 성능"
    ])
    
    with settings_tab1:
        show_system_settings()
    
    with settings_tab2:
        show_user_settings()
    
    with settings_tab3:
        show_data_settings()
    
    with settings_tab4:
        show_advanced_settings()
    
    with settings_tab5:
        show_environment_settings()
    
    with settings_tab6:
        show_logging_settings()
    
    with settings_tab7:
        show_performance_settings()

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

def show_environment_settings():
    """환경변수 설정 섹션"""
    st.markdown("### 🔑 환경변수 설정")
    
    # 환경 설정 상태 표시
    env_manager.show_environment_status()
    
    st.markdown("---")
    
    # 캐시 관리
    st.subheader("💾 캐시 관리")
    
    # 캐시 정보 표시
    cache_info = get_cache_info()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("캐시 파일 수", cache_info['cache_count'])
    with col2:
        st.metric("캐시 크기", f"{cache_info['total_size_mb']} MB")
    with col3:
        st.metric("캐시 디렉토리", cache_info['cache_dir'])
    
    # 캐시 삭제 버튼
    if st.button("🗑️ 모든 캐시 삭제", type="secondary"):
        if clear_all_caches():
            st.success("캐시가 성공적으로 삭제되었습니다.")
            st.rerun()
        else:
            st.error("캐시 삭제 중 오류가 발생했습니다.")
    
    st.markdown("---")
    
    # .env 파일 안내
    st.subheader("📝 환경설정 파일 (.env)")
    
    env_file = Path('.env')
    template_file = Path('.env.template')
    
    if not env_file.exists():
        st.warning("⚠️ .env 파일이 없습니다.")
        
        if template_file.exists():
            st.info("""
            📋 **설정 방법:**
            
            1. 터미널에서 다음 명령어 실행:
            ```bash
            cp .env.template .env
            ```
            
            2. .env 파일을 편집하여 필요한 설정 입력
            
            3. AlphaForge-UI 재시작
            """)
        else:
            st.error("❌ .env.template 파일도 없습니다. 개발자에게 문의하세요.")
    else:
        st.success("✅ .env 파일이 존재합니다.")
        
        with st.expander("📄 .env 파일 내용 보기"):
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    env_content = f.read()
                
                # 민감한 정보 마스킹
                lines = env_content.split('\n')
                masked_lines = []
                
                for line in lines:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.split('=', 1)
                        if any(sensitive in key.upper() for sensitive in ['KEY', 'SECRET', 'PASSWORD', 'TOKEN']):
                            masked_value = env_manager.mask_sensitive_value(value)
                            masked_lines.append(f"{key}={masked_value}")
                        else:
                            masked_lines.append(line)
                    else:
                        masked_lines.append(line)
                
                st.code('\n'.join(masked_lines))
                
            except Exception as e:
                st.error(f"파일을 읽을 수 없습니다: {e}")
    
    # 환경변수 예시
    with st.expander("🔧 환경변수 설정 예시"):
        st.code("""
# API 키 설정
YAHOO_FINANCE_API_KEY=your_api_key_here
ALPHA_VANTAGE_API_KEY=your_api_key_here

# 성능 설정
MAX_WORKERS=4
CACHE_TTL=3600

# 로깅 설정
LOG_LEVEL=INFO

# 보안 설정
SECRET_KEY=your_very_secure_secret_key_here
        """, language='bash')

def show_logging_settings():
    """로깅 설정 섹션"""
    st.markdown("### 📋 로깅 설정")
    
    # 로그 분석
    st.subheader("📊 로그 분석")
    
    analysis_hours = st.selectbox(
        "분석 기간 (시간)",
        [1, 6, 24, 72, 168],  # 1시간, 6시간, 1일, 3일, 1주일
        index=2  # 기본값: 24시간
    )
    
    if st.button("🔍 로그 분석 실행"):
        with st.spinner("로그 분석 중..."):
            analysis = analyze_logs(analysis_hours)
            
            if 'error' in analysis:
                st.error(f"로그 분석 실패: {analysis['error']}")
            else:
                # 분석 결과 표시
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("총 로그 수", analysis['total_entries'])
                
                with col2:
                    error_count = analysis['levels'].get('ERROR', 0) + analysis['levels'].get('CRITICAL', 0)
                    st.metric("에러 수", error_count)
                
                with col3:
                    st.metric("성능 이슈", len(analysis['performance_issues']))
                
                with col4:
                    st.metric("사용자 액션", len(analysis['user_actions']))
                
                # 로그 레벨별 분포
                if analysis['levels']:
                    st.subheader("📈 로그 레벨 분포")
                    import plotly.express as px
                    import pandas as pd
                    
                    levels_df = pd.DataFrame([
                        {'Level': level, 'Count': count} 
                        for level, count in analysis['levels'].items()
                    ])
                    
                    fig = px.bar(levels_df, x='Level', y='Count', 
                                title="로그 레벨별 분포")
                    st.plotly_chart(fig, use_container_width=True)
                
                # 최근 에러 목록
                if analysis['errors']:
                    st.subheader("🚨 최근 에러")
                    for error in analysis['errors'][-5:]:  # 최근 5개만
                        with st.expander(f"❌ {error.get('timestamp', 'Unknown time')} - {error.get('level', 'ERROR')}"):
                            st.json(error)
                
                # 성능 이슈
                if analysis['performance_issues']:
                    st.subheader("⚡ 성능 이슈")
                    for issue in analysis['performance_issues'][-5:]:  # 최근 5개만
                        duration = issue.get('extra', {}).get('duration_seconds', 0)
                        operation = issue.get('extra', {}).get('operation', 'Unknown')
                        st.warning(f"🐌 {operation}: {duration:.2f}초")
    
    st.markdown("---")
    
    # 로그 뷰어
    show_log_viewer()
    
    st.markdown("---")
    
    # 로깅 설정
    st.subheader("⚙️ 로깅 설정")
    
    logging_config = env_manager.get_logging_config()
    
    st.write("**현재 로깅 설정:**")
    st.json(logging_config)
    
    st.info("""
    📋 **로깅 설정 변경 방법:**
    
    .env 파일에서 다음 변수를 설정하세요:
    
    - `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR, CRITICAL
    - `LOG_FILE`: 로그 파일 경로 (예: logs/app.log)
    
    변경 후 앱을 재시작해주세요.
    """)

def show_performance_settings():
    """성능 설정 섹션"""
    st.markdown("### ⚡ 성능 모니터링 & 최적화")
    
    # 성능 최적화 상태
    st.subheader("🔧 성능 최적화 설정")
    
    from utils.performance_optimizer import performance_optimizer
    from utils.logger import analyze_logs
    import multiprocessing
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 현재 설정 표시
        st.metric("최대 워커 수", performance_optimizer.max_workers)
        st.metric("CPU 코어 수", multiprocessing.cpu_count())
        st.metric("기본 청크 크기", f"{performance_optimizer.chunk_size:,}")
        
        # 워커 수 조정
        new_workers = st.slider(
            "최대 워커 수 조정",
            min_value=1,
            max_value=multiprocessing.cpu_count(),
            value=performance_optimizer.max_workers,
            help="병렬 처리에 사용할 최대 워커 수"
        )
        
        if new_workers != performance_optimizer.max_workers:
            if st.button("워커 수 적용"):
                performance_optimizer.set_max_workers(new_workers)
                st.success(f"워커 수가 {new_workers}개로 설정되었습니다!")
                st.rerun()
    
    with col2:
        # 청크 크기 조정
        new_chunk_size = st.number_input(
            "청크 크기 설정",
            min_value=1000,
            max_value=100000,
            value=performance_optimizer.chunk_size,
            step=5000,
            help="대용량 데이터 처리 시 사용할 청크 크기"
        )
        
        if new_chunk_size != performance_optimizer.chunk_size:
            if st.button("청크 크기 적용"):
                performance_optimizer.set_chunk_size(new_chunk_size)
                st.success(f"청크 크기가 {new_chunk_size:,}로 설정되었습니다!")
                st.rerun()
    
    st.markdown("---")
    
    # 성능 분석
    st.subheader("📊 성능 분석")
    
    analysis_period = st.selectbox(
        "분석 기간",
        [1, 6, 24, 72],
        index=2,
        format_func=lambda x: f"{x}시간"
    )
    
    if st.button("🔍 성능 분석 실행"):
        with st.spinner("성능 분석 중..."):
            analysis = analyze_logs(analysis_period)
            
            if 'error' in analysis:
                st.error(f"분석 실패: {analysis['error']}")
            else:
                # 성능 메트릭 표시
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("총 연산 수", len(analysis.get('performance_issues', []) + 
                                           [entry for entry in analysis.get('user_actions', []) 
                                            if 'duration' in entry.get('extra', {})]))
                
                with col2:
                    perf_issues = analysis.get('performance_issues', [])
                    avg_duration = 0
                    if perf_issues:
                        durations = [issue.get('extra', {}).get('duration_seconds', 0) for issue in perf_issues]
                        avg_duration = sum(durations) / len(durations) if durations else 0
                    st.metric("평균 처리 시간", f"{avg_duration:.2f}초")
                
                with col3:
                    slow_ops = len([issue for issue in perf_issues 
                                  if issue.get('extra', {}).get('duration_seconds', 0) > 5.0])
                    st.metric("느린 연산 수", slow_ops)
                
                with col4:
                    memory_ops = len([entry for entry in analysis.get('user_actions', [])
                                    if 'memory' in entry.get('extra', {}).get('operation', '').lower()])
                    st.metric("메모리 최적화", memory_ops)
                
                # 성능 이슈 상세
                if analysis.get('performance_issues'):
                    st.subheader("🐌 성능 이슈")
                    
                    for issue in analysis['performance_issues'][-10:]:  # 최근 10개
                        duration = issue.get('extra', {}).get('duration_seconds', 0)
                        operation = issue.get('extra', {}).get('operation', 'Unknown')
                        timestamp = issue.get('timestamp', 'Unknown')
                        
                        with st.expander(f"⚠️ {operation} - {duration:.2f}초 ({timestamp})"):
                            st.json(issue)
    
    st.markdown("---")
    
    # 메모리 사용량 모니터링
    st.subheader("💾 메모리 모니터링")
    
    if 'processed_data' in st.session_state:
        data = st.session_state['processed_data']
        memory_info = performance_optimizer.get_memory_usage(data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 메모리", f"{memory_info['total_memory_mb']:.1f} MB")
        with col2:
            st.metric("데이터 행 수", f"{memory_info['rows']:,}")
        with col3:
            st.metric("행당 메모리", f"{memory_info['memory_per_row_bytes']:.0f} bytes")
        
        # 컬럼별 메모리 사용량
        with st.expander("📋 컬럼별 메모리 상세"):
            column_data = []
            for col_name, col_info in memory_info['column_details'].items():
                column_data.append({
                    '컬럼': col_name,
                    '메모리(MB)': f"{col_info['memory_mb']:.2f}",
                    '데이터타입': col_info['dtype'],
                    '고유값': col_info['unique_values'],
                    '결측치': col_info['null_count']
                })
            
            if column_data:
                import pandas as pd
                df_memory = pd.DataFrame(column_data)
                st.dataframe(df_memory, use_container_width=True)
    else:
        st.info("📊 데이터를 업로드하면 메모리 사용량을 분석할 수 있습니다.")
    
    # 최적화 권장 사항
    st.subheader("💡 최적화 권장 사항")
    
    recommendations = [
        "🔹 **대용량 데이터**: 10만 행 이상의 데이터는 자동으로 청크 처리됩니다",
        "🔹 **메모리 최적화**: 데이터 타입이 자동으로 최적화되어 메모리 사용량이 줄어듭니다",
        "🔹 **병렬 처리**: CPU 코어 수에 따라 자동으로 병렬 처리가 적용됩니다",
        "🔹 **캐시 활용**: 기술적 지표 계산 결과가 캐시되어 재사용됩니다",
        "🔹 **성능 모니터링**: 모든 연산의 성능이 자동으로 모니터링되고 로그에 기록됩니다"
    ]
    
    for rec in recommendations:
        st.markdown(rec)
    
    st.info("""
    ⚡ **성능 최적화 팁:**
    - 워커 수는 CPU 코어 수의 75% 정도가 적당합니다
    - 청크 크기는 메모리 용량에 따라 조정하세요 (기본: 10,000행)
    - 정기적으로 성능 분석을 실행하여 병목 지점을 파악하세요
    """)

if __name__ == "__main__":
    show_page() 
