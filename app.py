import streamlit as st
import streamlit_option_menu as option_menu
from pages import data_management, factor_mining, dynamic_combination, backtesting, reporting, settings
import utils.config as config
import utils.styles as styles

# 페이지 설정
st.set_page_config(
    page_title="AlphaForge-UI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 적용
styles.apply_custom_styles()

def main():
    """AlphaForge-UI 메인 애플리케이션"""
    
    # 사이드바 네비게이션
    with st.sidebar:
        # 로고 또는 제목 표시
        if config.LOGO_PATH and config.LOGO_PATH.exists():
            try:
                st.image(str(config.LOGO_PATH), width=200)
            except Exception as e:
                st.title("🚀 AlphaForge-UI")
        else:
            st.title("🚀 AlphaForge-UI")
        
        # 메뉴 구성
        selected = option_menu.option_menu(
            menu_title="Navigation",
            options=["📊 Dashboard", "📈 데이터 관리", "🧠 팩터 마이닝", "⚖️ 동적 결합", "📊 백테스팅", "📋 리포트", "⚙️ 설정"],
            icons=["house", "database", "brain", "gear", "graph-up", "file-text", "settings"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "18px"},
                "nav-link": {
                    "color": "#424242",
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee"
                },
                "nav-link-selected": {"background-color": "#1E88E5", "color": "white"},
            }
        )
        
        # 사이드바 하단 정보
        st.markdown("---")
        show_sidebar_info()
    
    # 메인 콘텐츠 영역
    if selected == "📊 Dashboard":
        show_dashboard()
    elif selected == "📈 데이터 관리":
        data_management.show_page()
    elif selected == "🧠 팩터 마이닝":
        factor_mining.show_page()
    elif selected == "⚖️ 동적 결합":
        dynamic_combination.show_page()
    elif selected == "📊 백테스팅":
        backtesting.show_page()
    elif selected == "📋 리포트":
        reporting.show_page()
    elif selected == "⚙️ 설정":
        settings.show_page()

def show_sidebar_info():
    """사이드바 하단에 정보를 표시합니다."""
    # 현재 데이터 상태
    if 'processed_data' in st.session_state:
        data = st.session_state['processed_data']
        st.success(f"✅ 처리된 데이터: {len(data):,}행")
        if 'Ticker' in data.columns:
            st.success(f"📈 종목: {data['Ticker'].nunique()}개")
    elif 'uploaded_data' in st.session_state:
        data = st.session_state['uploaded_data']
        st.info(f"📊 업로드된 데이터: {len(data):,}행")
        if 'Ticker' in data.columns:
            st.info(f"📈 종목: {data['Ticker'].nunique()}개")
        st.warning("⚠️ 데이터 처리가 필요합니다")
    
    # 팩터 마이닝 상태
    if 'mining_results' in st.session_state:
        mining_results = st.session_state['mining_results']
        st.success(f"🧠 팩터: {len(mining_results['factors'])}개")
    elif 'processed_data' in st.session_state:
        st.info("🧠 팩터 마이닝 필요")
    
    # 동적 결합 상태
    if 'combination_results' in st.session_state:
        st.success("⚖️ 동적 결합 완료")
    elif 'mining_results' in st.session_state:
        st.info("⚖️ 동적 결합 필요")
    
    # 시스템 상태
    st.markdown("### 🔧 시스템 상태")
    st.success("✅ 정상 작동")
    
    # 버전 정보
    st.markdown("### 📋 버전 정보")
    st.caption("AlphaForge-UI v1.0.0")
    st.caption("Python 3.13.0")
    
    # 도움말 링크
    st.markdown("### 💡 도움말")
    if st.button("📖 사용 가이드", use_container_width=True):
        st.info("사용 가이드를 확인하려면 대시보드로 이동하세요.")

def show_dashboard():
    """대시보드 페이지"""
    # 헤더 섹션
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 3.5rem; margin-bottom: 1rem;">🚀 AlphaForge-UI</h1>
        <p style="font-size: 1.5rem; color: #CBD5E1; margin-bottom: 2rem;">AI 기반 알파 팩터 발굴 플랫폼</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 상단 메트릭 카드들
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h3 style="color: #10B981; font-size: 2rem;">🎯</h3>
            <h4 style="color: #F8FAFC;">AI 팩터</h4>
            <p style="color: #CBD5E1; font-size: 1.2rem;">고품질 알파</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h3 style="color: #3B82F6; font-size: 2rem;">⚡</h3>
            <h4 style="color: #F8FAFC;">실시간</h4>
            <p style="color: #CBD5E1; font-size: 1.2rem;">동적 최적화</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h3 style="color: #EC4899; font-size: 2rem;">📊</h3>
            <h4 style="color: #F8FAFC;">백테스팅</h4>
            <p style="color: #CBD5E1; font-size: 1.2rem;">성과 분석</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h3 style="color: #F59E0B; font-size: 2rem;">🎨</h3>
            <h4 style="color: #F8FAFC;">시각화</h4>
            <p style="color: #CBD5E1; font-size: 1.2rem;">인터랙티브</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 메인 콘텐츠 섹션
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="chart-container">
            <h3 style="color: #A855F7; margin-bottom: 1rem;">🎯 AlphaForge 소개</h3>
            <div style="background: rgba(168, 85, 247, 0.1); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #A855F7;">
                <h4 style="color: #F8FAFC; margin-bottom: 1rem;">🚀 2단계 프레임워크</h4>
                <ul style="color: #CBD5E1; line-height: 1.8;">
                    <li><strong>팩터 마이닝</strong>: AI 기반 알파 팩터 자동 발굴</li>
                    <li><strong>동적 결합</strong>: 시장 변화에 적응하는 스마트 포트폴리오</li>
                    <li><strong>생성-예측 신경망</strong>: IC 최적화 기반 고품질 팩터 생성</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="chart-container">
            <h3 style="color: #EC4899; margin-bottom: 1rem;">📊 핵심 성과 지표</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div style="background: rgba(236, 72, 153, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(236, 72, 153, 0.3);">
                    <h5 style="color: #F8FAFC; margin-bottom: 0.5rem;">IC</h5>
                    <p style="color: #CBD5E1; font-size: 0.9rem;">팩터 예측력 측정</p>
                </div>
                <div style="background: rgba(59, 130, 246, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.3);">
                    <h5 style="color: #F8FAFC; margin-bottom: 0.5rem;">ICIR</h5>
                    <p style="color: #CBD5E1; font-size: 0.9rem;">팩터 안정성 평가</p>
                </div>
                <div style="background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.3);">
                    <h5 style="color: #F8FAFC; margin-bottom: 0.5rem;">RankIC</h5>
                    <p style="color: #CBD5E1; font-size: 0.9rem;">극값에 강건한 측정</p>
                </div>
                <div style="background: rgba(245, 158, 11, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(245, 158, 11, 0.3);">
                    <h5 style="color: #F8FAFC; margin-bottom: 0.5rem;">Sharpe</h5>
                    <p style="color: #CBD5E1; font-size: 0.9rem;">리스크 조정 수익률</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="chart-container">
            <h3 style="color: #10B981; margin-bottom: 1rem;">🛠️ 사용 가이드</h3>
            <div style="background: rgba(16, 185, 129, 0.1); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #10B981;">
                <ol style="color: #CBD5E1; line-height: 1.8;">
                    <li><strong>데이터 업로드</strong>: 주가 데이터 준비</li>
                    <li><strong>팩터 설정</strong>: 원하는 팩터 선택</li>
                    <li><strong>마이닝 실행</strong>: AI 기반 자동 발굴</li>
                    <li><strong>백테스팅</strong>: 성과 분석</li>
                    <li><strong>최적화</strong>: 동적 가중치 조정</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="chart-container">
            <h3 style="color: #F59E0B; margin-bottom: 1rem;">💡 Pro Tips</h3>
            <div style="background: rgba(245, 158, 11, 0.1); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #F59E0B;">
                <ul style="color: #CBD5E1; line-height: 1.6;">
                    <li>팩터 풀 크기: 10개 내외</li>
                    <li>IC > 0.02, ICIR > 0.5</li>
                    <li>월 단위 리밸런싱</li>
                    <li>리스크 관리 중요</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 퀵 액션 버튼
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0 2rem 0;">
        <h2 style="color: #F8FAFC; font-size: 2.5rem; margin-bottom: 1rem;">🚀 퀵 액션</h2>
        <p style="color: #CBD5E1; font-size: 1.2rem;">원하는 기능을 바로 시작하세요</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📁 데이터 업로드", use_container_width=True):
            st.switch_page("pages/data_management.py")
    
    with col2:
        if st.button("🧠 팩터 마이닝", use_container_width=True):
            st.switch_page("pages/factor_mining.py")
    
    with col3:
        if st.button("⚖️ 동적 결합", use_container_width=True):
            st.switch_page("pages/dynamic_combination.py")
    
    with col4:
        if st.button("📊 백테스팅", use_container_width=True):
            st.switch_page("pages/backtesting.py")
    
    # 최근 활동 및 통계
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0 2rem 0;">
        <h2 style="color: #F8FAFC; font-size: 2.5rem; margin-bottom: 1rem;">📊 실시간 상태</h2>
        <p style="color: #CBD5E1; font-size: 1.2rem;">현재 시스템 상태와 활동을 확인하세요</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="chart-container">
            <h3 style="color: #EC4899; margin-bottom: 1rem;">📈 최근 활동</h3>
            <div style="background: rgba(236, 72, 153, 0.1); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #EC4899;">
                <div style="text-align: center; padding: 2rem;">
                    <h4 style="color: #F8FAFC; font-size: 3rem; margin-bottom: 1rem;">📁</h4>
                    <p style="color: #CBD5E1; font-size: 1.1rem;">아직 데이터가 없습니다</p>
                    <p style="color: #CBD5E1; font-size: 0.9rem;">데이터를 업로드하여 시작하세요!</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="chart-container">
            <h3 style="color: #10B981; margin-bottom: 1rem;">🔧 시스템 상태</h3>
            <div style="background: rgba(16, 185, 129, 0.1); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #10B981;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div style="text-align: center; padding: 1rem; background: rgba(16, 185, 129, 0.2); border-radius: 8px;">
                        <h5 style="color: #F8FAFC; margin-bottom: 0.5rem;">✅ 상태</h5>
                        <p style="color: #CBD5E1; font-size: 0.9rem;">정상 작동</p>
                    </div>
                    <div style="text-align: center; padding: 1rem; background: rgba(59, 130, 246, 0.2); border-radius: 8px;">
                        <h5 style="color: #F8FAFC; margin-bottom: 0.5rem;">📊 데이터</h5>
                        <p style="color: #CBD5E1; font-size: 0.9rem;">0 행</p>
                    </div>
                    <div style="text-align: center; padding: 1rem; background: rgba(168, 85, 247, 0.2); border-radius: 8px;">
                        <h5 style="color: #F8FAFC; margin-bottom: 0.5rem;">🧠 팩터</h5>
                        <p style="color: #CBD5E1; font-size: 0.9rem;">0 개</p>
                    </div>
                    <div style="text-align: center; padding: 1rem; background: rgba(245, 158, 11, 0.2); border-radius: 8px;">
                        <h5 style="color: #F8FAFC; margin-bottom: 0.5rem;">⚡ 성능</h5>
                        <p style="color: #CBD5E1; font-size: 0.9rem;">최적</p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
