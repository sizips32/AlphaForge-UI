"""
AlphaForge-UI 스타일 모듈
커스텀 CSS 스타일과 UI 테마를 관리합니다.
"""

import streamlit as st
from utils.config import COLORS

def apply_custom_styles():
    """커스텀 CSS 스타일을 적용합니다."""
    
    st.markdown(f"""
    <style>
    /* 전체 앱 스타일링 - 밝은 모던 테마 */
    .stApp {{
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #cbd5e1 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }}
    
    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    /* 글로우 효과 */
    .glow {{
        box-shadow: 0 0 20px rgba(30, 136, 229, 0.3), 0 0 40px rgba(59, 130, 246, 0.2), 0 0 60px rgba(99, 102, 241, 0.1);
        animation: glow 2s ease-in-out infinite alternate;
    }}
    
    @keyframes glow {{
        from {{ box-shadow: 0 0 20px rgba(30, 136, 229, 0.3), 0 0 40px rgba(59, 130, 246, 0.2), 0 0 60px rgba(99, 102, 241, 0.1); }}
        to {{ box-shadow: 0 0 30px rgba(30, 136, 229, 0.4), 0 0 60px rgba(59, 130, 246, 0.3), 0 0 90px rgba(99, 102, 241, 0.2); }}
    }}
    
    /* 사이드바 스타일링 */
    .css-1d391kg {{
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 0, 0, 0.1);
    }}
    
    /* 사이드바 내부 요소 스타일링 */
    .css-1d391kg .stMarkdown {{
        color: #1e293b;
    }}
    
    .css-1d391kg .stButton > button {{
        background: linear-gradient(45deg, {COLORS['primary']}, {COLORS['secondary']});
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 14px;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
    }}
    
    .css-1d391kg .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(30, 136, 229, 0.3);
    }}
    
    /* 사이드바 정보 박스 */
    .css-1d391kg .stAlert {{
        background: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        border: 1px solid rgba(30, 136, 229, 0.2);
        margin: 8px 0;
    }}
    
    /* 사이드바 제목 */
    .css-1d391kg h3 {{
        color: {COLORS['primary']};
        font-size: 16px;
        font-weight: 600;
        margin: 16px 0 8px 0;
    }}
    
    /* 사이드바 캡션 - 이미 위에서 강화됨 */
    
    /* 메인 콘텐츠 영역 - 밝은 글래스모피즘 */
    .main .block-container {{
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(30, 136, 229, 0.3);
        position: relative;
        overflow: hidden;
    }}
    
    .main .block-container::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['secondary']}, #6366f1);
        animation: borderGlow 3s ease-in-out infinite;
    }}
    
    @keyframes borderGlow {{
        0%, 100% {{ opacity: 0.7; }}
        50% {{ opacity: 1; }}
    }}
    
    /* 헤더 스타일링 - 밝은 테마 */
    h1, h2, h3 {{
        color: #1e293b;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(30, 136, 229, 0.2);
    }}
    
    h1 {{
        font-size: 3rem;
        margin-bottom: 2rem;
        background: linear-gradient(45deg, {COLORS['primary']}, {COLORS['secondary']}, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 20px rgba(30, 136, 229, 0.3);
        animation: titleGlow 2s ease-in-out infinite alternate;
    }}
    
    @keyframes titleGlow {{
        from {{ text-shadow: 0 0 20px rgba(30, 136, 229, 0.3); }}
        to {{ text-shadow: 0 0 30px rgba(30, 136, 229, 0.5), 0 0 40px rgba(59, 130, 246, 0.3); }}
    }}
    
    /* 메트릭 카드 스타일링 - 밝은 카드 */
    .metric-card {{
        background: rgba(255, 255, 255, 0.98);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(30, 136, 229, 0.3);
        margin: 15px 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(30, 136, 229, 0.2);
        border-color: {COLORS['primary']};
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(30, 136, 229, 0.1), transparent);
        transition: left 0.5s;
    }}
    
    .metric-card:hover::before {{
        left: 100%;
    }}
    
    /* 차트 컨테이너 - 밝은 글래스모피즘 */
    .chart-container {{
        background: rgba(255, 255, 255, 0.98);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        margin: 15px 0;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(30, 136, 229, 0.3);
    }}
    
    /* 버튼 스타일링 - 밝은 테마 */
    .stButton > button {{
        background: linear-gradient(45deg, {COLORS['primary']}, {COLORS['secondary']}, #6366f1);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 12px 24px;
        font-weight: 700;
        font-size: 16px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }}
    
    .stButton > button::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 10px 25px rgba(30, 136, 229, 0.3), 0 0 20px rgba(59, 130, 246, 0.2);
    }}
    
    .stButton > button:hover::before {{
        left: 100%;
    }}
    
    /* 파일 업로더 스타일링 */
    .stFileUploader {{
        background: white;
        border-radius: 10px;
        padding: 20px;
        border: 2px dashed {COLORS['primary']};
    }}
    
    /* 데이터프레임 스타일링 */
    .dataframe {{
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        background: white;
    }}
    
    /* 탭 스타일링 */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: white;
        border-radius: 8px 8px 0 0;
        border: none;
        color: #64748b;
        font-weight: 500;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {COLORS['primary']};
        color: white;
    }}
    
    /* 프로그레스 바 스타일링 */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['secondary']});
    }}
    
    /* 알림 메시지 스타일링 */
    .stAlert {{
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        background: white;
    }}
    
    /* 익스팬더 스타일링 */
    .streamlit-expanderHeader {{
        background: linear-gradient(45deg, {COLORS['primary']}, {COLORS['secondary']});
        color: white;
        border-radius: 10px;
        font-weight: 600;
    }}
    
    /* 코드 블록 스타일링 */
    .stCodeBlock {{
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }}
    
    /* 테이블 스타일링 */
    .stTable {{
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        background: white;
    }}
    
    /* 선택 박스 스타일링 */
    .stSelectbox > div > div {{
        background: white;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }}
    
    /* 숫자 입력 스타일링 */
    .stNumberInput > div > div > input {{
        background: white;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        color: #1e293b;
    }}
    
    /* 텍스트 입력 스타일링 */
    .stTextInput > div > div > input {{
        background: white;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        color: #1e293b;
    }}
    
    /* 체크박스 스타일링 - 이미 위에서 강화됨 */
    
    /* 라디오 버튼 스타일링 - 이미 위에서 강화됨 */
    
    /* 슬라이더 스타일링 */
    .stSlider > div > div > div > div {{
        background: {COLORS['primary']};
    }}
    
    /* 텍스트 색상 개선 - 더 진하고 도드라지게 */
    .stMarkdown {{
        color: #0f172a !important;
        font-weight: 500;
    }}
    
    .stText {{
        color: #0f172a !important;
        font-weight: 500;
    }}
    
    /* 메트릭 값 색상 */
    .stMetric {{
        color: #0f172a !important;
    }}
    
    .stMetric > div > div > div {{
        color: #0f172a !important;
        font-weight: 600;
    }}
    
    /* 메트릭 라벨 색상 */
    .stMetric > div > div > div:first-child {{
        color: #374151 !important;
        font-weight: 600;
    }}
    
    /* 일반 텍스트 강화 */
    p, span, div {{
        color: #0f172a !important;
    }}
    
    /* 라벨 텍스트 강화 */
    label {{
        color: #0f172a !important;
        font-weight: 600;
    }}
    
    /* 캡션 텍스트 강화 */
    .stCaption {{
        color: #374151 !important;
        font-weight: 500;
    }}
    
    /* 정보 박스 텍스트 강화 */
    .stAlert {{
        color: #0f172a !important;
    }}
    
    .stAlert p {{
        color: #0f172a !important;
        font-weight: 500;
    }}
    
    /* 성공/경고/에러 메시지 텍스트 강화 */
    .stAlert[data-baseweb="notification"] {{
        color: #0f172a !important;
    }}
    
    .stAlert[data-baseweb="notification"] p {{
        color: #0f172a !important;
        font-weight: 500;
    }}
    
    /* 데이터프레임 텍스트 강화 */
    .dataframe {{
        color: #0f172a !important;
    }}
    
    .dataframe th {{
        color: #0f172a !important;
        font-weight: 600;
        background-color: #f8fafc !important;
    }}
    
    .dataframe td {{
        color: #0f172a !important;
        font-weight: 500;
    }}
    
    /* 테이블 텍스트 강화 */
    .stTable {{
        color: #0f172a !important;
    }}
    
    .stTable th {{
        color: #0f172a !important;
        font-weight: 600;
        background-color: #f8fafc !important;
    }}
    
    .stTable td {{
        color: #0f172a !important;
        font-weight: 500;
    }}
    
    /* 입력 필드 라벨 강화 */
    .stTextInput > label, .stNumberInput > label, .stSelectbox > label {{
        color: #0f172a !important;
        font-weight: 600;
    }}
    
    /* 체크박스 라벨 강화 */
    .stCheckbox > label {{
        color: #0f172a !important;
        font-weight: 600;
    }}
    
    /* 라디오 버튼 라벨 강화 */
    .stRadio > label {{
        color: #0f172a !important;
        font-weight: 600;
    }}
    
    /* 슬라이더 라벨 강화 */
    .stSlider > label {{
        color: #0f172a !important;
        font-weight: 600;
    }}
    
    /* 파일 업로더 텍스트 강화 */
    .stFileUploader {{
        color: #0f172a !important;
    }}
    
    .stFileUploader > div > div > div {{
        color: #0f172a !important;
        font-weight: 500;
    }}
    
    /* 익스팬더 내용 텍스트 강화 */
    .streamlit-expanderContent {{
        color: #0f172a !important;
    }}
    
    .streamlit-expanderContent p {{
        color: #0f172a !important;
        font-weight: 500;
    }}
    
    /* 탭 내용 텍스트 강화 */
    .stTabs [data-baseweb="tab-panel"] {{
        color: #0f172a !important;
    }}
    
    .stTabs [data-baseweb="tab-panel"] p {{
        color: #0f172a !important;
        font-weight: 500;
    }}
    
    /* 코드 블록 텍스트 강화 */
    .stCodeBlock {{
        color: #0f172a !important;
    }}
    
    .stCodeBlock pre {{
        color: #0f172a !important;
        font-weight: 500;
    }}
    
    /* 사이드바 텍스트 강화 */
    .css-1d391kg .stMarkdown {{
        color: #0f172a !important;
        font-weight: 500;
    }}
    
    .css-1d391kg .stCaption {{
        color: #374151 !important;
        font-weight: 500;
    }}
    
    .css-1d391kg h3 {{
        color: #1e40af !important;
        font-weight: 700;
    }}
    
    /* 헤더 텍스트 강화 */
    h1, h2, h3, h4, h5, h6 {{
        color: #0f172a !important;
        font-weight: 700;
    }}
    
    /* 링크 텍스트 강화 */
    a {{
        color: #1e40af !important;
        font-weight: 600;
        text-decoration: none;
    }}
    
    a:hover {{
        color: #1d4ed8 !important;
        text-decoration: underline;
    }}
    
    /* 강조 텍스트 */
    strong, b {{
        color: #0f172a !important;
        font-weight: 700;
    }}
    
    /* 인용 텍스트 */
    blockquote {{
        color: #374151 !important;
        font-weight: 500;
        border-left: 4px solid #1e88e5;
        padding-left: 1rem;
        margin: 1rem 0;
    }}
    
    /* 반응형 디자인 */
    @media (max-width: 768px) {{
        .main .block-container {{
            margin: 0.5rem;
            padding: 1rem;
        }}
        
        h1 {{
            font-size: 2rem;
        }}
    }}
    
    /* 애니메이션 효과 */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .main .block-container {{
        animation: fadeIn 0.5s ease-out;
    }}
    
    /* 커스텀 스크롤바 */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: #f1f1f1;
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['primary']};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['secondary']};
    }}
    </style>
    """, unsafe_allow_html=True)

def get_metric_style():
    """메트릭 카드용 스타일을 반환합니다."""
    return """
    <style>
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #1E88E5;
        margin: 10px 0;
        text-align: center;
        color: #0f172a;
        font-weight: 600;
    }
    
    .metric-card .metric-value {
        color: #0f172a;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    .metric-card .metric-label {
        color: #374151;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
    """

def get_chart_style():
    """차트 컨테이너용 스타일을 반환합니다."""
    return """
    <style>
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 10px 0;
    }
    </style>
    """ 
