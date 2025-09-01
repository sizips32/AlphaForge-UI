"""
현대적 대시보드 컴포넌트 시스템
반응형 레이아웃, 동적 테마, 인터랙티브 요소 제공
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import json
from datetime import datetime, timedelta
import uuid

class DashboardTheme(Enum):
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"

class LayoutType(Enum):
    GRID = "grid"
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    CUSTOM = "custom"

class ComponentType(Enum):
    METRIC_CARD = "metric_card"
    CHART = "chart"
    TABLE = "table"
    FILTER = "filter"
    ALERT = "alert"
    PROGRESS = "progress"

class ModernDashboard:
    """현대적 대시보드 컴포넌트 시스템"""
    
    def __init__(self, theme: DashboardTheme = DashboardTheme.AUTO):
        self.theme = theme
        self.components = {}
        self.layout_config = {}
        self.current_theme = self._detect_theme()
        self._initialize_styles()
    
    def _detect_theme(self) -> str:
        """테마 자동 감지"""
        if self.theme == DashboardTheme.AUTO:
            # 시간 기반 자동 테마 (예시)
            current_hour = datetime.now().hour
            return "dark" if 18 <= current_hour or current_hour <= 6 else "light"
        return self.theme.value
    
    def _initialize_styles(self):
        """CSS 스타일 초기화"""
        self.styles = {
            "light": {
                "background": "#ffffff",
                "surface": "#f8f9fa",
                "primary": "#2563eb",
                "secondary": "#64748b",
                "accent": "#7c3aed",
                "text": "#1e293b",
                "text_secondary": "#64748b",
                "border": "#e2e8f0",
                "success": "#059669",
                "warning": "#d97706",
                "error": "#dc2626",
                "shadow": "0 1px 3px 0 rgba(0, 0, 0, 0.1)"
            },
            "dark": {
                "background": "#0f172a",
                "surface": "#1e293b",
                "primary": "#3b82f6",
                "secondary": "#94a3b8",
                "accent": "#8b5cf6",
                "text": "#f1f5f9",
                "text_secondary": "#94a3b8",
                "border": "#334155",
                "success": "#10b981",
                "warning": "#f59e0b",
                "error": "#ef4444",
                "shadow": "0 1px 3px 0 rgba(0, 0, 0, 0.3)"
            }
        }
    
    def get_current_style(self) -> Dict[str, str]:
        """현재 테마 스타일 반환"""
        return self.styles[self.current_theme]
    
    def apply_dashboard_styles(self):
        """대시보드 전역 스타일 적용"""
        style = self.get_current_style()
        
        css = f"""
        <style>
        /* 전역 스타일 */
        .stApp {{
            background-color: {style['background']};
            color: {style['text']};
        }}
        
        /* 메트릭 카드 스타일 */
        .metric-card {{
            background: {style['surface']};
            border: 1px solid {style['border']};
            border-radius: 12px;
            padding: 24px;
            margin: 8px 0;
            box-shadow: {style['shadow']};
            transition: all 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.15);
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {style['primary']};
            margin: 0;
        }}
        
        .metric-label {{
            font-size: 0.875rem;
            color: {style['text_secondary']};
            margin-bottom: 8px;
        }}
        
        .metric-delta {{
            font-size: 0.875rem;
            font-weight: 500;
        }}
        
        .metric-delta.positive {{
            color: {style['success']};
        }}
        
        .metric-delta.negative {{
            color: {style['error']};
        }}
        
        /* 차트 컨테이너 */
        .chart-container {{
            background: {style['surface']};
            border: 1px solid {style['border']};
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
            box-shadow: {style['shadow']};
        }}
        
        /* 필터 패널 */
        .filter-panel {{
            background: {style['surface']};
            border: 1px solid {style['border']};
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
        }}
        
        /* 알림 스타일 */
        .alert {{
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
            border-left: 4px solid;
        }}
        
        .alert.success {{
            background-color: rgba(16, 185, 129, 0.1);
            border-left-color: {style['success']};
            color: {style['success']};
        }}
        
        .alert.warning {{
            background-color: rgba(245, 158, 11, 0.1);
            border-left-color: {style['warning']};
            color: {style['warning']};
        }}
        
        .alert.error {{
            background-color: rgba(239, 68, 68, 0.1);
            border-left-color: {style['error']};
            color: {style['error']};
        }}
        
        /* 사이드바 스타일링 */
        .css-1d391kg {{
            background-color: {style['surface']};
        }}
        
        /* 반응형 그리드 */
        .dashboard-grid {{
            display: grid;
            gap: 16px;
            margin: 16px 0;
        }}
        
        .grid-cols-1 {{ grid-template-columns: 1fr; }}
        .grid-cols-2 {{ grid-template-columns: repeat(2, 1fr); }}
        .grid-cols-3 {{ grid-template-columns: repeat(3, 1fr); }}
        .grid-cols-4 {{ grid-template-columns: repeat(4, 1fr); }}
        
        @media (max-width: 768px) {{
            .grid-cols-2, .grid-cols-3, .grid-cols-4 {{
                grid-template-columns: 1fr;
            }}
        }}
        
        @media (min-width: 769px) and (max-width: 1024px) {{
            .grid-cols-3, .grid-cols-4 {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        
        /* 로딩 애니메이션 */
        .loading-spinner {{
            border: 3px solid {style['border']};
            border-top: 3px solid {style['primary']};
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        /* 사용자 정의 버튼 */
        .custom-button {{
            background-color: {style['primary']};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .custom-button:hover {{
            background-color: {style['accent']};
            transform: translateY(-1px);
        }}
        </style>
        """
        
        st.markdown(css, unsafe_allow_html=True)
    
    def create_metric_card(self, 
                          label: str, 
                          value: Union[int, float, str],
                          delta: Optional[Union[int, float]] = None,
                          delta_color: str = "normal",
                          format_func: Optional[callable] = None) -> str:
        """현대적 메트릭 카드 생성"""
        
        if format_func:
            formatted_value = format_func(value)
        else:
            if isinstance(value, (int, float)):
                if abs(value) >= 1000000:
                    formatted_value = f"{value/1000000:.1f}M"
                elif abs(value) >= 1000:
                    formatted_value = f"{value/1000:.1f}K"
                else:
                    formatted_value = f"{value:,.0f}" if isinstance(value, int) else f"{value:,.2f}"
            else:
                formatted_value = str(value)
        
        delta_html = ""
        if delta is not None:
            delta_class = "positive" if delta > 0 else "negative" if delta < 0 else ""
            delta_symbol = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            delta_html = f'<div class="metric-delta {delta_class}">{delta_symbol} {abs(delta):,.2f}%</div>'
        
        card_html = f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{formatted_value}</div>
            {delta_html}
        </div>
        """
        
        return card_html
    
    def create_responsive_grid(self, components: List[str], cols: int = 3) -> str:
        """반응형 그리드 레이아웃 생성"""
        grid_html = f'<div class="dashboard-grid grid-cols-{cols}">'
        for component in components:
            grid_html += f'<div>{component}</div>'
        grid_html += '</div>'
        return grid_html
    
    def create_chart_container(self, chart_title: str, chart_content: str) -> str:
        """차트 컨테이너 생성"""
        return f"""
        <div class="chart-container">
            <h3 style="margin-top: 0; margin-bottom: 16px;">{chart_title}</h3>
            {chart_content}
        </div>
        """
    
    def create_alert(self, message: str, alert_type: str = "info") -> str:
        """알림 컴포넌트 생성"""
        return f'<div class="alert {alert_type}">{message}</div>'
    
    def create_filter_panel(self, filters: Dict[str, Any]) -> str:
        """필터 패널 생성"""
        panel_html = '<div class="filter-panel"><h4>필터</h4>'
        for filter_name, filter_config in filters.items():
            panel_html += f'<div class="filter-item">{filter_name}: {filter_config}</div>'
        panel_html += '</div>'
        return panel_html

class InteractiveComponents:
    """인터랙티브 컴포넌트 라이브러리"""
    
    @staticmethod
    def create_advanced_multiselect(options: List[str], 
                                  default: List[str] = None,
                                  key: str = None,
                                  help_text: str = None) -> List[str]:
        """고급 멀티셀렉트 컴포넌트"""
        with st.expander("🔍 필터 옵션", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected = st.multiselect(
                    "항목 선택",
                    options=options,
                    default=default or [],
                    key=key,
                    help=help_text
                )
            
            with col2:
                if st.button("전체 선택", key=f"select_all_{key}"):
                    st.session_state[key] = options
                    st.experimental_rerun()
                
                if st.button("전체 해제", key=f"clear_all_{key}"):
                    st.session_state[key] = []
                    st.experimental_rerun()
        
        return selected
    
    @staticmethod
    def create_date_range_picker(start_date: datetime = None,
                               end_date: datetime = None,
                               key: str = None) -> tuple:
        """날짜 범위 선택기"""
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            start = st.date_input(
                "시작 날짜",
                value=start_date or datetime.now() - timedelta(days=30),
                key=f"start_{key}"
            )
        
        with col2:
            end = st.date_input(
                "종료 날짜",
                value=end_date or datetime.now(),
                key=f"end_{key}"
            )
        
        with col3:
            st.write("빠른 선택")
            if st.button("최근 7일", key=f"7days_{key}"):
                end = datetime.now().date()
                start = end - timedelta(days=7)
                st.session_state[f"start_{key}"] = start
                st.session_state[f"end_{key}"] = end
                st.experimental_rerun()
            
            if st.button("최근 30일", key=f"30days_{key}"):
                end = datetime.now().date()
                start = end - timedelta(days=30)
                st.session_state[f"start_{key}"] = start
                st.session_state[f"end_{key}"] = end
                st.experimental_rerun()
        
        return start, end
    
    @staticmethod
    def create_advanced_slider(label: str,
                             min_value: float,
                             max_value: float,
                             default_value: tuple = None,
                             step: float = 0.01,
                             key: str = None) -> tuple:
        """고급 범위 슬라이더"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            values = st.slider(
                label,
                min_value=min_value,
                max_value=max_value,
                value=default_value or (min_value, max_value),
                step=step,
                key=key
            )
        
        with col2:
            st.write("범위 정보")
            st.write(f"최소: {values[0]:.2f}")
            st.write(f"최대: {values[1]:.2f}")
            st.write(f"범위: {values[1] - values[0]:.2f}")
        
        return values

class DashboardLayout:
    """대시보드 레이아웃 매니저"""
    
    def __init__(self, dashboard: ModernDashboard):
        self.dashboard = dashboard
        self.containers = {}
    
    def create_header(self, title: str, subtitle: str = None) -> None:
        """헤더 생성"""
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            st.markdown(f"# {title}")
            if subtitle:
                st.markdown(f"*{subtitle}*")
        
        with col2:
            # 검색 바
            search_query = st.text_input("🔍 검색", placeholder="데이터 검색...", key="header_search")
        
        with col3:
            # 테마 토글
            if st.button("🌓", help="테마 전환"):
                if st.session_state.get('theme') == 'dark':
                    st.session_state['theme'] = 'light'
                else:
                    st.session_state['theme'] = 'dark'
                st.experimental_rerun()
    
    def create_sidebar_navigation(self, pages: Dict[str, str]) -> str:
        """사이드바 네비게이션"""
        with st.sidebar:
            st.markdown("## 📊 AlphaForge UI")
            
            selected_page = st.radio(
                "페이지 선택",
                options=list(pages.keys()),
                format_func=lambda x: f"{pages[x]} {x}"
            )
            
            st.markdown("---")
            
            # 퀵 액세스
            st.markdown("### ⚡ 빠른 액세스")
            if st.button("🔄 데이터 새로고침"):
                st.cache_data.clear()
                st.success("캐시가 지워졌습니다!")
            
            if st.button("📥 데이터 내보내기"):
                st.info("내보내기 기능 준비 중...")
            
            # 시스템 상태
            st.markdown("### 📊 시스템 상태")
            st.metric("메모리 사용량", "75%", "5%")
            st.metric("캐시 적중률", "89%", "3%")
            
        return selected_page
    
    def create_kpi_row(self, metrics: Dict[str, Dict]) -> None:
        """KPI 행 생성"""
        cols = st.columns(len(metrics))
        cards_html = []
        
        for i, (key, metric_data) in enumerate(metrics.items()):
            card_html = self.dashboard.create_metric_card(
                label=metric_data.get('label', key),
                value=metric_data.get('value', 0),
                delta=metric_data.get('delta'),
                format_func=metric_data.get('format_func')
            )
            cards_html.append(card_html)
        
        # 반응형 그리드로 표시
        grid_html = self.dashboard.create_responsive_grid(cards_html, len(metrics))
        st.markdown(grid_html, unsafe_allow_html=True)