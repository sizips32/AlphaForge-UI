"""
현대적 UI 컴포넌트 라이브러리
재사용 가능한 UI 컴포넌트들과 레이아웃 시스템
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
import json
import uuid
from enum import Enum
from dataclasses import dataclass
import base64
from io import BytesIO

class ComponentSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    EXTRA_LARGE = "xl"

class ComponentVariant(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"
    GHOST = "ghost"
    OUTLINE = "outline"

class ButtonState(Enum):
    NORMAL = "normal"
    LOADING = "loading"
    DISABLED = "disabled"

@dataclass
class ComponentTheme:
    primary: str = "#2563eb"
    secondary: str = "#64748b"
    success: str = "#059669"
    warning: str = "#d97706"
    error: str = "#dc2626"
    info: str = "#0284c7"
    background: str = "#ffffff"
    surface: str = "#f8f9fa"
    text: str = "#1e293b"
    text_secondary: str = "#64748b"
    border: str = "#e2e8f0"

class ModernCard:
    """현대적 카드 컴포넌트"""
    
    def __init__(self, 
                 title: Optional[str] = None,
                 subtitle: Optional[str] = None,
                 theme: ComponentTheme = ComponentTheme()):
        self.title = title
        self.subtitle = subtitle
        self.theme = theme
    
    def render(self, 
               content: Union[str, Callable],
               elevation: int = 1,
               hover_effect: bool = True,
               padding: str = "24px",
               border_radius: str = "12px",
               actions: Optional[List[Dict[str, Any]]] = None) -> None:
        """카드 렌더링"""
        
        # 그림자 레벨
        shadows = {
            0: "none",
            1: "0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24)",
            2: "0 3px 6px rgba(0, 0, 0, 0.15), 0 2px 4px rgba(0, 0, 0, 0.12)",
            3: "0 10px 20px rgba(0, 0, 0, 0.15), 0 3px 6px rgba(0, 0, 0, 0.10)"
        }
        
        hover_style = """
        .modern-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15), 0 2px 6px rgba(0, 0, 0, 0.10);
        }
        """ if hover_effect else ""
        
        card_id = str(uuid.uuid4())
        
        card_html = f"""
        <div id="card-{card_id}" class="modern-card" style="
            background: {self.theme.background};
            border: 1px solid {self.theme.border};
            border-radius: {border_radius};
            padding: {padding};
            box-shadow: {shadows.get(elevation, shadows[1])};
            transition: all 0.3s ease;
            margin: 16px 0;
        ">
        """
        
        # 헤더
        if self.title or self.subtitle:
            card_html += '<div style="margin-bottom: 16px;">'
            
            if self.title:
                card_html += f"""
                <h3 style="margin: 0 0 4px 0; color: {self.theme.text}; 
                           font-size: 1.25rem; font-weight: 600;">
                    {self.title}
                </h3>
                """
            
            if self.subtitle:
                card_html += f"""
                <p style="margin: 0; color: {self.theme.text_secondary}; 
                          font-size: 0.875rem;">
                    {self.subtitle}
                </p>
                """
            
            card_html += '</div>'
        
        # 콘텐츠 영역
        card_html += '<div class="card-content">'
        
        # CSS 적용
        st.markdown(f"""
        <style>
        {hover_style}
        </style>
        {card_html}
        """, unsafe_allow_html=True)
        
        # 콘텐츠 렌더링
        if callable(content):
            content()
        else:
            st.markdown(content)
        
        # 액션 버튼들
        if actions:
            st.markdown('<div style="margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap;">')
            cols = st.columns(len(actions))
            
            for i, action in enumerate(actions):
                with cols[i]:
                    if st.button(
                        action['label'],
                        key=f"card_action_{card_id}_{i}",
                        type=action.get('type', 'secondary'),
                        disabled=action.get('disabled', False)
                    ):
                        if action.get('callback'):
                            action['callback']()
            
            st.markdown('</div>')
        
        # 카드 종료
        st.markdown('</div></div>', unsafe_allow_html=True)

class ModernButton:
    """현대적 버튼 컴포넌트"""
    
    def __init__(self, theme: ComponentTheme = ComponentTheme()):
        self.theme = theme
    
    def render(self,
               label: str,
               variant: ComponentVariant = ComponentVariant.PRIMARY,
               size: ComponentSize = ComponentSize.MEDIUM,
               state: ButtonState = ButtonState.NORMAL,
               icon: Optional[str] = None,
               full_width: bool = False,
               on_click: Optional[Callable] = None,
               key: Optional[str] = None) -> bool:
        """버튼 렌더링"""
        
        # 버튼 스타일 정의
        styles = self._get_button_styles(variant, size, state, full_width)
        button_id = key or str(uuid.uuid4())
        
        # 아이콘과 레이블 조합
        button_content = label
        if icon:
            button_content = f"{icon} {label}" if label else icon
        
        # 로딩 상태
        if state == ButtonState.LOADING:
            button_content = "⏳ 로딩 중..."
        
        # 비활성화 상태
        disabled = state in [ButtonState.DISABLED, ButtonState.LOADING]
        
        # Streamlit 버튼으로 렌더링 (커스텀 스타일 적용)
        st.markdown(f"""
        <style>
        .stButton > button[data-testid="baseButton-{variant.value}"] {{
            {styles}
        }}
        </style>
        """, unsafe_allow_html=True)
        
        clicked = st.button(
            button_content,
            disabled=disabled,
            key=button_id,
            type="primary" if variant == ComponentVariant.PRIMARY else "secondary",
            use_container_width=full_width
        )
        
        if clicked and on_click and not disabled:
            on_click()
        
        return clicked
    
    def _get_button_styles(self, 
                          variant: ComponentVariant, 
                          size: ComponentSize, 
                          state: ButtonState,
                          full_width: bool) -> str:
        """버튼 스타일 생성"""
        
        # 크기별 스타일
        size_styles = {
            ComponentSize.SMALL: "padding: 6px 12px; font-size: 0.875rem;",
            ComponentSize.MEDIUM: "padding: 8px 16px; font-size: 1rem;",
            ComponentSize.LARGE: "padding: 12px 24px; font-size: 1.125rem;",
            ComponentSize.EXTRA_LARGE: "padding: 16px 32px; font-size: 1.25rem;"
        }
        
        # 변형별 색상
        variant_styles = {
            ComponentVariant.PRIMARY: f"background-color: {self.theme.primary}; color: white; border: none;",
            ComponentVariant.SECONDARY: f"background-color: {self.theme.secondary}; color: white; border: none;",
            ComponentVariant.SUCCESS: f"background-color: {self.theme.success}; color: white; border: none;",
            ComponentVariant.WARNING: f"background-color: {self.theme.warning}; color: white; border: none;",
            ComponentVariant.ERROR: f"background-color: {self.theme.error}; color: white; border: none;",
            ComponentVariant.INFO: f"background-color: {self.theme.info}; color: white; border: none;",
            ComponentVariant.GHOST: f"background-color: transparent; color: {self.theme.primary}; border: none;",
            ComponentVariant.OUTLINE: f"background-color: transparent; color: {self.theme.primary}; border: 1px solid {self.theme.primary};"
        }
        
        # 상태별 스타일
        state_styles = {
            ButtonState.NORMAL: "",
            ButtonState.LOADING: "opacity: 0.7; cursor: not-allowed;",
            ButtonState.DISABLED: "opacity: 0.5; cursor: not-allowed;"
        }
        
        base_style = """
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        cursor: pointer;
        """
        
        width_style = "width: 100%;" if full_width else ""
        
        return f"""
        {base_style}
        {size_styles[size]}
        {variant_styles[variant]}
        {state_styles[state]}
        {width_style}
        """

class ModernInput:
    """현대적 입력 컴포넌트"""
    
    def __init__(self, theme: ComponentTheme = ComponentTheme()):
        self.theme = theme
    
    def text_input(self,
                   label: str,
                   placeholder: str = "",
                   help_text: Optional[str] = None,
                   error: Optional[str] = None,
                   icon: Optional[str] = None,
                   key: Optional[str] = None) -> str:
        """텍스트 입력"""
        
        # 레이블과 아이콘
        label_html = f"""
        <label style="display: block; margin-bottom: 8px; font-weight: 500; 
                      color: {self.theme.text}; font-size: 0.875rem;">
            {icon + ' ' if icon else ''}{label}
        </label>
        """
        
        st.markdown(label_html, unsafe_allow_html=True)
        
        # 입력 필드
        value = st.text_input(
            "",
            placeholder=placeholder,
            help=help_text,
            key=key,
            label_visibility="collapsed"
        )
        
        # 에러 메시지
        if error:
            st.markdown(f"""
            <div style="color: {self.theme.error}; font-size: 0.75rem; 
                        margin-top: 4px;">
                ⚠️ {error}
            </div>
            """, unsafe_allow_html=True)
        
        return value
    
    def select_input(self,
                     label: str,
                     options: List[Any],
                     format_func: Optional[Callable] = None,
                     help_text: Optional[str] = None,
                     icon: Optional[str] = None,
                     key: Optional[str] = None) -> Any:
        """셀렉트 입력"""
        
        # 레이블
        label_html = f"""
        <label style="display: block; margin-bottom: 8px; font-weight: 500; 
                      color: {self.theme.text}; font-size: 0.875rem;">
            {icon + ' ' if icon else ''}{label}
        </label>
        """
        
        st.markdown(label_html, unsafe_allow_html=True)
        
        # 셀렉트박스
        return st.selectbox(
            "",
            options=options,
            format_func=format_func,
            help=help_text,
            key=key,
            label_visibility="collapsed"
        )
    
    def number_input(self,
                     label: str,
                     min_value: Optional[float] = None,
                     max_value: Optional[float] = None,
                     value: Optional[float] = None,
                     step: Optional[float] = None,
                     format: Optional[str] = None,
                     help_text: Optional[str] = None,
                     icon: Optional[str] = None,
                     key: Optional[str] = None) -> float:
        """숫자 입력"""
        
        # 레이블
        label_html = f"""
        <label style="display: block; margin-bottom: 8px; font-weight: 500; 
                      color: {self.theme.text}; font-size: 0.875rem;">
            {icon + ' ' if icon else ''}{label}
        </label>
        """
        
        st.markdown(label_html, unsafe_allow_html=True)
        
        # 숫자 입력
        return st.number_input(
            "",
            min_value=min_value,
            max_value=max_value,
            value=value,
            step=step,
            format=format,
            help=help_text,
            key=key,
            label_visibility="collapsed"
        )

class ModernLayout:
    """현대적 레이아웃 시스템"""
    
    def __init__(self, theme: ComponentTheme = ComponentTheme()):
        self.theme = theme
    
    def container(self,
                  content: Callable,
                  max_width: str = "1200px",
                  padding: str = "20px",
                  center: bool = True):
        """컨테이너 레이아웃"""
        
        container_style = f"""
        max-width: {max_width};
        padding: {padding};
        """
        
        if center:
            container_style += "margin: 0 auto;"
        
        st.markdown(f"""
        <div style="{container_style}">
        """, unsafe_allow_html=True)
        
        content()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def grid(self,
             content_functions: List[Callable],
             columns: Union[int, List[int]] = None,
             gap: str = "16px"):
        """그리드 레이아웃"""
        
        if columns is None:
            columns = len(content_functions)
        
        if isinstance(columns, int):
            cols = st.columns(columns)
            for i, func in enumerate(content_functions):
                if i < len(cols):
                    with cols[i]:
                        func()
        else:
            cols = st.columns(columns)
            for i, func in enumerate(content_functions):
                if i < len(cols):
                    with cols[i]:
                        func()
    
    def sidebar_layout(self,
                       sidebar_content: Callable,
                       main_content: Callable,
                       sidebar_width: str = "300px"):
        """사이드바 레이아웃"""
        
        with st.sidebar:
            st.markdown(f"""
            <div style="width: {sidebar_width};">
            """, unsafe_allow_html=True)
            
            sidebar_content()
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        main_content()

class ModernDataDisplay:
    """현대적 데이터 표시 컴포넌트"""
    
    def __init__(self, theme: ComponentTheme = ComponentTheme()):
        self.theme = theme
    
    def metric_card(self,
                    title: str,
                    value: Union[str, int, float],
                    delta: Optional[Union[str, int, float]] = None,
                    delta_color: str = "normal",
                    icon: Optional[str] = None,
                    format_func: Optional[Callable] = None) -> None:
        """메트릭 카드"""
        
        # 값 포맷팅
        if format_func:
            formatted_value = format_func(value)
        else:
            if isinstance(value, (int, float)):
                if abs(value) >= 1000000:
                    formatted_value = f"{value/1000000:.1f}M"
                elif abs(value) >= 1000:
                    formatted_value = f"{value/1000:.1f}K"
                else:
                    formatted_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
            else:
                formatted_value = str(value)
        
        # 델타 색상
        delta_colors = {
            "normal": self.theme.text_secondary,
            "positive": self.theme.success,
            "negative": self.theme.error,
            "inverse": self.theme.warning
        }
        
        delta_html = ""
        if delta is not None:
            delta_symbol = "↑" if float(delta) > 0 else "↓" if float(delta) < 0 else "→"
            delta_html = f"""
            <div style="color: {delta_colors.get(delta_color, delta_colors['normal'])}; 
                        font-size: 0.875rem; font-weight: 500; margin-top: 4px;">
                {delta_symbol} {delta}%
            </div>
            """
        
        # 메트릭 카드 HTML
        card_html = f"""
        <div style="
            background: {self.theme.surface};
            border: 1px solid {self.theme.border};
            border-radius: 12px;
            padding: 20px;
            margin: 8px 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        " class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div style="color: {self.theme.text_secondary}; font-size: 0.875rem; 
                                margin-bottom: 4px; font-weight: 500;">
                        {title}
                    </div>
                    <div style="color: {self.theme.primary}; font-size: 2rem; 
                                font-weight: 700; line-height: 1;">
                        {formatted_value}
                    </div>
                    {delta_html}
                </div>
                {f'<div style="font-size: 2rem; opacity: 0.6;">{icon}</div>' if icon else ''}
            </div>
        </div>
        
        <style>
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }}
        </style>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
    
    def progress_ring(self,
                      value: float,
                      max_value: float = 100,
                      size: int = 120,
                      stroke_width: int = 8,
                      color: str = None) -> None:
        """원형 진행률"""
        
        color = color or self.theme.primary
        percentage = (value / max_value) * 100
        
        # SVG 원형 진행률
        radius = (size - stroke_width) / 2
        circumference = 2 * np.pi * radius
        stroke_dasharray = circumference
        stroke_dashoffset = circumference - (percentage / 100) * circumference
        
        ring_html = f"""
        <div style="display: flex; flex-direction: column; align-items: center; margin: 20px 0;">
            <svg width="{size}" height="{size}" style="transform: rotate(-90deg);">
                <circle
                    cx="{size/2}"
                    cy="{size/2}"
                    r="{radius}"
                    stroke="{self.theme.border}"
                    stroke-width="{stroke_width}"
                    fill="none"
                />
                <circle
                    cx="{size/2}"
                    cy="{size/2}"
                    r="{radius}"
                    stroke="{color}"
                    stroke-width="{stroke_width}"
                    stroke-dasharray="{stroke_dasharray}"
                    stroke-dashoffset="{stroke_dashoffset}"
                    stroke-linecap="round"
                    fill="none"
                    style="transition: stroke-dashoffset 0.5s ease-in-out;"
                />
                <text
                    x="{size/2}"
                    y="{size/2 + 6}"
                    text-anchor="middle"
                    style="font-size: {size/6}px; font-weight: 600; fill: {self.theme.text}; 
                           transform: rotate(90deg); transform-origin: {size/2}px {size/2}px;"
                >
                    {percentage:.1f}%
                </text>
            </svg>
        </div>
        """
        
        st.markdown(ring_html, unsafe_allow_html=True)

class ModernNavigation:
    """현대적 네비게이션 컴포넌트"""
    
    def __init__(self, theme: ComponentTheme = ComponentTheme()):
        self.theme = theme
    
    def tab_navigation(self,
                       tabs: List[Dict[str, Any]],
                       active_tab: str = None,
                       key: str = None) -> str:
        """탭 네비게이션"""
        
        tab_names = [tab['name'] for tab in tabs]
        tab_labels = [f"{tab.get('icon', '')} {tab['label']}" for tab in tabs]
        
        # Streamlit 탭 사용
        tab_objects = st.tabs(tab_labels)
        
        for i, (tab, tab_obj) in enumerate(zip(tabs, tab_objects)):
            with tab_obj:
                if tab.get('content'):
                    if callable(tab['content']):
                        tab['content']()
                    else:
                        st.write(tab['content'])
        
        return tab_names[0]  # 기본값
    
    def breadcrumb(self, items: List[Dict[str, Any]]):
        """브레드크럼 네비게이션"""
        
        breadcrumb_html = '<div style="margin: 16px 0; display: flex; align-items: center; flex-wrap: wrap;">'
        
        for i, item in enumerate(items):
            if i > 0:
                breadcrumb_html += f"""
                <span style="margin: 0 8px; color: {self.theme.text_secondary};">
                    /
                </span>
                """
            
            if item.get('link'):
                breadcrumb_html += f"""
                <a href="{item['link']}" style="
                    color: {self.theme.primary}; 
                    text-decoration: none;
                    font-weight: 500;
                " onmouseover="this.style.textDecoration='underline'" 
                  onmouseout="this.style.textDecoration='none'">
                    {item['label']}
                </a>
                """
            else:
                text_color = self.theme.text if i == len(items) - 1 else self.theme.text_secondary
                breadcrumb_html += f"""
                <span style="color: {text_color}; font-weight: {'600' if i == len(items) - 1 else '400'};">
                    {item['label']}
                </span>
                """
        
        breadcrumb_html += '</div>'
        st.markdown(breadcrumb_html, unsafe_allow_html=True)

# 전역 테마 및 컴포넌트 인스턴스
default_theme = ComponentTheme()
ui_card = ModernCard(theme=default_theme)
ui_button = ModernButton(theme=default_theme)
ui_input = ModernInput(theme=default_theme)
ui_layout = ModernLayout(theme=default_theme)
ui_data = ModernDataDisplay(theme=default_theme)
ui_nav = ModernNavigation(theme=default_theme)

# 편의 함수들
def create_card(title: str = None, subtitle: str = None, **kwargs):
    """카드 생성 편의 함수"""
    return ModernCard(title=title, subtitle=subtitle, theme=default_theme)

def create_metric_card(title: str, value: Union[str, int, float], **kwargs):
    """메트릭 카드 생성 편의 함수"""
    ui_data.metric_card(title=title, value=value, **kwargs)

def create_button(label: str, **kwargs) -> bool:
    """버튼 생성 편의 함수"""
    return ui_button.render(label=label, **kwargs)