"""
테마 관리 시스템
다크모드, 라이트모드, 커스텀 테마 지원
"""

import streamlit as st
import json
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime
from pathlib import Path

class ThemeMode(Enum):
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"
    CUSTOM = "custom"

class ThemeManager:
    """테마 관리 시스템"""
    
    def __init__(self, config_path: str = ".streamlit/theme_config.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(exist_ok=True)
        self.themes = self._load_default_themes()
        self.custom_themes = self._load_custom_themes()
        self.current_theme = self._detect_current_theme()
    
    def _load_default_themes(self) -> Dict[str, Dict[str, Any]]:
        """기본 테마 로드"""
        return {
            "light": {
                "name": "Light Mode",
                "primaryColor": "#2563eb",
                "backgroundColor": "#ffffff",
                "secondaryBackgroundColor": "#f8f9fa",
                "textColor": "#1e293b",
                "font": "sans serif",
                "base": "light",
                # 추가 색상 팔레트
                "colors": {
                    "surface": "#f8f9fa",
                    "surface_variant": "#e2e8f0",
                    "on_surface": "#1e293b",
                    "on_surface_variant": "#64748b",
                    "primary": "#2563eb",
                    "primary_variant": "#1d4ed8",
                    "on_primary": "#ffffff",
                    "secondary": "#64748b",
                    "secondary_variant": "#475569",
                    "on_secondary": "#ffffff",
                    "accent": "#7c3aed",
                    "accent_variant": "#6d28d9",
                    "on_accent": "#ffffff",
                    "success": "#059669",
                    "warning": "#d97706",
                    "error": "#dc2626",
                    "info": "#0284c7",
                    "border": "#e2e8f0",
                    "divider": "#cbd5e1",
                    "shadow": "rgba(0, 0, 0, 0.1)",
                    "overlay": "rgba(0, 0, 0, 0.5)"
                },
                # 타이포그래피
                "typography": {
                    "h1": {"size": "2.5rem", "weight": "700", "line_height": "1.2"},
                    "h2": {"size": "2rem", "weight": "600", "line_height": "1.25"},
                    "h3": {"size": "1.5rem", "weight": "600", "line_height": "1.3"},
                    "h4": {"size": "1.25rem", "weight": "500", "line_height": "1.4"},
                    "body": {"size": "1rem", "weight": "400", "line_height": "1.5"},
                    "caption": {"size": "0.875rem", "weight": "400", "line_height": "1.4"},
                    "small": {"size": "0.75rem", "weight": "400", "line_height": "1.3"}
                },
                # 그림자
                "shadows": {
                    "small": "0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24)",
                    "medium": "0 3px 6px rgba(0, 0, 0, 0.15), 0 2px 4px rgba(0, 0, 0, 0.12)",
                    "large": "0 10px 20px rgba(0, 0, 0, 0.15), 0 3px 6px rgba(0, 0, 0, 0.10)",
                    "elevated": "0 15px 25px rgba(0, 0, 0, 0.15), 0 5px 10px rgba(0, 0, 0, 0.05)"
                }
            },
            "dark": {
                "name": "Dark Mode",
                "primaryColor": "#3b82f6",
                "backgroundColor": "#0f172a",
                "secondaryBackgroundColor": "#1e293b",
                "textColor": "#f1f5f9",
                "font": "sans serif",
                "base": "dark",
                # 추가 색상 팔레트
                "colors": {
                    "surface": "#1e293b",
                    "surface_variant": "#334155",
                    "on_surface": "#f1f5f9",
                    "on_surface_variant": "#94a3b8",
                    "primary": "#3b82f6",
                    "primary_variant": "#2563eb",
                    "on_primary": "#ffffff",
                    "secondary": "#94a3b8",
                    "secondary_variant": "#64748b",
                    "on_secondary": "#000000",
                    "accent": "#8b5cf6",
                    "accent_variant": "#7c3aed",
                    "on_accent": "#ffffff",
                    "success": "#10b981",
                    "warning": "#f59e0b",
                    "error": "#ef4444",
                    "info": "#06b6d4",
                    "border": "#334155",
                    "divider": "#475569",
                    "shadow": "rgba(0, 0, 0, 0.3)",
                    "overlay": "rgba(0, 0, 0, 0.7)"
                },
                # 타이포그래피 (라이트 모드와 동일)
                "typography": {
                    "h1": {"size": "2.5rem", "weight": "700", "line_height": "1.2"},
                    "h2": {"size": "2rem", "weight": "600", "line_height": "1.25"},
                    "h3": {"size": "1.5rem", "weight": "600", "line_height": "1.3"},
                    "h4": {"size": "1.25rem", "weight": "500", "line_height": "1.4"},
                    "body": {"size": "1rem", "weight": "400", "line_height": "1.5"},
                    "caption": {"size": "0.875rem", "weight": "400", "line_height": "1.4"},
                    "small": {"size": "0.75rem", "weight": "400", "line_height": "1.3"}
                },
                # 그림자
                "shadows": {
                    "small": "0 1px 3px rgba(0, 0, 0, 0.5), 0 1px 2px rgba(0, 0, 0, 0.6)",
                    "medium": "0 3px 6px rgba(0, 0, 0, 0.4), 0 2px 4px rgba(0, 0, 0, 0.5)",
                    "large": "0 10px 20px rgba(0, 0, 0, 0.4), 0 3px 6px rgba(0, 0, 0, 0.3)",
                    "elevated": "0 15px 25px rgba(0, 0, 0, 0.4), 0 5px 10px rgba(0, 0, 0, 0.2)"
                }
            }
        }
    
    def _load_custom_themes(self) -> Dict[str, Dict[str, Any]]:
        """커스텀 테마 로드"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f).get('custom_themes', {})
            except Exception as e:
                st.warning(f"커스텀 테마 로드 실패: {e}")
        return {}
    
    def _detect_current_theme(self) -> str:
        """현재 테마 감지"""
        # 세션 상태에서 테마 확인
        if 'theme_mode' in st.session_state:
            return st.session_state.theme_mode
        
        # 시간 기반 자동 테마
        current_hour = datetime.now().hour
        if 6 <= current_hour < 18:
            return "light"
        else:
            return "dark"
    
    def get_theme(self, theme_name: str = None) -> Dict[str, Any]:
        """테마 정보 반환"""
        theme_name = theme_name or self.current_theme
        
        if theme_name in self.themes:
            return self.themes[theme_name]
        elif theme_name in self.custom_themes:
            return self.custom_themes[theme_name]
        else:
            return self.themes["light"]  # 기본값
    
    def apply_theme(self, theme_name: str = None):
        """테마 적용"""
        theme = self.get_theme(theme_name)
        
        # Streamlit 기본 테마 설정
        st.markdown(f"""
        <style>
        :root {{
            --primary-color: {theme['primaryColor']};
            --background-color: {theme['backgroundColor']};
            --secondary-background-color: {theme['secondaryBackgroundColor']};
            --text-color: {theme['textColor']};
        }}
        </style>
        """, unsafe_allow_html=True)
        
        # 추가 CSS 스타일 적용
        self._apply_advanced_styles(theme)
    
    def _apply_advanced_styles(self, theme: Dict[str, Any]):
        """고급 스타일 적용"""
        colors = theme.get('colors', {})
        typography = theme.get('typography', {})
        shadows = theme.get('shadows', {})
        
        css = f"""
        <style>
        /* 전역 CSS 변수 */
        :root {{
            --surface: {colors.get('surface', '#f8f9fa')};
            --surface-variant: {colors.get('surface_variant', '#e2e8f0')};
            --on-surface: {colors.get('on_surface', '#1e293b')};
            --on-surface-variant: {colors.get('on_surface_variant', '#64748b')};
            --primary: {colors.get('primary', '#2563eb')};
            --primary-variant: {colors.get('primary_variant', '#1d4ed8')};
            --on-primary: {colors.get('on_primary', '#ffffff')};
            --secondary: {colors.get('secondary', '#64748b')};
            --accent: {colors.get('accent', '#7c3aed')};
            --success: {colors.get('success', '#059669')};
            --warning: {colors.get('warning', '#d97706')};
            --error: {colors.get('error', '#dc2626')};
            --info: {colors.get('info', '#0284c7')};
            --border: {colors.get('border', '#e2e8f0')};
            --shadow-small: {shadows.get('small', '0 1px 3px rgba(0,0,0,0.12)')};
            --shadow-medium: {shadows.get('medium', '0 3px 6px rgba(0,0,0,0.15)')};
        }}
        
        /* 앱 전체 스타일 */
        .stApp {{
            background-color: {theme['backgroundColor']};
            color: {theme['textColor']};
        }}
        
        /* 사이드바 */
        .css-1d391kg {{
            background-color: var(--surface);
        }}
        
        /* 메인 컨텐츠 영역 */
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        
        /* 타이포그래피 */
        h1 {{ 
            font-size: {typography.get('h1', {}).get('size', '2.5rem')};
            font-weight: {typography.get('h1', {}).get('weight', '700')};
            line-height: {typography.get('h1', {}).get('line_height', '1.2')};
            color: var(--on-surface);
        }}
        
        h2 {{ 
            font-size: {typography.get('h2', {}).get('size', '2rem')};
            font-weight: {typography.get('h2', {}).get('weight', '600')};
            line-height: {typography.get('h2', {}).get('line_height', '1.25')};
            color: var(--on-surface);
        }}
        
        h3 {{ 
            font-size: {typography.get('h3', {}).get('size', '1.5rem')};
            font-weight: {typography.get('h3', {}).get('weight', '600')};
            line-height: {typography.get('h3', {}).get('line_height', '1.3')};
            color: var(--on-surface);
        }}
        
        /* 카드 스타일 */
        .theme-card {{
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: var(--shadow-small);
            transition: all 0.3s ease;
        }}
        
        .theme-card:hover {{
            box-shadow: var(--shadow-medium);
            transform: translateY(-2px);
        }}
        
        /* 버튼 스타일 */
        .stButton > button {{
            background-color: var(--primary);
            color: var(--on-primary);
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            background-color: var(--primary-variant);
            transform: translateY(-1px);
        }}
        
        /* 메트릭 카드 */
        [data-testid="metric-container"] {{
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: var(--shadow-small);
        }}
        
        /* 선택박스 */
        .stSelectbox > div > div {{
            background-color: var(--surface);
            border-color: var(--border);
        }}
        
        /* 체크박스 */
        .stCheckbox > label {{
            color: var(--on-surface);
        }}
        
        /* 알림 스타일 */
        .alert-success {{
            background-color: rgba(5, 150, 105, 0.1);
            border-left: 4px solid var(--success);
            padding: 1rem;
            border-radius: 4px;
            color: var(--success);
        }}
        
        .alert-warning {{
            background-color: rgba(217, 119, 6, 0.1);
            border-left: 4px solid var(--warning);
            padding: 1rem;
            border-radius: 4px;
            color: var(--warning);
        }}
        
        .alert-error {{
            background-color: rgba(220, 38, 38, 0.1);
            border-left: 4px solid var(--error);
            padding: 1rem;
            border-radius: 4px;
            color: var(--error);
        }}
        
        .alert-info {{
            background-color: rgba(2, 132, 199, 0.1);
            border-left: 4px solid var(--info);
            padding: 1rem;
            border-radius: 4px;
            color: var(--info);
        }}
        
        /* 로딩 스피너 */
        .theme-spinner {{
            border: 3px solid var(--border);
            border-top: 3px solid var(--primary);
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
        
        /* 반응형 디자인 */
        @media (max-width: 768px) {{
            .main .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}
            
            h1 {{ font-size: 2rem; }}
            h2 {{ font-size: 1.5rem; }}
            h3 {{ font-size: 1.25rem; }}
        }}
        </style>
        """)
        
        st.markdown(css, unsafe_allow_html=True)
    
    def create_theme_selector(self) -> str:
        """테마 선택기 위젯"""
        available_themes = {
            **{k: v['name'] for k, v in self.themes.items()},
            **{k: v.get('name', k) for k, v in self.custom_themes.items()}
        }
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            selected_theme = st.selectbox(
                "테마 선택",
                options=list(available_themes.keys()),
                format_func=lambda x: available_themes[x],
                index=list(available_themes.keys()).index(self.current_theme) if self.current_theme in available_themes else 0,
                key="theme_selector"
            )
        
        with col2:
            auto_switch = st.checkbox(
                "자동 테마 전환",
                help="시간에 따라 자동으로 라이트/다크 모드 전환",
                key="auto_theme"
            )
        
        with col3:
            if st.button("적용", key="apply_theme"):
                st.session_state.theme_mode = selected_theme
                if auto_switch:
                    st.session_state.auto_theme = True
                st.experimental_rerun()
        
        return selected_theme
    
    def create_theme_preview(self, theme_name: str):
        """테마 미리보기"""
        theme = self.get_theme(theme_name)
        colors = theme.get('colors', {})
        
        st.markdown("### 테마 미리보기")
        
        # 색상 팔레트
        col1, col2, col3, col4 = st.columns(4)
        
        color_samples = [
            ('Primary', colors.get('primary', '#2563eb')),
            ('Secondary', colors.get('secondary', '#64748b')),
            ('Accent', colors.get('accent', '#7c3aed')),
            ('Success', colors.get('success', '#059669')),
        ]
        
        for i, (name, color) in enumerate(color_samples):
            with [col1, col2, col3, col4][i]:
                st.markdown(f"""
                <div style="
                    background-color: {color};
                    height: 60px;
                    border-radius: 8px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: 500;
                    margin-bottom: 8px;
                ">
                    {name}
                </div>
                <div style="text-align: center; font-size: 0.8em; color: #666;">
                    {color}
                </div>
                """, unsafe_allow_html=True)
        
        # 컴포넌트 미리보기
        st.markdown("#### 컴포넌트 미리보기")
        
        # 메트릭 카드
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 수익률", "24.5%", "2.1%")
        with col2:
            st.metric("샤프 비율", "1.85", "-0.05")
        with col3:
            st.metric("최대 낙폭", "-8.2%", "1.3%")
        
        # 알림 예시
        st.markdown(f"""
        <div class="alert-success">
            ✅ 성공: 포트폴리오가 성공적으로 업데이트되었습니다.
        </div>
        <div class="alert-warning">
            ⚠️ 경고: 리스크가 높은 자산이 포함되어 있습니다.
        </div>
        <div class="alert-error">
            ❌ 오류: 데이터를 불러오는 중 문제가 발생했습니다.
        </div>
        <div class="alert-info">
            ℹ️ 정보: 새로운 팩터가 추가되었습니다.
        </div>
        """, unsafe_allow_html=True)
    
    def save_custom_theme(self, theme_name: str, theme_config: Dict[str, Any]):
        """커스텀 테마 저장"""
        self.custom_themes[theme_name] = theme_config
        
        config = {
            'custom_themes': self.custom_themes,
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            st.success(f"커스텀 테마 '{theme_name}'가 저장되었습니다!")
        except Exception as e:
            st.error(f"테마 저장 실패: {e}")
    
    def create_theme_editor(self):
        """테마 편집기"""
        st.markdown("### 🎨 커스텀 테마 편집기")
        
        with st.expander("새 테마 만들기", expanded=False):
            theme_name = st.text_input("테마 이름", placeholder="my_custom_theme")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 기본 색상")
                primary_color = st.color_picker("Primary Color", "#2563eb")
                background_color = st.color_picker("Background Color", "#ffffff")
                surface_color = st.color_picker("Surface Color", "#f8f9fa")
                text_color = st.color_picker("Text Color", "#1e293b")
            
            with col2:
                st.markdown("#### 액센트 색상")
                accent_color = st.color_picker("Accent Color", "#7c3aed")
                success_color = st.color_picker("Success Color", "#059669")
                warning_color = st.color_picker("Warning Color", "#d97706")
                error_color = st.color_picker("Error Color", "#dc2626")
            
            if st.button("테마 저장") and theme_name:
                custom_theme = {
                    "name": theme_name.title(),
                    "primaryColor": primary_color,
                    "backgroundColor": background_color,
                    "secondaryBackgroundColor": surface_color,
                    "textColor": text_color,
                    "font": "sans serif",
                    "base": "light" if background_color == "#ffffff" else "dark",
                    "colors": {
                        "surface": surface_color,
                        "on_surface": text_color,
                        "primary": primary_color,
                        "accent": accent_color,
                        "success": success_color,
                        "warning": warning_color,
                        "error": error_color
                    }
                }
                
                self.save_custom_theme(theme_name, custom_theme)

class ThemeUtils:
    """테마 유틸리티 함수들"""
    
    @staticmethod
    def get_contrast_color(hex_color: str) -> str:
        """색상에 대한 대비 색상 반환 (흰색 또는 검은색)"""
        # hex를 RGB로 변환
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # 휘도 계산
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        
        return "#000000" if luminance > 0.5 else "#ffffff"
    
    @staticmethod
    def lighten_color(hex_color: str, factor: float = 0.2) -> str:
        """색상을 밝게 만들기"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    @staticmethod
    def darken_color(hex_color: str, factor: float = 0.2) -> str:
        """색상을 어둡게 만들기"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        r = max(0, int(r * (1 - factor)))
        g = max(0, int(g * (1 - factor)))
        b = max(0, int(b * (1 - factor)))
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    @staticmethod
    def create_color_scheme(base_color: str) -> Dict[str, str]:
        """기본 색상에서 색상 스키마 생성"""
        return {
            "primary": base_color,
            "primary_light": ThemeUtils.lighten_color(base_color, 0.3),
            "primary_dark": ThemeUtils.darken_color(base_color, 0.3),
            "contrast": ThemeUtils.get_contrast_color(base_color)
        }

# 전역 테마 매니저 인스턴스
theme_manager = ThemeManager()