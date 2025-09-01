"""
UX 개선 컴포넌트 라이브러리
향상된 사용자 경험을 위한 인터랙티브 컴포넌트들
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
import time
import uuid
from dataclasses import dataclass
from enum import Enum
import json
import asyncio

class NotificationType(Enum):
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"

class ToastPosition(Enum):
    TOP_RIGHT = "top-right"
    TOP_LEFT = "top-left"
    BOTTOM_RIGHT = "bottom-right"
    BOTTOM_LEFT = "bottom-left"
    TOP_CENTER = "top-center"
    BOTTOM_CENTER = "bottom-center"

@dataclass
class Notification:
    id: str
    type: NotificationType
    title: str
    message: str
    timestamp: datetime
    duration: int = 5000  # milliseconds
    action_label: Optional[str] = None
    action_callback: Optional[Callable] = None

class NotificationManager:
    """알림 관리 시스템"""
    
    def __init__(self):
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
        if 'notification_settings' not in st.session_state:
            st.session_state.notification_settings = {
                'enabled': True,
                'position': ToastPosition.TOP_RIGHT,
                'default_duration': 5000
            }
    
    def add_notification(self, 
                        type: NotificationType, 
                        title: str, 
                        message: str,
                        duration: int = None,
                        action_label: str = None,
                        action_callback: Callable = None) -> str:
        """알림 추가"""
        notification_id = str(uuid.uuid4())
        
        notification = Notification(
            id=notification_id,
            type=type,
            title=title,
            message=message,
            timestamp=datetime.now(),
            duration=duration or st.session_state.notification_settings['default_duration'],
            action_label=action_label,
            action_callback=action_callback
        )
        
        st.session_state.notifications.append(notification)
        return notification_id
    
    def remove_notification(self, notification_id: str):
        """알림 제거"""
        st.session_state.notifications = [
            n for n in st.session_state.notifications 
            if n.id != notification_id
        ]
    
    def render_notifications(self):
        """알림 렌더링"""
        if not st.session_state.notification_settings['enabled']:
            return
        
        notifications = st.session_state.notifications.copy()
        current_time = datetime.now()
        
        # 만료된 알림 제거
        active_notifications = [
            n for n in notifications
            if (current_time - n.timestamp).total_seconds() * 1000 < n.duration
        ]
        
        st.session_state.notifications = active_notifications
        
        if not active_notifications:
            return
        
        # 알림 CSS 스타일
        position = st.session_state.notification_settings['position']
        position_styles = self._get_position_styles(position)
        
        notifications_html = f"""
        <div id="notification-container" style="{position_styles}">
        """
        
        for notification in active_notifications[-5:]:  # 최대 5개만 표시
            icon = self._get_notification_icon(notification.type)
            color = self._get_notification_color(notification.type)
            
            action_button = ""
            if notification.action_label:
                action_button = f"""
                <button onclick="handleNotificationAction('{notification.id}')" 
                        style="background: none; border: 1px solid currentColor; 
                               color: inherit; padding: 4px 12px; border-radius: 4px;
                               font-size: 0.8em; cursor: pointer; margin-left: 8px;">
                    {notification.action_label}
                </button>
                """
            
            notifications_html += f"""
            <div class="notification notification-{notification.type.value}" 
                 style="background: {color['bg']}; border-left: 4px solid {color['border']};
                        color: {color['text']}; padding: 12px 16px; margin-bottom: 8px;
                        border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        display: flex; align-items: center; justify-content: space-between;
                        animation: slideIn 0.3s ease-out;">
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 1.2em; margin-right: 8px;">{icon}</span>
                    <div>
                        <div style="font-weight: 600; margin-bottom: 2px;">{notification.title}</div>
                        <div style="font-size: 0.9em; opacity: 0.9;">{notification.message}</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center;">
                    {action_button}
                    <button onclick="removeNotification('{notification.id}')" 
                            style="background: none; border: none; color: inherit; 
                                   font-size: 1.2em; cursor: pointer; margin-left: 8px;">
                        ×
                    </button>
                </div>
            </div>
            """
        
        notifications_html += """
        </div>
        
        <style>
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        .notification:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        </style>
        
        <script>
        function removeNotification(id) {
            // Streamlit에서 제거하는 로직은 세션 상태로 처리
            console.log('Remove notification:', id);
        }
        
        function handleNotificationAction(id) {
            console.log('Handle notification action:', id);
        }
        </script>
        """
        
        st.markdown(notifications_html, unsafe_allow_html=True)
    
    def _get_position_styles(self, position: ToastPosition) -> str:
        """위치별 스타일"""
        base_style = "position: fixed; z-index: 1000; max-width: 400px; min-width: 300px;"
        
        position_map = {
            ToastPosition.TOP_RIGHT: "top: 20px; right: 20px;",
            ToastPosition.TOP_LEFT: "top: 20px; left: 20px;",
            ToastPosition.BOTTOM_RIGHT: "bottom: 20px; right: 20px;",
            ToastPosition.BOTTOM_LEFT: "bottom: 20px; left: 20px;",
            ToastPosition.TOP_CENTER: "top: 20px; left: 50%; transform: translateX(-50%);",
            ToastPosition.BOTTOM_CENTER: "bottom: 20px; left: 50%; transform: translateX(-50%);"
        }
        
        return base_style + position_map.get(position, position_map[ToastPosition.TOP_RIGHT])
    
    def _get_notification_icon(self, type: NotificationType) -> str:
        """알림 타입별 아이콘"""
        icons = {
            NotificationType.SUCCESS: "✅",
            NotificationType.WARNING: "⚠️",
            NotificationType.ERROR: "❌",
            NotificationType.INFO: "ℹ️"
        }
        return icons.get(type, "📢")
    
    def _get_notification_color(self, type: NotificationType) -> Dict[str, str]:
        """알림 타입별 색상"""
        colors = {
            NotificationType.SUCCESS: {
                "bg": "rgba(16, 185, 129, 0.1)",
                "border": "#10b981",
                "text": "#065f46"
            },
            NotificationType.WARNING: {
                "bg": "rgba(245, 158, 11, 0.1)",
                "border": "#f59e0b",
                "text": "#92400e"
            },
            NotificationType.ERROR: {
                "bg": "rgba(239, 68, 68, 0.1)",
                "border": "#ef4444",
                "text": "#991b1b"
            },
            NotificationType.INFO: {
                "bg": "rgba(59, 130, 246, 0.1)",
                "border": "#3b82f6",
                "text": "#1e3a8a"
            }
        }
        return colors.get(type, colors[NotificationType.INFO])

class ProgressTracker:
    """진행률 추적기"""
    
    def __init__(self, title: str, total_steps: int):
        self.title = title
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
        self.step_times = []
        
        # 진행률 컨테이너 생성
        self.container = st.empty()
        self.status_container = st.empty()
    
    def update(self, step: int, message: str = ""):
        """진행률 업데이트"""
        self.current_step = step
        self.step_times.append(datetime.now())
        
        progress_percentage = (step / self.total_steps) * 100
        
        # 예상 완료 시간 계산
        if len(self.step_times) > 1:
            avg_step_time = sum([
                (self.step_times[i] - self.step_times[i-1]).total_seconds()
                for i in range(1, len(self.step_times))
            ]) / (len(self.step_times) - 1)
            
            remaining_steps = self.total_steps - step
            eta = datetime.now() + timedelta(seconds=avg_step_time * remaining_steps)
            eta_text = eta.strftime("%H:%M:%S")
        else:
            eta_text = "계산 중..."
        
        # 진행률 바
        self.container.progress(progress_percentage / 100, text=f"{self.title} - {step}/{self.total_steps} ({progress_percentage:.1f}%)")
        
        # 상태 정보
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        status_html = f"""
        <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; margin: 8px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>{message or f"단계 {step}/{self.total_steps}"}</strong>
                </div>
                <div style="text-align: right; font-size: 0.9em; color: #666;">
                    <div>경과 시간: {elapsed_time:.1f}초</div>
                    <div>예상 완료: {eta_text}</div>
                </div>
            </div>
        </div>
        """
        
        self.status_container.markdown(status_html, unsafe_allow_html=True)
    
    def complete(self, success_message: str = "완료!"):
        """완료 처리"""
        self.container.progress(1.0, text=f"{self.title} - 완료!")
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        completion_html = f"""
        <div style="background: #d1fae5; border: 1px solid #10b981; padding: 12px; 
                    border-radius: 8px; margin: 8px 0;">
            <div style="color: #065f46;">
                <strong>✅ {success_message}</strong>
                <br>
                <small>총 소요 시간: {total_time:.1f}초</small>
            </div>
        </div>
        """
        
        self.status_container.markdown(completion_html, unsafe_allow_html=True)

class LoadingSpinner:
    """로딩 스피너"""
    
    def __init__(self, message: str = "로딩 중..."):
        self.message = message
        self.container = st.empty()
        self.is_active = False
    
    def start(self):
        """스피너 시작"""
        self.is_active = True
        spinner_html = f"""
        <div style="display: flex; flex-direction: column; align-items: center; 
                    padding: 20px; margin: 20px 0;">
            <div style="border: 4px solid #f3f3f3; border-top: 4px solid #3498db;
                        border-radius: 50%; width: 40px; height: 40px;
                        animation: spin 1s linear infinite;">
            </div>
            <div style="margin-top: 16px; font-weight: 500; color: #666;">
                {self.message}
            </div>
        </div>
        
        <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        """
        
        self.container.markdown(spinner_html, unsafe_allow_html=True)
    
    def stop(self):
        """스피너 중지"""
        self.is_active = False
        self.container.empty()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

class InteractiveTable:
    """인터랙티브 테이블"""
    
    def __init__(self, data: pd.DataFrame, key: str = None):
        self.data = data
        self.key = key or str(uuid.uuid4())
        self.filtered_data = data.copy()
    
    def render(self, 
               show_filters: bool = True,
               show_export: bool = True,
               show_pagination: bool = True,
               page_size: int = 50,
               selectable: bool = False) -> Dict[str, Any]:
        """테이블 렌더링"""
        
        result = {
            'filtered_data': self.filtered_data,
            'selected_rows': [],
            'export_data': None
        }
        
        # 상단 컨트롤
        if show_filters or show_export:
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                if show_filters:
                    search_query = st.text_input(
                        "🔍 검색", 
                        placeholder="테이블 검색...",
                        key=f"table_search_{self.key}"
                    )
                    
                    if search_query:
                        self.filtered_data = self._apply_search(search_query)
            
            with col2:
                if show_filters and len(self.data.select_dtypes(include=['object']).columns) > 0:
                    categorical_cols = self.data.select_dtypes(include=['object']).columns
                    filter_col = st.selectbox(
                        "필터 컬럼",
                        ["전체"] + categorical_cols.tolist(),
                        key=f"table_filter_col_{self.key}"
                    )
                    
                    if filter_col != "전체":
                        unique_values = self.data[filter_col].unique()
                        selected_values = st.multiselect(
                            "값 선택",
                            unique_values,
                            default=unique_values,
                            key=f"table_filter_values_{self.key}"
                        )
                        
                        if selected_values != list(unique_values):
                            self.filtered_data = self.filtered_data[
                                self.filtered_data[filter_col].isin(selected_values)
                            ]
            
            with col3:
                if show_export:
                    export_format = st.selectbox(
                        "내보내기",
                        ["선택", "CSV", "Excel", "JSON"],
                        key=f"table_export_{self.key}"
                    )
                    
                    if export_format != "선택":
                        result['export_data'] = {
                            'format': export_format,
                            'data': self.filtered_data
                        }
        
        # 테이블 정보
        total_rows = len(self.data)
        filtered_rows = len(self.filtered_data)
        
        info_html = f"""
        <div style="background: #f8f9fa; padding: 8px 12px; border-radius: 4px; 
                    margin: 8px 0; font-size: 0.9em; color: #666;">
            표시 중: {filtered_rows:,}개 (전체 {total_rows:,}개 중)
            {f" | 컬럼: {len(self.filtered_data.columns)}개" if not self.filtered_data.empty else ""}
        </div>
        """
        st.markdown(info_html, unsafe_allow_html=True)
        
        # 페이지네이션
        if show_pagination and len(self.filtered_data) > page_size:
            total_pages = (len(self.filtered_data) - 1) // page_size + 1
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                page_number = st.number_input(
                    "페이지",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    key=f"table_page_{self.key}"
                )
            
            with col2:
                st.write(f"페이지 {page_number} / {total_pages}")
            
            with col3:
                page_size = st.selectbox(
                    "페이지 크기",
                    [25, 50, 100, 200],
                    index=1,
                    key=f"table_page_size_{self.key}"
                )
            
            start_idx = (page_number - 1) * page_size
            end_idx = start_idx + page_size
            display_data = self.filtered_data.iloc[start_idx:end_idx]
        else:
            display_data = self.filtered_data
        
        # 테이블 표시
        if selectable:
            # 선택 가능한 테이블
            selection = st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True,
                selection_mode="multi-row",
                on_select="rerun",
                key=f"table_selection_{self.key}"
            )
            
            if hasattr(selection, 'selection') and selection.selection.get('rows'):
                selected_indices = selection.selection['rows']
                result['selected_rows'] = display_data.iloc[selected_indices]
        else:
            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )
        
        result['filtered_data'] = self.filtered_data
        return result
    
    def _apply_search(self, query: str) -> pd.DataFrame:
        """검색 적용"""
        if not query:
            return self.data
        
        masks = []
        for col in self.data.select_dtypes(include=['object', 'string']).columns:
            mask = self.data[col].astype(str).str.contains(query, case=False, na=False)
            masks.append(mask)
        
        if masks:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask | mask
            
            return self.data[combined_mask]
        
        return self.data

class SmartSuggestions:
    """스마트 제안 시스템"""
    
    def __init__(self):
        if 'user_actions' not in st.session_state:
            st.session_state.user_actions = []
        if 'suggestion_settings' not in st.session_state:
            st.session_state.suggestion_settings = {
                'enabled': True,
                'frequency': 'smart',  # always, smart, minimal
                'categories': ['performance', 'workflow', 'insights']
            }
    
    def log_action(self, action_type: str, context: Dict[str, Any]):
        """사용자 액션 로깅"""
        action = {
            'type': action_type,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'session_id': st.session_state.get('session_id', 'unknown')
        }
        
        st.session_state.user_actions.append(action)
        
        # 최근 100개 액션만 유지
        if len(st.session_state.user_actions) > 100:
            st.session_state.user_actions = st.session_state.user_actions[-100:]
    
    def generate_suggestions(self, current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """제안 생성"""
        if not st.session_state.suggestion_settings['enabled']:
            return []
        
        suggestions = []
        recent_actions = st.session_state.user_actions[-10:]  # 최근 10개 액션
        
        # 성능 제안
        if 'performance' in st.session_state.suggestion_settings['categories']:
            perf_suggestions = self._generate_performance_suggestions(recent_actions, current_context)
            suggestions.extend(perf_suggestions)
        
        # 워크플로우 제안
        if 'workflow' in st.session_state.suggestion_settings['categories']:
            workflow_suggestions = self._generate_workflow_suggestions(recent_actions, current_context)
            suggestions.extend(workflow_suggestions)
        
        # 인사이트 제안
        if 'insights' in st.session_state.suggestion_settings['categories']:
            insight_suggestions = self._generate_insight_suggestions(recent_actions, current_context)
            suggestions.extend(insight_suggestions)
        
        return suggestions[:5]  # 최대 5개 제안
    
    def render_suggestions(self, context: Dict[str, Any]):
        """제안 렌더링"""
        suggestions = self.generate_suggestions(context)
        
        if not suggestions:
            return
        
        st.markdown("### 💡 스마트 제안")
        
        for i, suggestion in enumerate(suggestions):
            with st.expander(f"💡 {suggestion['title']}", expanded=i == 0):
                st.write(suggestion['description'])
                
                if suggestion.get('action_label'):
                    if st.button(suggestion['action_label'], key=f"suggestion_action_{i}"):
                        if suggestion.get('action_callback'):
                            suggestion['action_callback']()
                        st.success("제안이 적용되었습니다!")
                
                # 제안 평가
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("👍 도움됨", key=f"suggestion_helpful_{i}"):
                        self._rate_suggestion(suggestion['id'], 'helpful')
                with col2:
                    if st.button("👎 도움안됨", key=f"suggestion_not_helpful_{i}"):
                        self._rate_suggestion(suggestion['id'], 'not_helpful')
                with col3:
                    if st.button("🚫 제안 숨기기", key=f"suggestion_hide_{i}"):
                        self._hide_suggestion(suggestion['id'])
    
    def _generate_performance_suggestions(self, recent_actions: List[Dict], context: Dict) -> List[Dict]:
        """성능 개선 제안"""
        suggestions = []
        
        # 데이터 크기 기반 제안
        if context.get('data_size', 0) > 10000:
            suggestions.append({
                'id': 'perf_large_data',
                'title': '대용량 데이터 최적화',
                'description': '현재 데이터가 크니 샘플링이나 배치 처리를 고려해보세요.',
                'category': 'performance',
                'action_label': '샘플링 적용',
                'action_callback': lambda: st.info("데이터 샘플링이 적용됩니다.")
            })
        
        # 반복적인 계산 감지
        calc_actions = [a for a in recent_actions if a['type'] in ['calculate', 'analyze']]
        if len(calc_actions) > 3:
            suggestions.append({
                'id': 'perf_caching',
                'title': '계산 결과 캐싱',
                'description': '반복적인 계산을 감지했습니다. 캐싱을 활성화하면 속도가 향상됩니다.',
                'category': 'performance',
                'action_label': '캐싱 활성화',
                'action_callback': lambda: st.info("캐싱이 활성화됩니다.")
            })
        
        return suggestions
    
    def _generate_workflow_suggestions(self, recent_actions: List[Dict], context: Dict) -> List[Dict]:
        """워크플로우 개선 제안"""
        suggestions = []
        
        # 필터 사용 패턴 분석
        filter_actions = [a for a in recent_actions if a['type'] == 'filter']
        if len(filter_actions) > 2:
            suggestions.append({
                'id': 'workflow_save_filter',
                'title': '필터 저장',
                'description': '자주 사용하는 필터 조건을 저장하여 빠르게 재사용할 수 있습니다.',
                'category': 'workflow',
                'action_label': '현재 필터 저장',
                'action_callback': lambda: st.info("필터가 저장됩니다.")
            })
        
        return suggestions
    
    def _generate_insight_suggestions(self, recent_actions: List[Dict], context: Dict) -> List[Dict]:
        """인사이트 제안"""
        suggestions = []
        
        # 데이터 패턴 기반 제안
        if context.get('has_time_series'):
            suggestions.append({
                'id': 'insight_trend_analysis',
                'title': '트렌드 분석',
                'description': '시계열 데이터가 감지되었습니다. 트렌드 분석을 수행해보세요.',
                'category': 'insights',
                'action_label': '트렌드 분석 시작',
                'action_callback': lambda: st.info("트렌드 분석을 시작합니다.")
            })
        
        return suggestions
    
    def _rate_suggestion(self, suggestion_id: str, rating: str):
        """제안 평가"""
        if 'suggestion_ratings' not in st.session_state:
            st.session_state.suggestion_ratings = {}
        
        st.session_state.suggestion_ratings[suggestion_id] = {
            'rating': rating,
            'timestamp': datetime.now().isoformat()
        }
    
    def _hide_suggestion(self, suggestion_id: str):
        """제안 숨기기"""
        if 'hidden_suggestions' not in st.session_state:
            st.session_state.hidden_suggestions = set()
        
        st.session_state.hidden_suggestions.add(suggestion_id)

# 전역 인스턴스
notification_manager = NotificationManager()
smart_suggestions = SmartSuggestions()

def create_success_toast(title: str, message: str):
    """성공 토스트 생성"""
    notification_manager.add_notification(NotificationType.SUCCESS, title, message)

def create_warning_toast(title: str, message: str):
    """경고 토스트 생성"""
    notification_manager.add_notification(NotificationType.WARNING, title, message)

def create_error_toast(title: str, message: str):
    """에러 토스트 생성"""
    notification_manager.add_notification(NotificationType.ERROR, title, message)

def create_info_toast(title: str, message: str):
    """정보 토스트 생성"""
    notification_manager.add_notification(NotificationType.INFO, title, message)