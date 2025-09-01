"""
실시간 업데이트 시스템
WebSocket을 사용한 실시간 진행 상황 및 결과 업데이트
"""

import json
import time
import asyncio
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import uuid

# WebSocket 관련 imports (선택적)
try:
    import websockets
    import asyncio
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

import streamlit as st
from utils.logger import get_logger
from utils.error_handler import handle_error, ErrorCategory, ErrorSeverity


class MessageType(Enum):
    """메시지 타입"""
    TASK_UPDATE = "task_update"
    PROGRESS = "progress"
    RESULT = "result"
    ERROR = "error"
    STATUS = "status"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"


class UpdateType(Enum):
    """업데이트 타입"""
    DATA_PROCESSING = "data_processing"
    FACTOR_MINING = "factor_mining"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    CHART_UPDATE = "chart_update"
    TABLE_UPDATE = "table_update"
    METRIC_UPDATE = "metric_update"


@dataclass
class RealtimeMessage:
    """실시간 메시지"""
    id: str
    type: MessageType
    update_type: UpdateType
    timestamp: datetime
    data: Dict[str, Any]
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class RealtimeUpdateManager:
    """실시간 업데이트 관리자"""
    
    def __init__(self, websocket_port: int = 8765):
        """초기화"""
        self.logger = get_logger("realtime_updates")
        self.websocket_port = websocket_port
        self.use_websocket = WEBSOCKET_AVAILABLE
        
        # 연결된 클라이언트 관리
        self.connected_clients: Dict[str, Any] = {}
        self.message_queues: Dict[str, queue.Queue] = {}
        
        # 구독자 관리
        self.subscribers: Dict[UpdateType, List[str]] = {}
        
        # 메시지 히스토리 (최근 100개)
        self.message_history: List[RealtimeMessage] = []
        self.max_history = 100
        
        # WebSocket 서버 (비동기)
        self.websocket_server = None
        self.server_task = None
        
        # Streamlit 통합을 위한 상태
        self.streamlit_updates: Dict[str, Any] = {}
        
        if self.use_websocket:
            self._start_websocket_server()
        else:
            self.logger.warning("WebSocket not available, using polling mode")
    
    def _start_websocket_server(self):
        """WebSocket 서버 시작"""
        if not WEBSOCKET_AVAILABLE:
            return
        
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def handle_client(websocket, path):
                client_id = str(uuid.uuid4())
                self.connected_clients[client_id] = {
                    'websocket': websocket,
                    'connected_at': datetime.now(),
                    'last_heartbeat': datetime.now()
                }
                
                self.logger.info(f"Client connected: {client_id}")
                
                try:
                    # 환영 메시지
                    await websocket.send(json.dumps({
                        'type': 'connection',
                        'client_id': client_id,
                        'timestamp': datetime.now().isoformat()
                    }))
                    
                    # 메시지 처리 루프
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self._handle_client_message(client_id, data)
                        except json.JSONDecodeError:
                            self.logger.warning(f"Invalid JSON from client {client_id}")
                        except Exception as e:
                            self.logger.error(f"Error handling message from {client_id}: {e}")
                
                except websockets.exceptions.ConnectionClosed:
                    pass
                except Exception as e:
                    self.logger.error(f"WebSocket error for client {client_id}: {e}")
                finally:
                    # 클라이언트 연결 해제
                    if client_id in self.connected_clients:
                        del self.connected_clients[client_id]
                    if client_id in self.message_queues:
                        del self.message_queues[client_id]
                    
                    self.logger.info(f"Client disconnected: {client_id}")
            
            # 서버 시작
            start_server = websockets.serve(
                handle_client, 
                "localhost", 
                self.websocket_port
            )
            
            self.websocket_server = loop.run_until_complete(start_server)
            self.logger.info(f"WebSocket server started on port {self.websocket_port}")
            
            # 하트비트 태스크
            loop.create_task(self._heartbeat_task())
            
            loop.run_forever()
        
        # 백그라운드 스레드에서 실행
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
    
    async def _handle_client_message(self, client_id: str, data: Dict[str, Any]):
        """클라이언트 메시지 처리"""
        message_type = data.get('type')
        
        if message_type == 'subscribe':
            # 구독 요청
            update_types = data.get('update_types', [])
            for update_type_str in update_types:
                try:
                    update_type = UpdateType(update_type_str)
                    if update_type not in self.subscribers:
                        self.subscribers[update_type] = []
                    if client_id not in self.subscribers[update_type]:
                        self.subscribers[update_type].append(client_id)
                except ValueError:
                    self.logger.warning(f"Invalid update type: {update_type_str}")
        
        elif message_type == 'unsubscribe':
            # 구독 해제
            update_types = data.get('update_types', [])
            for update_type_str in update_types:
                try:
                    update_type = UpdateType(update_type_str)
                    if update_type in self.subscribers and client_id in self.subscribers[update_type]:
                        self.subscribers[update_type].remove(client_id)
                except ValueError:
                    pass
        
        elif message_type == 'heartbeat':
            # 하트비트 응답
            if client_id in self.connected_clients:
                self.connected_clients[client_id]['last_heartbeat'] = datetime.now()
    
    async def _heartbeat_task(self):
        """하트비트 태스크"""
        while True:
            try:
                await asyncio.sleep(30)  # 30초마다
                
                current_time = datetime.now()
                disconnected_clients = []
                
                for client_id, client_info in self.connected_clients.items():
                    last_heartbeat = client_info['last_heartbeat']
                    if (current_time - last_heartbeat).total_seconds() > 60:  # 1분 타임아웃
                        disconnected_clients.append(client_id)
                    else:
                        # 하트비트 전송
                        try:
                            await client_info['websocket'].send(json.dumps({
                                'type': 'heartbeat',
                                'timestamp': current_time.isoformat()
                            }))
                        except Exception:
                            disconnected_clients.append(client_id)
                
                # 연결 해제된 클라이언트 정리
                for client_id in disconnected_clients:
                    if client_id in self.connected_clients:
                        del self.connected_clients[client_id]
                
            except Exception as e:
                self.logger.error(f"Heartbeat task error: {e}")
    
    def send_update(
        self,
        update_type: UpdateType,
        data: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """업데이트 전송"""
        try:
            # 메시지 생성
            message = RealtimeMessage(
                id=str(uuid.uuid4()),
                type=MessageType.TASK_UPDATE,
                update_type=update_type,
                timestamp=datetime.now(),
                data=data,
                session_id=session_id,
                user_id=user_id
            )
            
            # 히스토리에 추가
            self._add_to_history(message)
            
            # Streamlit 상태 업데이트
            self._update_streamlit_state(message)
            
            # WebSocket 클라이언트에 전송
            if self.use_websocket:
                self._send_to_websocket_clients(message)
            
            self.logger.debug(f"Update sent: {update_type.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to send update: {e}")
            handle_error(
                e,
                ErrorCategory.SYSTEM,
                ErrorSeverity.MEDIUM,
                context={'update_type': update_type.value, 'data': str(data)[:200]}
            )
    
    def _add_to_history(self, message: RealtimeMessage):
        """메시지 히스토리에 추가"""
        self.message_history.append(message)
        
        # 히스토리 크기 제한
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]
    
    def _update_streamlit_state(self, message: RealtimeMessage):
        """Streamlit 상태 업데이트"""
        update_key = f"{message.update_type.value}_{message.session_id}"
        self.streamlit_updates[update_key] = {
            'message': message,
            'timestamp': message.timestamp,
            'data': message.data
        }
        
        # Streamlit 세션 상태에도 저장
        if hasattr(st, 'session_state'):
            if 'realtime_updates' not in st.session_state:
                st.session_state['realtime_updates'] = {}
            
            st.session_state['realtime_updates'][update_key] = message.data
    
    def _send_to_websocket_clients(self, message: RealtimeMessage):
        """WebSocket 클라이언트에 메시지 전송"""
        if not self.use_websocket or message.update_type not in self.subscribers:
            return
        
        # 구독자에게 전송
        subscribers = self.subscribers[message.update_type]
        message_json = json.dumps({
            'id': message.id,
            'type': message.type.value,
            'update_type': message.update_type.value,
            'timestamp': message.timestamp.isoformat(),
            'data': message.data,
            'session_id': message.session_id,
            'user_id': message.user_id
        })
        
        disconnected_clients = []
        
        for client_id in subscribers:
            if client_id in self.connected_clients:
                try:
                    client_ws = self.connected_clients[client_id]['websocket']
                    # 비동기 전송을 동기 방식으로 처리
                    asyncio.run(client_ws.send(message_json))
                except Exception as e:
                    self.logger.warning(f"Failed to send message to client {client_id}: {e}")
                    disconnected_clients.append(client_id)
        
        # 연결 해제된 클라이언트 정리
        for client_id in disconnected_clients:
            if client_id in self.connected_clients:
                del self.connected_clients[client_id]
            for sub_list in self.subscribers.values():
                if client_id in sub_list:
                    sub_list.remove(client_id)
    
    def send_progress_update(
        self,
        task_id: str,
        progress: float,
        current_step: str,
        session_id: Optional[str] = None
    ):
        """진행 상황 업데이트 전송"""
        self.send_update(
            update_type=UpdateType.DATA_PROCESSING,
            data={
                'task_id': task_id,
                'progress': progress,
                'current_step': current_step,
                'timestamp': datetime.now().isoformat()
            },
            session_id=session_id
        )
    
    def send_result_update(
        self,
        task_type: str,
        result_data: Dict[str, Any],
        session_id: Optional[str] = None
    ):
        """결과 업데이트 전송"""
        update_type_mapping = {
            'data_processing': UpdateType.DATA_PROCESSING,
            'factor_mining': UpdateType.FACTOR_MINING,
            'performance_analysis': UpdateType.PERFORMANCE_ANALYSIS
        }
        
        update_type = update_type_mapping.get(task_type, UpdateType.DATA_PROCESSING)
        
        self.send_update(
            update_type=update_type,
            data={
                'type': 'result',
                'task_type': task_type,
                'result': result_data,
                'timestamp': datetime.now().isoformat()
            },
            session_id=session_id
        )
    
    def send_chart_update(
        self,
        chart_id: str,
        chart_data: Dict[str, Any],
        session_id: Optional[str] = None
    ):
        """차트 업데이트 전송"""
        self.send_update(
            update_type=UpdateType.CHART_UPDATE,
            data={
                'chart_id': chart_id,
                'chart_data': chart_data,
                'timestamp': datetime.now().isoformat()
            },
            session_id=session_id
        )
    
    def send_notification(
        self,
        title: str,
        message: str,
        level: str = 'info',
        session_id: Optional[str] = None
    ):
        """알림 메시지 전송"""
        message_obj = RealtimeMessage(
            id=str(uuid.uuid4()),
            type=MessageType.NOTIFICATION,
            update_type=UpdateType.DATA_PROCESSING,  # 기본값
            timestamp=datetime.now(),
            data={
                'title': title,
                'message': message,
                'level': level,
                'timestamp': datetime.now().isoformat()
            },
            session_id=session_id
        )
        
        self._add_to_history(message_obj)
        self._update_streamlit_state(message_obj)
        
        if self.use_websocket:
            self._send_to_websocket_clients(message_obj)
    
    def get_latest_updates(self, session_id: Optional[str] = None) -> List[RealtimeMessage]:
        """최신 업데이트 조회"""
        if session_id:
            return [msg for msg in self.message_history 
                   if msg.session_id == session_id or msg.session_id is None]
        return self.message_history
    
    def get_client_count(self) -> int:
        """연결된 클라이언트 수 조회"""
        return len(self.connected_clients)
    
    def get_subscription_stats(self) -> Dict[str, int]:
        """구독 통계 조회"""
        stats = {}
        for update_type, subscribers in self.subscribers.items():
            stats[update_type.value] = len(subscribers)
        return stats


# 글로벌 실시간 업데이트 관리자
realtime_manager = RealtimeUpdateManager()


# Streamlit 통합 함수들
def show_realtime_status():
    """실시간 상태 표시"""
    st.subheader("📡 실시간 연결 상태")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("연결된 클라이언트", realtime_manager.get_client_count())
    
    with col2:
        st.metric("메시지 히스토리", len(realtime_manager.message_history))
    
    with col3:
        ws_status = "활성" if realtime_manager.use_websocket else "비활성"
        st.metric("WebSocket", ws_status)
    
    # 구독 통계
    subscription_stats = realtime_manager.get_subscription_stats()
    if subscription_stats:
        st.subheader("📊 구독 통계")
        for update_type, count in subscription_stats.items():
            st.text(f"{update_type}: {count}명")


def show_realtime_updates():
    """실시간 업데이트 표시"""
    st.subheader("🔄 실시간 업데이트")
    
    # 세션 ID 가져오기
    session_id = st.session_state.get('session_id')
    
    # 최신 업데이트 조회
    updates = realtime_manager.get_latest_updates(session_id)
    
    if not updates:
        st.info("업데이트가 없습니다.")
        return
    
    # 최신 10개 업데이트 표시
    for update in reversed(updates[-10:]):
        with st.expander(
            f"{update.update_type.value} - {update.timestamp.strftime('%H:%M:%S')}",
            expanded=False
        ):
            st.json(update.data)


def setup_realtime_polling():
    """실시간 폴링 설정 (WebSocket 미사용시)"""
    if not realtime_manager.use_websocket:
        # 폴링 방식으로 업데이트 확인
        if st.checkbox("실시간 업데이트 활성화 (3초마다 새로고침)"):
            time.sleep(3)
            st.experimental_rerun()


# JavaScript 클라이언트 코드 생성
def generate_websocket_client_code(websocket_url: str = "ws://localhost:8765") -> str:
    """WebSocket 클라이언트 JavaScript 코드 생성"""
    return f"""
    <script>
    class AlphaForgeWebSocket {{
        constructor(url) {{
            this.url = url;
            this.socket = null;
            this.reconnectInterval = 5000;
            this.isConnected = false;
        }}
        
        connect() {{
            try {{
                this.socket = new WebSocket(this.url);
                
                this.socket.onopen = (event) => {{
                    console.log('WebSocket connected');
                    this.isConnected = true;
                    this.onConnected();
                }};
                
                this.socket.onmessage = (event) => {{
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                }};
                
                this.socket.onclose = (event) => {{
                    console.log('WebSocket disconnected');
                    this.isConnected = false;
                    this.onDisconnected();
                    
                    // 자동 재연결
                    setTimeout(() => {{
                        this.connect();
                    }}, this.reconnectInterval);
                }};
                
                this.socket.onerror = (error) => {{
                    console.error('WebSocket error:', error);
                }};
            }} catch (error) {{
                console.error('Failed to connect WebSocket:', error);
            }}
        }}
        
        subscribe(updateTypes) {{
            if (this.isConnected) {{
                this.socket.send(JSON.stringify({{
                    type: 'subscribe',
                    update_types: updateTypes
                }}));
            }}
        }}
        
        handleMessage(data) {{
            switch(data.type) {{
                case 'task_update':
                    this.onTaskUpdate(data);
                    break;
                case 'notification':
                    this.onNotification(data);
                    break;
                case 'heartbeat':
                    // 하트비트 응답
                    this.socket.send(JSON.stringify({{
                        type: 'heartbeat'
                    }}));
                    break;
                default:
                    console.log('Unknown message type:', data.type);
            }}
        }}
        
        onConnected() {{
            // 구독 설정
            this.subscribe([
                'data_processing',
                'factor_mining', 
                'performance_analysis',
                'chart_update'
            ]);
            
            // 연결 상태 표시
            const statusElement = document.getElementById('ws-status');
            if (statusElement) {{
                statusElement.textContent = '연결됨';
                statusElement.className = 'status-connected';
            }}
        }}
        
        onDisconnected() {{
            const statusElement = document.getElementById('ws-status');
            if (statusElement) {{
                statusElement.textContent = '연결 해제';
                statusElement.className = 'status-disconnected';
            }}
        }}
        
        onTaskUpdate(data) {{
            console.log('Task update:', data);
            
            // 진행률 업데이트
            if (data.data.progress !== undefined) {{
                const progressElement = document.getElementById('progress-bar');
                if (progressElement) {{
                    progressElement.value = data.data.progress * 100;
                }}
                
                const progressText = document.getElementById('progress-text');
                if (progressText) {{
                    progressText.textContent = `${{(data.data.progress * 100).toFixed(1)}}% - ${{data.data.current_step}}`;
                }}
            }}
            
            // 결과 업데이트
            if (data.data.type === 'result') {{
                this.onResultUpdate(data.data.result);
            }}
        }}
        
        onNotification(data) {{
            console.log('Notification:', data);
            
            // 알림 표시
            const notification = document.createElement('div');
            notification.className = `notification notification-${{data.data.level}}`;
            notification.innerHTML = `
                <strong>${{data.data.title}}</strong>
                <p>${{data.data.message}}</p>
            `;
            
            const container = document.getElementById('notifications');
            if (container) {{
                container.appendChild(notification);
                
                // 5초 후 자동 제거
                setTimeout(() => {{
                    notification.remove();
                }}, 5000);
            }}
        }}
        
        onResultUpdate(result) {{
            // Streamlit 페이지 새로고침 트리거
            if (window.parent) {{
                window.parent.postMessage({{
                    type: 'streamlit:refresh'
                }}, '*');
            }}
        }}
    }}
    
    // WebSocket 클라이언트 시작
    const wsClient = new AlphaForgeWebSocket('{websocket_url}');
    wsClient.connect();
    </script>
    
    <style>
    .status-connected {{
        color: green;
        font-weight: bold;
    }}
    
    .status-disconnected {{
        color: red;
        font-weight: bold;
    }}
    
    .notification {{
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        border-left: 4px solid;
    }}
    
    .notification-info {{
        background-color: #e3f2fd;
        border-color: #2196f3;
    }}
    
    .notification-success {{
        background-color: #e8f5e8;
        border-color: #4caf50;
    }}
    
    .notification-warning {{
        background-color: #fff3cd;
        border-color: #ff9800;
    }}
    
    .notification-error {{
        background-color: #ffebee;
        border-color: #f44336;
    }}
    </style>
    
    <div id="ws-status">연결 중...</div>
    <div id="notifications"></div>
    """


def embed_websocket_client():
    """WebSocket 클라이언트를 Streamlit 페이지에 임베드"""
    if realtime_manager.use_websocket:
        websocket_url = f"ws://localhost:{realtime_manager.websocket_port}"
        client_code = generate_websocket_client_code(websocket_url)
        st.components.v1.html(client_code, height=100)