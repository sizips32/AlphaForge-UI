"""
모니터링 및 알림 시스템
시스템 메트릭 수집, 성능 모니터링, 알림 관리
"""

import streamlit as st
import pandas as pd
import numpy as np
import psutil
import time
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import json
import sqlite3
from pathlib import Path
import logging
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MetricType(Enum):
    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    USER = "user"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"

class AlertStatus(Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class Metric:
    name: str
    type: MetricType
    value: Union[int, float, str]
    timestamp: datetime
    labels: Dict[str, str] = None
    unit: Optional[str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}

@dataclass
class Alert:
    id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    timestamp: datetime
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    resolved_at: Optional[datetime] = None
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}

@dataclass
class AlertRule:
    name: str
    metric_name: str
    condition: str  # >, <, >=, <=, ==, !=
    threshold: float
    severity: AlertSeverity
    duration_seconds: int = 60
    message_template: str = "{metric_name} is {condition} {threshold}"
    enabled: bool = True
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}

class MetricsCollector:
    """메트릭 수집기"""
    
    def __init__(self):
        self.collectors = []
        self.metrics_queue = queue.Queue()
        self.is_running = False
        self.collection_interval = 10  # 10초마다 수집
    
    def add_collector(self, collector_func: Callable[[], List[Metric]]):
        """수집기 함수 추가"""
        self.collectors.append(collector_func)
    
    def start_collection(self):
        """메트릭 수집 시작"""
        if self.is_running:
            return
        
        self.is_running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
    
    def stop_collection(self):
        """메트릭 수집 중지"""
        self.is_running = False
        if hasattr(self, 'collection_thread'):
            self.collection_thread.join(timeout=5)
    
    def _collection_loop(self):
        """메트릭 수집 루프"""
        while self.is_running:
            try:
                timestamp = datetime.now()
                
                for collector in self.collectors:
                    try:
                        metrics = collector()
                        for metric in metrics:
                            self.metrics_queue.put(metric)
                    except Exception as e:
                        logging.error(f"메트릭 수집 오류: {str(e)}")
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logging.error(f"메트릭 수집 루프 오류: {str(e)}")
                time.sleep(self.collection_interval)
    
    def get_metrics(self) -> List[Metric]:
        """수집된 메트릭 반환"""
        metrics = []
        while not self.metrics_queue.empty():
            try:
                metric = self.metrics_queue.get_nowait()
                metrics.append(metric)
            except queue.Empty:
                break
        return metrics

class SystemMetricsCollector:
    """시스템 메트릭 수집기"""
    
    @staticmethod
    def collect_system_metrics() -> List[Metric]:
        """시스템 메트릭 수집"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(Metric(
                name="system_cpu_usage_percent",
                type=MetricType.SYSTEM,
                value=cpu_percent,
                timestamp=timestamp,
                unit="percent"
            ))
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            metrics.append(Metric(
                name="system_memory_usage_percent",
                type=MetricType.SYSTEM,
                value=memory.percent,
                timestamp=timestamp,
                unit="percent"
            ))
            
            metrics.append(Metric(
                name="system_memory_available_gb",
                type=MetricType.SYSTEM,
                value=round(memory.available / (1024**3), 2),
                timestamp=timestamp,
                unit="GB"
            ))
            
            # 디스크 사용률
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(Metric(
                name="system_disk_usage_percent",
                type=MetricType.SYSTEM,
                value=round(disk_percent, 2),
                timestamp=timestamp,
                unit="percent"
            ))
            
            # 네트워크 I/O
            network = psutil.net_io_counters()
            metrics.append(Metric(
                name="system_network_bytes_sent",
                type=MetricType.SYSTEM,
                value=network.bytes_sent,
                timestamp=timestamp,
                unit="bytes"
            ))
            
            metrics.append(Metric(
                name="system_network_bytes_recv",
                type=MetricType.SYSTEM,
                value=network.bytes_recv,
                timestamp=timestamp,
                unit="bytes"
            ))
            
            # 프로세스 수
            process_count = len(psutil.pids())
            metrics.append(Metric(
                name="system_process_count",
                type=MetricType.SYSTEM,
                value=process_count,
                timestamp=timestamp,
                unit="count"
            ))
            
        except Exception as e:
            logging.error(f"시스템 메트릭 수집 실패: {str(e)}")
        
        return metrics

class ApplicationMetricsCollector:
    """애플리케이션 메트릭 수집기"""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.active_users = set()
    
    def collect_application_metrics(self) -> List[Metric]:
        """애플리케이션 메트릭 수집"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Streamlit 세션 정보 (가능한 경우)
            session_count = 0
            if hasattr(st, 'session_state') and st.session_state:
                session_count = 1  # 현재 세션
            
            metrics.append(Metric(
                name="app_active_sessions",
                type=MetricType.APPLICATION,
                value=session_count,
                timestamp=timestamp,
                unit="count"
            ))
            
            # 요청 수
            metrics.append(Metric(
                name="app_request_count",
                type=MetricType.APPLICATION,
                value=self.request_count,
                timestamp=timestamp,
                unit="count"
            ))
            
            # 에러 수
            metrics.append(Metric(
                name="app_error_count",
                type=MetricType.APPLICATION,
                value=self.error_count,
                timestamp=timestamp,
                unit="count"
            ))
            
            # 평균 응답 시간
            if self.response_times:
                avg_response_time = sum(self.response_times) / len(self.response_times)
                metrics.append(Metric(
                    name="app_avg_response_time_ms",
                    type=MetricType.APPLICATION,
                    value=round(avg_response_time, 2),
                    timestamp=timestamp,
                    unit="ms"
                ))
                
                # 응답 시간 초기화 (sliding window)
                self.response_times = self.response_times[-100:]  # 최근 100개만 유지
            
            # 캐시 히트율 (세션 상태에서 가져오기)
            cache_hit_rate = st.session_state.get('cache_hit_rate', 0)
            metrics.append(Metric(
                name="app_cache_hit_rate_percent",
                type=MetricType.APPLICATION,
                value=cache_hit_rate,
                timestamp=timestamp,
                unit="percent"
            ))
            
        except Exception as e:
            logging.error(f"애플리케이션 메트릭 수집 실패: {str(e)}")
        
        return metrics
    
    def record_request(self, response_time_ms: float):
        """요청 기록"""
        self.request_count += 1
        self.response_times.append(response_time_ms)
    
    def record_error(self):
        """에러 기록"""
        self.error_count += 1

class MetricsStorage:
    """메트릭 저장소"""
    
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    labels TEXT,
                    unit TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                ON metrics(name, timestamp)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    metric_name TEXT,
                    threshold_value REAL,
                    current_value REAL,
                    resolved_at DATETIME,
                    labels TEXT
                )
            """)
    
    def store_metrics(self, metrics: List[Metric]):
        """메트릭 저장"""
        with sqlite3.connect(self.db_path) as conn:
            for metric in metrics:
                conn.execute("""
                    INSERT INTO metrics (name, type, value, timestamp, labels, unit)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metric.name,
                    metric.type.value,
                    str(metric.value),
                    metric.timestamp.isoformat(),
                    json.dumps(metric.labels),
                    metric.unit
                ))
    
    def get_metrics(self, 
                   metric_name: str,
                   start_time: datetime,
                   end_time: datetime) -> pd.DataFrame:
        """메트릭 조회"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT name, type, value, timestamp, labels, unit
                FROM metrics
                WHERE name = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            
            df = pd.read_sql_query(
                query,
                conn,
                params=[metric_name, start_time.isoformat(), end_time.isoformat()]
            )
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            return df
    
    def store_alert(self, alert: Alert):
        """알림 저장"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alerts 
                (id, name, severity, status, message, timestamp, metric_name, 
                 threshold_value, current_value, resolved_at, labels)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id,
                alert.name,
                alert.severity.value,
                alert.status.value,
                alert.message,
                alert.timestamp.isoformat(),
                alert.metric_name,
                alert.threshold_value,
                alert.current_value,
                alert.resolved_at.isoformat() if alert.resolved_at else None,
                json.dumps(alert.labels)
            ))
    
    def get_active_alerts(self) -> List[Alert]:
        """활성 알림 조회"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT id, name, severity, status, message, timestamp,
                       metric_name, threshold_value, current_value, resolved_at, labels
                FROM alerts
                WHERE status = 'active'
                ORDER BY timestamp DESC
            """
            
            cursor = conn.execute(query)
            alerts = []
            
            for row in cursor.fetchall():
                alert = Alert(
                    id=row[0],
                    name=row[1],
                    severity=AlertSeverity(row[2]),
                    status=AlertStatus(row[3]),
                    message=row[4],
                    timestamp=datetime.fromisoformat(row[5]),
                    metric_name=row[6],
                    threshold_value=row[7],
                    current_value=row[8],
                    resolved_at=datetime.fromisoformat(row[9]) if row[9] else None,
                    labels=json.loads(row[10]) if row[10] else {}
                )
                alerts.append(alert)
            
            return alerts
    
    def cleanup_old_metrics(self, retention_days: int = 30):
        """오래된 메트릭 정리"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                DELETE FROM metrics 
                WHERE timestamp < ?
            """, (cutoff_date.isoformat(),))

class AlertManager:
    """알림 관리자"""
    
    def __init__(self, storage: MetricsStorage):
        self.storage = storage
        self.rules: List[AlertRule] = []
        self.notification_handlers = []
        self.alert_state = {}  # 알림 상태 추적
    
    def add_rule(self, rule: AlertRule):
        """알림 규칙 추가"""
        self.rules.append(rule)
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """알림 핸들러 추가"""
        self.notification_handlers.append(handler)
    
    def evaluate_metrics(self, metrics: List[Metric]):
        """메트릭 평가 및 알림 생성"""
        for metric in metrics:
            for rule in self.rules:
                if rule.enabled and rule.metric_name == metric.name:
                    self._evaluate_rule(rule, metric)
    
    def _evaluate_rule(self, rule: AlertRule, metric: Metric):
        """개별 규칙 평가"""
        try:
            current_value = float(metric.value)
            threshold = rule.threshold
            
            # 조건 평가
            condition_met = False
            if rule.condition == '>':
                condition_met = current_value > threshold
            elif rule.condition == '<':
                condition_met = current_value < threshold
            elif rule.condition == '>=':
                condition_met = current_value >= threshold
            elif rule.condition == '<=':
                condition_met = current_value <= threshold
            elif rule.condition == '==':
                condition_met = abs(current_value - threshold) < 0.0001
            elif rule.condition == '!=':
                condition_met = abs(current_value - threshold) >= 0.0001
            
            alert_key = f"{rule.name}_{metric.name}"
            
            if condition_met:
                # 알림 발생
                if alert_key not in self.alert_state:
                    self.alert_state[alert_key] = {
                        'first_trigger': metric.timestamp,
                        'last_trigger': metric.timestamp
                    }
                else:
                    self.alert_state[alert_key]['last_trigger'] = metric.timestamp
                
                # 지속 시간 확인
                duration = (metric.timestamp - self.alert_state[alert_key]['first_trigger']).total_seconds()
                
                if duration >= rule.duration_seconds:
                    alert_id = f"alert_{alert_key}_{int(metric.timestamp.timestamp())}"
                    
                    message = rule.message_template.format(
                        metric_name=metric.name,
                        condition=rule.condition,
                        threshold=threshold,
                        current_value=current_value
                    )
                    
                    alert = Alert(
                        id=alert_id,
                        name=rule.name,
                        severity=rule.severity,
                        status=AlertStatus.ACTIVE,
                        message=message,
                        timestamp=metric.timestamp,
                        metric_name=metric.name,
                        threshold_value=threshold,
                        current_value=current_value,
                        labels={**rule.labels, **metric.labels}
                    )
                    
                    self.storage.store_alert(alert)
                    
                    # 알림 전송
                    for handler in self.notification_handlers:
                        try:
                            handler(alert)
                        except Exception as e:
                            logging.error(f"알림 핸들러 실행 실패: {str(e)}")
            
            else:
                # 조건이 해결됨
                if alert_key in self.alert_state:
                    del self.alert_state[alert_key]
                    
                    # 활성 알림이 있다면 해결됨으로 표시
                    # (실제 구현에서는 더 정교한 로직 필요)
        
        except (ValueError, TypeError) as e:
            logging.error(f"알림 규칙 평가 실패 {rule.name}: {str(e)}")

class NotificationHandler:
    """알림 핸들러"""
    
    @staticmethod
    def console_handler(alert: Alert):
        """콘솔 알림 핸들러"""
        severity_icons = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.FATAL: "💥"
        }
        
        icon = severity_icons.get(alert.severity, "🔔")
        print(f"{icon} [{alert.severity.value.upper()}] {alert.name}: {alert.message}")
    
    @staticmethod
    def email_handler(alert: Alert, smtp_config: Dict[str, Any]):
        """이메일 알림 핸들러"""
        try:
            msg = MimeMultipart()
            msg['From'] = smtp_config['from_email']
            msg['To'] = ', '.join(smtp_config['to_emails'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.name}"
            
            body = f"""
            알림: {alert.name}
            심각도: {alert.severity.value}
            메시지: {alert.message}
            시간: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            메트릭: {alert.metric_name}
            현재 값: {alert.current_value}
            임계값: {alert.threshold_value}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port'])
            if smtp_config.get('use_tls'):
                server.starttls()
            if smtp_config.get('username'):
                server.login(smtp_config['username'], smtp_config['password'])
            
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logging.error(f"이메일 전송 실패: {str(e)}")
    
    @staticmethod
    def slack_handler(alert: Alert, webhook_url: str):
        """Slack 알림 핸들러"""
        try:
            severity_colors = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9500",
                AlertSeverity.CRITICAL: "#ff0000",
                AlertSeverity.FATAL: "#800000"
            }
            
            payload = {
                "attachments": [{
                    "color": severity_colors.get(alert.severity, "#36a64f"),
                    "title": f"🔔 {alert.name}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "심각도",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "메트릭",
                            "value": alert.metric_name or "N/A",
                            "short": True
                        },
                        {
                            "title": "현재 값",
                            "value": str(alert.current_value) if alert.current_value else "N/A",
                            "short": True
                        },
                        {
                            "title": "임계값",
                            "value": str(alert.threshold_value) if alert.threshold_value else "N/A",
                            "short": True
                        }
                    ],
                    "footer": "AlphaForge Monitoring",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
        except Exception as e:
            logging.error(f"Slack 전송 실패: {str(e)}")

class MonitoringDashboard:
    """모니터링 대시보드"""
    
    def __init__(self, storage: MetricsStorage):
        self.storage = storage
    
    def render_system_overview(self):
        """시스템 개요 대시보드"""
        st.markdown("### 📊 시스템 모니터링")
        
        # 현재 시스템 상태
        current_metrics = SystemMetricsCollector.collect_system_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        for i, metric in enumerate(current_metrics[:4]):
            with [col1, col2, col3, col4][i]:
                delta_color = "normal"
                if "cpu" in metric.name and metric.value > 80:
                    delta_color = "inverse"
                elif "memory" in metric.name and metric.value > 85:
                    delta_color = "inverse"
                elif "disk" in metric.name and metric.value > 90:
                    delta_color = "inverse"
                
                st.metric(
                    label=metric.name.replace('system_', '').replace('_', ' ').title(),
                    value=f"{metric.value}{metric.unit if metric.unit else ''}",
                    delta_color=delta_color
                )
    
    def render_metrics_charts(self, metric_names: List[str], time_range_hours: int = 24):
        """메트릭 차트 렌더링"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_range_hours)
        
        if len(metric_names) == 1:
            # 단일 메트릭 차트
            df = self.storage.get_metrics(metric_names[0], start_time, end_time)
            
            if not df.empty:
                fig = px.line(
                    df, 
                    x='timestamp', 
                    y='value',
                    title=f"{metric_names[0]} - Last {time_range_hours}h"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"메트릭 '{metric_names[0]}'에 대한 데이터가 없습니다.")
        
        else:
            # 다중 메트릭 차트
            fig = make_subplots(
                rows=len(metric_names), 
                cols=1,
                subplot_titles=metric_names,
                shared_xaxes=True
            )
            
            for i, metric_name in enumerate(metric_names):
                df = self.storage.get_metrics(metric_name, start_time, end_time)
                
                if not df.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df['value'],
                            mode='lines',
                            name=metric_name
                        ),
                        row=i+1, col=1
                    )
            
            fig.update_layout(height=200 * len(metric_names))
            st.plotly_chart(fig, use_container_width=True)
    
    def render_alerts_panel(self, alert_manager: AlertManager):
        """알림 패널 렌더링"""
        st.markdown("### 🚨 활성 알림")
        
        active_alerts = self.storage.get_active_alerts()
        
        if not active_alerts:
            st.success("현재 활성 알림이 없습니다.")
            return
        
        # 심각도별 그룹화
        severity_groups = {}
        for alert in active_alerts:
            if alert.severity not in severity_groups:
                severity_groups[alert.severity] = []
            severity_groups[alert.severity].append(alert)
        
        # 심각도 순으로 표시
        severity_order = [AlertSeverity.FATAL, AlertSeverity.CRITICAL, AlertSeverity.WARNING, AlertSeverity.INFO]
        
        for severity in severity_order:
            if severity in severity_groups:
                alerts = severity_groups[severity]
                
                severity_icons = {
                    AlertSeverity.FATAL: "💥",
                    AlertSeverity.CRITICAL: "🚨",
                    AlertSeverity.WARNING: "⚠️",
                    AlertSeverity.INFO: "ℹ️"
                }
                
                icon = severity_icons.get(severity, "🔔")
                
                with st.expander(f"{icon} {severity.value.upper()} ({len(alerts)})", expanded=severity in [AlertSeverity.FATAL, AlertSeverity.CRITICAL]):
                    for alert in alerts:
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.write(f"**{alert.name}**")
                            st.write(alert.message)
                        
                        with col2:
                            st.write(f"발생 시간:")
                            st.write(alert.timestamp.strftime("%H:%M:%S"))
                        
                        with col3:
                            if st.button("해결", key=f"resolve_{alert.id}"):
                                # 알림 해결 처리
                                alert.status = AlertStatus.RESOLVED
                                alert.resolved_at = datetime.now()
                                self.storage.store_alert(alert)
                                st.experimental_rerun()

# 전역 모니터링 시스템 인스턴스
metrics_storage = MetricsStorage()
metrics_collector = MetricsCollector()
alert_manager = AlertManager(metrics_storage)
monitoring_dashboard = MonitoringDashboard(metrics_storage)

# 기본 수집기 등록
app_metrics_collector = ApplicationMetricsCollector()
metrics_collector.add_collector(SystemMetricsCollector.collect_system_metrics)
metrics_collector.add_collector(app_metrics_collector.collect_application_metrics)

# 기본 알림 규칙 등록
default_rules = [
    AlertRule(
        name="High CPU Usage",
        metric_name="system_cpu_usage_percent",
        condition=">",
        threshold=80.0,
        severity=AlertSeverity.WARNING,
        duration_seconds=60
    ),
    AlertRule(
        name="Critical CPU Usage",
        metric_name="system_cpu_usage_percent", 
        condition=">",
        threshold=95.0,
        severity=AlertSeverity.CRITICAL,
        duration_seconds=30
    ),
    AlertRule(
        name="High Memory Usage",
        metric_name="system_memory_usage_percent",
        condition=">",
        threshold=85.0,
        severity=AlertSeverity.WARNING,
        duration_seconds=120
    ),
    AlertRule(
        name="Critical Memory Usage",
        metric_name="system_memory_usage_percent",
        condition=">",
        threshold=95.0,
        severity=AlertSeverity.CRITICAL,
        duration_seconds=60
    ),
    AlertRule(
        name="High Disk Usage",
        metric_name="system_disk_usage_percent",
        condition=">",
        threshold=90.0,
        severity=AlertSeverity.WARNING,
        duration_seconds=300
    )
]

for rule in default_rules:
    alert_manager.add_rule(rule)

# 기본 알림 핸들러 등록
alert_manager.add_notification_handler(NotificationHandler.console_handler)