"""
구조화된 로깅 시스템
JSON 형태의 구조화된 로그와 다양한 로그 레벨을 지원합니다.
"""

import logging
import logging.handlers
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import streamlit as st
import pandas as pd
from utils.env_manager import env_manager

# 로그 디렉토리 생성
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

class JSONFormatter(logging.Formatter):
    """JSON 형태로 로그를 포맷하는 클래스"""
    
    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드를 JSON 형태로 포맷합니다."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # 예외 정보가 있으면 추가
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # 추가 속성이 있으면 포함
        extra_attrs = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 'exc_text', 
                          'stack_info']:
                try:
                    json.dumps(value)  # JSON 직렬화 가능한지 확인
                    extra_attrs[key] = value
                except (TypeError, ValueError):
                    extra_attrs[key] = str(value)
        
        if extra_attrs:
            log_data['extra'] = extra_attrs
        
        return json.dumps(log_data, ensure_ascii=False)

class ColoredFormatter(logging.Formatter):
    """컬러 출력을 지원하는 포맷터"""
    
    # ANSI 색상 코드
    COLORS = {
        'DEBUG': '\033[36m',      # 청록색
        'INFO': '\033[32m',       # 초록색
        'WARNING': '\033[33m',    # 노란색
        'ERROR': '\033[31m',      # 빨간색
        'CRITICAL': '\033[35m',   # 자홍색
        'RESET': '\033[0m'        # 리셋
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """컬러가 적용된 로그를 포맷합니다."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # 기본 포맷 적용
        formatted = super().format(record)
        
        # 컬러 적용 (터미널에서만)
        if sys.stderr.isatty():
            return f"{color}{formatted}{reset}"
        else:
            return formatted

class AlphaForgeLogger:
    """AlphaForge-UI 전용 로깅 클래스"""
    
    def __init__(self):
        self.loggers: Dict[str, logging.Logger] = {}
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """루트 로거를 설정합니다."""
        # 환경변수에서 로그 설정 가져오기
        log_level = env_manager.get_env('LOG_LEVEL', 'INFO').upper()
        log_file = env_manager.get_env('LOG_FILE', 'logs/app.log')
        
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        # 기존 핸들러 제거
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 파일 핸들러 추가 (JSON 형태)
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
        
        # 콘솔 핸들러 추가 (컬러 적용)
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.WARNING)  # 콘솔은 WARNING 이상만
        root_logger.addHandler(console_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """특정 이름의 로거를 반환합니다."""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def log_user_action(self, action: str, details: Optional[Dict[str, Any]] = None):
        """사용자 액션을 로깅합니다."""
        logger = self.get_logger('user_action')
        
        log_data = {
            'action': action,
            'session_id': st.session_state.get('session_id', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        if details:
            log_data['details'] = details
        
        logger.info("User action performed", extra=log_data)
    
    def log_performance(self, operation: str, duration: float, details: Optional[Dict[str, Any]] = None):
        """성능 메트릭을 로깅합니다."""
        logger = self.get_logger('performance')
        
        log_data = {
            'operation': operation,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        if details:
            log_data['details'] = details
        
        # 성능 임계값에 따라 로그 레벨 조정
        if duration > 10.0:
            logger.warning("Slow operation detected", extra=log_data)
        elif duration > 5.0:
            logger.info("Moderate operation duration", extra=log_data)
        else:
            logger.debug("Operation completed", extra=log_data)
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """에러를 구조화된 형태로 로깅합니다."""
        logger = self.get_logger('error')
        
        log_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat()
        }
        
        if context:
            log_data['context'] = context
        
        logger.error("Error occurred", extra=log_data, exc_info=True)
    
    def log_system_event(self, event: str, details: Optional[Dict[str, Any]] = None):
        """시스템 이벤트를 로깅합니다."""
        logger = self.get_logger('system')
        
        log_data = {
            'event': event,
            'timestamp': datetime.now().isoformat()
        }
        
        if details:
            log_data['details'] = details
        
        logger.info("System event", extra=log_data)
    
    def log_security_event(self, event: str, severity: str = 'info', details: Optional[Dict[str, Any]] = None):
        """보안 이벤트를 로깅합니다."""
        logger = self.get_logger('security')
        
        log_data = {
            'security_event': event,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        
        if details:
            log_data['details'] = details
        
        level = getattr(logging, severity.upper(), logging.INFO)
        logger.log(level, "Security event", extra=log_data)

# 전역 로거 인스턴스
alpha_logger = AlphaForgeLogger()

# 편의 함수들
def get_logger(name: str) -> logging.Logger:
    """로거 인스턴스를 반환합니다."""
    return alpha_logger.get_logger(name)

def log_user_action(action: str, details: Optional[Dict[str, Any]] = None):
    """사용자 액션 로깅"""
    alpha_logger.log_user_action(action, details)

def log_performance(operation: str, duration: float, details: Optional[Dict[str, Any]] = None):
    """성능 메트릭 로깅"""
    alpha_logger.log_performance(operation, duration, details)

def log_error(error: Exception, context: Optional[Dict[str, Any]] = None):
    """에러 로깅"""
    alpha_logger.log_error(error, context)

def log_system_event(event: str, details: Optional[Dict[str, Any]] = None):
    """시스템 이벤트 로깅"""
    alpha_logger.log_system_event(event, details)

def log_security_event(event: str, severity: str = 'info', details: Optional[Dict[str, Any]] = None):
    """보안 이벤트 로깅"""
    alpha_logger.log_security_event(event, severity, details)

# 데코레이터
def log_execution_time(operation_name: Optional[str] = None):
    """함수 실행 시간을 로깅하는 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            start_time = time.time()
            operation = operation_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                log_performance(operation, duration, {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                })
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                log_error(e, {
                    'function': func.__name__,
                    'module': func.__module__,
                    'operation': operation,
                    'duration': duration
                })
                
                raise
        
        return wrapper
    return decorator

def log_user_interaction(action_name: Optional[str] = None):
    """사용자 상호작용을 로깅하는 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            action = action_name or f"{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                
                log_user_action(action, {
                    'function': func.__name__,
                    'success': True
                })
                
                return result
                
            except Exception as e:
                log_user_action(action, {
                    'function': func.__name__,
                    'success': False,
                    'error': str(e)
                })
                
                raise
        
        return wrapper
    return decorator

# Streamlit 통합을 위한 함수들
def setup_streamlit_logging():
    """Streamlit 앱 시작시 로깅 설정"""
    if 'session_id' not in st.session_state:
        import uuid
        st.session_state['session_id'] = str(uuid.uuid4())[:8]
    
    log_system_event('streamlit_session_started', {
        'session_id': st.session_state['session_id']
    })

def show_log_viewer():
    """로그 뷰어를 표시합니다."""
    st.subheader("📋 로그 뷰어")
    
    # 로그 파일 선택
    log_files = list(LOG_DIR.glob("*.log*"))
    
    if not log_files:
        st.warning("로그 파일이 없습니다.")
        return
    
    selected_file = st.selectbox("로그 파일 선택", log_files)
    
    if selected_file:
        try:
            # 로그 파일 읽기 (최근 100줄만)
            with open(selected_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                recent_lines = lines[-100:] if len(lines) > 100 else lines
            
            st.text_area(
                f"최근 로그 ({len(recent_lines)}줄)",
                ''.join(recent_lines),
                height=400
            )
            
        except Exception as e:
            st.error(f"로그 파일을 읽을 수 없습니다: {e}")

# 로그 분석을 위한 유틸리티 함수들
def analyze_logs(hours: int = 24) -> Dict[str, Any]:
    """지정된 시간 동안의 로그를 분석합니다."""
    log_file = Path(env_manager.get_env('LOG_FILE', 'logs/app.log'))
    
    if not log_file.exists():
        return {'error': 'Log file not found'}
    
    try:
        analysis = {
            'total_entries': 0,
            'levels': {},
            'errors': [],
            'performance_issues': [],
            'user_actions': [],
            'security_events': [],
            'system_events': [],
            'top_errors': {},
            'performance_trends': [],
            'session_statistics': {}
        }
        
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    
                    # 시간 필터링
                    log_time = datetime.fromisoformat(log_entry['timestamp']).timestamp()
                    if log_time < cutoff_time:
                        continue
                    
                    analysis['total_entries'] += 1
                    
                    # 레벨별 카운트
                    level = log_entry['level']
                    analysis['levels'][level] = analysis['levels'].get(level, 0) + 1
                    
                    # 에러 수집 및 분석
                    if level in ['ERROR', 'CRITICAL']:
                        analysis['errors'].append(log_entry)
                        error_type = log_entry.get('extra', {}).get('error_type', 'Unknown')
                        analysis['top_errors'][error_type] = analysis['top_errors'].get(error_type, 0) + 1
                    
                    # 성능 이슈 수집
                    if 'extra' in log_entry and 'duration_seconds' in log_entry['extra']:
                        duration = log_entry['extra']['duration_seconds']
                        if duration > 5.0:
                            analysis['performance_issues'].append(log_entry)
                        
                        # 성능 트렌드 분석
                        analysis['performance_trends'].append({
                            'timestamp': log_entry['timestamp'],
                            'operation': log_entry['extra'].get('operation', 'unknown'),
                            'duration': duration
                        })
                    
                    # 사용자 액션 수집
                    if log_entry.get('logger') == 'user_action':
                        analysis['user_actions'].append(log_entry)
                        
                        # 세션 통계
                        session_id = log_entry.get('extra', {}).get('session_id', 'unknown')
                        if session_id not in analysis['session_statistics']:
                            analysis['session_statistics'][session_id] = {
                                'actions': 0,
                                'first_action': log_entry['timestamp'],
                                'last_action': log_entry['timestamp']
                            }
                        analysis['session_statistics'][session_id]['actions'] += 1
                        analysis['session_statistics'][session_id]['last_action'] = log_entry['timestamp']
                    
                    # 보안 이벤트 수집
                    if log_entry.get('logger') == 'security':
                        analysis['security_events'].append(log_entry)
                    
                    # 시스템 이벤트 수집
                    if log_entry.get('logger') == 'system':
                        analysis['system_events'].append(log_entry)
                
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # 상위 에러 정렬
        analysis['top_errors'] = dict(sorted(
            analysis['top_errors'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10])
        
        return analysis
        
    except Exception as e:
        return {'error': str(e)}


def generate_log_report(hours: int = 24) -> str:
    """로그 분석 리포트 생성"""
    analysis = analyze_logs(hours)
    
    if 'error' in analysis:
        return f"로그 분석 실패: {analysis['error']}"
    
    report = []
    report.append(f"📊 로그 분석 리포트 (최근 {hours}시간)")
    report.append("=" * 50)
    
    # 전체 통계
    report.append(f"총 로그 엔트리: {analysis['total_entries']}")
    
    if analysis['levels']:
        report.append("\n📈 로그 레벨별 분포:")
        for level, count in sorted(analysis['levels'].items()):
            report.append(f"  {level}: {count}")
    
    # 에러 분석
    if analysis['errors']:
        report.append(f"\n❌ 에러 발생: {len(analysis['errors'])}건")
        
        if analysis['top_errors']:
            report.append("\n상위 에러 유형:")
            for error_type, count in list(analysis['top_errors'].items())[:5]:
                report.append(f"  {error_type}: {count}회")
    
    # 성능 이슈
    if analysis['performance_issues']:
        report.append(f"\n⚠️ 성능 이슈: {len(analysis['performance_issues'])}건")
        
        # 가장 느린 작업 찾기
        slowest_operations = sorted(
            analysis['performance_trends'], 
            key=lambda x: x['duration'], 
            reverse=True
        )[:3]
        
        if slowest_operations:
            report.append("\n가장 느린 작업들:")
            for op in slowest_operations:
                report.append(f"  {op['operation']}: {op['duration']:.2f}초")
    
    # 사용자 활동
    if analysis['user_actions']:
        report.append(f"\n👥 사용자 액션: {len(analysis['user_actions'])}건")
        report.append(f"활성 세션: {len(analysis['session_statistics'])}개")
    
    # 보안 이벤트
    if analysis['security_events']:
        report.append(f"\n🔒 보안 이벤트: {len(analysis['security_events'])}건")
    
    return "\n".join(report)


def export_logs_to_csv(hours: int = 24) -> pd.DataFrame:
    """로그를 CSV 형태로 내보내기"""
    log_file = Path(env_manager.get_env('LOG_FILE', 'logs/app.log'))
    
    if not log_file.exists():
        return pd.DataFrame()
    
    logs_data = []
    cutoff_time = datetime.now().timestamp() - (hours * 3600)
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    
                    # 시간 필터링
                    log_time = datetime.fromisoformat(log_entry['timestamp']).timestamp()
                    if log_time < cutoff_time:
                        continue
                    
                    # 플랫 구조로 변환
                    flat_entry = {
                        'timestamp': log_entry['timestamp'],
                        'level': log_entry['level'],
                        'logger': log_entry['logger'],
                        'module': log_entry.get('module', ''),
                        'function': log_entry.get('function', ''),
                        'line': log_entry.get('line', ''),
                        'message': log_entry['message']
                    }
                    
                    # 추가 필드 처리
                    if 'extra' in log_entry:
                        for key, value in log_entry['extra'].items():
                            flat_entry[f'extra_{key}'] = value
                    
                    if 'exception' in log_entry:
                        flat_entry['exception_type'] = log_entry['exception'].get('type', '')
                        flat_entry['exception_message'] = log_entry['exception'].get('message', '')
                    
                    logs_data.append(flat_entry)
                
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return pd.DataFrame(logs_data)
        
    except Exception as e:
        return pd.DataFrame({'error': [str(e)]})


def show_advanced_log_viewer():
    """향상된 로그 뷰어 표시"""
    st.subheader("📋 고급 로그 뷰어")
    
    # 시간 범위 선택
    col1, col2 = st.columns(2)
    with col1:
        hours = st.selectbox(
            "분석 기간",
            [1, 6, 12, 24, 48, 72],
            index=3,
            help="분석할 로그의 시간 범위를 선택하세요"
        )
    
    with col2:
        log_level = st.selectbox(
            "로그 레벨 필터",
            ['ALL', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            index=0
        )
    
    # 로그 분석 실행
    if st.button("🔍 로그 분석 실행"):
        with st.spinner("로그를 분석 중입니다..."):
            # 리포트 생성
            report = generate_log_report(hours)
            st.text_area("분석 리포트", report, height=300)
            
            # 상세 분석 표시
            analysis = analyze_logs(hours)
            
            if 'error' not in analysis:
                # 메트릭 표시
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("총 로그 수", analysis['total_entries'])
                
                with col2:
                    error_count = len(analysis['errors'])
                    st.metric("에러 수", error_count, delta=f"{error_count}/day" if error_count > 0 else None)
                
                with col3:
                    performance_issues = len(analysis['performance_issues'])
                    st.metric("성능 이슈", performance_issues)
                
                with col4:
                    active_sessions = len(analysis['session_statistics'])
                    st.metric("활성 세션", active_sessions)
                
                # 차트 표시
                if analysis['levels']:
                    st.subheader("로그 레벨 분포")
                    level_df = pd.DataFrame(
                        list(analysis['levels'].items()),
                        columns=['Level', 'Count']
                    )
                    st.bar_chart(level_df.set_index('Level'))
                
                # 에러 분석
                if analysis['top_errors']:
                    st.subheader("상위 에러 유형")
                    error_df = pd.DataFrame(
                        list(analysis['top_errors'].items()),
                        columns=['Error Type', 'Count']
                    )
                    st.bar_chart(error_df.set_index('Error Type'))
                
                # 성능 트렌드
                if analysis['performance_trends']:
                    st.subheader("성능 트렌드")
                    perf_df = pd.DataFrame(analysis['performance_trends'])
                    if not perf_df.empty:
                        perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'])
                        st.line_chart(perf_df.set_index('timestamp')['duration'])
    
    # 로그 내보내기
    st.subheader("📤 로그 내보내기")
    
    if st.button("CSV로 내보내기"):
        with st.spinner("로그를 내보내는 중..."):
            df = export_logs_to_csv(hours)
            
            if not df.empty and 'error' not in df.columns:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="다운로드 CSV",
                    data=csv,
                    file_name=f"logs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                st.success(f"총 {len(df)}개의 로그 엔트리를 내보냈습니다.")
            else:
                st.error("로그 내보내기 실패")


# 실시간 로그 모니터링을 위한 추가 기능
class RealTimeLogMonitor:
    """실시간 로그 모니터링 클래스"""
    
    def __init__(self):
        self.alerts = []
        self.thresholds = {
            'error_rate': 10,  # 시간당 에러 수
            'slow_operations': 5,  # 5초 이상 작업
            'memory_usage': 0.8   # 80% 메모리 사용량
        }
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """알림 조건 확인"""
        analysis = analyze_logs(1)  # 최근 1시간
        alerts = []
        
        if 'error' not in analysis:
            # 에러율 확인
            error_count = len(analysis['errors'])
            if error_count > self.thresholds['error_rate']:
                alerts.append({
                    'type': 'error_rate',
                    'message': f"높은 에러율 감지: {error_count}개/시간",
                    'severity': 'high',
                    'timestamp': datetime.now().isoformat()
                })
            
            # 성능 이슈 확인
            slow_ops = len(analysis['performance_issues'])
            if slow_ops > 0:
                alerts.append({
                    'type': 'performance',
                    'message': f"느린 작업 감지: {slow_ops}개",
                    'severity': 'medium',
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def show_monitoring_dashboard(self):
        """실시간 모니터링 대시보드 표시"""
        st.subheader("🔴 실시간 모니터링")
        
        # 자동 새로고침
        if st.checkbox("자동 새로고침 (30초)", value=False):
            import time
            time.sleep(30)
            st.experimental_rerun()
        
        # 알림 확인
        alerts = self.check_alerts()
        
        if alerts:
            st.error("🚨 알림이 있습니다!")
            for alert in alerts:
                if alert['severity'] == 'high':
                    st.error(f"🔥 {alert['message']}")
                elif alert['severity'] == 'medium':
                    st.warning(f"⚠️ {alert['message']}")
                else:
                    st.info(f"ℹ️ {alert['message']}")
        else:
            st.success("✅ 모든 시스템이 정상 작동 중입니다")
        
        # 실시간 메트릭
        analysis = analyze_logs(1)
        if 'error' not in analysis:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "최근 1시간 로그",
                    analysis['total_entries'],
                    delta=None
                )
            
            with col2:
                error_count = len(analysis['errors'])
                st.metric(
                    "에러 수",
                    error_count,
                    delta=-error_count if error_count == 0 else error_count,
                    delta_color="inverse"
                )
            
            with col3:
                perf_issues = len(analysis['performance_issues'])
                st.metric(
                    "성능 이슈",
                    perf_issues,
                    delta=-perf_issues if perf_issues == 0 else perf_issues,
                    delta_color="inverse"
                )


# 글로벌 실시간 모니터 인스턴스
realtime_monitor = RealTimeLogMonitor()