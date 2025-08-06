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
from typing import Any, Dict, Optional, Union
import streamlit as st
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
            'user_actions': []
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
                    
                    # 에러 수집
                    if level in ['ERROR', 'CRITICAL']:
                        analysis['errors'].append(log_entry)
                    
                    # 성능 이슈 수집
                    if 'extra' in log_entry and 'duration_seconds' in log_entry['extra']:
                        duration = log_entry['extra']['duration_seconds']
                        if duration > 5.0:
                            analysis['performance_issues'].append(log_entry)
                    
                    # 사용자 액션 수집
                    if log_entry.get('logger') == 'user_action':
                        analysis['user_actions'].append(log_entry)
                
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return analysis
        
    except Exception as e:
        return {'error': str(e)}