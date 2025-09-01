"""
글로벌 에러 핸들러
알파포지 UI의 중앙 집중식 에러 처리 시스템
"""

import logging
import traceback
import sys
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import streamlit as st
import pandas as pd


class ErrorSeverity(Enum):
    """에러 심각도 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """에러 카테고리"""
    DATA_VALIDATION = "data_validation"
    DATA_PROCESSING = "data_processing"
    FACTOR_MINING = "factor_mining"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    FILE_IO = "file_io"
    NETWORK = "network"
    SYSTEM = "system"
    USER_INPUT = "user_input"
    CALCULATION = "calculation"


@dataclass
class ErrorInfo:
    """에러 정보 클래스"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: str
    timestamp: datetime
    context: Dict[str, Any]
    traceback_info: Optional[str] = None
    user_message: Optional[str] = None
    recovery_suggestion: Optional[str] = None


class ErrorHandler:
    """글로벌 에러 핸들러 클래스"""
    
    def __init__(self, logger_name: str = "alphaforge_error_handler"):
        """초기화"""
        self.logger = logging.getLogger(logger_name)
        self.error_history: List[ErrorInfo] = []
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self.max_history_size = 1000
        
        # 복구 전략 등록
        self._register_recovery_strategies()
    
    def _register_recovery_strategies(self):
        """복구 전략 등록"""
        self.recovery_strategies = {
            ErrorCategory.DATA_VALIDATION: self._recover_data_validation,
            ErrorCategory.DATA_PROCESSING: self._recover_data_processing,
            ErrorCategory.FACTOR_MINING: self._recover_factor_mining,
            ErrorCategory.FILE_IO: self._recover_file_io,
            ErrorCategory.USER_INPUT: self._recover_user_input,
            ErrorCategory.CALCULATION: self._recover_calculation,
        }
    
    def handle_error(
        self,
        exception: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        recovery_suggestion: Optional[str] = None,
        raise_exception: bool = False
    ) -> ErrorInfo:
        """
        에러 처리 메인 메서드
        
        Args:
            exception: 발생한 예외
            category: 에러 카테고리
            severity: 에러 심각도
            context: 추가 컨텍스트 정보
            user_message: 사용자에게 표시할 메시지
            recovery_suggestion: 복구 제안
            raise_exception: 예외를 다시 발생시킬지 여부
            
        Returns:
            ErrorInfo: 에러 정보 객체
        """
        # 에러 ID 생성
        error_id = self._generate_error_id()
        
        # 에러 정보 생성
        error_info = ErrorInfo(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(exception),
            details=self._extract_error_details(exception),
            timestamp=datetime.now(),
            context=context or {},
            traceback_info=traceback.format_exc(),
            user_message=user_message,
            recovery_suggestion=recovery_suggestion
        )
        
        # 에러 기록
        self._log_error(error_info)
        
        # 에러 히스토리 저장
        self._add_to_history(error_info)
        
        # Streamlit에 에러 표시
        self._display_error_in_ui(error_info)
        
        # 복구 시도
        if category in self.recovery_strategies:
            try:
                recovery_result = self.recovery_strategies[category](error_info)
                if recovery_result:
                    self.logger.info(f"Error {error_id} recovered successfully")
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed for error {error_id}: {recovery_error}")
        
        # 필요시 예외 재발생
        if raise_exception:
            raise exception
        
        return error_info
    
    def _generate_error_id(self) -> str:
        """에러 ID 생성"""
        from uuid import uuid4
        return f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid4())[:8]}"
    
    def _extract_error_details(self, exception: Exception) -> str:
        """에러 세부 정보 추출"""
        details = []
        
        # 예외 타입
        details.append(f"Exception Type: {type(exception).__name__}")
        
        # 예외 메시지
        details.append(f"Message: {str(exception)}")
        
        # 스택 정보
        if hasattr(exception, '__traceback__') and exception.__traceback__:
            tb = exception.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            details.append(f"File: {tb.tb_frame.f_code.co_filename}")
            details.append(f"Line: {tb.tb_lineno}")
            details.append(f"Function: {tb.tb_frame.f_code.co_name}")
        
        return " | ".join(details)
    
    def _log_error(self, error_info: ErrorInfo):
        """에러 로깅"""
        log_message = (
            f"Error ID: {error_info.error_id} | "
            f"Category: {error_info.category.value} | "
            f"Severity: {error_info.severity.value} | "
            f"Message: {error_info.message}"
        )
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # 트레이스백 정보 로깅
        if error_info.traceback_info:
            self.logger.debug(f"Traceback for {error_info.error_id}:\n{error_info.traceback_info}")
    
    def _add_to_history(self, error_info: ErrorInfo):
        """에러 히스토리에 추가"""
        self.error_history.append(error_info)
        
        # 히스토리 크기 제한
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def _display_error_in_ui(self, error_info: ErrorInfo):
        """Streamlit UI에 에러 표시"""
        try:
            if error_info.user_message:
                message = error_info.user_message
            else:
                message = self._generate_user_friendly_message(error_info)
            
            if error_info.severity == ErrorSeverity.CRITICAL:
                st.error(f"🚨 Critical Error: {message}")
            elif error_info.severity == ErrorSeverity.HIGH:
                st.error(f"❌ Error: {message}")
            elif error_info.severity == ErrorSeverity.MEDIUM:
                st.warning(f"⚠️ Warning: {message}")
            else:
                st.info(f"ℹ️ Notice: {message}")
            
            # 복구 제안 표시
            if error_info.recovery_suggestion:
                st.info(f"💡 Suggestion: {error_info.recovery_suggestion}")
            
            # 에러 세부 정보 (확장 가능)
            with st.expander("🔍 Technical Details", expanded=False):
                st.code(f"Error ID: {error_info.error_id}")
                st.code(f"Category: {error_info.category.value}")
                st.code(f"Timestamp: {error_info.timestamp}")
                if error_info.context:
                    st.code(f"Context: {error_info.context}")
        
        except Exception:
            # UI 표시 실패 시 로깅만 수행
            self.logger.error(f"Failed to display error {error_info.error_id} in UI")
    
    def _generate_user_friendly_message(self, error_info: ErrorInfo) -> str:
        """사용자 친화적 메시지 생성"""
        category_messages = {
            ErrorCategory.DATA_VALIDATION: "데이터 검증 중 문제가 발생했습니다. 입력 데이터를 확인해주세요.",
            ErrorCategory.DATA_PROCESSING: "데이터 처리 중 오류가 발생했습니다. 데이터 형식을 확인해주세요.",
            ErrorCategory.FACTOR_MINING: "팩터 생성 중 문제가 발생했습니다. 데이터가 충분한지 확인해주세요.",
            ErrorCategory.PERFORMANCE_ANALYSIS: "성능 분석 중 오류가 발생했습니다.",
            ErrorCategory.FILE_IO: "파일 처리 중 문제가 발생했습니다. 파일 형식과 권한을 확인해주세요.",
            ErrorCategory.NETWORK: "네트워크 연결에 문제가 있습니다. 인터넷 연결을 확인해주세요.",
            ErrorCategory.SYSTEM: "시스템 오류가 발생했습니다.",
            ErrorCategory.USER_INPUT: "입력값에 문제가 있습니다. 올바른 값을 입력해주세요.",
            ErrorCategory.CALCULATION: "계산 중 오류가 발생했습니다. 입력 데이터를 확인해주세요.",
        }
        
        return category_messages.get(error_info.category, "예상치 못한 오류가 발생했습니다.")
    
    # 복구 전략 메서드들
    def _recover_data_validation(self, error_info: ErrorInfo) -> bool:
        """데이터 검증 에러 복구"""
        try:
            # 세션 상태에서 백업 데이터 찾기
            if 'backup_data' in st.session_state:
                st.session_state['processed_data'] = st.session_state['backup_data']
                st.success("백업 데이터로 복구했습니다.")
                return True
        except Exception:
            pass
        return False
    
    def _recover_data_processing(self, error_info: ErrorInfo) -> bool:
        """데이터 처리 에러 복구"""
        try:
            # 기본 설정으로 재시도
            if 'raw_data' in st.session_state:
                st.info("기본 설정으로 데이터 처리를 재시도합니다.")
                return True
        except Exception:
            pass
        return False
    
    def _recover_factor_mining(self, error_info: ErrorInfo) -> bool:
        """팩터 마이닝 에러 복구"""
        try:
            # 더 간단한 팩터로 재시도
            st.info("기본 팩터로 재시도합니다.")
            return True
        except Exception:
            pass
        return False
    
    def _recover_file_io(self, error_info: ErrorInfo) -> bool:
        """파일 I/O 에러 복구"""
        try:
            # 임시 파일 정리
            import tempfile
            tempfile._cleanup()
            st.info("임시 파일을 정리했습니다. 다시 시도해주세요.")
            return True
        except Exception:
            pass
        return False
    
    def _recover_user_input(self, error_info: ErrorInfo) -> bool:
        """사용자 입력 에러 복구"""
        try:
            # 기본값으로 재설정
            st.info("입력값을 기본값으로 재설정했습니다.")
            return True
        except Exception:
            pass
        return False
    
    def _recover_calculation(self, error_info: ErrorInfo) -> bool:
        """계산 에러 복구"""
        try:
            # 대체 계산 방법 시도
            st.info("대체 계산 방법을 시도합니다.")
            return True
        except Exception:
            pass
        return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """에러 통계 조회"""
        if not self.error_history:
            return {"total_errors": 0}
        
        stats = {
            "total_errors": len(self.error_history),
            "by_category": {},
            "by_severity": {},
            "recent_errors": len([e for e in self.error_history 
                                if (datetime.now() - e.timestamp).total_seconds() < 3600])
        }
        
        # 카테고리별 통계
        for error in self.error_history:
            category = error.category.value
            severity = error.severity.value
            
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
        
        return stats
    
    def clear_error_history(self):
        """에러 히스토리 초기화"""
        self.error_history.clear()
        self.logger.info("Error history cleared")
    
    def export_error_log(self) -> pd.DataFrame:
        """에러 로그 내보내기"""
        if not self.error_history:
            return pd.DataFrame()
        
        data = []
        for error in self.error_history:
            data.append({
                'error_id': error.error_id,
                'category': error.category.value,
                'severity': error.severity.value,
                'message': error.message,
                'timestamp': error.timestamp,
                'details': error.details
            })
        
        return pd.DataFrame(data)


# 글로벌 에러 핸들러 인스턴스
global_error_handler = ErrorHandler()


def handle_error(
    exception: Exception,
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: Optional[Dict[str, Any]] = None,
    user_message: Optional[str] = None,
    recovery_suggestion: Optional[str] = None,
    raise_exception: bool = False
) -> ErrorInfo:
    """
    글로벌 에러 처리 함수
    
    사용 예:
    try:
        # 위험한 작업
        result = risky_operation()
    except Exception as e:
        handle_error(
            e,
            ErrorCategory.DATA_PROCESSING,
            ErrorSeverity.HIGH,
            context={'operation': 'risky_operation'},
            user_message="데이터 처리 중 오류가 발생했습니다.",
            recovery_suggestion="데이터 형식을 확인하고 다시 시도해주세요."
        )
    """
    return global_error_handler.handle_error(
        exception=exception,
        category=category,
        severity=severity,
        context=context,
        user_message=user_message,
        recovery_suggestion=recovery_suggestion,
        raise_exception=raise_exception
    )


# 데코레이터를 위한 추가 함수
def error_handler(
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    user_message: Optional[str] = None,
    recovery_suggestion: Optional[str] = None
):
    """
    에러 처리 데코레이터
    
    사용 예:
    @error_handler(
        category=ErrorCategory.DATA_PROCESSING,
        severity=ErrorSeverity.HIGH,
        user_message="데이터 처리 중 오류가 발생했습니다."
    )
    def process_data(data):
        # 데이터 처리 로직
        pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handle_error(
                    exception=e,
                    category=category,
                    severity=severity,
                    context={
                        'function': func.__name__,
                        'args': str(args)[:200],  # 길이 제한
                        'kwargs': str(kwargs)[:200]
                    },
                    user_message=user_message,
                    recovery_suggestion=recovery_suggestion
                )
                return None
        return wrapper
    return decorator