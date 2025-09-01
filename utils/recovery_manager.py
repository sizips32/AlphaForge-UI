"""
복구 관리자
AlphaForge-UI의 자동 복구 및 장애 대응 시스템
"""

import logging
import json
import pickle
import os
import time
from typing import Dict, Any, Optional, Callable, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import streamlit as st
from pathlib import Path


class RecoveryStatus(Enum):
    """복구 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


class RecoveryStrategy(Enum):
    """복구 전략"""
    RETRY = "retry"
    FALLBACK = "fallback"
    BACKUP_RESTORE = "backup_restore"
    RESET = "reset"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class RecoveryPoint:
    """복구 지점 정보"""
    id: str
    timestamp: datetime
    operation_name: str
    data_state: Dict[str, Any]
    metadata: Dict[str, Any]
    file_path: Optional[str] = None


@dataclass
class RecoveryAction:
    """복구 액션 정보"""
    id: str
    strategy: RecoveryStrategy
    target_operation: str
    recovery_point_id: Optional[str]
    parameters: Dict[str, Any]
    timestamp: datetime
    status: RecoveryStatus
    retry_count: int = 0
    max_retries: int = 3
    result_message: Optional[str] = None


class RecoveryManager:
    """복구 관리자 클래스"""
    
    def __init__(self, backup_dir: str = "./backups"):
        """초기화"""
        self.logger = logging.getLogger("recovery_manager")
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # 복구 지점 및 액션 저장소
        self.recovery_points: Dict[str, RecoveryPoint] = {}
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        
        # 복구 전략 매핑
        self.recovery_strategies: Dict[str, Callable] = {
            RecoveryStrategy.RETRY.value: self._execute_retry,
            RecoveryStrategy.FALLBACK.value: self._execute_fallback,
            RecoveryStrategy.BACKUP_RESTORE.value: self._execute_backup_restore,
            RecoveryStrategy.RESET.value: self._execute_reset,
            RecoveryStrategy.MANUAL_INTERVENTION.value: self._execute_manual_intervention
        }
        
        # 최대 복구 지점 수
        self.max_recovery_points = 50
        
        # 기존 복구 지점 로드
        self._load_recovery_points()
    
    def create_recovery_point(
        self,
        operation_name: str,
        data_state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        복구 지점 생성
        
        Args:
            operation_name: 작업 이름
            data_state: 현재 데이터 상태
            metadata: 추가 메타데이터
            
        Returns:
            str: 복구 지점 ID
        """
        recovery_point_id = self._generate_recovery_point_id(operation_name)
        
        recovery_point = RecoveryPoint(
            id=recovery_point_id,
            timestamp=datetime.now(),
            operation_name=operation_name,
            data_state=data_state.copy(),
            metadata=metadata or {}
        )
        
        # 복구 지점 저장
        self.recovery_points[recovery_point_id] = recovery_point
        
        # 파일 시스템에 백업
        backup_file = self._save_recovery_point_to_file(recovery_point)
        recovery_point.file_path = backup_file
        
        # 오래된 복구 지점 정리
        self._cleanup_old_recovery_points()
        
        self.logger.info(f"Recovery point created: {recovery_point_id} for {operation_name}")
        
        return recovery_point_id
    
    def create_recovery_action(
        self,
        strategy: RecoveryStrategy,
        target_operation: str,
        recovery_point_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> str:
        """
        복구 액션 생성
        
        Args:
            strategy: 복구 전략
            target_operation: 대상 작업
            recovery_point_id: 복구 지점 ID
            parameters: 추가 파라미터
            max_retries: 최대 재시도 횟수
            
        Returns:
            str: 복구 액션 ID
        """
        action_id = self._generate_action_id()
        
        recovery_action = RecoveryAction(
            id=action_id,
            strategy=strategy,
            target_operation=target_operation,
            recovery_point_id=recovery_point_id,
            parameters=parameters or {},
            timestamp=datetime.now(),
            status=RecoveryStatus.PENDING,
            max_retries=max_retries
        )
        
        self.recovery_actions[action_id] = recovery_action
        
        self.logger.info(f"Recovery action created: {action_id} with strategy {strategy.value}")
        
        return action_id
    
    def execute_recovery(self, action_id: str) -> bool:
        """
        복구 실행
        
        Args:
            action_id: 복구 액션 ID
            
        Returns:
            bool: 복구 성공 여부
        """
        if action_id not in self.recovery_actions:
            self.logger.error(f"Recovery action not found: {action_id}")
            return False
        
        action = self.recovery_actions[action_id]
        action.status = RecoveryStatus.IN_PROGRESS
        
        try:
            # 복구 전략 실행
            strategy_func = self.recovery_strategies.get(action.strategy.value)
            if not strategy_func:
                raise ValueError(f"Unknown recovery strategy: {action.strategy}")
            
            success = strategy_func(action)
            
            if success:
                action.status = RecoveryStatus.SUCCESS
                action.result_message = "Recovery completed successfully"
                self.logger.info(f"Recovery action {action_id} completed successfully")
            else:
                action.status = RecoveryStatus.FAILED
                action.result_message = "Recovery failed"
                self.logger.error(f"Recovery action {action_id} failed")
            
            return success
            
        except Exception as e:
            action.status = RecoveryStatus.FAILED
            action.result_message = f"Recovery failed with exception: {str(e)}"
            self.logger.error(f"Recovery action {action_id} failed with exception: {e}")
            return False
    
    def auto_recover(
        self,
        operation_name: str,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        자동 복구 시도
        
        Args:
            operation_name: 실패한 작업 이름
            exception: 발생한 예외
            context: 추가 컨텍스트
            
        Returns:
            bool: 복구 성공 여부
        """
        # 복구 전략 결정
        strategy = self._determine_recovery_strategy(operation_name, exception, context)
        
        # 적절한 복구 지점 찾기
        recovery_point_id = self._find_suitable_recovery_point(operation_name)
        
        # 복구 액션 생성
        action_id = self.create_recovery_action(
            strategy=strategy,
            target_operation=operation_name,
            recovery_point_id=recovery_point_id,
            parameters=context or {}
        )
        
        # 복구 실행
        return self.execute_recovery(action_id)
    
    def _determine_recovery_strategy(
        self,
        operation_name: str,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> RecoveryStrategy:
        """복구 전략 결정"""
        exception_type = type(exception).__name__
        
        # 예외 타입별 전략 매핑
        strategy_map = {
            'FileNotFoundError': RecoveryStrategy.BACKUP_RESTORE,
            'PermissionError': RecoveryStrategy.RETRY,
            'ValueError': RecoveryStrategy.RESET,
            'KeyError': RecoveryStrategy.FALLBACK,
            'ConnectionError': RecoveryStrategy.RETRY,
            'MemoryError': RecoveryStrategy.RESET,
            'TimeoutError': RecoveryStrategy.RETRY,
        }
        
        # 컨텍스트 기반 조정
        if context and context.get('retry_count', 0) > 2:
            return RecoveryStrategy.BACKUP_RESTORE
        
        return strategy_map.get(exception_type, RecoveryStrategy.FALLBACK)
    
    def _find_suitable_recovery_point(self, operation_name: str) -> Optional[str]:
        """적절한 복구 지점 찾기"""
        # 같은 작업의 가장 최근 복구 지점 찾기
        suitable_points = [
            rp for rp in self.recovery_points.values()
            if rp.operation_name == operation_name
        ]
        
        if not suitable_points:
            return None
        
        # 가장 최근 것 반환
        latest_point = max(suitable_points, key=lambda rp: rp.timestamp)
        return latest_point.id
    
    # 복구 전략 실행 메서드들
    def _execute_retry(self, action: RecoveryAction) -> bool:
        """재시도 전략 실행"""
        if action.retry_count >= action.max_retries:
            action.result_message = f"Max retries ({action.max_retries}) exceeded"
            return False
        
        action.retry_count += 1
        
        # 재시도 간격 (지수 백오프)
        wait_time = min(2 ** action.retry_count, 30)
        time.sleep(wait_time)
        
        # Streamlit UI에 재시도 알림
        if 'streamlit_running' in globals():
            st.info(f"Retrying operation: {action.target_operation} (Attempt {action.retry_count})")
        
        action.result_message = f"Retry attempt {action.retry_count}"
        return True
    
    def _execute_fallback(self, action: RecoveryAction) -> bool:
        """대안 실행 전략"""
        operation_name = action.target_operation
        
        # 작업별 대안 매핑
        fallback_operations = {
            'advanced_factor_mining': 'basic_factor_mining',
            'complex_optimization': 'simple_optimization',
            'detailed_analysis': 'basic_analysis',
            'full_backtest': 'quick_backtest'
        }
        
        fallback_op = fallback_operations.get(operation_name)
        if not fallback_op:
            action.result_message = f"No fallback available for {operation_name}"
            return False
        
        # Streamlit UI에 대안 실행 알림
        if 'streamlit_running' in globals():
            st.warning(f"Using fallback operation: {fallback_op} instead of {operation_name}")
        
        # 세션 상태에 대안 작업 플래그 설정
        if hasattr(st, 'session_state'):
            st.session_state[f'fallback_mode_{operation_name}'] = fallback_op
        
        action.result_message = f"Fallback to {fallback_op}"
        return True
    
    def _execute_backup_restore(self, action: RecoveryAction) -> bool:
        """백업 복원 전략"""
        if not action.recovery_point_id:
            action.result_message = "No recovery point specified"
            return False
        
        recovery_point = self.recovery_points.get(action.recovery_point_id)
        if not recovery_point:
            action.result_message = f"Recovery point not found: {action.recovery_point_id}"
            return False
        
        try:
            # 데이터 상태 복원
            if hasattr(st, 'session_state'):
                for key, value in recovery_point.data_state.items():
                    st.session_state[key] = value
            
            # 파일 백업 복원
            if recovery_point.file_path and os.path.exists(recovery_point.file_path):
                self._restore_from_backup_file(recovery_point.file_path)
            
            # UI 알림
            if 'streamlit_running' in globals():
                st.success(f"Data restored from recovery point: {recovery_point.timestamp}")
            
            action.result_message = f"Restored from recovery point {action.recovery_point_id}"
            return True
            
        except Exception as e:
            action.result_message = f"Backup restore failed: {str(e)}"
            return False
    
    def _execute_reset(self, action: RecoveryAction) -> bool:
        """리셋 전략"""
        try:
            # 세션 상태 초기화
            if hasattr(st, 'session_state'):
                keys_to_reset = ['processed_data', 'factors_data', 'analysis_results', 'error_state']
                for key in keys_to_reset:
                    if key in st.session_state:
                        del st.session_state[key]
            
            # UI 알림
            if 'streamlit_running' in globals():
                st.info("Application state has been reset. Please reload your data.")
            
            action.result_message = "Application state reset completed"
            return True
            
        except Exception as e:
            action.result_message = f"Reset failed: {str(e)}"
            return False
    
    def _execute_manual_intervention(self, action: RecoveryAction) -> bool:
        """수동 개입 요청"""
        # UI에 수동 개입 요청 표시
        if 'streamlit_running' in globals():
            st.error("🚨 Manual intervention required!")
            st.error(f"Operation: {action.target_operation}")
            st.error("Please contact system administrator or restart the application.")
            
            # 로그 다운로드 버튼 제공
            if hasattr(self, 'get_error_logs'):
                error_logs = self.get_error_logs()
                st.download_button(
                    label="Download Error Logs",
                    data=error_logs,
                    file_name=f"error_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        action.result_message = "Manual intervention requested"
        return False  # 수동 개입은 자동 복구 실패로 간주
    
    def _save_recovery_point_to_file(self, recovery_point: RecoveryPoint) -> str:
        """복구 지점을 파일에 저장"""
        filename = f"recovery_point_{recovery_point.id}_{recovery_point.timestamp.strftime('%Y%m%d_%H%M%S')}.pkl"
        filepath = self.backup_dir / filename
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(recovery_point, f)
            
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save recovery point to file: {e}")
            return ""
    
    def _restore_from_backup_file(self, filepath: str) -> bool:
        """백업 파일에서 복원"""
        try:
            with open(filepath, 'rb') as f:
                recovery_point = pickle.load(f)
            
            # 데이터 복원 로직
            if hasattr(st, 'session_state'):
                for key, value in recovery_point.data_state.items():
                    st.session_state[key] = value
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore from backup file: {e}")
            return False
    
    def _load_recovery_points(self):
        """저장된 복구 지점 로드"""
        try:
            for filepath in self.backup_dir.glob("recovery_point_*.pkl"):
                try:
                    with open(filepath, 'rb') as f:
                        recovery_point = pickle.load(f)
                    
                    self.recovery_points[recovery_point.id] = recovery_point
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load recovery point from {filepath}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to load recovery points: {e}")
    
    def _cleanup_old_recovery_points(self):
        """오래된 복구 지점 정리"""
        if len(self.recovery_points) <= self.max_recovery_points:
            return
        
        # 오래된 복구 지점들을 제거
        sorted_points = sorted(
            self.recovery_points.values(),
            key=lambda rp: rp.timestamp,
            reverse=True
        )
        
        points_to_remove = sorted_points[self.max_recovery_points:]
        
        for point in points_to_remove:
            # 메모리에서 제거
            if point.id in self.recovery_points:
                del self.recovery_points[point.id]
            
            # 파일에서 제거
            if point.file_path and os.path.exists(point.file_path):
                try:
                    os.remove(point.file_path)
                except Exception as e:
                    self.logger.warning(f"Failed to remove backup file {point.file_path}: {e}")
    
    def _generate_recovery_point_id(self, operation_name: str) -> str:
        """복구 지점 ID 생성"""
        from uuid import uuid4
        return f"RP_{operation_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid4())[:8]}"
    
    def _generate_action_id(self) -> str:
        """액션 ID 생성"""
        from uuid import uuid4
        return f"RA_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid4())[:8]}"
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """복구 통계 조회"""
        return {
            "total_recovery_points": len(self.recovery_points),
            "total_recovery_actions": len(self.recovery_actions),
            "successful_recoveries": len([a for a in self.recovery_actions.values() 
                                        if a.status == RecoveryStatus.SUCCESS]),
            "failed_recoveries": len([a for a in self.recovery_actions.values() 
                                    if a.status == RecoveryStatus.FAILED]),
            "backup_storage_size": self._get_backup_storage_size()
        }
    
    def _get_backup_storage_size(self) -> int:
        """백업 저장소 크기 조회"""
        total_size = 0
        try:
            for filepath in self.backup_dir.glob("*"):
                if filepath.is_file():
                    total_size += filepath.stat().st_size
        except Exception:
            pass
        
        return total_size
    
    def cleanup_all_backups(self):
        """모든 백업 정리"""
        self.recovery_points.clear()
        self.recovery_actions.clear()
        
        try:
            for filepath in self.backup_dir.glob("*"):
                if filepath.is_file():
                    filepath.unlink()
        except Exception as e:
            self.logger.error(f"Failed to cleanup backups: {e}")


# 글로벌 복구 관리자 인스턴스
global_recovery_manager = RecoveryManager()


# 편의 함수들
def create_recovery_point(operation_name: str, **data_state) -> str:
    """복구 지점 생성 편의 함수"""
    return global_recovery_manager.create_recovery_point(operation_name, data_state)


def auto_recover(operation_name: str, exception: Exception, **context) -> bool:
    """자동 복구 편의 함수"""
    return global_recovery_manager.auto_recover(operation_name, exception, context)


def recovery_point_decorator(operation_name: str):
    """복구 지점 자동 생성 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 함수 실행 전 복구 지점 생성
            if hasattr(st, 'session_state'):
                recovery_point_id = create_recovery_point(
                    operation_name=operation_name,
                    session_state=dict(st.session_state)
                )
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # 자동 복구 시도
                recovery_success = auto_recover(
                    operation_name=operation_name,
                    exception=e,
                    function=func.__name__,
                    args=str(args)[:200],
                    kwargs=str(kwargs)[:200]
                )
                
                if not recovery_success:
                    raise e
                
                return None
        
        return wrapper
    return decorator