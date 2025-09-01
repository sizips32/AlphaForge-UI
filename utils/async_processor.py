"""
비동기 처리 시스템
Celery를 사용한 백그라운드 작업 처리 및 실시간 진행 상황 추적
"""

import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path

# Celery 관련 imports (선택적)
try:
    from celery import Celery, current_task
    from celery.result import AsyncResult
    from kombu import Queue
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    # Fallback for development without Redis/Celery
    class MockCelery:
        def task(self, *args, **kwargs):
            def decorator(func):
                func.delay = lambda *a, **kw: MockAsyncResult(func(*a, **kw))
                return func
            return decorator
    
    class MockAsyncResult:
        def __init__(self, result):
            self._result = result
            self.id = f"mock_{int(time.time())}"
        
        @property
        def result(self):
            return self._result
        
        @property
        def state(self):
            return 'SUCCESS'
        
        def ready(self):
            return True

import streamlit as st
from utils.logger import get_logger, log_performance
from utils.error_handler import handle_error, ErrorCategory, ErrorSeverity


class TaskStatus(Enum):
    """작업 상태"""
    PENDING = "PENDING"
    STARTED = "STARTED"
    PROGRESS = "PROGRESS"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


class TaskPriority(Enum):
    """작업 우선순위"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class TaskProgress:
    """작업 진행 상황"""
    task_id: str
    status: TaskStatus
    progress: float  # 0.0 - 1.0
    current_step: str
    total_steps: int
    completed_steps: int
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None


class AsyncTaskManager:
    """비동기 작업 관리자"""
    
    def __init__(self, broker_url: str = None, result_backend: str = None):
        """초기화"""
        self.logger = get_logger("async_task_manager")
        
        # Celery 설정
        if CELERY_AVAILABLE and broker_url and result_backend:
            self.celery_app = Celery(
                'alphaforge_tasks',
                broker=broker_url,
                backend=result_backend,
                include=['utils.async_processor']
            )
            
            # Celery 설정
            self.celery_app.conf.update(
                task_serializer='json',
                accept_content=['json'],
                result_serializer='json',
                timezone='UTC',
                enable_utc=True,
                task_track_started=True,
                task_time_limit=1800,  # 30분
                task_soft_time_limit=1500,  # 25분
                worker_prefetch_multiplier=1,
                task_acks_late=True,
                worker_max_tasks_per_child=1000,
                task_routes={
                    'utils.async_processor.process_data_async': {'queue': 'data_processing'},
                    'utils.async_processor.mine_factors_async': {'queue': 'factor_mining'},
                    'utils.async_processor.analyze_performance_async': {'queue': 'analysis'},
                    'utils.async_processor.generate_report_async': {'queue': 'reporting'},
                }
            )
            
            self.use_celery = True
        else:
            # Fallback 모드
            self.celery_app = MockCelery()
            self.use_celery = False
            self.logger.warning("Celery not available, using synchronous processing")
        
        # 작업 진행 상황 추적
        self.task_progress: Dict[str, TaskProgress] = {}
        self.active_tasks: Dict[str, Any] = {}
    
    def submit_task(
        self,
        task_func: str,
        args: tuple = (),
        kwargs: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        eta: Optional[datetime] = None,
        countdown: Optional[int] = None
    ) -> str:
        """
        작업 제출
        
        Args:
            task_func: 실행할 함수명
            args: 함수 인수
            kwargs: 함수 키워드 인수
            priority: 작업 우선순위
            eta: 예상 실행 시간
            countdown: 실행 대기 시간 (초)
            
        Returns:
            str: 작업 ID
        """
        kwargs = kwargs or {}
        
        try:
            # 작업 제출
            if self.use_celery:
                # Celery를 사용한 비동기 실행
                task = getattr(self.celery_app, task_func)
                
                apply_kwargs = {
                    'args': args,
                    'kwargs': kwargs,
                    'priority': priority.value
                }
                
                if eta:
                    apply_kwargs['eta'] = eta
                elif countdown:
                    apply_kwargs['countdown'] = countdown
                
                result = task.apply_async(**apply_kwargs)
                task_id = result.id
            else:
                # 동기 실행 (Fallback)
                task_func_obj = globals().get(task_func)
                if not task_func_obj:
                    raise ValueError(f"Task function {task_func} not found")
                
                result = task_func_obj(*args, **kwargs)
                task_id = f"sync_{int(time.time() * 1000)}"
            
            # 진행 상황 초기화
            self.task_progress[task_id] = TaskProgress(
                task_id=task_id,
                status=TaskStatus.PENDING,
                progress=0.0,
                current_step="작업 대기 중",
                total_steps=1,
                completed_steps=0,
                start_time=datetime.now()
            )
            
            self.active_tasks[task_id] = result if self.use_celery else None
            
            self.logger.info(f"Task submitted: {task_id} ({task_func})")
            return task_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit task {task_func}: {e}")
            handle_error(
                e, 
                ErrorCategory.SYSTEM, 
                ErrorSeverity.HIGH,
                context={'task_func': task_func, 'args': str(args)[:200]}
            )
            raise
    
    def get_task_status(self, task_id: str) -> Optional[TaskProgress]:
        """작업 상태 조회"""
        if task_id not in self.task_progress:
            return None
        
        progress = self.task_progress[task_id]
        
        # Celery 작업 상태 업데이트
        if self.use_celery and task_id in self.active_tasks:
            celery_result = self.active_tasks[task_id]
            
            if celery_result:
                celery_state = celery_result.state
                
                # 상태 매핑
                status_mapping = {
                    'PENDING': TaskStatus.PENDING,
                    'STARTED': TaskStatus.STARTED,
                    'PROGRESS': TaskStatus.PROGRESS,
                    'SUCCESS': TaskStatus.SUCCESS,
                    'FAILURE': TaskStatus.FAILURE,
                    'RETRY': TaskStatus.RETRY,
                    'REVOKED': TaskStatus.REVOKED
                }
                
                progress.status = status_mapping.get(celery_state, TaskStatus.PENDING)
                
                # 결과 또는 에러 정보 업데이트
                if celery_state == 'SUCCESS':
                    progress.result_data = celery_result.result
                    progress.progress = 1.0
                    progress.completed_steps = progress.total_steps
                elif celery_state == 'FAILURE':
                    progress.error_message = str(celery_result.result)
                elif celery_state == 'PROGRESS' and celery_result.info:
                    # 진행 상황 정보 업데이트
                    info = celery_result.info
                    if isinstance(info, dict):
                        progress.progress = info.get('progress', progress.progress)
                        progress.current_step = info.get('current_step', progress.current_step)
                        progress.completed_steps = info.get('completed_steps', progress.completed_steps)
        
        return progress
    
    def update_task_progress(
        self,
        task_id: str,
        progress: float,
        current_step: str,
        completed_steps: Optional[int] = None
    ):
        """작업 진행 상황 업데이트"""
        if task_id in self.task_progress:
            task_progress = self.task_progress[task_id]
            task_progress.progress = min(max(progress, 0.0), 1.0)
            task_progress.current_step = current_step
            task_progress.status = TaskStatus.PROGRESS
            
            if completed_steps is not None:
                task_progress.completed_steps = completed_steps
            
            # 완료 시간 추정
            if progress > 0:
                elapsed_time = datetime.now() - task_progress.start_time
                estimated_total_time = elapsed_time / progress
                task_progress.estimated_completion = task_progress.start_time + estimated_total_time
            
            # Celery 진행 상황 업데이트
            if self.use_celery and current_task:
                current_task.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': progress,
                        'current_step': current_step,
                        'completed_steps': completed_steps
                    }
                )
    
    def wait_for_task(self, task_id: str, timeout: Optional[int] = None) -> Optional[Any]:
        """작업 완료 대기"""
        if not self.use_celery:
            # 동기 모드에서는 즉시 반환
            progress = self.get_task_status(task_id)
            return progress.result_data if progress else None
        
        if task_id not in self.active_tasks:
            return None
        
        try:
            celery_result = self.active_tasks[task_id]
            result = celery_result.get(timeout=timeout)
            
            # 최종 상태 업데이트
            if task_id in self.task_progress:
                self.task_progress[task_id].status = TaskStatus.SUCCESS
                self.task_progress[task_id].result_data = result
                self.task_progress[task_id].progress = 1.0
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task {task_id} failed: {e}")
            if task_id in self.task_progress:
                self.task_progress[task_id].status = TaskStatus.FAILURE
                self.task_progress[task_id].error_message = str(e)
            return None
    
    def cancel_task(self, task_id: str) -> bool:
        """작업 취소"""
        try:
            if self.use_celery and task_id in self.active_tasks:
                celery_result = self.active_tasks[task_id]
                celery_result.revoke(terminate=True)
                
                if task_id in self.task_progress:
                    self.task_progress[task_id].status = TaskStatus.REVOKED
                
                self.logger.info(f"Task cancelled: {task_id}")
                return True
            else:
                # 동기 모드에서는 취소 불가
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    def get_active_tasks(self) -> List[TaskProgress]:
        """활성 작업 목록 조회"""
        active_tasks = []
        
        for task_id in list(self.task_progress.keys()):
            progress = self.get_task_status(task_id)
            if progress and progress.status not in [TaskStatus.SUCCESS, TaskStatus.FAILURE, TaskStatus.REVOKED]:
                active_tasks.append(progress)
        
        return active_tasks
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """완료된 작업 정리"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        tasks_to_remove = []
        for task_id, progress in self.task_progress.items():
            if (progress.status in [TaskStatus.SUCCESS, TaskStatus.FAILURE, TaskStatus.REVOKED] and
                progress.start_time < cutoff_time):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.task_progress[task_id]
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
        
        self.logger.info(f"Cleaned up {len(tasks_to_remove)} completed tasks")


# 글로벌 비동기 작업 관리자
async_manager = AsyncTaskManager(
    broker_url=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)


# Celery 작업 함수들
def process_data_async(data_dict: Dict[str, Any], processing_options: Dict[str, Any] = None) -> Dict[str, Any]:
    """비동기 데이터 처리"""
    try:
        from utils.data_processor import DataProcessor
        
        processor = DataProcessor()
        task_id = current_task.request.id if CELERY_AVAILABLE and current_task else "sync"
        
        # 진행 상황 업데이트
        async_manager.update_task_progress(task_id, 0.1, "데이터 로딩 중")
        
        # DataFrame 복원
        df = pd.DataFrame(data_dict)
        
        async_manager.update_task_progress(task_id, 0.3, "데이터 검증 중")
        
        # 데이터 처리
        processed_data = processor.process_data(df, **(processing_options or {}))
        
        async_manager.update_task_progress(task_id, 0.8, "결과 준비 중")
        
        # 결과 직렬화
        result = {
            'processed_data': processed_data.to_dict('records'),
            'columns': processed_data.columns.tolist(),
            'shape': processed_data.shape,
            'processing_time': time.time()
        }
        
        async_manager.update_task_progress(task_id, 1.0, "완료")
        
        return result
        
    except Exception as e:
        async_manager.update_task_progress(task_id, 0.0, f"오류 발생: {str(e)}")
        raise


def mine_factors_async(data_dict: Dict[str, Any], mining_options: Dict[str, Any] = None) -> Dict[str, Any]:
    """비동기 팩터 마이닝"""
    try:
        from utils.factor_miner import FactorMiner
        
        miner = FactorMiner()
        task_id = current_task.request.id if CELERY_AVAILABLE and current_task else "sync"
        
        async_manager.update_task_progress(task_id, 0.1, "팩터 마이닝 시작")
        
        # DataFrame 복원
        df = pd.DataFrame(data_dict)
        
        async_manager.update_task_progress(task_id, 0.3, "기본 팩터 생성 중")
        
        # 팩터 생성
        factors = miner.generate_basic_factors(df)
        
        async_manager.update_task_progress(task_id, 0.6, "팩터 성능 분석 중")
        
        # 성능 분석
        performance = miner.calculate_factor_performance(factors)
        
        async_manager.update_task_progress(task_id, 0.9, "결과 준비 중")
        
        result = {
            'factors': factors.to_dict('records'),
            'performance': performance,
            'factor_count': len([col for col in factors.columns if col.startswith('factor_')]),
            'mining_time': time.time()
        }
        
        async_manager.update_task_progress(task_id, 1.0, "완료")
        
        return result
        
    except Exception as e:
        async_manager.update_task_progress(task_id, 0.0, f"오류 발생: {str(e)}")
        raise


def analyze_performance_async(data_dict: Dict[str, Any], analysis_options: Dict[str, Any] = None) -> Dict[str, Any]:
    """비동기 성능 분석"""
    try:
        from utils.performance_analyzer import PerformanceAnalyzer
        
        analyzer = PerformanceAnalyzer()
        task_id = current_task.request.id if CELERY_AVAILABLE and current_task else "sync"
        
        async_manager.update_task_progress(task_id, 0.1, "성능 분석 시작")
        
        # 데이터 복원
        df = pd.DataFrame(data_dict)
        returns = df.get('returns', pd.Series())
        
        async_manager.update_task_progress(task_id, 0.4, "성능 메트릭 계산 중")
        
        # 기본 메트릭 계산
        metrics = {}
        if not returns.empty:
            metrics['sharpe_ratio'] = analyzer.calculate_sharpe_ratio(returns)
            metrics['max_drawdown'] = analyzer.calculate_max_drawdown((1 + returns).cumprod())
            
        async_manager.update_task_progress(task_id, 0.7, "리스크 분석 중")
        
        # 리스크 메트릭
        risk_metrics = analyzer.calculate_risk_metrics(returns) if not returns.empty else {}
        
        async_manager.update_task_progress(task_id, 0.9, "리포트 생성 중")
        
        result = {
            'performance_metrics': metrics,
            'risk_metrics': risk_metrics,
            'analysis_time': time.time()
        }
        
        async_manager.update_task_progress(task_id, 1.0, "완료")
        
        return result
        
    except Exception as e:
        async_manager.update_task_progress(task_id, 0.0, f"오류 발생: {str(e)}")
        raise


# Celery 작업 등록 (Celery가 사용 가능한 경우)
if CELERY_AVAILABLE and async_manager.use_celery:
    process_data_async = async_manager.celery_app.task(process_data_async)
    mine_factors_async = async_manager.celery_app.task(mine_factors_async)
    analyze_performance_async = async_manager.celery_app.task(analyze_performance_async)


# Streamlit 통합 함수들
def show_task_monitor():
    """작업 모니터링 UI 표시"""
    st.subheader("🔄 작업 모니터링")
    
    # 활성 작업 조회
    active_tasks = async_manager.get_active_tasks()
    
    if not active_tasks:
        st.info("현재 실행 중인 작업이 없습니다.")
        return
    
    for task_progress in active_tasks:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.text(f"작업 ID: {task_progress.task_id}")
                st.text(f"현재 단계: {task_progress.current_step}")
                
                # 진행률 표시
                progress_bar = st.progress(task_progress.progress)
                st.text(f"진행률: {task_progress.progress*100:.1f}%")
            
            with col2:
                st.metric("상태", task_progress.status.value)
                st.metric("완료 단계", f"{task_progress.completed_steps}/{task_progress.total_steps}")
            
            with col3:
                # 예상 완료 시간
                if task_progress.estimated_completion:
                    remaining = task_progress.estimated_completion - datetime.now()
                    if remaining.total_seconds() > 0:
                        st.metric("남은 시간", f"{remaining.total_seconds():.0f}초")
                
                # 작업 취소 버튼
                if st.button(f"취소", key=f"cancel_{task_progress.task_id}"):
                    if async_manager.cancel_task(task_progress.task_id):
                        st.success("작업이 취소되었습니다.")
                        st.experimental_rerun()
                    else:
                        st.error("작업 취소에 실패했습니다.")
        
        st.divider()


def submit_async_task(task_name: str, data: pd.DataFrame, options: Dict[str, Any] = None) -> str:
    """비동기 작업 제출 편의 함수"""
    
    # 데이터 직렬화
    data_dict = data.to_dict('records') if not data.empty else []
    
    # 작업 함수 매핑
    task_functions = {
        'data_processing': 'process_data_async',
        'factor_mining': 'mine_factors_async',
        'performance_analysis': 'analyze_performance_async'
    }
    
    task_func = task_functions.get(task_name)
    if not task_func:
        raise ValueError(f"Unknown task: {task_name}")
    
    # 작업 제출
    task_id = async_manager.submit_task(
        task_func=task_func,
        kwargs={
            'data_dict': data_dict,
            f'{task_name.split("_")[0]}_options': options or {}
        },
        priority=TaskPriority.NORMAL
    )
    
    return task_id