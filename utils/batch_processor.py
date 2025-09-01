"""
배치 처리 최적화 시스템
대용량 데이터의 효율적인 배치 처리 및 메모리 최적화
"""

import os
import gc
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Iterator, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import tempfile

from utils.logger import get_logger, log_performance
from utils.error_handler import handle_error, ErrorCategory, ErrorSeverity
from utils.realtime_updates import realtime_manager, UpdateType


class ProcessingMode(Enum):
    """처리 모드"""
    SINGLE_THREAD = "single_thread"
    MULTI_THREAD = "multi_thread"
    MULTI_PROCESS = "multi_process"
    HYBRID = "hybrid"


class BatchSize(Enum):
    """배치 크기"""
    SMALL = 1000
    MEDIUM = 5000
    LARGE = 10000
    XLARGE = 25000
    AUTO = -1


@dataclass
class BatchConfig:
    """배치 처리 설정"""
    batch_size: int
    processing_mode: ProcessingMode
    max_workers: int
    memory_limit_mb: int
    timeout_seconds: int
    checkpoint_interval: int
    temp_dir: Optional[str] = None
    use_compression: bool = True
    progress_callback: Optional[Callable] = None


@dataclass
class BatchResult:
    """배치 처리 결과"""
    total_items: int
    processed_items: int
    failed_items: int
    processing_time: float
    memory_usage_mb: float
    results: List[Any]
    errors: List[str]
    checkpoints: List[str]


class MemoryMonitor:
    """메모리 모니터링"""
    
    def __init__(self, limit_mb: int = 4000):
        """초기화"""
        self.limit_mb = limit_mb
        self.logger = get_logger("memory_monitor")
        self.initial_memory = self.get_current_memory()
    
    def get_current_memory(self) -> float:
        """현재 메모리 사용량 조회 (MB)"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb
    
    def check_memory_limit(self) -> bool:
        """메모리 제한 확인"""
        current_memory = self.get_current_memory()
        return current_memory > self.limit_mb
    
    def force_garbage_collection(self):
        """강제 가비지 컬렉션"""
        collected = gc.collect()
        self.logger.debug(f"Garbage collection freed {collected} objects")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """메모리 통계 조회"""
        current = self.get_current_memory()
        return {
            'current_mb': current,
            'limit_mb': self.limit_mb,
            'usage_ratio': current / self.limit_mb,
            'increase_mb': current - self.initial_memory
        }


class BatchProcessor:
    """배치 처리기"""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """초기화"""
        self.logger = get_logger("batch_processor")
        
        # 기본 설정
        if config is None:
            config = BatchConfig(
                batch_size=BatchSize.AUTO.value,
                processing_mode=ProcessingMode.MULTI_THREAD,
                max_workers=min(8, mp.cpu_count()),
                memory_limit_mb=4000,
                timeout_seconds=1800,
                checkpoint_interval=10,
                temp_dir=tempfile.gettempdir()
            )
        
        self.config = config
        self.memory_monitor = MemoryMonitor(config.memory_limit_mb)
        
        # 상태 추적
        self.is_running = False
        self.current_batch = 0
        self.total_batches = 0
        self.start_time = None
        
        # 체크포인트 관리
        self.checkpoint_dir = Path(config.temp_dir or tempfile.gettempdir()) / "alphaforge_checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        process_func: Callable,
        *args,
        **kwargs
    ) -> BatchResult:
        """DataFrame 배치 처리"""
        self.logger.info(f"Starting batch processing of {len(df)} rows")
        self.start_time = time.time()
        self.is_running = True
        
        try:
            # 배치 크기 자동 결정
            if self.config.batch_size == BatchSize.AUTO.value:
                batch_size = self._determine_optimal_batch_size(df)
            else:
                batch_size = self.config.batch_size
            
            # 배치 분할
            batches = self._split_dataframe(df, batch_size)
            self.total_batches = len(batches)
            
            self.logger.info(f"Processing {self.total_batches} batches of size {batch_size}")
            
            # 결과 수집
            all_results = []
            all_errors = []
            processed_items = 0
            checkpoints = []
            
            # 배치 처리 실행
            for batch_idx, batch_df in enumerate(batches):
                self.current_batch = batch_idx + 1
                
                try:
                    # 메모리 체크
                    if self.memory_monitor.check_memory_limit():
                        self.memory_monitor.force_garbage_collection()
                        
                        if self.memory_monitor.check_memory_limit():
                            self.logger.warning("Memory limit exceeded, creating checkpoint")
                            checkpoint_path = self._create_checkpoint(all_results, batch_idx)
                            checkpoints.append(checkpoint_path)
                            
                            # 결과 메모리에서 제거
                            all_results.clear()
                            gc.collect()
                    
                    # 배치 처리 실행
                    batch_result = self._process_batch(batch_df, process_func, *args, **kwargs)
                    
                    if batch_result is not None:
                        if isinstance(batch_result, list):
                            all_results.extend(batch_result)
                        else:
                            all_results.append(batch_result)
                        
                        processed_items += len(batch_df)
                    
                    # 진행 상황 업데이트
                    progress = self.current_batch / self.total_batches
                    self._update_progress(progress, f"배치 {self.current_batch}/{self.total_batches} 처리 완료")
                    
                    # 체크포인트 생성
                    if (batch_idx + 1) % self.config.checkpoint_interval == 0:
                        checkpoint_path = self._create_checkpoint(all_results, batch_idx)
                        checkpoints.append(checkpoint_path)
                
                except Exception as e:
                    error_msg = f"Batch {batch_idx + 1} failed: {str(e)}"
                    self.logger.error(error_msg)
                    all_errors.append(error_msg)
                    
                    handle_error(
                        e,
                        ErrorCategory.DATA_PROCESSING,
                        ErrorSeverity.MEDIUM,
                        context={'batch_idx': batch_idx, 'batch_size': len(batch_df)}
                    )
            
            # 최종 결과 생성
            processing_time = time.time() - self.start_time
            memory_stats = self.memory_monitor.get_memory_stats()
            
            result = BatchResult(
                total_items=len(df),
                processed_items=processed_items,
                failed_items=len(df) - processed_items,
                processing_time=processing_time,
                memory_usage_mb=memory_stats['current_mb'],
                results=all_results,
                errors=all_errors,
                checkpoints=checkpoints
            )
            
            # 성능 로깅
            log_performance(
                "batch_processing",
                processing_time,
                {
                    'total_items': len(df),
                    'processed_items': processed_items,
                    'batch_count': self.total_batches,
                    'processing_mode': self.config.processing_mode.value,
                    'memory_usage_mb': memory_stats['current_mb']
                }
            )
            
            self.logger.info(f"Batch processing completed: {processed_items}/{len(df)} items processed")
            
            return result
            
        finally:
            self.is_running = False
            self.current_batch = 0
            self.total_batches = 0
    
    def _determine_optimal_batch_size(self, df: pd.DataFrame) -> int:
        """최적 배치 크기 결정"""
        # 메모리 기반 계산
        memory_per_row = df.memory_usage(deep=True).sum() / len(df) / 1024 / 1024  # MB per row
        available_memory = self.config.memory_limit_mb * 0.5  # 50% 여유
        
        optimal_batch_size = int(available_memory / memory_per_row)
        
        # 최소/최대 제한
        optimal_batch_size = max(100, min(optimal_batch_size, 50000))
        
        self.logger.info(f"Determined optimal batch size: {optimal_batch_size} (memory per row: {memory_per_row:.2f} MB)")
        
        return optimal_batch_size
    
    def _split_dataframe(self, df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
        """DataFrame 배치 분할"""
        batches = []
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size].copy()
            batches.append(batch_df)
        
        return batches
    
    def _process_batch(
        self,
        batch_df: pd.DataFrame,
        process_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """단일 배치 처리"""
        try:
            if self.config.processing_mode == ProcessingMode.SINGLE_THREAD:
                return process_func(batch_df, *args, **kwargs)
            
            elif self.config.processing_mode == ProcessingMode.MULTI_THREAD:
                return self._process_batch_multithreaded(batch_df, process_func, *args, **kwargs)
            
            elif self.config.processing_mode == ProcessingMode.MULTI_PROCESS:
                return self._process_batch_multiprocess(batch_df, process_func, *args, **kwargs)
            
            elif self.config.processing_mode == ProcessingMode.HYBRID:
                return self._process_batch_hybrid(batch_df, process_func, *args, **kwargs)
            
            else:
                return process_func(batch_df, *args, **kwargs)
                
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            raise
    
    def _process_batch_multithreaded(
        self,
        batch_df: pd.DataFrame,
        process_func: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """멀티스레드 배치 처리"""
        # DataFrame을 더 작은 청크로 분할
        chunk_size = max(1, len(batch_df) // self.config.max_workers)
        chunks = [batch_df.iloc[i:i + chunk_size] for i in range(0, len(batch_df), chunk_size)]
        
        results = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_chunk = {
                executor.submit(process_func, chunk, *args, **kwargs): chunk
                for chunk in chunks
            }
            
            for future in as_completed(future_to_chunk, timeout=self.config.timeout_seconds):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Thread processing failed: {e}")
        
        return results
    
    def _process_batch_multiprocess(
        self,
        batch_df: pd.DataFrame,
        process_func: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """멀티프로세스 배치 처리"""
        # 프로세스 간 데이터 공유를 위해 임시 파일 사용
        temp_files = []
        chunk_size = max(1, len(batch_df) // self.config.max_workers)
        
        try:
            # 청크를 임시 파일로 저장
            chunks_info = []
            for i in range(0, len(batch_df), chunk_size):
                chunk = batch_df.iloc[i:i + chunk_size]
                
                temp_file = tempfile.NamedTemporaryFile(
                    suffix='.parquet',
                    dir=self.config.temp_dir,
                    delete=False
                )
                chunk.to_parquet(temp_file.name, compression='gzip' if self.config.use_compression else None)
                temp_files.append(temp_file.name)
                chunks_info.append(temp_file.name)
            
            # 멀티프로세싱 실행
            results = []
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_file = {
                    executor.submit(self._process_chunk_from_file, file_path, process_func, *args, **kwargs): file_path
                    for file_path in chunks_info
                }
                
                for future in as_completed(future_to_file, timeout=self.config.timeout_seconds):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        self.logger.error(f"Process processing failed: {e}")
            
            return results
            
        finally:
            # 임시 파일 정리
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
    
    def _process_batch_hybrid(
        self,
        batch_df: pd.DataFrame,
        process_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """하이브리드 배치 처리 (프로세스 + 스레드)"""
        # CPU 집약적 작업은 멀티프로세스, I/O 집약적 작업은 멀티스레드
        if len(batch_df) > 10000:
            return self._process_batch_multiprocess(batch_df, process_func, *args, **kwargs)
        else:
            return self._process_batch_multithreaded(batch_df, process_func, *args, **kwargs)
    
    def _process_chunk_from_file(
        self,
        file_path: str,
        process_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """파일에서 청크를 로드하여 처리"""
        try:
            chunk_df = pd.read_parquet(file_path)
            return process_func(chunk_df, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"Failed to process chunk from file {file_path}: {e}")
            return None
    
    def _create_checkpoint(self, results: List[Any], batch_idx: int) -> str:
        """체크포인트 생성"""
        checkpoint_filename = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{batch_idx}.pkl"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        try:
            import pickle
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'batch_idx': batch_idx,
                    'results': results,
                    'timestamp': datetime.now(),
                    'memory_stats': self.memory_monitor.get_memory_stats()
                }, f)
            
            self.logger.info(f"Checkpoint created: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            return ""
    
    def _load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """체크포인트 로드"""
        try:
            import pickle
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def _update_progress(self, progress: float, message: str):
        """진행 상황 업데이트"""
        if self.config.progress_callback:
            self.config.progress_callback(progress, message)
        
        # 실시간 업데이트 전송
        realtime_manager.send_progress_update(
            task_id=f"batch_processing_{id(self)}",
            progress=progress,
            current_step=message
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 조회"""
        memory_stats = self.memory_monitor.get_memory_stats()
        
        stats = {
            'is_running': self.is_running,
            'current_batch': self.current_batch,
            'total_batches': self.total_batches,
            'progress': self.current_batch / self.total_batches if self.total_batches > 0 else 0,
            'memory_stats': memory_stats,
            'config': {
                'batch_size': self.config.batch_size,
                'processing_mode': self.config.processing_mode.value,
                'max_workers': self.config.max_workers
            }
        }
        
        if self.start_time:
            stats['elapsed_time'] = time.time() - self.start_time
        
        return stats
    
    def cleanup_checkpoints(self, max_age_hours: int = 24):
        """오래된 체크포인트 정리"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        removed_count = 0
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pkl"):
            try:
                file_time = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
                if file_time < cutoff_time:
                    checkpoint_file.unlink()
                    removed_count += 1
            except OSError:
                pass
        
        self.logger.info(f"Cleaned up {removed_count} old checkpoints")


# 특화된 배치 처리 함수들
def batch_factor_mining(
    df: pd.DataFrame,
    config: Optional[BatchConfig] = None
) -> BatchResult:
    """팩터 마이닝 배치 처리"""
    from utils.factor_miner import FactorMiner
    
    def process_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
        miner = FactorMiner()
        return miner.generate_basic_factors(chunk_df)
    
    processor = BatchProcessor(config)
    return processor.process_dataframe(df, process_chunk)


def batch_performance_analysis(
    df: pd.DataFrame,
    config: Optional[BatchConfig] = None
) -> BatchResult:
    """성능 분석 배치 처리"""
    from utils.performance_analyzer import PerformanceAnalyzer
    
    def process_chunk(chunk_df: pd.DataFrame) -> Dict[str, Any]:
        analyzer = PerformanceAnalyzer()
        if 'returns' in chunk_df.columns and not chunk_df['returns'].empty:
            return {
                'sharpe_ratio': analyzer.calculate_sharpe_ratio(chunk_df['returns']),
                'max_drawdown': analyzer.calculate_max_drawdown((1 + chunk_df['returns']).cumprod()),
                'chunk_size': len(chunk_df)
            }
        return {'chunk_size': len(chunk_df)}
    
    processor = BatchProcessor(config)
    return processor.process_dataframe(df, process_chunk)


def batch_data_validation(
    df: pd.DataFrame,
    config: Optional[BatchConfig] = None
) -> BatchResult:
    """데이터 검증 배치 처리"""
    from utils.validators import DataValidator
    
    def process_chunk(chunk_df: pd.DataFrame) -> Dict[str, Any]:
        validator = DataValidator()
        result = validator.validate_data(chunk_df)
        return {
            'is_valid': result['is_valid'],
            'quality_score': result['quality_score'],
            'errors': result.get('errors', []),
            'chunk_size': len(chunk_df)
        }
    
    processor = BatchProcessor(config)
    return processor.process_dataframe(df, process_chunk)


# Streamlit 통합
def show_batch_processing_monitor(processor: BatchProcessor):
    """배치 처리 모니터링 UI"""
    import streamlit as st
    
    st.subheader("⚡ 배치 처리 모니터")
    
    stats = processor.get_processing_stats()
    
    # 진행 상황 표시
    if stats['is_running']:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("현재 배치", f"{stats['current_batch']}/{stats['total_batches']}")
        
        with col2:
            st.metric("진행률", f"{stats['progress']*100:.1f}%")
        
        with col3:
            if 'elapsed_time' in stats:
                st.metric("경과 시간", f"{stats['elapsed_time']:.1f}초")
        
        # 진행률 바
        st.progress(stats['progress'])
        
        # 메모리 사용량
        memory_stats = stats['memory_stats']
        st.subheader("📊 메모리 사용량")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "현재 사용량",
                f"{memory_stats['current_mb']:.1f} MB",
                delta=f"{memory_stats['increase_mb']:.1f} MB"
            )
        
        with col2:
            usage_ratio = memory_stats['usage_ratio']
            st.metric(
                "사용률",
                f"{usage_ratio*100:.1f}%",
                delta_color="inverse" if usage_ratio > 0.8 else "normal"
            )
        
        # 메모리 사용률 바
        st.progress(min(usage_ratio, 1.0))
        
    else:
        st.info("현재 실행 중인 배치 처리가 없습니다.")
    
    # 설정 정보
    with st.expander("⚙️ 배치 처리 설정"):
        config_stats = stats['config']
        st.json(config_stats)


def create_optimized_batch_config(
    df_size: int,
    available_memory_mb: int = 4000,
    cpu_cores: int = None
) -> BatchConfig:
    """최적화된 배치 설정 생성"""
    if cpu_cores is None:
        cpu_cores = mp.cpu_count()
    
    # 데이터 크기별 최적 설정
    if df_size < 10000:
        # 소규모 데이터
        return BatchConfig(
            batch_size=BatchSize.SMALL.value,
            processing_mode=ProcessingMode.SINGLE_THREAD,
            max_workers=1,
            memory_limit_mb=available_memory_mb,
            timeout_seconds=300,
            checkpoint_interval=5
        )
    elif df_size < 100000:
        # 중간 규모 데이터
        return BatchConfig(
            batch_size=BatchSize.MEDIUM.value,
            processing_mode=ProcessingMode.MULTI_THREAD,
            max_workers=min(4, cpu_cores),
            memory_limit_mb=available_memory_mb,
            timeout_seconds=900,
            checkpoint_interval=10
        )
    else:
        # 대규모 데이터
        return BatchConfig(
            batch_size=BatchSize.AUTO.value,
            processing_mode=ProcessingMode.HYBRID,
            max_workers=min(8, cpu_cores),
            memory_limit_mb=available_memory_mb,
            timeout_seconds=1800,
            checkpoint_interval=5,
            use_compression=True
        )