"""
Dask 대용량 데이터 처리 시스템
분산 컴퓨팅을 위한 Dask 통합 및 대용량 데이터셋 처리
"""

import os
import time
import warnings
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path

# Dask 관련 imports (선택적)
try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster, as_completed
    from dask import delayed, compute
    from dask.diagnostics import ProgressBar
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    # Fallback 클래스들
    class MockDaskDataFrame:
        def __init__(self, df):
            self._df = df
        
        def compute(self):
            return self._df
        
        def __getattr__(self, name):
            return getattr(self._df, name)

from utils.logger import get_logger, log_performance
from utils.error_handler import handle_error, ErrorCategory, ErrorSeverity
from utils.realtime_updates import realtime_manager, UpdateType


class DaskBackend(Enum):
    """Dask 백엔드"""
    THREADS = "threads"
    PROCESSES = "processes"
    SYNCHRONOUS = "synchronous"
    DISTRIBUTED = "distributed"


class ComputeMode(Enum):
    """계산 모드"""
    LAZY = "lazy"
    EAGER = "eager"
    STREAMING = "streaming"


@dataclass
class DaskConfig:
    """Dask 설정"""
    backend: DaskBackend
    n_workers: int
    threads_per_worker: int
    memory_limit: str
    chunk_size: Union[int, str]
    compute_mode: ComputeMode
    scheduler_address: Optional[str] = None
    temporary_directory: Optional[str] = None
    dashboard_address: Optional[str] = None


@dataclass
class ProcessingResult:
    """처리 결과"""
    result: Any
    processing_time: float
    memory_usage: Dict[str, Any]
    task_graph_size: int
    chunk_info: Dict[str, Any]


class DaskProcessor:
    """Dask 대용량 데이터 처리기"""
    
    def __init__(self, config: Optional[DaskConfig] = None):
        """초기화"""
        self.logger = get_logger("dask_processor")
        
        if not DASK_AVAILABLE:
            self.logger.warning("Dask not available, falling back to pandas")
            self.use_dask = False
            return
        
        self.use_dask = True
        
        # 기본 설정
        if config is None:
            config = DaskConfig(
                backend=DaskBackend.THREADS,
                n_workers=4,
                threads_per_worker=2,
                memory_limit="2GB",
                chunk_size="100MB",
                compute_mode=ComputeMode.LAZY
            )
        
        self.config = config
        self.client = None
        self.cluster = None
        
        # Dask 설정 적용
        self._configure_dask()
        self._setup_client()
        
        # 성능 추적
        self.task_history: List[Dict[str, Any]] = []
    
    def _configure_dask(self):
        """Dask 전역 설정"""
        # 경고 억제
        warnings.filterwarnings('ignore', category=UserWarning, module='dask')
        
        # 기본 설정
        dask.config.set({
            'dataframe.query-planning': False,  # 안정성을 위해 비활성화
            'array.slicing.split_large_chunks': True,
            'optimization.fuse': {},  # 최적화 비활성화 (안정성 우선)
            'distributed.worker.daemon': False
        })
        
        if self.config.temporary_directory:
            dask.config.set({'temporary-directory': self.config.temporary_directory})
    
    def _setup_client(self):
        """Dask 클라이언트 설정"""
        try:
            if self.config.backend == DaskBackend.DISTRIBUTED and self.config.scheduler_address:
                # 기존 분산 클러스터에 연결
                self.client = Client(self.config.scheduler_address)
                self.logger.info(f"Connected to distributed cluster: {self.config.scheduler_address}")
            
            elif self.config.backend == DaskBackend.DISTRIBUTED:
                # 로컬 클러스터 생성
                self.cluster = LocalCluster(
                    n_workers=self.config.n_workers,
                    threads_per_worker=self.config.threads_per_worker,
                    memory_limit=self.config.memory_limit,
                    dashboard_address=self.config.dashboard_address or ':0'
                )
                self.client = Client(self.cluster)
                self.logger.info(f"Created local cluster with {self.config.n_workers} workers")
            
            else:
                # 스레드 또는 프로세스 백엔드
                self.client = None
                self.logger.info(f"Using {self.config.backend.value} backend")
        
        except Exception as e:
            self.logger.error(f"Failed to setup Dask client: {e}")
            self.client = None
            self.cluster = None
    
    def load_data(
        self,
        data_source: Union[str, pd.DataFrame, List[str]],
        file_format: str = 'auto',
        **read_kwargs
    ) -> Union[dd.DataFrame, pd.DataFrame]:
        """데이터 로드"""
        if not self.use_dask:
            # Pandas fallback
            if isinstance(data_source, str):
                if data_source.endswith('.csv'):
                    return pd.read_csv(data_source, **read_kwargs)
                elif data_source.endswith('.parquet'):
                    return pd.read_parquet(data_source, **read_kwargs)
            elif isinstance(data_source, pd.DataFrame):
                return data_source
            else:
                raise ValueError("Unsupported data source for pandas fallback")
        
        try:
            if isinstance(data_source, pd.DataFrame):
                # DataFrame을 Dask DataFrame으로 변환
                chunk_size = self._parse_chunk_size(len(data_source))
                ddf = dd.from_pandas(data_source, npartitions=max(1, len(data_source) // chunk_size))
                
                self.logger.info(f"Converted pandas DataFrame to Dask with {ddf.npartitions} partitions")
                return ddf
            
            elif isinstance(data_source, str):
                # 파일에서 로드
                if file_format == 'auto':
                    file_format = Path(data_source).suffix[1:]  # 확장자에서 . 제거
                
                if file_format.lower() == 'csv':
                    ddf = dd.read_csv(
                        data_source,
                        blocksize=self.config.chunk_size,
                        **read_kwargs
                    )
                elif file_format.lower() == 'parquet':
                    ddf = dd.read_parquet(data_source, **read_kwargs)
                elif file_format.lower() == 'json':
                    ddf = dd.read_json(
                        data_source,
                        blocksize=self.config.chunk_size,
                        **read_kwargs
                    )
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")
                
                self.logger.info(f"Loaded {file_format} data with {ddf.npartitions} partitions")
                return ddf
            
            elif isinstance(data_source, list):
                # 여러 파일에서 로드
                if all(f.endswith('.csv') for f in data_source):
                    ddf = dd.read_csv(data_source, blocksize=self.config.chunk_size, **read_kwargs)
                elif all(f.endswith('.parquet') for f in data_source):
                    ddf = dd.read_parquet(data_source, **read_kwargs)
                else:
                    raise ValueError("Mixed file formats not supported")
                
                self.logger.info(f"Loaded {len(data_source)} files with {ddf.npartitions} partitions")
                return ddf
            
            else:
                raise ValueError(f"Unsupported data source type: {type(data_source)}")
        
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            handle_error(
                e,
                ErrorCategory.DATA_PROCESSING,
                ErrorSeverity.HIGH,
                context={'data_source': str(data_source)[:200]}
            )
            raise
    
    def process_data(
        self,
        ddf: Union[dd.DataFrame, pd.DataFrame],
        operations: List[Callable],
        compute_result: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """데이터 처리"""
        start_time = time.time()
        
        try:
            # Dask가 사용 불가능한 경우 pandas로 처리
            if not self.use_dask or isinstance(ddf, pd.DataFrame):
                return self._process_with_pandas(ddf, operations, start_time, progress_callback)
            
            self.logger.info(f"Starting Dask processing with {len(operations)} operations")
            
            # 작업 그래프 정보 수집
            initial_graph_size = len(ddf.__dask_graph__()) if hasattr(ddf, '__dask_graph__') else 0
            
            # 연산 적용 (지연 실행)
            result_ddf = ddf
            for i, operation in enumerate(operations):
                if progress_callback:
                    progress_callback(i / len(operations), f"연산 {i+1}/{len(operations)} 적용 중")
                
                result_ddf = operation(result_ddf)
                
                # 실시간 업데이트
                realtime_manager.send_progress_update(
                    task_id=f"dask_processing_{id(self)}",
                    progress=i / len(operations),
                    current_step=f"연산 {i+1}/{len(operations)} 적용 중"
                )
            
            # 청크 정보
            chunk_info = {
                'n_partitions': result_ddf.npartitions,
                'partition_sizes': [partition.nbytes for partition in result_ddf.to_delayed()],
                'total_size_estimate': sum(result_ddf.map_partitions(len).compute())
            }
            
            # 계산 실행
            if compute_result and self.config.compute_mode != ComputeMode.LAZY:
                if progress_callback:
                    progress_callback(0.8, "최종 계산 실행 중")
                
                # 진행률 표시와 함께 계산
                with ProgressBar():
                    if self.client:
                        # 분산 실행
                        future = self.client.compute(result_ddf)
                        result = future.result()
                    else:
                        # 로컬 실행
                        result = result_ddf.compute(scheduler=self.config.backend.value)
            else:
                # 지연 실행 - 계산 그래프만 반환
                result = result_ddf
            
            # 최종 그래프 크기
            final_graph_size = len(result.__dask_graph__()) if hasattr(result, '__dask_graph__') else 0
            
            processing_time = time.time() - start_time
            
            # 메모리 사용량 추정
            memory_usage = self._estimate_memory_usage(result)
            
            # 결과 생성
            processing_result = ProcessingResult(
                result=result,
                processing_time=processing_time,
                memory_usage=memory_usage,
                task_graph_size=final_graph_size - initial_graph_size,
                chunk_info=chunk_info
            )
            
            # 성능 로깅
            log_performance(
                "dask_processing",
                processing_time,
                {
                    'n_partitions': chunk_info['n_partitions'],
                    'operations_count': len(operations),
                    'backend': self.config.backend.value,
                    'compute_mode': self.config.compute_mode.value,
                    'graph_size_increase': final_graph_size - initial_graph_size
                }
            )
            
            # 작업 히스토리 저장
            self.task_history.append({
                'timestamp': datetime.now(),
                'processing_time': processing_time,
                'operations_count': len(operations),
                'n_partitions': chunk_info['n_partitions'],
                'memory_usage': memory_usage
            })
            
            self.logger.info(f"Dask processing completed in {processing_time:.2f}s")
            
            return processing_result
        
        except Exception as e:
            self.logger.error(f"Dask processing failed: {e}")
            handle_error(
                e,
                ErrorCategory.DATA_PROCESSING,
                ErrorSeverity.HIGH,
                context={
                    'operations_count': len(operations),
                    'backend': self.config.backend.value
                }
            )
            raise
    
    def _process_with_pandas(
        self,
        df: pd.DataFrame,
        operations: List[Callable],
        start_time: float,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """Pandas를 사용한 대체 처리"""
        result_df = df.copy()
        
        for i, operation in enumerate(operations):
            if progress_callback:
                progress_callback(i / len(operations), f"연산 {i+1}/{len(operations)} 적용 중")
            
            result_df = operation(result_df)
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            result=result_df,
            processing_time=processing_time,
            memory_usage={'pandas_fallback': True},
            task_graph_size=0,
            chunk_info={'n_partitions': 1, 'fallback_mode': True}
        )
    
    def parallel_apply(
        self,
        ddf: Union[dd.DataFrame, pd.DataFrame],
        func: Callable,
        axis: int = 0,
        meta: Optional[Any] = None,
        **kwargs
    ) -> Union[dd.DataFrame, pd.DataFrame]:
        """병렬 함수 적용"""
        if not self.use_dask or isinstance(ddf, pd.DataFrame):
            return ddf.apply(func, axis=axis, **kwargs)
        
        try:
            # 메타데이터 추론
            if meta is None:
                sample = ddf.head(1)
                meta = sample.apply(func, axis=axis, **kwargs)
            
            return ddf.map_partitions(
                lambda partition: partition.apply(func, axis=axis, **kwargs),
                meta=meta
            )
        
        except Exception as e:
            self.logger.error(f"Parallel apply failed: {e}")
            raise
    
    def optimize_dataframe(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """DataFrame 최적화"""
        if not self.use_dask:
            return ddf
        
        try:
            # 불필요한 컬럼 제거
            # (실제 구현에서는 사용하지 않는 컬럼 식별 로직 필요)
            
            # 파티션 재분할
            optimal_partitions = self._calculate_optimal_partitions(ddf)
            if ddf.npartitions != optimal_partitions:
                ddf = ddf.repartition(npartitions=optimal_partitions)
                self.logger.info(f"Repartitioned to {optimal_partitions} partitions")
            
            # 인덱스 최적화
            if not ddf.known_divisions:
                # 정렬된 컬럼이 있다면 인덱스로 설정
                if 'date' in ddf.columns:
                    ddf = ddf.set_index('date', sorted=True)
                    self.logger.info("Set date column as sorted index")
            
            return ddf
        
        except Exception as e:
            self.logger.warning(f"DataFrame optimization failed: {e}")
            return ddf
    
    def save_data(
        self,
        ddf: Union[dd.DataFrame, pd.DataFrame],
        output_path: str,
        file_format: str = 'parquet',
        **write_kwargs
    ):
        """데이터 저장"""
        try:
            if not self.use_dask or isinstance(ddf, pd.DataFrame):
                # Pandas 저장
                if file_format == 'parquet':
                    ddf.to_parquet(output_path, **write_kwargs)
                elif file_format == 'csv':
                    ddf.to_csv(output_path, **write_kwargs)
                else:
                    raise ValueError(f"Unsupported format for pandas: {file_format}")
            else:
                # Dask 저장
                if file_format == 'parquet':
                    ddf.to_parquet(output_path, **write_kwargs)
                elif file_format == 'csv':
                    ddf.to_csv(output_path, **write_kwargs)
                elif file_format == 'hdf':
                    ddf.to_hdf(output_path, key='data', **write_kwargs)
                else:
                    raise ValueError(f"Unsupported format: {file_format}")
            
            self.logger.info(f"Data saved to {output_path} in {file_format} format")
        
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
            handle_error(
                e,
                ErrorCategory.FILE_IO,
                ErrorSeverity.HIGH,
                context={'output_path': output_path, 'format': file_format}
            )
            raise
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """클러스터 정보 조회"""
        if not self.client:
            return {'status': 'no_cluster', 'backend': self.config.backend.value}
        
        try:
            scheduler_info = self.client.scheduler_info()
            return {
                'status': 'active',
                'scheduler_address': self.client.scheduler.address if self.client.scheduler else 'local',
                'n_workers': len(scheduler_info.get('workers', {})),
                'total_cores': sum(w.get('nthreads', 0) for w in scheduler_info.get('workers', {}).values()),
                'total_memory': sum(w.get('memory_limit', 0) for w in scheduler_info.get('workers', {}).values()),
                'dashboard_link': getattr(self.cluster, 'dashboard_link', None)
            }
        except Exception as e:
            self.logger.error(f"Failed to get cluster info: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _parse_chunk_size(self, total_size: int) -> int:
        """청크 크기 파싱"""
        if isinstance(self.config.chunk_size, str):
            if self.config.chunk_size.endswith('MB'):
                mb_size = int(self.config.chunk_size[:-2])
                # 대략적으로 1MB당 1000 rows로 추정
                return mb_size * 1000
            elif self.config.chunk_size.endswith('GB'):
                gb_size = int(self.config.chunk_size[:-2])
                return gb_size * 1000000
            else:
                return int(self.config.chunk_size)
        else:
            return self.config.chunk_size
    
    def _calculate_optimal_partitions(self, ddf: dd.DataFrame) -> int:
        """최적 파티션 수 계산"""
        # 데이터 크기 기반
        try:
            total_size = ddf.map_partitions(len).sum().compute()
            optimal_size_per_partition = 100000  # 파티션당 10만 행
            
            optimal_partitions = max(1, min(
                total_size // optimal_size_per_partition,
                self.config.n_workers * 4  # 워커당 최대 4개 파티션
            ))
            
            return int(optimal_partitions)
        
        except Exception:
            # 계산 실패시 현재 파티션 수 유지
            return ddf.npartitions
    
    def _estimate_memory_usage(self, result: Any) -> Dict[str, Any]:
        """메모리 사용량 추정"""
        try:
            if hasattr(result, 'memory_usage'):
                # DataFrame의 경우
                memory_bytes = result.memory_usage(deep=True).sum()
                return {
                    'estimated_mb': memory_bytes / 1024 / 1024,
                    'type': 'dataframe_computed'
                }
            elif hasattr(result, '__dask_graph__'):
                # Dask 객체의 경우
                return {
                    'estimated_mb': 'unknown',
                    'type': 'dask_lazy',
                    'graph_size': len(result.__dask_graph__())
                }
            else:
                return {
                    'estimated_mb': 'unknown',
                    'type': type(result).__name__
                }
        except Exception:
            return {'estimated_mb': 'error', 'type': 'unknown'}
    
    def close(self):
        """리소스 정리"""
        try:
            if self.client:
                self.client.close()
                self.client = None
                self.logger.info("Dask client closed")
            
            if self.cluster:
                self.cluster.close()
                self.cluster = None
                self.logger.info("Dask cluster closed")
        
        except Exception as e:
            self.logger.error(f"Error closing Dask resources: {e}")
    
    def __del__(self):
        """소멸자"""
        self.close()


# 특화된 Dask 처리 함수들
def dask_factor_analysis(
    data_path: Union[str, pd.DataFrame],
    config: Optional[DaskConfig] = None
) -> ProcessingResult:
    """Dask를 사용한 팩터 분석"""
    processor = DaskProcessor(config)
    
    try:
        # 데이터 로드
        ddf = processor.load_data(data_path)
        
        # 팩터 생성 연산 정의
        def generate_momentum_factors(df):
            df = df.copy()
            if 'close' in df.columns:
                df['momentum_5'] = df['close'].pct_change(5)
                df['momentum_20'] = df['close'].pct_change(20)
            return df
        
        def generate_volatility_factors(df):
            df = df.copy()
            if 'close' in df.columns:
                df['volatility_20'] = df['close'].pct_change().rolling(20).std()
            return df
        
        operations = [generate_momentum_factors, generate_volatility_factors]
        
        # 처리 실행
        result = processor.process_data(ddf, operations)
        
        return result
    
    finally:
        processor.close()


def dask_performance_calculation(
    data_path: Union[str, pd.DataFrame],
    config: Optional[DaskConfig] = None
) -> ProcessingResult:
    """Dask를 사용한 성능 계산"""
    processor = DaskProcessor(config)
    
    try:
        ddf = processor.load_data(data_path)
        
        def calculate_returns(df):
            df = df.copy()
            if 'close' in df.columns:
                df['returns'] = df['close'].pct_change()
                df['cumulative_returns'] = (1 + df['returns']).cumprod()
            return df
        
        def calculate_rolling_metrics(df):
            df = df.copy()
            if 'returns' in df.columns:
                df['rolling_sharpe'] = df['returns'].rolling(252).mean() / df['returns'].rolling(252).std() * np.sqrt(252)
                df['rolling_volatility'] = df['returns'].rolling(252).std() * np.sqrt(252)
            return df
        
        operations = [calculate_returns, calculate_rolling_metrics]
        result = processor.process_data(ddf, operations)
        
        return result
    
    finally:
        processor.close()


# Streamlit 통합
def show_dask_dashboard():
    """Dask 대시보드 표시"""
    import streamlit as st
    
    st.subheader("🔧 Dask 클러스터 정보")
    
    if not DASK_AVAILABLE:
        st.error("Dask가 설치되지 않았습니다.")
        return
    
    # 임시 프로세서로 클러스터 정보 확인
    temp_processor = DaskProcessor()
    cluster_info = temp_processor.get_cluster_info()
    temp_processor.close()
    
    if cluster_info['status'] == 'active':
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("워커 수", cluster_info['n_workers'])
        
        with col2:
            st.metric("총 코어 수", cluster_info['total_cores'])
        
        with col3:
            memory_gb = cluster_info['total_memory'] / (1024**3) if cluster_info['total_memory'] else 0
            st.metric("총 메모리", f"{memory_gb:.1f} GB")
        
        # 대시보드 링크
        if cluster_info.get('dashboard_link'):
            st.markdown(f"[Dask 대시보드 열기]({cluster_info['dashboard_link']})")
    
    elif cluster_info['status'] == 'no_cluster':
        st.info(f"클러스터 없음 - {cluster_info['backend']} 백엔드 사용 중")
    
    else:
        st.error(f"클러스터 오류: {cluster_info.get('error', 'Unknown error')}")


def create_dask_config_ui() -> DaskConfig:
    """Dask 설정 UI 생성"""
    import streamlit as st
    
    st.subheader("⚙️ Dask 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        backend = st.selectbox(
            "백엔드",
            options=[b.value for b in DaskBackend],
            index=0
        )
        
        n_workers = st.number_input("워커 수", min_value=1, max_value=16, value=4)
        
        threads_per_worker = st.number_input("워커당 스레드", min_value=1, max_value=8, value=2)
    
    with col2:
        memory_limit = st.selectbox(
            "메모리 제한",
            options=["1GB", "2GB", "4GB", "8GB"],
            index=1
        )
        
        chunk_size = st.selectbox(
            "청크 크기",
            options=["50MB", "100MB", "200MB", "500MB"],
            index=1
        )
        
        compute_mode = st.selectbox(
            "계산 모드",
            options=[m.value for m in ComputeMode],
            index=0
        )
    
    return DaskConfig(
        backend=DaskBackend(backend),
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        chunk_size=chunk_size,
        compute_mode=ComputeMode(compute_mode)
    )