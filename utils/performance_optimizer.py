"""
성능 최적화 모듈
대용량 데이터 처리를 위한 최적화 기능을 제공합니다.
"""

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Tuple
import multiprocessing
from functools import partial
import gc
import streamlit as st
from utils.logger import log_performance, log_system_event
import time

class PerformanceOptimizer:
    """성능 최적화 클래스"""
    
    def __init__(self, max_workers: Optional[int] = None) -> None:
        """
        초기화
        
        Args:
            max_workers: 최대 워커 수. None일 경우 CPU 코어 수의 75% 사용
        """
        self.max_workers = max_workers or max(1, int(multiprocessing.cpu_count() * 0.75))
        self.chunk_size = 10000  # 기본 청크 크기
        
        log_system_event('performance_optimizer_initialized', {
            'max_workers': self.max_workers,
            'cpu_count': multiprocessing.cpu_count()
        })
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터프레임 메모리 사용량을 최적화합니다.
        
        Args:
            df: 최적화할 데이터프레임
            
        Returns:
            최적화된 데이터프레임
        """
        start_time = time.time()
        original_memory = df.memory_usage(deep=True).sum()
        
        optimized_df = df.copy()
        
        # 숫자 컬럼 최적화
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            col_min = optimized_df[col].min()
            col_max = optimized_df[col].max()
            
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                optimized_df[col] = optimized_df[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                optimized_df[col] = optimized_df[col].astype(np.int16)
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                optimized_df[col] = optimized_df[col].astype(np.int32)
        
        # Float 컬럼 최적화
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        # Object 컬럼을 category로 변환 (카디널리티가 낮은 경우)
        for col in optimized_df.select_dtypes(include=['object']).columns:
            if optimized_df[col].nunique() / len(optimized_df) < 0.5:  # 50% 미만이 고유값인 경우
                optimized_df[col] = optimized_df[col].astype('category')
        
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        memory_reduction = (original_memory - optimized_memory) / original_memory * 100
        
        duration = time.time() - start_time
        log_performance('dataframe_memory_optimization', duration, {
            'rows': len(df),
            'cols': len(df.columns),
            'original_memory_mb': original_memory / 1024 / 1024,
            'optimized_memory_mb': optimized_memory / 1024 / 1024,
            'memory_reduction_percent': memory_reduction
        })
        
        return optimized_df
    
    def parallel_apply(self, 
                      df: pd.DataFrame, 
                      func: Callable, 
                      group_by: str,
                      *args, 
                      **kwargs) -> pd.DataFrame:
        """
        그룹별 병렬 처리를 수행합니다.
        
        Args:
            df: 처리할 데이터프레임
            func: 적용할 함수
            group_by: 그룹핑 기준 컬럼
            *args, **kwargs: 함수에 전달할 추가 인자
            
        Returns:
            병렬 처리 결과
        """
        start_time = time.time()
        
        # 그룹별로 데이터 분할
        groups = [group for name, group in df.groupby(group_by)]
        
        if len(groups) <= 1:
            # 그룹이 1개 이하면 병렬 처리할 필요 없음
            result = func(df, *args, **kwargs)
        else:
            # 병렬 처리
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 각 그룹에 함수 적용
                future_to_group = {
                    executor.submit(func, group, *args, **kwargs): group 
                    for group in groups
                }
                
                results = []
                for future in as_completed(future_to_group):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        log_system_event('parallel_apply_error', {
                            'error': str(e),
                            'group_size': len(future_to_group[future])
                        })
                        raise
                
                # 결과 병합
                result = pd.concat(results, ignore_index=True)
        
        duration = time.time() - start_time
        log_performance('parallel_apply', duration, {
            'rows': len(df),
            'groups': len(groups),
            'workers': self.max_workers,
            'function': func.__name__
        })
        
        return result
    
    def chunked_processing(self, 
                          df: pd.DataFrame, 
                          func: Callable,
                          chunk_size: Optional[int] = None,
                          *args, 
                          **kwargs) -> pd.DataFrame:
        """
        청크 단위로 데이터를 처리합니다.
        
        Args:
            df: 처리할 데이터프레임
            func: 적용할 함수
            chunk_size: 청크 크기
            *args, **kwargs: 함수에 전달할 추가 인자
            
        Returns:
            처리된 결과
        """
        start_time = time.time()
        chunk_size = chunk_size or self.chunk_size
        
        # 데이터가 청크 크기보다 작으면 그대로 처리
        if len(df) <= chunk_size:
            result = func(df, *args, **kwargs)
        else:
            # 청크 단위로 처리
            results = []
            total_chunks = (len(df) - 1) // chunk_size + 1
            
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                
                try:
                    chunk_result = func(chunk, *args, **kwargs)
                    results.append(chunk_result)
                    
                    # 진행 상황 표시 (Streamlit에서)
                    if hasattr(st, 'progress'):
                        progress = (i // chunk_size + 1) / total_chunks
                        if 'progress_bar' in kwargs:
                            kwargs['progress_bar'].progress(progress)
                    
                    # 메모리 정리
                    del chunk
                    if i % (chunk_size * 10) == 0:  # 10개 청크마다 GC 실행
                        gc.collect()
                        
                except Exception as e:
                    log_system_event('chunked_processing_error', {
                        'error': str(e),
                        'chunk_index': i // chunk_size,
                        'chunk_size': len(chunk)
                    })
                    raise
            
            # 결과 병합
            if results:
                result = pd.concat(results, ignore_index=True)
            else:
                result = df.copy()
        
        duration = time.time() - start_time
        log_performance('chunked_processing', duration, {
            'total_rows': len(df),
            'chunk_size': chunk_size,
            'total_chunks': (len(df) - 1) // chunk_size + 1,
            'function': func.__name__
        })
        
        return result
    
    def parallel_chunked_processing(self, 
                                   df: pd.DataFrame, 
                                   func: Callable,
                                   chunk_size: Optional[int] = None,
                                   *args, 
                                   **kwargs) -> pd.DataFrame:
        """
        병렬 + 청크 처리를 결합합니다.
        
        Args:
            df: 처리할 데이터프레임
            func: 적용할 함수
            chunk_size: 청크 크기
            *args, **kwargs: 함수에 전달할 추가 인자
            
        Returns:
            처리된 결과
        """
        start_time = time.time()
        chunk_size = chunk_size or self.chunk_size
        
        # 청크로 데이터 분할
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        if len(chunks) <= 1:
            # 청크가 1개면 병렬 처리할 필요 없음
            result = func(df, *args, **kwargs)
        else:
            # 병렬로 각 청크 처리
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_chunk = {
                    executor.submit(func, chunk, *args, **kwargs): i 
                    for i, chunk in enumerate(chunks)
                }
                
                results = [None] * len(chunks)
                for future in as_completed(future_to_chunk):
                    try:
                        chunk_index = future_to_chunk[future]
                        results[chunk_index] = future.result()
                    except Exception as e:
                        log_system_event('parallel_chunked_processing_error', {
                            'error': str(e),
                            'chunk_index': future_to_chunk[future]
                        })
                        raise
                
                # 결과 병합 (순서 유지)
                valid_results = [r for r in results if r is not None]
                if valid_results:
                    result = pd.concat(valid_results, ignore_index=True)
                else:
                    result = df.copy()
        
        duration = time.time() - start_time
        log_performance('parallel_chunked_processing', duration, {
            'total_rows': len(df),
            'chunk_size': chunk_size,
            'total_chunks': len(chunks),
            'workers': self.max_workers,
            'function': func.__name__
        })
        
        return result
    
    def optimize_technical_indicators(self, 
                                    data: pd.DataFrame, 
                                    group_by: str = 'Ticker') -> pd.DataFrame:
        """
        기술적 지표 계산을 최적화합니다.
        
        Args:
            data: 주가 데이터
            group_by: 그룹핑 기준 컬럼
            
        Returns:
            기술적 지표가 추가된 데이터
        """
        def calculate_indicators_for_group(group_data: pd.DataFrame) -> pd.DataFrame:
            """그룹별 기술적 지표 계산"""
            result = group_data.copy()
            
            if len(result) < 20:  # 데이터가 부족하면 기본값으로 채움
                for col in ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']:
                    result[col] = 0
                return result
            
            # 벡터화된 연산 사용
            close_prices = result['Close'].values
            
            # 단순 이동평균 (numpy 사용)
            result['SMA_20'] = pd.Series(close_prices).rolling(20).mean()
            result['SMA_50'] = pd.Series(close_prices).rolling(50).mean()
            
            # 수익률
            result['Returns'] = pd.Series(close_prices).pct_change()
            
            # RSI (벡터화)
            returns = pd.Series(close_prices).diff()
            gains = returns.where(returns > 0, 0)
            losses = -returns.where(returns < 0, 0)
            
            avg_gains = gains.rolling(14).mean()
            avg_losses = losses.rolling(14).mean()
            
            rs = avg_gains / (avg_losses + 1e-8)
            result['RSI'] = 100 - (100 / (1 + rs))
            
            # 볼린저 밴드
            sma20 = result['SMA_20']
            std20 = pd.Series(close_prices).rolling(20).std()
            result['BB_Upper'] = sma20 + (std20 * 2)
            result['BB_Lower'] = sma20 - (std20 * 2)
            
            return result
        
        # 그룹별 병렬 처리
        return self.parallel_apply(data, calculate_indicators_for_group, group_by)
    
    def get_memory_usage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        데이터프레임의 메모리 사용량 정보를 반환합니다.
        
        Args:
            df: 분석할 데이터프레임
            
        Returns:
            메모리 사용량 정보
        """
        memory_info = {
            'total_memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'rows': len(df),
            'columns': len(df.columns),
            'memory_per_row_bytes': df.memory_usage(deep=True).sum() / len(df),
        }
        
        # 컬럼별 메모리 사용량
        column_memory = {}
        for col in df.columns:
            column_memory[col] = {
                'memory_mb': df[col].memory_usage(deep=True) / 1024 / 1024,
                'dtype': str(df[col].dtype),
                'unique_values': df[col].nunique(),
                'null_count': df[col].isnull().sum()
            }
        
        memory_info['column_details'] = column_memory
        
        return memory_info
    
    def set_chunk_size(self, size: int) -> None:
        """청크 크기를 설정합니다."""
        self.chunk_size = size
        log_system_event('chunk_size_updated', {'new_size': size})
    
    def set_max_workers(self, workers: int) -> None:
        """최대 워커 수를 설정합니다."""
        self.max_workers = workers
        log_system_event('max_workers_updated', {'new_workers': workers})

# 전역 인스턴스
performance_optimizer = PerformanceOptimizer()

# 편의 함수들
def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """메모리 최적화 편의 함수"""
    return performance_optimizer.optimize_dataframe_memory(df)

def parallel_apply(df: pd.DataFrame, func: Callable, group_by: str, *args, **kwargs) -> pd.DataFrame:
    """병렬 적용 편의 함수"""
    return performance_optimizer.parallel_apply(df, func, group_by, *args, **kwargs)

def chunked_process(df: pd.DataFrame, func: Callable, chunk_size: Optional[int] = None, *args, **kwargs) -> pd.DataFrame:
    """청크 처리 편의 함수"""
    return performance_optimizer.chunked_processing(df, func, chunk_size, *args, **kwargs)

def get_memory_info(df: pd.DataFrame) -> Dict[str, Any]:
    """메모리 정보 편의 함수"""
    return performance_optimizer.get_memory_usage(df)