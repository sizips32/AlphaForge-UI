"""
스트리밍 데이터 처리 시스템
실시간 데이터 스트림 처리 및 온라인 학습
"""

import time
import threading
import queue
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Generator, Iterator, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import asyncio
from pathlib import Path

from utils.logger import get_logger, log_performance
from utils.error_handler import handle_error, ErrorCategory, ErrorSeverity
from utils.realtime_updates import realtime_manager, UpdateType


class StreamingMode(Enum):
    """스트리밍 모드"""
    CONTINUOUS = "continuous"
    MICRO_BATCH = "micro_batch"
    EVENT_DRIVEN = "event_driven"
    WINDOWED = "windowed"


class WindowType(Enum):
    """윈도우 타입"""
    TUMBLING = "tumbling"      # 겹치지 않는 고정 크기 윈도우
    SLIDING = "sliding"        # 겹치는 고정 크기 윈도우
    SESSION = "session"        # 활동 기반 가변 크기 윈도우
    COUNT = "count"           # 개수 기반 윈도우


@dataclass
class StreamWindow:
    """스트림 윈도우"""
    window_id: str
    window_type: WindowType
    size: Union[int, timedelta]
    slide: Optional[Union[int, timedelta]] = None
    data: deque = field(default_factory=deque)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    is_complete: bool = False


@dataclass
class StreamingConfig:
    """스트리밍 설정"""
    mode: StreamingMode
    buffer_size: int
    batch_size: int
    window_size: Union[int, timedelta]
    window_type: WindowType
    slide_interval: Optional[Union[int, timedelta]]
    max_latency_ms: int
    checkpoint_interval: int
    enable_backpressure: bool
    max_memory_mb: int


@dataclass
class StreamMetrics:
    """스트림 메트릭"""
    processed_records: int = 0
    failed_records: int = 0
    avg_processing_time_ms: float = 0
    throughput_per_second: float = 0
    current_latency_ms: float = 0
    buffer_utilization: float = 0
    memory_usage_mb: float = 0
    last_update: datetime = field(default_factory=datetime.now)


class StreamingProcessor:
    """스트리밍 데이터 처리기"""
    
    def __init__(self, config: StreamingConfig):
        """초기화"""
        self.config = config
        self.logger = get_logger("streaming_processor")
        
        # 스트림 상태
        self.is_running = False
        self.is_paused = False
        
        # 데이터 버퍼
        self.input_buffer = queue.Queue(maxsize=config.buffer_size)
        self.output_buffer = queue.Queue()
        
        # 윈도우 관리
        self.windows: Dict[str, StreamWindow] = {}
        self.window_counter = 0
        
        # 메트릭
        self.metrics = StreamMetrics()
        
        # 처리 함수들
        self.processors: List[Callable] = []
        self.output_handlers: List[Callable] = []
        
        # 스레드 관리
        self.processing_thread = None
        self.output_thread = None
        
        # 체크포인트
        self.last_checkpoint = datetime.now()
        
        # 백프레셔 관리
        self.backpressure_active = False
    
    def add_processor(self, processor: Callable):
        """처리 함수 추가"""
        self.processors.append(processor)
        self.logger.info(f"Added processor: {processor.__name__}")
    
    def add_output_handler(self, handler: Callable):
        """출력 핸들러 추가"""
        self.output_handlers.append(handler)
        self.logger.info(f"Added output handler: {handler.__name__}")
    
    def start_stream(self):
        """스트림 처리 시작"""
        if self.is_running:
            self.logger.warning("Stream is already running")
            return
        
        self.is_running = True
        self.is_paused = False
        
        self.logger.info(f"Starting stream processing in {self.config.mode.value} mode")
        
        # 처리 스레드 시작
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name="StreamProcessor"
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # 출력 스레드 시작
        self.output_thread = threading.Thread(
            target=self._output_loop,
            name="StreamOutput"
        )
        self.output_thread.daemon = True
        self.output_thread.start()
        
        # 메트릭 업데이트 스레드
        metrics_thread = threading.Thread(
            target=self._metrics_loop,
            name="StreamMetrics"
        )
        metrics_thread.daemon = True
        metrics_thread.start()
        
        self.logger.info("Stream processing started")
    
    def stop_stream(self):
        """스트림 처리 중지"""
        self.logger.info("Stopping stream processing")
        self.is_running = False
        
        # 스레드 종료 대기
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        if self.output_thread:
            self.output_thread.join(timeout=5)
        
        # 남은 데이터 처리
        self._flush_remaining_data()
        
        self.logger.info("Stream processing stopped")
    
    def pause_stream(self):
        """스트림 처리 일시 정지"""
        self.is_paused = True
        self.logger.info("Stream processing paused")
    
    def resume_stream(self):
        """스트림 처리 재개"""
        self.is_paused = False
        self.logger.info("Stream processing resumed")
    
    def push_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> bool:
        """데이터 입력"""
        if not self.is_running:
            self.logger.warning("Stream is not running")
            return False
        
        try:
            # 백프레셔 확인
            if self.config.enable_backpressure and self.input_buffer.qsize() > self.config.buffer_size * 0.9:
                if not self.backpressure_active:
                    self.logger.warning("Backpressure activated")
                    self.backpressure_active = True
                return False
            
            self.backpressure_active = False
            
            # 데이터 정규화
            normalized_data = self._normalize_input_data(data)
            
            # 타임스탬프 추가
            for record in normalized_data:
                if 'timestamp' not in record:
                    record['timestamp'] = datetime.now()
            
            # 버퍼에 추가
            for record in normalized_data:
                try:
                    self.input_buffer.put_nowait(record)
                except queue.Full:
                    self.logger.warning("Input buffer is full, dropping data")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to push data: {e}")
            handle_error(
                e,
                ErrorCategory.DATA_PROCESSING,
                ErrorSeverity.MEDIUM,
                context={'data_type': type(data).__name__}
            )
            return False
    
    def _normalize_input_data(self, data: Union[Dict, List, pd.DataFrame]) -> List[Dict[str, Any]]:
        """입력 데이터 정규화"""
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        elif isinstance(data, pd.DataFrame):
            return data.to_dict('records')
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _processing_loop(self):
        """메인 처리 루프"""
        batch = []
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                current_time = time.time()
                
                # 데이터 수집
                try:
                    # 타임아웃으로 데이터 가져오기
                    timeout = max(0.001, (self.config.max_latency_ms - (current_time - last_batch_time) * 1000) / 1000)
                    record = self.input_buffer.get(timeout=timeout)
                    batch.append(record)
                    
                except queue.Empty:
                    # 타임아웃 발생시 현재 배치 처리
                    pass
                
                # 배치 처리 조건 확인
                should_process = (
                    len(batch) >= self.config.batch_size or
                    (batch and (current_time - last_batch_time) * 1000 > self.config.max_latency_ms)
                )
                
                if should_process and batch:
                    self._process_batch(batch)
                    batch = []
                    last_batch_time = current_time
                
                # 체크포인트
                if (datetime.now() - self.last_checkpoint).seconds > self.config.checkpoint_interval:
                    self._create_checkpoint()
                    self.last_checkpoint = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Processing loop error: {e}")
                handle_error(
                    e,
                    ErrorCategory.DATA_PROCESSING,
                    ErrorSeverity.MEDIUM,
                    context={'batch_size': len(batch)}
                )
                time.sleep(0.1)
        
        # 종료시 남은 배치 처리
        if batch:
            self._process_batch(batch)
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """배치 처리"""
        start_time = time.time()
        
        try:
            # DataFrame으로 변환
            df = pd.DataFrame(batch)
            
            # 윈도우 처리
            if self.config.mode == StreamingMode.WINDOWED:
                self._process_windowed_batch(df)
            else:
                # 직접 처리
                results = self._apply_processors(df)
                
                # 출력 버퍼에 추가
                if results is not None:
                    self.output_buffer.put({
                        'timestamp': datetime.now(),
                        'batch_size': len(batch),
                        'results': results
                    })
            
            # 메트릭 업데이트
            processing_time = (time.time() - start_time) * 1000
            self.metrics.processed_records += len(batch)
            self._update_processing_metrics(processing_time, len(batch))
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            self.metrics.failed_records += len(batch)
            raise
    
    def _process_windowed_batch(self, df: pd.DataFrame):
        """윈도우 기반 배치 처리"""
        if self.config.window_type == WindowType.TUMBLING:
            self._process_tumbling_window(df)
        elif self.config.window_type == WindowType.SLIDING:
            self._process_sliding_window(df)
        elif self.config.window_type == WindowType.COUNT:
            self._process_count_window(df)
        else:
            self.logger.warning(f"Unsupported window type: {self.config.window_type}")
    
    def _process_tumbling_window(self, df: pd.DataFrame):
        """텀블링 윈도우 처리"""
        for _, record in df.iterrows():
            record_dict = record.to_dict()
            
            # 현재 활성 윈도우 찾기
            current_window = None
            for window in self.windows.values():
                if not window.is_complete:
                    current_window = window
                    break
            
            # 새 윈도우 생성
            if current_window is None:
                self.window_counter += 1
                current_window = StreamWindow(
                    window_id=f"window_{self.window_counter}",
                    window_type=self.config.window_type,
                    size=self.config.window_size,
                    start_time=datetime.now()
                )
                self.windows[current_window.window_id] = current_window
            
            # 윈도우에 데이터 추가
            current_window.data.append(record_dict)
            
            # 윈도우 완료 확인
            if self._is_window_complete(current_window):
                current_window.is_complete = True
                current_window.end_time = datetime.now()
                
                # 윈도우 처리
                self._process_completed_window(current_window)
    
    def _process_sliding_window(self, df: pd.DataFrame):
        """슬라이딩 윈도우 처리"""
        # 구현: 슬라이딩 윈도우는 겹치는 윈도우들을 관리
        pass
    
    def _process_count_window(self, df: pd.DataFrame):
        """카운트 기반 윈도우 처리"""
        # 구현: 고정 개수 기반 윈도우
        pass
    
    def _is_window_complete(self, window: StreamWindow) -> bool:
        """윈도우 완료 여부 확인"""
        if isinstance(self.config.window_size, int):
            return len(window.data) >= self.config.window_size
        elif isinstance(self.config.window_size, timedelta):
            if window.start_time:
                return datetime.now() - window.start_time >= self.config.window_size
        return False
    
    def _process_completed_window(self, window: StreamWindow):
        """완료된 윈도우 처리"""
        try:
            # 윈도우 데이터를 DataFrame으로 변환
            window_df = pd.DataFrame(list(window.data))
            
            # 처리 함수 적용
            results = self._apply_processors(window_df)
            
            # 결과 출력
            if results is not None:
                self.output_buffer.put({
                    'window_id': window.window_id,
                    'window_type': window.window_type.value,
                    'start_time': window.start_time,
                    'end_time': window.end_time,
                    'record_count': len(window.data),
                    'results': results
                })
            
            # 완료된 윈도우 정리
            del self.windows[window.window_id]
            
        except Exception as e:
            self.logger.error(f"Window processing failed: {e}")
    
    def _apply_processors(self, df: pd.DataFrame) -> Optional[Any]:
        """처리 함수들 적용"""
        try:
            result = df
            
            for processor in self.processors:
                result = processor(result)
                if result is None:
                    break
            
            return result
            
        except Exception as e:
            self.logger.error(f"Processor application failed: {e}")
            return None
    
    def _output_loop(self):
        """출력 처리 루프"""
        while self.is_running:
            try:
                # 출력 데이터 가져오기
                try:
                    output_data = self.output_buffer.get(timeout=1.0)
                    
                    # 출력 핸들러들 실행
                    for handler in self.output_handlers:
                        try:
                            handler(output_data)
                        except Exception as e:
                            self.logger.error(f"Output handler failed: {e}")
                    
                    # 실시간 업데이트 전송
                    realtime_manager.send_update(
                        update_type=UpdateType.DATA_PROCESSING,
                        data={
                            'type': 'streaming_result',
                            'timestamp': output_data.get('timestamp', datetime.now()).isoformat(),
                            'record_count': output_data.get('record_count', 0)
                        }
                    )
                    
                except queue.Empty:
                    continue
                    
            except Exception as e:
                self.logger.error(f"Output loop error: {e}")
                time.sleep(0.1)
    
    def _metrics_loop(self):
        """메트릭 업데이트 루프"""
        while self.is_running:
            try:
                self._update_system_metrics()
                
                # 메트릭을 실시간으로 전송
                realtime_manager.send_update(
                    update_type=UpdateType.METRIC_UPDATE,
                    data={
                        'type': 'streaming_metrics',
                        'metrics': {
                            'processed_records': self.metrics.processed_records,
                            'throughput_per_second': self.metrics.throughput_per_second,
                            'current_latency_ms': self.metrics.current_latency_ms,
                            'buffer_utilization': self.metrics.buffer_utilization,
                            'memory_usage_mb': self.metrics.memory_usage_mb
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                )
                
                time.sleep(1)  # 1초마다 메트릭 업데이트
                
            except Exception as e:
                self.logger.error(f"Metrics loop error: {e}")
    
    def _update_processing_metrics(self, processing_time_ms: float, batch_size: int):
        """처리 메트릭 업데이트"""
        # 평균 처리 시간 업데이트 (지수 이동 평균)
        alpha = 0.1
        self.metrics.avg_processing_time_ms = (
            alpha * processing_time_ms + (1 - alpha) * self.metrics.avg_processing_time_ms
        )
        
        # 처리량 계산 (지수 이동 평균)
        current_throughput = batch_size / (processing_time_ms / 1000) if processing_time_ms > 0 else 0
        self.metrics.throughput_per_second = (
            alpha * current_throughput + (1 - alpha) * self.metrics.throughput_per_second
        )
        
        # 지연시간 업데이트
        self.metrics.current_latency_ms = processing_time_ms
        
        self.metrics.last_update = datetime.now()
    
    def _update_system_metrics(self):
        """시스템 메트릭 업데이트"""
        try:
            # 버퍼 사용률
            self.metrics.buffer_utilization = self.input_buffer.qsize() / self.config.buffer_size
            
            # 메모리 사용량 (추정)
            import psutil
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            
        except Exception as e:
            self.logger.debug(f"System metrics update failed: {e}")
    
    def _create_checkpoint(self):
        """체크포인트 생성"""
        try:
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    'processed_records': self.metrics.processed_records,
                    'failed_records': self.metrics.failed_records,
                    'avg_processing_time_ms': self.metrics.avg_processing_time_ms,
                    'throughput_per_second': self.metrics.throughput_per_second
                },
                'active_windows': len([w for w in self.windows.values() if not w.is_complete]),
                'buffer_size': self.input_buffer.qsize()
            }
            
            # 체크포인트 저장 (실제 구현에서는 파일이나 데이터베이스에 저장)
            self.logger.debug(f"Checkpoint created: {json.dumps(checkpoint_data, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Checkpoint creation failed: {e}")
    
    def _flush_remaining_data(self):
        """남은 데이터 처리"""
        try:
            # 입력 버퍼의 남은 데이터 처리
            remaining_batch = []
            while not self.input_buffer.empty():
                try:
                    record = self.input_buffer.get_nowait()
                    remaining_batch.append(record)
                except queue.Empty:
                    break
            
            if remaining_batch:
                self.logger.info(f"Processing {len(remaining_batch)} remaining records")
                self._process_batch(remaining_batch)
            
            # 미완료 윈도우 처리
            for window in list(self.windows.values()):
                if not window.is_complete and window.data:
                    self.logger.info(f"Processing incomplete window: {window.window_id}")
                    window.is_complete = True
                    window.end_time = datetime.now()
                    self._process_completed_window(window)
            
        except Exception as e:
            self.logger.error(f"Failed to flush remaining data: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """메트릭 조회"""
        return {
            'processed_records': self.metrics.processed_records,
            'failed_records': self.metrics.failed_records,
            'avg_processing_time_ms': self.metrics.avg_processing_time_ms,
            'throughput_per_second': self.metrics.throughput_per_second,
            'current_latency_ms': self.metrics.current_latency_ms,
            'buffer_utilization': self.metrics.buffer_utilization,
            'memory_usage_mb': self.metrics.memory_usage_mb,
            'active_windows': len([w for w in self.windows.values() if not w.is_complete]),
            'backpressure_active': self.backpressure_active,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'last_update': self.metrics.last_update.isoformat()
        }


# 특화된 스트리밍 프로세서들
class FactorStreamProcessor:
    """팩터 계산 스트리밍 프로세서"""
    
    def __init__(self):
        self.logger = get_logger("factor_stream_processor")
        self.price_history = deque(maxlen=100)  # 최근 100개 가격
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """팩터 계산"""
        try:
            result = df.copy()
            
            # 가격 히스토리 업데이트
            if 'close' in df.columns:
                for price in df['close']:
                    self.price_history.append(price)
            
            # 실시간 팩터 계산
            if len(self.price_history) >= 5:
                prices = list(self.price_history)
                
                # 모멘텀 팩터
                if 'close' in result.columns:
                    result['momentum_5'] = result['close'].pct_change(min(5, len(prices)))
                    result['rsi'] = self._calculate_rsi(prices[-14:] if len(prices) >= 14 else prices)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Factor calculation failed: {e}")
            return df
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI 계산"""
        if len(prices) < 2:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        if len(gains) == 0 or len(losses) == 0:
            return 50.0
        
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class AlertStreamProcessor:
    """알림 스트리밍 프로세서"""
    
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
        self.logger = get_logger("alert_stream_processor")
    
    def __call__(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """알림 조건 확인"""
        alerts = []
        
        try:
            for _, row in df.iterrows():
                for column, threshold in self.thresholds.items():
                    if column in row and abs(row[column]) > threshold:
                        alerts.append({
                            'timestamp': row.get('timestamp', datetime.now()).isoformat(),
                            'column': column,
                            'value': row[column],
                            'threshold': threshold,
                            'severity': 'high' if abs(row[column]) > threshold * 1.5 else 'medium'
                        })
            
            return {'alerts': alerts} if alerts else None
            
        except Exception as e:
            self.logger.error(f"Alert processing failed: {e}")
            return None


# Streamlit 통합
def show_streaming_dashboard(processor: StreamingProcessor):
    """스트리밍 대시보드 표시"""
    import streamlit as st
    
    st.subheader("🌊 스트리밍 처리 대시보드")
    
    # 상태 표시
    metrics = processor.get_metrics()
    
    # 상태 인디케이터
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "실행 중" if metrics['is_running'] else "중지"
        if metrics['is_paused']:
            status += " (일시정지)"
        st.metric("상태", status)
    
    with col2:
        st.metric("처리된 레코드", metrics['processed_records'])
    
    with col3:
        st.metric("처리량 (/초)", f"{metrics['throughput_per_second']:.1f}")
    
    with col4:
        st.metric("평균 지연시간 (ms)", f"{metrics['avg_processing_time_ms']:.1f}")
    
    # 버퍼 사용률
    st.subheader("📊 시스템 메트릭")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("버퍼 사용률", f"{metrics['buffer_utilization']*100:.1f}%")
        st.progress(metrics['buffer_utilization'])
    
    with col2:
        st.metric("메모리 사용량", f"{metrics['memory_usage_mb']:.1f} MB")
    
    # 백프레셔 상태
    if metrics['backpressure_active']:
        st.error("⚠️ 백프레셔 활성화: 입력 속도를 줄이세요")
    
    # 제어 버튼
    st.subheader("🎛️ 제어")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("일시정지" if metrics['is_running'] and not metrics['is_paused'] else "재개"):
            if metrics['is_running'] and not metrics['is_paused']:
                processor.pause_stream()
            else:
                processor.resume_stream()
            st.experimental_rerun()
    
    with col2:
        if st.button("중지"):
            processor.stop_stream()
            st.experimental_rerun()
    
    with col3:
        if st.button("메트릭 새로고침"):
            st.experimental_rerun()


def create_streaming_config_ui() -> StreamingConfig:
    """스트리밍 설정 UI 생성"""
    import streamlit as st
    
    st.subheader("⚙️ 스트리밍 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mode = st.selectbox(
            "스트리밍 모드",
            options=[m.value for m in StreamingMode],
            index=0
        )
        
        buffer_size = st.number_input("버퍼 크기", min_value=100, max_value=10000, value=1000)
        
        batch_size = st.number_input("배치 크기", min_value=1, max_value=1000, value=100)
        
        max_latency_ms = st.number_input("최대 지연시간 (ms)", min_value=10, max_value=10000, value=1000)
    
    with col2:
        window_type = st.selectbox(
            "윈도우 타입",
            options=[w.value for w in WindowType],
            index=0
        )
        
        window_size = st.number_input("윈도우 크기", min_value=10, max_value=10000, value=100)
        
        checkpoint_interval = st.number_input("체크포인트 간격 (초)", min_value=10, max_value=3600, value=60)
        
        enable_backpressure = st.checkbox("백프레셔 활성화", value=True)
    
    return StreamingConfig(
        mode=StreamingMode(mode),
        buffer_size=buffer_size,
        batch_size=batch_size,
        window_size=window_size,
        window_type=WindowType(window_type),
        slide_interval=None,
        max_latency_ms=max_latency_ms,
        checkpoint_interval=checkpoint_interval,
        enable_backpressure=enable_backpressure,
        max_memory_mb=1000
    )


# 예제 사용법
def create_example_streaming_pipeline():
    """예제 스트리밍 파이프라인 생성"""
    # 설정 생성
    config = StreamingConfig(
        mode=StreamingMode.MICRO_BATCH,
        buffer_size=1000,
        batch_size=50,
        window_size=100,
        window_type=WindowType.TUMBLING,
        max_latency_ms=500,
        checkpoint_interval=30,
        enable_backpressure=True,
        max_memory_mb=500
    )
    
    # 프로세서 생성
    processor = StreamingProcessor(config)
    
    # 처리 함수 추가
    factor_processor = FactorStreamProcessor()
    processor.add_processor(factor_processor)
    
    # 알림 프로세서 추가
    alert_processor = AlertStreamProcessor({'momentum_5': 0.05, 'rsi': 80})
    processor.add_processor(alert_processor)
    
    # 출력 핸들러 추가
    def print_output(data):
        print(f"Processed at {data.get('timestamp')}: {data.get('record_count')} records")
    
    processor.add_output_handler(print_output)
    
    return processor