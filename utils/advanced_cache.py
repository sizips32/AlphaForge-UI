"""
고급 캐싱 전략 시스템
다층 캐시, 지능형 만료, 분산 캐시, 캐시 히트율 최적화
"""

import os
import time
import threading
import pickle
import json
import hashlib
import lru
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import weakref
import gc
import psutil
from pathlib import Path
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd

# Redis 관련 imports (선택적)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from utils.logger import get_logger, log_performance
from utils.error_handler import handle_error, ErrorCategory, ErrorSeverity


class CacheLevel(Enum):
    """캐시 레벨"""
    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"
    PERSISTENT = "persistent"


class CachePolicy(Enum):
    """캐시 정책"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"


class EvictionStrategy(Enum):
    """제거 전략"""
    SIZE_BASED = "size_based"
    TIME_BASED = "time_based"
    ACCESS_BASED = "access_based"
    PRIORITY_BASED = "priority_based"
    ML_BASED = "ml_based"


@dataclass
class CacheEntry:
    """캐시 엔트리"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl: Optional[int] = None
    priority: int = 1
    cost: float = 1.0
    tags: List[str] = field(default_factory=list)
    dependency_keys: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """만료 여부 확인"""
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl
    
    def update_access(self):
        """접근 정보 업데이트"""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """캐시 통계"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    avg_access_time_ms: float = 0
    hit_ratio: float = 0
    memory_usage_mb: float = 0
    last_reset: datetime = field(default_factory=datetime.now)
    
    def update_hit_ratio(self):
        """히트율 업데이트"""
        total = self.hits + self.misses
        self.hit_ratio = self.hits / total if total > 0 else 0


class IntelligentCache:
    """지능형 다층 캐시 시스템"""
    
    def __init__(
        self,
        max_memory_mb: int = 500,
        max_disk_mb: int = 2000,
        default_ttl: int = 3600,
        policy: CachePolicy = CachePolicy.ADAPTIVE,
        enable_distributed: bool = False,
        redis_url: Optional[str] = None
    ):
        """초기화"""
        self.logger = get_logger("intelligent_cache")
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_disk_bytes = max_disk_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.policy = policy
        
        # 캐시 저장소들
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.disk_cache_dir = Path(".cache/advanced")
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Redis 분산 캐시
        self.redis_client = None
        if enable_distributed and REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                self.logger.info("Connected to Redis distributed cache")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Redis: {e}")
                self.redis_client = None
        
        # 통계 및 메트릭
        self.stats = CacheStats()
        
        # 접근 패턴 추적
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.prediction_model = None
        
        # LRU 캐시 (메모리 레벨)
        self.lru_cache = lru.LRU(1000)  # 최대 1000개 엔트리
        
        # 스레드 안전성
        self.lock = threading.RLock()
        
        # 백그라운드 작업
        self.cleanup_thread = None
        self.start_background_tasks()
        
        # 캐시 워밍업 상태
        self.warmup_status = {'completed': 0, 'total': 0, 'active': False}
    
    def start_background_tasks(self):
        """백그라운드 작업 시작"""
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="CacheCleanup",
            daemon=True
        )
        self.cleanup_thread.start()
    
    def get(self, key: str, level_preference: Optional[CacheLevel] = None) -> Optional[Any]:
        """캐시에서 값 조회"""
        start_time = time.time()
        
        with self.lock:
            try:
                # 1. 메모리 캐시 확인
                if key in self.memory_cache:
                    entry = self.memory_cache[key]
                    if not entry.is_expired():
                        entry.update_access()
                        self._record_access_pattern(key)
                        self.stats.hits += 1
                        self._update_access_time(start_time)
                        self.logger.debug(f"Memory cache hit: {key}")
                        return entry.value
                    else:
                        # 만료된 엔트리 제거
                        del self.memory_cache[key]
                
                # 2. 디스크 캐시 확인
                disk_entry = self._load_from_disk(key)
                if disk_entry and not disk_entry.is_expired():
                    # 메모리로 승격
                    self._promote_to_memory(disk_entry)
                    disk_entry.update_access()
                    self._record_access_pattern(key)
                    self.stats.hits += 1
                    self._update_access_time(start_time)
                    self.logger.debug(f"Disk cache hit: {key}")
                    return disk_entry.value
                
                # 3. 분산 캐시 확인
                if self.redis_client:
                    distributed_value = self._load_from_redis(key)
                    if distributed_value is not None:
                        # 로컬 캐시로 복사
                        self.set(key, distributed_value, cache_locally=True)
                        self.stats.hits += 1
                        self._update_access_time(start_time)
                        self.logger.debug(f"Distributed cache hit: {key}")
                        return distributed_value
                
                # 캐시 미스
                self.stats.misses += 1
                self._update_access_time(start_time)
                return None
                
            except Exception as e:
                self.logger.error(f"Cache get failed for key {key}: {e}")
                self.stats.misses += 1
                return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        priority: int = 1,
        tags: Optional[List[str]] = None,
        level: Optional[CacheLevel] = None,
        cache_locally: bool = True
    ) -> bool:
        """캐시에 값 저장"""
        with self.lock:
            try:
                ttl = ttl or self.default_ttl
                tags = tags or []
                
                # 값 크기 계산
                size_bytes = self._calculate_size(value)
                
                # 캐시 엔트리 생성
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    size_bytes=size_bytes,
                    ttl=ttl,
                    priority=priority,
                    tags=tags
                )
                
                # 저장 레벨 결정
                target_level = self._determine_storage_level(entry, level)
                
                if target_level == CacheLevel.MEMORY and cache_locally:
                    self._store_in_memory(entry)
                elif target_level == CacheLevel.DISK:
                    self._store_on_disk(entry)
                
                # 분산 캐시에도 저장 (조건부)
                if self.redis_client and (priority >= 3 or size_bytes < 1024 * 100):  # 100KB 미만 또는 높은 우선순위
                    self._store_in_redis(key, value, ttl)
                
                self.stats.entry_count += 1
                self.stats.size_bytes += size_bytes
                
                return True
                
            except Exception as e:
                self.logger.error(f"Cache set failed for key {key}: {e}")
                handle_error(
                    e,
                    ErrorCategory.SYSTEM,
                    ErrorSeverity.MEDIUM,
                    context={'key': key, 'size_bytes': size_bytes}
                )
                return False
    
    def invalidate(self, key: str, propagate: bool = True) -> bool:
        """캐시 무효화"""
        with self.lock:
            try:
                removed = False
                
                # 메모리에서 제거
                if key in self.memory_cache:
                    entry = self.memory_cache[key]
                    self.stats.size_bytes -= entry.size_bytes
                    del self.memory_cache[key]
                    removed = True
                
                # 디스크에서 제거
                disk_path = self.disk_cache_dir / f"{key}.pkl"
                if disk_path.exists():
                    disk_path.unlink()
                    removed = True
                
                # 분산 캐시에서 제거
                if self.redis_client and propagate:
                    self.redis_client.delete(key)
                    removed = True
                
                if removed:
                    self.stats.entry_count -= 1
                    self.logger.debug(f"Invalidated cache key: {key}")
                
                return removed
                
            except Exception as e:
                self.logger.error(f"Cache invalidation failed for key {key}: {e}")
                return False
    
    def invalidate_by_tags(self, tags: List[str]) -> int:
        """태그로 캐시 무효화"""
        with self.lock:
            invalidated_count = 0
            
            # 메모리 캐시에서 태그 기반 무효화
            keys_to_remove = []
            for key, entry in self.memory_cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                if self.invalidate(key):
                    invalidated_count += 1
            
            # 디스크 캐시는 메타데이터 파일 스캔 필요 (성능상 생략)
            
            self.logger.info(f"Invalidated {invalidated_count} cache entries by tags: {tags}")
            return invalidated_count
    
    def warm_up(self, warm_up_functions: List[Callable[[], Dict[str, Any]]]) -> Dict[str, Any]:
        """캐시 워밍업"""
        self.logger.info("Starting cache warm-up")
        start_time = time.time()
        
        self.warmup_status = {
            'completed': 0,
            'total': len(warm_up_functions),
            'active': True,
            'start_time': start_time
        }
        
        warmed_keys = []
        failed_functions = []
        
        for i, warm_up_func in enumerate(warm_up_functions):
            try:
                # 워밍업 함수 실행
                result = warm_up_func()
                
                if isinstance(result, dict):
                    for key, value in result.items():
                        if self.set(key, value, priority=2):  # 중간 우선순위
                            warmed_keys.append(key)
                
                self.warmup_status['completed'] = i + 1
                
            except Exception as e:
                self.logger.error(f"Warm-up function failed: {e}")
                failed_functions.append(str(e))
        
        self.warmup_status['active'] = False
        elapsed_time = time.time() - start_time
        
        self.logger.info(f"Cache warm-up completed: {len(warmed_keys)} keys in {elapsed_time:.2f}s")
        
        return {
            'warmed_keys': warmed_keys,
            'failed_functions': failed_functions,
            'elapsed_time': elapsed_time,
            'success_rate': len(warmed_keys) / max(1, len(warm_up_functions))
        }
    
    def get_predictive_suggestions(self, current_key: str) -> List[str]:
        """예측적 캐시 제안"""
        if self.policy != CachePolicy.PREDICTIVE:
            return []
        
        suggestions = []
        
        try:
            # 현재 키의 접근 패턴 분석
            if current_key in self.access_patterns:
                current_pattern = self.access_patterns[current_key]
                
                # 비슷한 패턴의 키 찾기
                for other_key, other_pattern in self.access_patterns.items():
                    if other_key != current_key and len(other_pattern) > 5:
                        # 간단한 패턴 유사도 계산
                        similarity = self._calculate_pattern_similarity(current_pattern, other_pattern)
                        if similarity > 0.7:
                            suggestions.append(other_key)
            
            return suggestions[:5]  # 상위 5개만 반환
            
        except Exception as e:
            self.logger.error(f"Predictive suggestions failed: {e}")
            return []
    
    def optimize_cache_placement(self):
        """캐시 배치 최적화"""
        with self.lock:
            try:
                self.logger.info("Starting cache placement optimization")
                
                # 접근 빈도 분석
                access_frequency = {}
                for key, entry in self.memory_cache.items():
                    access_frequency[key] = entry.access_count
                
                # 상위 20% -> 메모리 유지
                # 중간 60% -> 디스크로 이동
                # 하위 20% -> 제거 고려
                
                sorted_keys = sorted(access_frequency.keys(), key=lambda k: access_frequency[k], reverse=True)
                
                total_keys = len(sorted_keys)
                high_freq_count = int(total_keys * 0.2)
                medium_freq_count = int(total_keys * 0.6)
                
                # 중간 빈도 키들을 디스크로 이동
                for key in sorted_keys[high_freq_count:high_freq_count + medium_freq_count]:
                    if key in self.memory_cache:
                        entry = self.memory_cache[key]
                        self._store_on_disk(entry)
                        del self.memory_cache[key]
                
                # 메모리 사용량 업데이트
                self._update_memory_stats()
                
                self.logger.info("Cache placement optimization completed")
                
            except Exception as e:
                self.logger.error(f"Cache optimization failed: {e}")
    
    def _determine_storage_level(self, entry: CacheEntry, preferred_level: Optional[CacheLevel]) -> CacheLevel:
        """저장 레벨 결정"""
        if preferred_level:
            return preferred_level
        
        # 적응적 결정
        if self.policy == CachePolicy.ADAPTIVE:
            # 크기 기반
            if entry.size_bytes > 10 * 1024 * 1024:  # 10MB 이상
                return CacheLevel.DISK
            
            # 우선순위 기반
            if entry.priority >= 4:
                return CacheLevel.MEMORY
            
            # 메모리 사용량 기반
            memory_usage_ratio = self.stats.size_bytes / self.max_memory_bytes
            if memory_usage_ratio > 0.8:
                return CacheLevel.DISK
            
            return CacheLevel.MEMORY
        
        return CacheLevel.MEMORY
    
    def _store_in_memory(self, entry: CacheEntry):
        """메모리에 저장"""
        # 메모리 제한 확인
        if self.stats.size_bytes + entry.size_bytes > self.max_memory_bytes:
            self._evict_from_memory()
        
        self.memory_cache[entry.key] = entry
        self.lru_cache[entry.key] = entry.value
    
    def _store_on_disk(self, entry: CacheEntry):
        """디스크에 저장"""
        try:
            cache_file = self.disk_cache_dir / f"{entry.key}.pkl"
            meta_file = self.disk_cache_dir / f"{entry.key}.meta"
            
            # 데이터 저장
            with open(cache_file, 'wb') as f:
                pickle.dump(entry.value, f)
            
            # 메타데이터 저장
            meta_data = {
                'created_at': entry.created_at.isoformat(),
                'ttl': entry.ttl,
                'size_bytes': entry.size_bytes,
                'priority': entry.priority,
                'tags': entry.tags
            }
            
            with open(meta_file, 'w') as f:
                json.dump(meta_data, f)
            
        except Exception as e:
            self.logger.error(f"Disk storage failed for key {entry.key}: {e}")
    
    def _store_in_redis(self, key: str, value: Any, ttl: int):
        """Redis에 저장"""
        try:
            serialized_value = pickle.dumps(value)
            self.redis_client.setex(key, ttl, serialized_value)
        except Exception as e:
            self.logger.error(f"Redis storage failed for key {key}: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        """디스크에서 로드"""
        try:
            cache_file = self.disk_cache_dir / f"{key}.pkl"
            meta_file = self.disk_cache_dir / f"{key}.meta"
            
            if not cache_file.exists() or not meta_file.exists():
                return None
            
            # 메타데이터 로드
            with open(meta_file, 'r') as f:
                meta_data = json.load(f)
            
            # 만료 확인
            created_at = datetime.fromisoformat(meta_data['created_at'])
            ttl = meta_data.get('ttl')
            if ttl and (datetime.now() - created_at).total_seconds() > ttl:
                # 만료된 파일 삭제
                cache_file.unlink(missing_ok=True)
                meta_file.unlink(missing_ok=True)
                return None
            
            # 데이터 로드
            with open(cache_file, 'rb') as f:
                value = pickle.load(f)
            
            # CacheEntry 재구성
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=created_at,
                last_accessed=datetime.now(),
                access_count=1,
                size_bytes=meta_data.get('size_bytes', 0),
                ttl=ttl,
                priority=meta_data.get('priority', 1),
                tags=meta_data.get('tags', [])
            )
            
            return entry
            
        except Exception as e:
            self.logger.error(f"Disk load failed for key {key}: {e}")
            return None
    
    def _load_from_redis(self, key: str) -> Optional[Any]:
        """Redis에서 로드"""
        try:
            serialized_value = self.redis_client.get(key)
            if serialized_value:
                return pickle.loads(serialized_value)
            return None
        except Exception as e:
            self.logger.error(f"Redis load failed for key {key}: {e}")
            return None
    
    def _promote_to_memory(self, entry: CacheEntry):
        """메모리로 승격"""
        self._store_in_memory(entry)
        
        # 디스크에서 제거
        cache_file = self.disk_cache_dir / f"{entry.key}.pkl"
        meta_file = self.disk_cache_dir / f"{entry.key}.meta"
        cache_file.unlink(missing_ok=True)
        meta_file.unlink(missing_ok=True)
    
    def _evict_from_memory(self):
        """메모리에서 제거"""
        if not self.memory_cache:
            return
        
        # 정책별 제거 전략
        if self.policy == CachePolicy.LRU:
            # 가장 오래 사용되지 않은 항목 제거
            oldest_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].last_accessed
            )
            entry = self.memory_cache[oldest_key]
            
        elif self.policy == CachePolicy.LFU:
            # 가장 적게 사용된 항목 제거
            least_used_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].access_count
            )
            entry = self.memory_cache[least_used_key]
            
        else:
            # 기본: 가장 큰 항목 제거
            largest_key = max(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].size_bytes
            )
            entry = self.memory_cache[largest_key]
        
        # 디스크로 이동 (우선순위가 높은 경우)
        if entry.priority >= 2:
            self._store_on_disk(entry)
        
        # 메모리에서 제거
        self.stats.size_bytes -= entry.size_bytes
        del self.memory_cache[entry.key]
        self.stats.evictions += 1
        
        self.logger.debug(f"Evicted from memory: {entry.key}")
    
    def _calculate_size(self, value: Any) -> int:
        """객체 크기 계산"""
        try:
            if isinstance(value, (str, int, float, bool)):
                return len(str(value).encode())
            elif isinstance(value, (list, tuple, dict)):
                return len(pickle.dumps(value))
            elif isinstance(value, pd.DataFrame):
                return value.memory_usage(deep=True).sum()
            elif isinstance(value, np.ndarray):
                return value.nbytes
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # 기본값
    
    def _record_access_pattern(self, key: str):
        """접근 패턴 기록"""
        current_time = datetime.now()
        self.access_patterns[key].append(current_time)
        
        # 최근 100개 접근만 유지
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-100:]
    
    def _calculate_pattern_similarity(self, pattern1: List[datetime], pattern2: List[datetime]) -> float:
        """패턴 유사도 계산"""
        try:
            # 간단한 접근 간격 기반 유사도
            if len(pattern1) < 3 or len(pattern2) < 3:
                return 0.0
            
            intervals1 = [(pattern1[i] - pattern1[i-1]).total_seconds() 
                         for i in range(1, min(len(pattern1), 10))]
            intervals2 = [(pattern2[i] - pattern2[i-1]).total_seconds() 
                         for i in range(1, min(len(pattern2), 10))]
            
            # 평균 간격 비교
            avg1 = sum(intervals1) / len(intervals1)
            avg2 = sum(intervals2) / len(intervals2)
            
            # 유사도 계산 (차이가 적을수록 높은 점수)
            max_diff = max(avg1, avg2, 1)
            similarity = 1.0 - abs(avg1 - avg2) / max_diff
            
            return max(0.0, similarity)
            
        except Exception:
            return 0.0
    
    def _update_access_time(self, start_time: float):
        """접근 시간 업데이트"""
        access_time_ms = (time.time() - start_time) * 1000
        
        # 지수 이동 평균으로 업데이트
        alpha = 0.1
        self.stats.avg_access_time_ms = (
            alpha * access_time_ms + (1 - alpha) * self.stats.avg_access_time_ms
        )
    
    def _update_memory_stats(self):
        """메모리 통계 업데이트"""
        self.stats.size_bytes = sum(entry.size_bytes for entry in self.memory_cache.values())
        self.stats.entry_count = len(self.memory_cache)
        self.stats.update_hit_ratio()
        
        # 시스템 메모리 사용량
        process = psutil.Process()
        self.stats.memory_usage_mb = process.memory_info().rss / 1024 / 1024
    
    def _cleanup_loop(self):
        """백그라운드 정리 루프"""
        while True:
            try:
                time.sleep(300)  # 5분마다 실행
                
                with self.lock:
                    # 만료된 엔트리 정리
                    expired_keys = [
                        key for key, entry in self.memory_cache.items()
                        if entry.is_expired()
                    ]
                    
                    for key in expired_keys:
                        self.invalidate(key)
                    
                    # 디스크 캐시 정리
                    self._cleanup_disk_cache()
                    
                    # 통계 업데이트
                    self._update_memory_stats()
                    
                    if expired_keys:
                        self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
    
    def _cleanup_disk_cache(self):
        """디스크 캐시 정리"""
        try:
            for cache_file in self.disk_cache_dir.glob("*.pkl"):
                meta_file = cache_file.with_suffix('.meta')
                
                if meta_file.exists():
                    try:
                        with open(meta_file, 'r') as f:
                            meta_data = json.load(f)
                        
                        created_at = datetime.fromisoformat(meta_data['created_at'])
                        ttl = meta_data.get('ttl')
                        
                        if ttl and (datetime.now() - created_at).total_seconds() > ttl:
                            cache_file.unlink(missing_ok=True)
                            meta_file.unlink(missing_ok=True)
                    
                    except Exception:
                        # 손상된 메타데이터 파일 제거
                        cache_file.unlink(missing_ok=True)
                        meta_file.unlink(missing_ok=True)
        
        except Exception as e:
            self.logger.error(f"Disk cache cleanup failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        with self.lock:
            self._update_memory_stats()
            
            return {
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'hit_ratio': self.stats.hit_ratio,
                'evictions': self.stats.evictions,
                'memory_entries': len(self.memory_cache),
                'memory_size_mb': self.stats.size_bytes / 1024 / 1024,
                'memory_usage_mb': self.stats.memory_usage_mb,
                'avg_access_time_ms': self.stats.avg_access_time_ms,
                'disk_cache_files': len(list(self.disk_cache_dir.glob("*.pkl"))),
                'redis_connected': self.redis_client is not None,
                'policy': self.policy.value,
                'warmup_status': self.warmup_status
            }
    
    def export_metrics(self) -> Dict[str, Any]:
        """메트릭 내보내기"""
        stats = self.get_stats()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cache_performance': {
                'hit_ratio': stats['hit_ratio'],
                'avg_access_time_ms': stats['avg_access_time_ms'],
                'memory_efficiency': stats['memory_size_mb'] / max(1, stats['memory_entries'])
            },
            'resource_usage': {
                'memory_usage_mb': stats['memory_usage_mb'],
                'disk_usage_files': stats['disk_cache_files']
            },
            'cache_stats': stats
        }
    
    def clear_all(self):
        """모든 캐시 삭제"""
        with self.lock:
            # 메모리 캐시 삭제
            self.memory_cache.clear()
            self.lru_cache.clear()
            
            # 디스크 캐시 삭제
            for cache_file in self.disk_cache_dir.glob("*"):
                cache_file.unlink(missing_ok=True)
            
            # Redis 캐시 삭제 (주의: 전체 Redis DB 삭제)
            if self.redis_client:
                try:
                    self.redis_client.flushdb()
                except Exception as e:
                    self.logger.error(f"Redis flush failed: {e}")
            
            # 통계 초기화
            self.stats = CacheStats()
            self.access_patterns.clear()
            
            self.logger.info("All caches cleared")


# 글로벌 고급 캐시 인스턴스
advanced_cache = IntelligentCache(
    max_memory_mb=int(os.getenv('CACHE_MEMORY_MB', '500')),
    max_disk_mb=int(os.getenv('CACHE_DISK_MB', '2000')),
    enable_distributed=os.getenv('REDIS_URL') is not None,
    redis_url=os.getenv('REDIS_URL')
)


# 캐시 데코레이터들
def intelligent_cache(
    ttl: int = 3600,
    priority: int = 1,
    tags: Optional[List[str]] = None,
    level: Optional[CacheLevel] = None
):
    """지능형 캐시 데코레이터"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # 캐시 키 생성
            key_data = {
                'func': func.__name__,
                'module': func.__module__,
                'args': str(args)[:200],  # 길이 제한
                'kwargs': str(sorted(kwargs.items()))[:200]
            }
            cache_key = hashlib.md5(str(key_data).encode()).hexdigest()
            
            # 캐시에서 조회
            cached_result = advanced_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 함수 실행
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # 실행 시간이 긴 경우 높은 우선순위로 캐시
            cache_priority = priority
            if execution_time > 1.0:  # 1초 이상
                cache_priority = max(priority, 3)
            
            # 캐시에 저장
            advanced_cache.set(
                cache_key,
                result,
                ttl=ttl,
                priority=cache_priority,
                tags=tags,
                level=level
            )
            
            return result
        
        return wrapper
    return decorator


# Streamlit 통합
def show_advanced_cache_dashboard():
    """고급 캐시 대시보드 표시"""
    import streamlit as st
    
    st.subheader("🧠 지능형 캐시 시스템")
    
    # 캐시 통계
    stats = advanced_cache.get_stats()
    
    # 메인 메트릭
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("히트율", f"{stats['hit_ratio']*100:.1f}%")
    
    with col2:
        st.metric("평균 접근 시간", f"{stats['avg_access_time_ms']:.1f}ms")
    
    with col3:
        st.metric("메모리 사용량", f"{stats['memory_size_mb']:.1f}MB")
    
    with col4:
        st.metric("총 엔트리", stats['memory_entries'])
    
    # 상세 통계
    st.subheader("📊 상세 통계")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("캐시 히트", stats['hits'])
        st.metric("캐시 미스", stats['misses'])
        st.metric("제거된 항목", stats['evictions'])
    
    with col2:
        st.metric("디스크 캐시 파일", stats['disk_cache_files'])
        st.metric("Redis 연결", "활성" if stats['redis_connected'] else "비활성")
        st.metric("캐시 정책", stats['policy'].upper())
    
    # 워밍업 상태
    warmup = stats['warmup_status']
    if warmup['active']:
        st.subheader("🔥 캐시 워밍업 진행 중")
        progress = warmup['completed'] / max(1, warmup['total'])
        st.progress(progress)
        st.text(f"진행률: {warmup['completed']}/{warmup['total']}")
    
    # 제어 버튼
    st.subheader("🎛️ 캐시 관리")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("캐시 최적화"):
            advanced_cache.optimize_cache_placement()
            st.success("캐시 최적화가 완료되었습니다.")
            st.experimental_rerun()
    
    with col2:
        if st.button("통계 초기화"):
            advanced_cache.stats = CacheStats()
            st.success("통계가 초기화되었습니다.")
            st.experimental_rerun()
    
    with col3:
        if st.button("⚠️ 전체 캐시 삭제"):
            advanced_cache.clear_all()
            st.success("모든 캐시가 삭제되었습니다.")
            st.experimental_rerun()
    
    # 메트릭 내보내기
    if st.button("메트릭 내보내기"):
        metrics = advanced_cache.export_metrics()
        st.json(metrics)


# 캐시 워밍업 함수들
def create_cache_warmup_functions() -> List[Callable]:
    """캐시 워밍업 함수들 생성"""
    warmup_functions = []
    
    def warmup_common_calculations():
        """공통 계산 워밍업"""
        import numpy as np
        return {
            'pi': np.pi,
            'e': np.e,
            'sqrt2': np.sqrt(2),
            'common_dates': pd.date_range('2020-01-01', '2023-12-31', freq='D').tolist()
        }
    
    def warmup_sample_data():
        """샘플 데이터 워밍업"""
        return {
            'sample_prices': np.random.randn(1000) * 10 + 100,
            'sample_returns': np.random.randn(1000) * 0.02,
            'sample_volumes': np.random.randint(1000000, 10000000, 1000)
        }
    
    warmup_functions.extend([
        warmup_common_calculations,
        warmup_sample_data
    ])
    
    return warmup_functions