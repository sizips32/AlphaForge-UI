"""
캐싱 유틸리티 모듈
Streamlit 캐시 데코레이터와 커스텀 캐싱 로직을 관리합니다.
"""

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import pickle
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional, Dict, Tuple
import functools

# 캐시 디렉토리 설정
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_data_hash(data: pd.DataFrame) -> str:
    """데이터프레임의 해시값을 생성합니다."""
    try:
        # 데이터의 기본 정보를 해시화
        info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': str(data.dtypes.to_dict()),
            'sample': str(data.head().to_dict()) if not data.empty else "",
            'memory': data.memory_usage().sum()
        }
        
        # 딕셔너리를 문자열로 변환하고 해시 생성
        info_str = str(sorted(info.items()))
        return hashlib.md5(info_str.encode()).hexdigest()
    except Exception:
        # 해시 생성 실패시 현재 시간 기반 해시 반환
        return hashlib.md5(str(datetime.now()).encode()).hexdigest()

def get_settings_hash(settings: Dict) -> str:
    """설정 딕셔너리의 해시값을 생성합니다."""
    try:
        settings_str = str(sorted(settings.items()))
        return hashlib.md5(settings_str.encode()).hexdigest()
    except Exception:
        return hashlib.md5(str(datetime.now()).encode()).hexdigest()

@st.cache_data(ttl=3600, show_spinner=True)
def cached_data_processing(data_hash: str, data: pd.DataFrame) -> pd.DataFrame:
    """데이터 처리 결과를 캐싱합니다."""
    from utils.data_processor import DataProcessor
    
    processor = DataProcessor()
    return processor.process_data(data)

@st.cache_data(ttl=1800, show_spinner=True)
def cached_factor_mining(data_hash: str, settings_hash: str, data: pd.DataFrame, settings: Dict) -> Dict:
    """팩터 마이닝 결과를 캐싱합니다."""
    from utils.factor_miner import FactorMiner
    
    miner = FactorMiner(settings)
    factors = miner.generate_basic_factors(data)
    
    return {
        'factors': factors,
        'settings': settings,
        'timestamp': datetime.now().isoformat()
    }

@st.cache_data(ttl=1800, show_spinner=True)
def cached_dynamic_combination(factors_hash: str, data_hash: str, settings_hash: str, 
                             factors: list, data: pd.DataFrame, settings: Dict) -> Dict:
    """동적 결합 결과를 캐싱합니다."""
    from utils.dynamic_combiner import DynamicCombiner
    
    combiner = DynamicCombiner(settings)
    result = combiner.combine_factors(factors, data)
    
    return {
        'mega_alpha': result,
        'settings': settings,
        'timestamp': datetime.now().isoformat()
    }

@st.cache_data(ttl=3600, show_spinner=True)
def cached_backtest(mega_alpha_hash: str, data_hash: str, settings_hash: str,
                   mega_alpha: Any, data: pd.DataFrame, settings: Dict) -> Dict:
    """백테스팅 결과를 캐싱합니다."""
    from utils.backtester import Backtester
    
    backtester = Backtester(settings)
    result = backtester.run_backtest(mega_alpha, data)
    
    return {
        'backtest_results': result,
        'settings': settings,
        'timestamp': datetime.now().isoformat()
    }

@st.cache_data(ttl=7200, show_spinner=True)  # 2시간 캐시
def cached_yahoo_finance_download(tickers: Tuple[str, ...], start_date: str, end_date: str) -> pd.DataFrame:
    """야후 파이낸스 데이터 다운로드를 캐싱합니다."""
    from utils.yahoo_finance_downloader import YahooFinanceDownloader
    
    downloader = YahooFinanceDownloader()
    return downloader.download_data(list(tickers), start_date, end_date)

@st.cache_data(ttl=1800, show_spinner=True)
def cached_technical_indicators(data_hash: str, data: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 계산을 캐싱합니다."""
    from utils.data_processor import DataProcessor
    
    processor = DataProcessor()
    return processor._calculate_technical_indicators(data)

@st.cache_data(ttl=1800, show_spinner=True)
def cached_performance_analysis(data_hash: str, results_hash: str, 
                              data: pd.DataFrame, results: Dict) -> Dict:
    """성과 분석을 캐싱합니다."""
    from utils.performance_analyzer import PerformanceAnalyzer
    
    analyzer = PerformanceAnalyzer()
    return analyzer.analyze_performance(data, results)

class CustomCache:
    """커스텀 파일 기반 캐시 클래스"""
    
    def __init__(self, cache_dir: str = ".cache", ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl
    
    def _get_cache_path(self, key: str) -> Path:
        """캐시 파일 경로를 반환합니다."""
        return self.cache_dir / f"{key}.pkl"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """캐시가 유효한지 확인합니다."""
        if not cache_path.exists():
            return False
        
        # TTL 확인
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        current_time = datetime.now()
        
        return (current_time - cache_time).seconds < self.ttl
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값을 가져옵니다."""
        try:
            cache_path = self._get_cache_path(key)
            
            if not self._is_cache_valid(cache_path):
                return None
            
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        except Exception as e:
            st.warning(f"캐시 로드 실패: {e}")
            return None
    
    def set(self, key: str, value: Any) -> bool:
        """캐시에 값을 저장합니다."""
        try:
            cache_path = self._get_cache_path(key)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            return True
        
        except Exception as e:
            st.warning(f"캐시 저장 실패: {e}")
            return False
    
    def clear(self) -> bool:
        """캐시를 삭제합니다."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            return True
        
        except Exception as e:
            st.error(f"캐시 삭제 실패: {e}")
            return False
    
    def cache_decorator(self, ttl: Optional[int] = None):
        """캐시 데코레이터"""
        cache_ttl = ttl or self.ttl
        
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # 캐시 키 생성
                key_data = {
                    'func': func.__name__,
                    'args': str(args),
                    'kwargs': str(sorted(kwargs.items()))
                }
                cache_key = hashlib.md5(str(key_data).encode()).hexdigest()
                
                # 캐시에서 조회
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # 함수 실행 및 캐시 저장
                result = func(*args, **kwargs)
                self.set(cache_key, result)
                
                return result
            
            return wrapper
        return decorator

# 전역 캐시 인스턴스
custom_cache = CustomCache()

def clear_all_caches():
    """모든 캐시를 삭제합니다."""
    try:
        # Streamlit 캐시 삭제
        st.cache_data.clear()
        
        # 커스텀 캐시 삭제  
        custom_cache.clear()
        
        st.success("모든 캐시가 삭제되었습니다.")
        return True
        
    except Exception as e:
        st.error(f"캐시 삭제 중 오류 발생: {e}")
        return False

def get_cache_info() -> Dict[str, Any]:
    """캐시 정보를 반환합니다."""
    try:
        cache_files = list(CACHE_DIR.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_count': len(cache_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_dir': str(CACHE_DIR),
            'oldest_cache': min((f.stat().st_mtime for f in cache_files), default=0),
            'newest_cache': max((f.stat().st_mtime for f in cache_files), default=0)
        }
    
    except Exception:
        return {
            'cache_count': 0,
            'total_size_mb': 0,
            'cache_dir': str(CACHE_DIR),
            'oldest_cache': 0,
            'newest_cache': 0
        }

# 캐시 워밍업 함수들
def warm_up_data_cache(data: pd.DataFrame):
    """데이터 처리 캐시를 사전 로드합니다."""
    if data is not None and not data.empty:
        data_hash = get_data_hash(data)
        cached_data_processing(data_hash, data)

def warm_up_factor_cache(data: pd.DataFrame, settings: Dict):
    """팩터 마이닝 캐시를 사전 로드합니다."""
    if data is not None and not data.empty:
        data_hash = get_data_hash(data)
        settings_hash = get_settings_hash(settings)
        cached_factor_mining(data_hash, settings_hash, data, settings)