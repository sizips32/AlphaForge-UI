"""
AlphaForge-UI 설정 모듈
앱의 전역 설정, 상수, 경로 등을 관리합니다.
"""

import os
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent

# 데이터 경로
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"
LOGS_DIR = PROJECT_ROOT / "logs"

# 로고 경로
LOGO_PATH = ASSETS_DIR / "logo.png" if (ASSETS_DIR / "logo.png").exists() else None

# 색상 팔레트 - 밝은 모던 테마
COLORS = {
    "primary": "#1E88E5",      # 파란색
    "secondary": "#42A5F5",    # 밝은 파란색
    "accent": "#FF6B6B",       # 산호색
    "warning": "#FFA726",      # 주황색
    "error": "#EF5350",        # 빨간색
    "success": "#66BB6A",      # 초록색
    "info": "#26C6DA",         # 청록색
    "background": "#F8FAFC",   # 밝은 회색
    "card": "#FFFFFF",         # 흰색
    "sidebar": "#FFFFFF",      # 흰색
    "text_primary": "#1E293B", # 진한 회색
    "text_secondary": "#64748B", # 중간 회색
    "gradient_start": "#E3F2FD", # 밝은 파란색 그라데이션 시작
    "gradient_end": "#F3E5F5",   # 밝은 보라색 그라데이션 끝
    "neon_purple": "#7C3AED",     # 보라색
    "neon_pink": "#EC4899",       # 핑크
    "neon_blue": "#3B82F6",       # 파란색
    "neon_green": "#10B981",      # 초록색
    "neon_yellow": "#F59E0B",     # 노란색
    "neon_cyan": "#06B6D4"        # 청록색
}

# 데이터 검증 규칙
DATA_VALIDATION = {
    "required_columns": ["Date", "Ticker", "Close"],
    "optional_columns": ["Open", "High", "Low", "Volume"],
    "date_format": "%Y-%m-%d",
    "min_history_days": 252,  # 최소 1년 데이터
    "max_missing_ratio": 0.05,  # 5% 미만 결측치
    "price_positive": True
}

# 팩터 설정
FACTOR_SETTINGS = {
    "default_factors": [
        "Momentum", "Value", "Quality", "Size", "Low Volatility"
    ],
    "factor_pool_size": 10,
    "min_ic": 0.02,
    "min_icir": 0.5,
    "rebalancing_frequency": "monthly"
}

# 백테스팅 설정
BACKTEST_SETTINGS = {
    "default_start_date": "2020-01-01",
    "default_end_date": "2024-12-31",
    "benchmark": "SPY",  # S&P 500 ETF
    "risk_free_rate": 0.02,  # 2% 무위험 수익률
    "transaction_cost": 0.001,  # 0.1% 거래 비용
    "max_position_size": 0.1  # 최대 10% 포지션
}

# UI 설정
UI_SETTINGS = {
    "page_title": "AlphaForge-UI",
    "page_icon": "🚀",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "max_upload_size": 200 * 1024 * 1024,  # 200MB
    "chart_height": 500,
    "table_height": 400
}

# 성능 설정
PERFORMANCE_SETTINGS = {
    "cache_ttl": 3600,  # 1시간 캐시
    "max_workers": 4,
    "chunk_size": 10000,
    "progress_bar": True
}

# 로깅 설정
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "app.log" if LOGS_DIR.exists() else None
}

# 필요한 디렉토리 생성
def ensure_directories():
    """필요한 디렉토리들을 생성합니다."""
    directories = [DATA_DIR, ASSETS_DIR, LOGS_DIR]
    for directory in directories:
        directory.mkdir(exist_ok=True)

# 초기화 시 디렉토리 생성
ensure_directories() 
