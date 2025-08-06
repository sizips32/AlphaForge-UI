"""
환경변수 관리 모듈
보안이 중요한 API 키와 설정값들을 안전하게 관리합니다.
"""

import os
import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# .env 파일 로드를 위한 python-dotenv 사용
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    st.warning("💡 python-dotenv가 설치되지 않았습니다. .env 파일을 사용하려면 'pip install python-dotenv'를 실행하세요.")

class EnvironmentManager:
    """환경변수 관리 클래스"""
    
    def __init__(self):
        self.env_file = Path('.env')
        self.template_file = Path('.env.template')
        
        # .env 파일 존재 여부 확인
        if not self.env_file.exists() and self.template_file.exists():
            self._show_env_setup_guide()
    
    def _show_env_setup_guide(self):
        """환경 설정 안내를 표시합니다."""
        st.info("""
        📋 **환경 설정 안내**
        
        1. `.env.template` 파일을 `.env`로 복사하세요
        2. 필요한 API 키와 설정값을 입력하세요
        3. `.env` 파일은 버전 관리에 포함되지 않습니다
        
        ```bash
        cp .env.template .env
        ```
        """)
    
    def get_env(self, key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        """환경변수 값을 안전하게 가져옵니다."""
        try:
            value = os.getenv(key, default)
            
            if required and not value:
                st.error(f"❌ 필수 환경변수 '{key}'가 설정되지 않았습니다.")
                return None
            
            return value
        
        except Exception as e:
            logging.error(f"환경변수 {key} 조회 실패: {e}")
            return default
    
    def get_bool_env(self, key: str, default: bool = False) -> bool:
        """불린 환경변수를 가져옵니다."""
        value = self.get_env(key)
        if value is None:
            return default
        
        return value.lower() in ('true', '1', 'yes', 'on')
    
    def get_int_env(self, key: str, default: int = 0) -> int:
        """정수 환경변수를 가져옵니다."""
        value = self.get_env(key)
        if value is None:
            return default
        
        try:
            return int(value)
        except ValueError:
            logging.warning(f"환경변수 {key}의 값 '{value}'를 정수로 변환할 수 없습니다. 기본값 {default}를 사용합니다.")
            return default
    
    def get_float_env(self, key: str, default: float = 0.0) -> float:
        """실수 환경변수를 가져옵니다."""
        value = self.get_env(key)
        if value is None:
            return default
        
        try:
            return float(value)
        except ValueError:
            logging.warning(f"환경변수 {key}의 값 '{value}'를 실수로 변환할 수 없습니다. 기본값 {default}를 사용합니다.")
            return default
    
    def get_list_env(self, key: str, separator: str = ',', default: Optional[list] = None) -> list:
        """리스트 환경변수를 가져옵니다."""
        value = self.get_env(key)
        if value is None:
            return default or []
        
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    def get_api_config(self) -> Dict[str, Any]:
        """API 설정을 반환합니다."""
        return {
            'yahoo_finance_api_key': self.get_env('YAHOO_FINANCE_API_KEY'),
            'alpha_vantage_api_key': self.get_env('ALPHA_VANTAGE_API_KEY'),
            'quandl_api_key': self.get_env('QUANDL_API_KEY'),
        }
    
    def get_app_config(self) -> Dict[str, Any]:
        """애플리케이션 설정을 반환합니다."""
        return {
            'app_name': self.get_env('APP_NAME', 'AlphaForge-UI'),
            'app_version': self.get_env('APP_VERSION', '1.0.0'),
            'debug': self.get_bool_env('DEBUG', False),
            'secret_key': self.get_env('SECRET_KEY', 'default_secret_key_change_in_production'),
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """캐시 설정을 반환합니다."""
        return {
            'cache_ttl': self.get_int_env('CACHE_TTL', 3600),
            'cache_dir': self.get_env('CACHE_DIR', '.cache'),
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """성능 설정을 반환합니다."""
        return {
            'max_workers': self.get_int_env('MAX_WORKERS', 4),
            'chunk_size': self.get_int_env('CHUNK_SIZE', 10000),
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """로깅 설정을 반환합니다."""
        return {
            'log_level': self.get_env('LOG_LEVEL', 'INFO'),
            'log_file': self.get_env('LOG_FILE', 'logs/app.log'),
        }
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """Streamlit 설정을 반환합니다."""
        return {
            'server_port': self.get_int_env('STREAMLIT_SERVER_PORT', 8503),
            'server_address': self.get_env('STREAMLIT_SERVER_ADDRESS', 'localhost'),
        }
    
    def validate_environment(self) -> Dict[str, Any]:
        """환경 설정 유효성을 검증합니다."""
        issues = []
        warnings = []
        
        # .env 파일 확인
        if not self.env_file.exists():
            warnings.append(".env 파일이 없습니다. 기본 설정을 사용합니다.")
        
        # API 키 확인
        api_config = self.get_api_config()
        if not any(api_config.values()):
            warnings.append("API 키가 설정되지 않았습니다. 일부 기능이 제한될 수 있습니다.")
        
        # 보안 키 확인
        secret_key = self.get_env('SECRET_KEY', 'default_secret_key_change_in_production')
        if secret_key == 'default_secret_key_change_in_production':
            issues.append("SECRET_KEY를 변경해주세요.")
        
        # 로그 디렉토리 확인
        log_file = Path(self.get_env('LOG_FILE', 'logs/app.log'))
        log_file.parent.mkdir(exist_ok=True)
        
        # 캐시 디렉토리 확인
        cache_dir = Path(self.get_env('CACHE_DIR', '.cache'))
        cache_dir.mkdir(exist_ok=True)
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def show_environment_status(self):
        """환경 설정 상태를 표시합니다."""
        validation_result = self.validate_environment()
        
        if validation_result['is_valid']:
            st.success("✅ 환경 설정이 올바르게 구성되었습니다.")
        else:
            st.error("❌ 환경 설정에 문제가 있습니다:")
            for issue in validation_result['issues']:
                st.error(f"- {issue}")
        
        if validation_result['warnings']:
            for warning in validation_result['warnings']:
                st.warning(f"⚠️ {warning}")
        
        # 설정 정보 표시 (민감한 정보는 마스킹)
        with st.expander("🔧 현재 환경 설정"):
            app_config = self.get_app_config()
            cache_config = self.get_cache_config()
            performance_config = self.get_performance_config()
            
            st.write("**애플리케이션 설정:**")
            st.json({
                'app_name': app_config['app_name'],
                'app_version': app_config['app_version'],
                'debug': app_config['debug']
            })
            
            st.write("**성능 설정:**")
            st.json(performance_config)
            
            st.write("**캐시 설정:**")
            st.json(cache_config)
            
            st.write("**API 키 상태:**")
            api_config = self.get_api_config()
            api_status = {}
            for key, value in api_config.items():
                api_status[key] = "✅ 설정됨" if value else "❌ 미설정"
            st.json(api_status)
    
    def mask_sensitive_value(self, value: str, mask_char: str = '*', show_last: int = 4) -> str:
        """민감한 값을 마스킹합니다."""
        if not value or len(value) <= show_last:
            return mask_char * 8
        
        return mask_char * (len(value) - show_last) + value[-show_last:]

# 전역 환경변수 매니저 인스턴스
env_manager = EnvironmentManager()

# 편의 함수들
def get_env(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """환경변수를 가져옵니다."""
    return env_manager.get_env(key, default, required)

def get_api_key(service: str) -> Optional[str]:
    """API 키를 안전하게 가져옵니다."""
    key_mapping = {
        'yahoo_finance': 'YAHOO_FINANCE_API_KEY',
        'alpha_vantage': 'ALPHA_VANTAGE_API_KEY',
        'quandl': 'QUANDL_API_KEY'
    }
    
    env_key = key_mapping.get(service.lower())
    if not env_key:
        logging.warning(f"알 수 없는 서비스: {service}")
        return None
    
    return env_manager.get_env(env_key)