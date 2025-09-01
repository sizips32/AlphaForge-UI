"""
보안 관리 시스템
인증, 인가, 보안 모니터링, 데이터 보호
"""

import streamlit as st
import pandas as pd
import hashlib
import hmac
import secrets
import jwt
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import re
import json
import sqlite3
from pathlib import Path
import logging
import time
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class UserRole(Enum):
    ADMIN = "admin"
    USER = "user"
    ANALYST = "analyst"
    VIEWER = "viewer"

class PermissionLevel(Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"

class SecurityEvent(Enum):
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PERMISSION_DENIED = "permission_denied"
    DATA_ACCESS = "data_access"
    CONFIG_CHANGE = "config_change"
    SECURITY_VIOLATION = "security_violation"

@dataclass
class User:
    id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    permissions: List[str]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    def __post_init__(self):
        if isinstance(self.permissions, str):
            self.permissions = json.loads(self.permissions)

@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True

@dataclass
class SecurityLog:
    id: str
    event_type: SecurityEvent
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    timestamp: datetime
    details: Dict[str, Any]
    risk_score: int = 0

class PasswordPolicy:
    """비밀번호 정책"""
    
    MIN_LENGTH = 8
    MAX_LENGTH = 128
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGITS = True
    REQUIRE_SPECIAL = True
    SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    @classmethod
    def validate_password(cls, password: str) -> Tuple[bool, List[str]]:
        """비밀번호 유효성 검사"""
        errors = []
        
        if len(password) < cls.MIN_LENGTH:
            errors.append(f"비밀번호는 최소 {cls.MIN_LENGTH}자 이상이어야 합니다.")
        
        if len(password) > cls.MAX_LENGTH:
            errors.append(f"비밀번호는 최대 {cls.MAX_LENGTH}자 이하여야 합니다.")
        
        if cls.REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
            errors.append("비밀번호에 대문자가 포함되어야 합니다.")
        
        if cls.REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
            errors.append("비밀번호에 소문자가 포함되어야 합니다.")
        
        if cls.REQUIRE_DIGITS and not re.search(r'\\d', password):
            errors.append("비밀번호에 숫자가 포함되어야 합니다.")
        
        if cls.REQUIRE_SPECIAL and not re.search(f'[{re.escape(cls.SPECIAL_CHARS)}]', password):
            errors.append("비밀번호에 특수문자가 포함되어야 합니다.")
        
        # 일반적인 취약한 패턴 검사
        weak_patterns = [
            r'(.)\\1{3,}',  # 연속된 같은 문자 4개 이상
            r'1234|abcd|qwer',  # 연속된 순서
            r'password|admin|user'  # 일반적인 단어
        ]
        
        for pattern in weak_patterns:
            if re.search(pattern, password.lower()):
                errors.append("취약한 패턴이 감지되었습니다.")
                break
        
        return len(errors) == 0, errors

class PasswordManager:
    """비밀번호 관리자"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """비밀번호 해시화"""
        salt = secrets.token_hex(32)
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return salt + pwdhash.hex()
    
    @staticmethod
    def verify_password(password: str, stored_hash: str) -> bool:
        """비밀번호 검증"""
        try:
            salt = stored_hash[:64]
            stored_pwdhash = stored_hash[64:]
            pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
            return pwdhash.hex() == stored_pwdhash
        except Exception:
            return False

class EncryptionManager:
    """암호화 관리자"""
    
    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            key = self._generate_key()
        self.fernet = Fernet(key)
    
    def _generate_key(self) -> bytes:
        """암호화 키 생성"""
        return Fernet.generate_key()
    
    def encrypt(self, data: str) -> str:
        """데이터 암호화"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """데이터 복호화"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_sensitive_data(self, data: Dict[str, Any]) -> str:
        """민감한 데이터 암호화"""
        json_data = json.dumps(data)
        return self.encrypt(json_data)
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> Dict[str, Any]:
        """민감한 데이터 복호화"""
        json_data = self.decrypt(encrypted_data)
        return json.loads(json_data)

class JWTManager:
    """JWT 토큰 관리자"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or os.environ.get('JWT_SECRET_KEY', secrets.token_urlsafe(32))
        self.algorithm = 'HS256'
        self.token_expiry_hours = 24
    
    def generate_token(self, user_id: str, role: str, permissions: List[str]) -> str:
        """JWT 토큰 생성"""
        payload = {
            'user_id': user_id,
            'role': role,
            'permissions': permissions,
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            'iat': datetime.utcnow(),
            'jti': secrets.token_urlsafe(16)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """JWT 토큰 검증"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

class UserManager:
    """사용자 관리자"""
    
    def __init__(self, db_path: str = "security.db"):
        self.db_path = db_path
        self.password_manager = PasswordManager()
        self.jwt_manager = JWTManager()
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            # 사용자 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    last_login DATETIME,
                    is_active BOOLEAN DEFAULT TRUE,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until DATETIME
                )
            """)
            
            # 세션 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    last_activity DATETIME NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # 보안 로그 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS security_logs (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp DATETIME NOT NULL,
                    details TEXT,
                    risk_score INTEGER DEFAULT 0
                )
            """)
            
            # 기본 관리자 계정 생성
            self._create_default_admin()
    
    def _create_default_admin(self):
        """기본 관리자 계정 생성"""
        admin_exists = self.get_user_by_username("admin")
        if not admin_exists:
            admin_user = User(
                id="admin_001",
                username="admin",
                email="admin@alphaforge.ai",
                password_hash=self.password_manager.hash_password("Admin123!"),
                role=UserRole.ADMIN,
                permissions=["read", "write", "execute", "admin"],
                created_at=datetime.now()
            )
            self.create_user(admin_user)
    
    def create_user(self, user: User) -> bool:
        """사용자 생성"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO users (id, username, email, password_hash, role, permissions, 
                                     created_at, last_login, is_active, failed_login_attempts, locked_until)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user.id, user.username, user.email, user.password_hash,
                    user.role.value, json.dumps(user.permissions), 
                    user.created_at.isoformat(), 
                    user.last_login.isoformat() if user.last_login else None,
                    user.is_active, user.failed_login_attempts,
                    user.locked_until.isoformat() if user.locked_until else None
                ))
            return True
        except Exception as e:
            logging.error(f"사용자 생성 실패: {str(e)}")
            return False
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """사용자명으로 사용자 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, username, email, password_hash, role, permissions, 
                       created_at, last_login, is_active, failed_login_attempts, locked_until
                FROM users WHERE username = ?
            """, (username,))
            
            row = cursor.fetchone()
            if row:
                return User(
                    id=row[0], username=row[1], email=row[2], password_hash=row[3],
                    role=UserRole(row[4]), permissions=json.loads(row[5]),
                    created_at=datetime.fromisoformat(row[6]),
                    last_login=datetime.fromisoformat(row[7]) if row[7] else None,
                    is_active=bool(row[8]), failed_login_attempts=row[9],
                    locked_until=datetime.fromisoformat(row[10]) if row[10] else None
                )
        return None
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: str = "", user_agent: str = "") -> Optional[str]:
        """사용자 인증"""
        user = self.get_user_by_username(username)
        
        if not user:
            self._log_security_event(SecurityEvent.LOGIN_FAILURE, None, ip_address, user_agent, 
                                   {"reason": "user_not_found", "username": username})
            return None
        
        # 계정 잠금 확인
        if user.locked_until and user.locked_until > datetime.now():
            self._log_security_event(SecurityEvent.LOGIN_FAILURE, user.id, ip_address, user_agent,
                                   {"reason": "account_locked", "locked_until": user.locked_until.isoformat()})
            return None
        
        # 계정 비활성 확인
        if not user.is_active:
            self._log_security_event(SecurityEvent.LOGIN_FAILURE, user.id, ip_address, user_agent,
                                   {"reason": "account_inactive"})
            return None
        
        # 비밀번호 확인
        if not self.password_manager.verify_password(password, user.password_hash):
            self._handle_failed_login(user, ip_address, user_agent)
            return None
        
        # 성공적인 로그인 처리
        self._handle_successful_login(user, ip_address, user_agent)
        
        # JWT 토큰 생성
        token = self.jwt_manager.generate_token(user.id, user.role.value, user.permissions)
        
        # 세션 생성
        session_id = self._create_session(user.id, ip_address, user_agent)
        
        return token
    
    def _handle_failed_login(self, user: User, ip_address: str, user_agent: str):
        """로그인 실패 처리"""
        user.failed_login_attempts += 1
        
        # 5회 실패 시 계정 잠금
        if user.failed_login_attempts >= 5:
            user.locked_until = datetime.now() + timedelta(minutes=30)
        
        self._update_user_security_info(user)
        
        self._log_security_event(SecurityEvent.LOGIN_FAILURE, user.id, ip_address, user_agent,
                               {"reason": "invalid_password", "attempts": user.failed_login_attempts})
    
    def _handle_successful_login(self, user: User, ip_address: str, user_agent: str):
        """로그인 성공 처리"""
        user.last_login = datetime.now()
        user.failed_login_attempts = 0
        user.locked_until = None
        
        self._update_user_security_info(user)
        
        self._log_security_event(SecurityEvent.LOGIN_SUCCESS, user.id, ip_address, user_agent, {})
    
    def _update_user_security_info(self, user: User):
        """사용자 보안 정보 업데이트"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE users 
                SET last_login = ?, failed_login_attempts = ?, locked_until = ?
                WHERE id = ?
            """, (
                user.last_login.isoformat() if user.last_login else None,
                user.failed_login_attempts,
                user.locked_until.isoformat() if user.locked_until else None,
                user.id
            ))
    
    def _create_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """세션 생성"""
        session_id = secrets.token_urlsafe(32)
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions (session_id, user_id, created_at, last_activity, 
                                    ip_address, user_agent, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id, session.user_id, session.created_at.isoformat(),
                session.last_activity.isoformat(), session.ip_address, 
                session.user_agent, session.is_active
            ))
        
        return session_id
    
    def _log_security_event(self, event_type: SecurityEvent, user_id: Optional[str],
                          ip_address: str, user_agent: str, details: Dict[str, Any],
                          risk_score: int = 0):
        """보안 이벤트 로깅"""
        log_entry = SecurityLog(
            id=secrets.token_urlsafe(16),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.now(),
            details=details,
            risk_score=risk_score
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO security_logs (id, event_type, user_id, ip_address, 
                                         user_agent, timestamp, details, risk_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log_entry.id, log_entry.event_type.value, log_entry.user_id,
                log_entry.ip_address, log_entry.user_agent, log_entry.timestamp.isoformat(),
                json.dumps(log_entry.details), log_entry.risk_score
            ))

class AuthorizationManager:
    """권한 관리자"""
    
    def __init__(self, user_manager: UserManager):
        self.user_manager = user_manager
        self.role_permissions = {
            UserRole.ADMIN: ["read", "write", "execute", "admin"],
            UserRole.ANALYST: ["read", "write", "execute"],
            UserRole.USER: ["read", "write"],
            UserRole.VIEWER: ["read"]
        }
    
    def check_permission(self, user_id: str, required_permission: str) -> bool:
        """권한 확인"""
        user = self._get_user_from_cache(user_id)
        if not user:
            return False
        
        return required_permission in user.permissions
    
    def require_permission(self, required_permission: str):
        """권한 요구 데코레이터"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Streamlit 세션에서 사용자 정보 확인
                if 'user_id' not in st.session_state:
                    st.error("로그인이 필요합니다.")
                    st.stop()
                
                if not self.check_permission(st.session_state.user_id, required_permission):
                    st.error("권한이 부족합니다.")
                    
                    # 권한 거부 이벤트 로깅
                    self.user_manager._log_security_event(
                        SecurityEvent.PERMISSION_DENIED,
                        st.session_state.user_id,
                        st.session_state.get('ip_address', ''),
                        st.session_state.get('user_agent', ''),
                        {"required_permission": required_permission}
                    )
                    st.stop()
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def _get_user_from_cache(self, user_id: str) -> Optional[User]:
        """캐시된 사용자 정보 조회"""
        # 실제로는 Redis나 메모리 캐시 사용
        if 'current_user' in st.session_state and st.session_state.current_user.id == user_id:
            return st.session_state.current_user
        
        # DB에서 조회하여 캐시
        with sqlite3.connect(self.user_manager.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, username, email, password_hash, role, permissions, 
                       created_at, last_login, is_active, failed_login_attempts, locked_until
                FROM users WHERE id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            if row:
                user = User(
                    id=row[0], username=row[1], email=row[2], password_hash=row[3],
                    role=UserRole(row[4]), permissions=json.loads(row[5]),
                    created_at=datetime.fromisoformat(row[6]),
                    last_login=datetime.fromisoformat(row[7]) if row[7] else None,
                    is_active=bool(row[8]), failed_login_attempts=row[9],
                    locked_until=datetime.fromisoformat(row[10]) if row[10] else None
                )
                st.session_state.current_user = user
                return user
        
        return None

class SecurityAuditor:
    """보안 감사자"""
    
    def __init__(self, user_manager: UserManager):
        self.user_manager = user_manager
    
    def get_security_report(self, days: int = 7) -> Dict[str, Any]:
        """보안 리포트 생성"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        with sqlite3.connect(self.user_manager.db_path) as conn:
            # 로그인 시도 통계
            login_stats = conn.execute("""
                SELECT event_type, COUNT(*) as count
                FROM security_logs 
                WHERE timestamp BETWEEN ? AND ? 
                AND event_type IN ('login_success', 'login_failure')
                GROUP BY event_type
            """, (start_date.isoformat(), end_date.isoformat())).fetchall()
            
            # 위험 점수별 이벤트
            risk_events = conn.execute("""
                SELECT risk_score, COUNT(*) as count
                FROM security_logs 
                WHERE timestamp BETWEEN ? AND ?
                AND risk_score > 0
                GROUP BY risk_score
                ORDER BY risk_score DESC
            """, (start_date.isoformat(), end_date.isoformat())).fetchall()
            
            # 최근 보안 이벤트
            recent_events = conn.execute("""
                SELECT event_type, user_id, ip_address, timestamp, details, risk_score
                FROM security_logs 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
                LIMIT 50
            """, (start_date.isoformat(), end_date.isoformat())).fetchall()
            
            # 활성 세션 수
            active_sessions = conn.execute("""
                SELECT COUNT(*) 
                FROM sessions 
                WHERE is_active = 1
            """).fetchone()[0]
            
            # 잠긴 계정 수
            locked_accounts = conn.execute("""
                SELECT COUNT(*) 
                FROM users 
                WHERE locked_until > ?
            """, (datetime.now().isoformat(),)).fetchone()[0]
        
        return {
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "login_stats": dict(login_stats),
            "risk_events": dict(risk_events),
            "recent_events": [
                {
                    "event_type": event[0],
                    "user_id": event[1],
                    "ip_address": event[2],
                    "timestamp": event[3],
                    "details": json.loads(event[4]) if event[4] else {},
                    "risk_score": event[5]
                }
                for event in recent_events
            ],
            "active_sessions": active_sessions,
            "locked_accounts": locked_accounts
        }
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """이상 행위 탐지"""
        anomalies = []
        
        with sqlite3.connect(self.user_manager.db_path) as conn:
            # 비정상적인 로그인 시도
            suspicious_ips = conn.execute("""
                SELECT ip_address, COUNT(*) as failed_attempts
                FROM security_logs 
                WHERE event_type = 'login_failure' 
                AND timestamp > ?
                GROUP BY ip_address
                HAVING failed_attempts > 10
            """, ((datetime.now() - timedelta(hours=1)).isoformat(),)).fetchall()
            
            for ip, attempts in suspicious_ips:
                anomalies.append({
                    "type": "brute_force_attempt",
                    "description": f"IP {ip}에서 1시간 내 {attempts}회 로그인 실패",
                    "risk_score": min(attempts * 2, 100),
                    "ip_address": ip
                })
            
            # 비정상적인 시간대 접근
            night_logins = conn.execute("""
                SELECT user_id, COUNT(*) as night_logins
                FROM security_logs 
                WHERE event_type = 'login_success'
                AND time(timestamp) BETWEEN '22:00' AND '06:00'
                AND timestamp > ?
                GROUP BY user_id
                HAVING night_logins > 3
            """, ((datetime.now() - timedelta(days=7)).isoformat(),)).fetchall()
            
            for user_id, logins in night_logins:
                anomalies.append({
                    "type": "unusual_hours_access",
                    "description": f"사용자 {user_id}가 7일간 {logins}회 야간 로그인",
                    "risk_score": logins * 5,
                    "user_id": user_id
                })
        
        return sorted(anomalies, key=lambda x: x["risk_score"], reverse=True)

class DataProtectionManager:
    """데이터 보호 관리자"""
    
    def __init__(self):
        self.encryption_manager = EncryptionManager()
        self.sensitive_fields = [
            'password', 'ssn', 'credit_card', 'phone', 'email',
            'api_key', 'secret', 'token', 'private_key'
        ]
    
    def mask_sensitive_data(self, data: Union[Dict, str], field_name: str = "") -> Union[Dict, str]:
        """민감한 데이터 마스킹"""
        if isinstance(data, dict):
            masked_data = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in self.sensitive_fields):
                    if isinstance(value, str) and len(value) > 4:
                        masked_data[key] = value[:2] + "*" * (len(value) - 4) + value[-2:]
                    else:
                        masked_data[key] = "***"
                else:
                    masked_data[key] = self.mask_sensitive_data(value, key)
            return masked_data
        
        elif isinstance(data, str):
            if any(sensitive in field_name.lower() for sensitive in self.sensitive_fields):
                if len(data) > 4:
                    return data[:2] + "*" * (len(data) - 4) + data[-2:]
                else:
                    return "***"
        
        return data
    
    def encrypt_dataframe(self, df: pd.DataFrame, sensitive_columns: List[str]) -> pd.DataFrame:
        """데이터프레임 암호화"""
        encrypted_df = df.copy()
        
        for column in sensitive_columns:
            if column in encrypted_df.columns:
                encrypted_df[column] = encrypted_df[column].apply(
                    lambda x: self.encryption_manager.encrypt(str(x)) if pd.notna(x) else x
                )
        
        return encrypted_df
    
    def decrypt_dataframe(self, df: pd.DataFrame, sensitive_columns: List[str]) -> pd.DataFrame:
        """데이터프레임 복호화"""
        decrypted_df = df.copy()
        
        for column in sensitive_columns:
            if column in decrypted_df.columns:
                decrypted_df[column] = decrypted_df[column].apply(
                    lambda x: self.encryption_manager.decrypt(str(x)) if pd.notna(x) and isinstance(x, str) else x
                )
        
        return decrypted_df

class SecurityInterface:
    """보안 인터페이스"""
    
    def __init__(self):
        self.user_manager = UserManager()
        self.authorization_manager = AuthorizationManager(self.user_manager)
        self.security_auditor = SecurityAuditor(self.user_manager)
        self.data_protection = DataProtectionManager()
    
    def render_login_form(self) -> bool:
        """로그인 폼 렌더링"""
        st.markdown("### 🔐 로그인")
        
        with st.form("login_form"):
            username = st.text_input("사용자명")
            password = st.text_input("비밀번호", type="password")
            submit_button = st.form_submit_button("로그인")
            
            if submit_button:
                if username and password:
                    # IP 주소와 User Agent 가져오기 (Streamlit에서는 제한적)
                    ip_address = st.session_state.get('ip_address', '127.0.0.1')
                    user_agent = st.session_state.get('user_agent', 'Streamlit App')
                    
                    token = self.user_manager.authenticate_user(username, password, ip_address, user_agent)
                    
                    if token:
                        # JWT 토큰 검증
                        payload = self.user_manager.jwt_manager.validate_token(token)
                        if payload:
                            st.session_state.authenticated = True
                            st.session_state.user_id = payload['user_id']
                            st.session_state.user_role = payload['role']
                            st.session_state.user_permissions = payload['permissions']
                            st.session_state.jwt_token = token
                            
                            st.success("로그인 성공!")
                            st.experimental_rerun()
                            return True
                    else:
                        st.error("로그인 실패. 사용자명과 비밀번호를 확인하세요.")
                else:
                    st.error("사용자명과 비밀번호를 모두 입력하세요.")
        
        return False
    
    def render_security_dashboard(self):
        """보안 대시보드 렌더링"""
        if not self.authorization_manager.check_permission(st.session_state.user_id, "admin"):
            st.error("관리자 권한이 필요합니다.")
            return
        
        st.markdown("### 🛡️ 보안 대시보드")
        
        # 보안 리포트
        report = self.security_auditor.get_security_report()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            success_logins = report['login_stats'].get('login_success', 0)
            st.metric("성공한 로그인", success_logins)
        
        with col2:
            failed_logins = report['login_stats'].get('login_failure', 0)
            st.metric("실패한 로그인", failed_logins)
        
        with col3:
            st.metric("활성 세션", report['active_sessions'])
        
        with col4:
            st.metric("잠긴 계정", report['locked_accounts'])
        
        # 이상 행위 탐지
        st.markdown("#### 🚨 이상 행위 탐지")
        anomalies = self.security_auditor.detect_anomalies()
        
        if anomalies:
            for anomaly in anomalies[:5]:  # 상위 5개만 표시
                severity = "🔴" if anomaly['risk_score'] > 50 else "🟡" if anomaly['risk_score'] > 20 else "🟢"
                st.warning(f"{severity} {anomaly['description']} (위험도: {anomaly['risk_score']})")
        else:
            st.success("감지된 이상 행위가 없습니다.")
        
        # 최근 보안 이벤트
        st.markdown("#### 📋 최근 보안 이벤트")
        
        events_df = pd.DataFrame(report['recent_events'])
        if not events_df.empty:
            events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
            events_df = events_df.sort_values('timestamp', ascending=False)
            
            # 민감한 데이터 마스킹
            events_df_masked = events_df.copy()
            events_df_masked['details'] = events_df_masked['details'].apply(
                lambda x: self.data_protection.mask_sensitive_data(x)
            )
            
            st.dataframe(events_df_masked, use_container_width=True)
        else:
            st.info("최근 보안 이벤트가 없습니다.")
    
    def check_authentication(self) -> bool:
        """인증 상태 확인"""
        if not st.session_state.get('authenticated', False):
            return self.render_login_form()
        
        # JWT 토큰 유효성 확인
        token = st.session_state.get('jwt_token')
        if token:
            payload = self.user_manager.jwt_manager.validate_token(token)
            if not payload:
                # 토큰이 만료되었거나 유효하지 않음
                st.session_state.authenticated = False
                st.warning("세션이 만료되었습니다. 다시 로그인해주세요.")
                return self.render_login_form()
        
        return True
    
    def logout(self):
        """로그아웃"""
        if st.session_state.get('user_id'):
            # 로그아웃 이벤트 로깅
            self.user_manager._log_security_event(
                SecurityEvent.LOGOUT,
                st.session_state.user_id,
                st.session_state.get('ip_address', ''),
                st.session_state.get('user_agent', ''),
                {}
            )
        
        # 세션 정리
        for key in list(st.session_state.keys()):
            if key.startswith(('authenticated', 'user_', 'jwt_token')):
                del st.session_state[key]
        
        st.success("로그아웃되었습니다.")
        st.experimental_rerun()

# 전역 보안 관리자 인스턴스
security_interface = SecurityInterface()
user_manager = security_interface.user_manager
authorization_manager = security_interface.authorization_manager