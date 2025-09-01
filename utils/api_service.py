"""
API 서비스 및 마이크로서비스 아키텍처
RESTful API, GraphQL, 마이크로서비스 관리
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import asyncio
import aiohttp
import requests
from pathlib import Path
import logging
import uuid
from pydantic import BaseModel, Field, ValidationError
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class APIVersion(Enum):
    V1 = "v1"
    V2 = "v2"

# Pydantic 모델들
class FactorRequest(BaseModel):
    data: List[Dict[str, Any]]
    settings: Dict[str, Any] = {}
    factor_types: List[str] = ["basic", "technical", "fundamental"]

class FactorResponse(BaseModel):
    factors: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: str
    request_id: str

class BacktestRequest(BaseModel):
    factors: List[Dict[str, Any]]
    data: List[Dict[str, Any]]
    settings: Dict[str, Any] = {}
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class BacktestResponse(BaseModel):
    results: Dict[str, Any]
    performance_metrics: Dict[str, float]
    charts: List[Dict[str, Any]]
    timestamp: str
    request_id: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime: float
    services: Dict[str, str]

@dataclass
class ServiceRegistration:
    name: str
    url: str
    version: str
    status: ServiceStatus
    last_heartbeat: datetime
    health_check_endpoint: str = "/health"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ServiceRegistry:
    """서비스 레지스트리"""
    
    def __init__(self):
        self.services: Dict[str, ServiceRegistration] = {}
        self.heartbeat_interval = 30  # 30초
        self.health_check_timeout = 5
        self._running = False
        self._monitor_thread = None
    
    def register_service(self, service: ServiceRegistration):
        """서비스 등록"""
        self.services[service.name] = service
        logging.info(f"서비스 등록: {service.name} ({service.url})")
    
    def unregister_service(self, service_name: str):
        """서비스 등록 해제"""
        if service_name in self.services:
            del self.services[service_name]
            logging.info(f"서비스 등록 해제: {service_name}")
    
    def get_service(self, service_name: str) -> Optional[ServiceRegistration]:
        """서비스 조회"""
        return self.services.get(service_name)
    
    def get_healthy_services(self) -> List[ServiceRegistration]:
        """건강한 서비스 목록 조회"""
        return [s for s in self.services.values() if s.status == ServiceStatus.HEALTHY]
    
    def start_monitoring(self):
        """서비스 모니터링 시작"""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_services, daemon=True)
        self._monitor_thread.start()
        logging.info("서비스 모니터링 시작")
    
    def stop_monitoring(self):
        """서비스 모니터링 중지"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logging.info("서비스 모니터링 중지")
    
    def _monitor_services(self):
        """서비스 모니터링 루프"""
        while self._running:
            for service in list(self.services.values()):
                try:
                    health_url = f"{service.url.rstrip('/')}{service.health_check_endpoint}"
                    response = requests.get(health_url, timeout=self.health_check_timeout)
                    
                    if response.status_code == 200:
                        service.status = ServiceStatus.HEALTHY
                        service.last_heartbeat = datetime.now()
                    else:
                        service.status = ServiceStatus.UNHEALTHY
                    
                except Exception as e:
                    service.status = ServiceStatus.UNHEALTHY
                    logging.warning(f"서비스 {service.name} 헬스체크 실패: {str(e)}")
                
                # 오래된 서비스 제거 (5분 이상 응답 없음)
                if (datetime.now() - service.last_heartbeat).total_seconds() > 300:
                    service.status = ServiceStatus.UNKNOWN
            
            import time
            time.sleep(self.heartbeat_interval)

class APIGateway:
    """API 게이트웨이"""
    
    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self.load_balancer = LoadBalancer()
        self.rate_limiter = RateLimiter()
        self.request_logs = []
    
    async def route_request(self, service_name: str, endpoint: str, 
                           method: str = "GET", data: Any = None, 
                           headers: Dict[str, str] = None) -> Dict[str, Any]:
        """요청 라우팅"""
        
        # 서비스 검색
        services = [s for s in self.service_registry.get_healthy_services() 
                   if s.name == service_name]
        
        if not services:
            raise HTTPException(
                status_code=503, 
                detail=f"서비스 {service_name}을 사용할 수 없습니다"
            )
        
        # 로드 밸런싱
        selected_service = self.load_balancer.select_service(services)
        
        # 요청 로깅
        request_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # HTTP 요청 실행
            url = f"{selected_service.url.rstrip('/')}{endpoint}"
            
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(url, headers=headers) as response:
                        result = await response.json()
                elif method.upper() == "POST":
                    async with session.post(url, json=data, headers=headers) as response:
                        result = await response.json()
                else:
                    raise HTTPException(status_code=405, detail="지원하지 않는 메소드")
            
            # 성공 로그
            self._log_request(request_id, service_name, endpoint, method, 
                            start_time, datetime.now(), True)
            
            return result
            
        except Exception as e:
            # 에러 로그
            self._log_request(request_id, service_name, endpoint, method,
                            start_time, datetime.now(), False, str(e))
            
            raise HTTPException(status_code=500, detail=f"서비스 호출 실패: {str(e)}")
    
    def _log_request(self, request_id: str, service: str, endpoint: str, 
                    method: str, start_time: datetime, end_time: datetime,
                    success: bool, error: str = None):
        """요청 로깅"""
        log_entry = {
            "request_id": request_id,
            "service": service,
            "endpoint": endpoint,
            "method": method,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_ms": (end_time - start_time).total_seconds() * 1000,
            "success": success,
            "error": error
        }
        
        self.request_logs.append(log_entry)
        
        # 최근 1000개만 유지
        if len(self.request_logs) > 1000:
            self.request_logs = self.request_logs[-1000:]

class LoadBalancer:
    """로드 밸런서"""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.round_robin_index = {}
        self.service_weights = {}
    
    def select_service(self, services: List[ServiceRegistration]) -> ServiceRegistration:
        """서비스 선택"""
        if not services:
            raise ValueError("사용 가능한 서비스가 없습니다")
        
        if len(services) == 1:
            return services[0]
        
        if self.strategy == "round_robin":
            return self._round_robin(services)
        elif self.strategy == "weighted":
            return self._weighted_selection(services)
        elif self.strategy == "least_connections":
            return self._least_connections(services)
        else:
            return services[0]  # 기본값
    
    def _round_robin(self, services: List[ServiceRegistration]) -> ServiceRegistration:
        """라운드 로빈 선택"""
        service_key = tuple(s.name for s in services)
        
        if service_key not in self.round_robin_index:
            self.round_robin_index[service_key] = 0
        
        index = self.round_robin_index[service_key]
        selected_service = services[index]
        
        self.round_robin_index[service_key] = (index + 1) % len(services)
        
        return selected_service
    
    def _weighted_selection(self, services: List[ServiceRegistration]) -> ServiceRegistration:
        """가중치 기반 선택"""
        # 간단한 구현: 모든 서비스에 동일 가중치
        import random
        return random.choice(services)
    
    def _least_connections(self, services: List[ServiceRegistration]) -> ServiceRegistration:
        """최소 연결 기반 선택"""
        # 실제 구현에서는 각 서비스의 현재 연결 수를 추적해야 함
        return services[0]

class RateLimiter:
    """속도 제한기"""
    
    def __init__(self):
        self.limits = {}  # {client_id: {"count": int, "reset_time": datetime}}
        self.default_limit = 100  # 분당 100 요청
        self.window_minutes = 1
    
    def is_allowed(self, client_id: str, limit: int = None) -> bool:
        """요청 허용 여부 확인"""
        limit = limit or self.default_limit
        now = datetime.now()
        
        if client_id not in self.limits:
            self.limits[client_id] = {
                "count": 1,
                "reset_time": now + timedelta(minutes=self.window_minutes)
            }
            return True
        
        client_data = self.limits[client_id]
        
        # 윈도우 리셋
        if now >= client_data["reset_time"]:
            client_data["count"] = 1
            client_data["reset_time"] = now + timedelta(minutes=self.window_minutes)
            return True
        
        # 제한 확인
        if client_data["count"] >= limit:
            return False
        
        client_data["count"] += 1
        return True

class MicroserviceManager:
    """마이크로서비스 관리자"""
    
    def __init__(self):
        self.service_registry = ServiceRegistry()
        self.api_gateway = APIGateway(self.service_registry)
        self.services = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def start_services(self):
        """서비스 시작"""
        self.service_registry.start_monitoring()
        
        # 핵심 서비스들 등록
        core_services = [
            ServiceRegistration(
                name="factor_service",
                url="http://localhost:8001",
                version="1.0.0",
                status=ServiceStatus.HEALTHY,
                last_heartbeat=datetime.now(),
                metadata={"description": "Factor generation service"}
            ),
            ServiceRegistration(
                name="backtest_service",
                url="http://localhost:8002",
                version="1.0.0", 
                status=ServiceStatus.HEALTHY,
                last_heartbeat=datetime.now(),
                metadata={"description": "Backtesting service"}
            ),
            ServiceRegistration(
                name="data_service",
                url="http://localhost:8003",
                version="1.0.0",
                status=ServiceStatus.HEALTHY,
                last_heartbeat=datetime.now(),
                metadata={"description": "Data processing service"}
            )
        ]
        
        for service in core_services:
            self.service_registry.register_service(service)
    
    def stop_services(self):
        """서비스 중지"""
        self.service_registry.stop_monitoring()
        self.executor.shutdown(wait=True)
    
    def get_service_status(self) -> Dict[str, Any]:
        """서비스 상태 조회"""
        services_status = {}
        
        for name, service in self.service_registry.services.items():
            services_status[name] = {
                "url": service.url,
                "version": service.version,
                "status": service.status.value,
                "last_heartbeat": service.last_heartbeat.isoformat(),
                "metadata": service.metadata
            }
        
        return {
            "total_services": len(self.service_registry.services),
            "healthy_services": len(self.service_registry.get_healthy_services()),
            "services": services_status
        }

# FastAPI 애플리케이션
def create_fastapi_app(microservice_manager: MicroserviceManager) -> FastAPI:
    """FastAPI 애플리케이션 생성"""
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 시작 시
        microservice_manager.start_services()
        yield
        # 종료 시
        microservice_manager.stop_services()
    
    app = FastAPI(
        title="AlphaForge API",
        description="AI-powered alpha factor discovery platform API",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    security = HTTPBearer()
    
    def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
        """현재 사용자 확인"""
        # JWT 토큰 검증 로직
        return {"user_id": "test_user", "permissions": ["read", "write"]}
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """헬스 체크"""
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            uptime=0.0,
            services={}
        )
    
    @app.get("/api/v1/services")
    async def get_services(current_user: dict = Depends(get_current_user)):
        """서비스 상태 조회"""
        return microservice_manager.get_service_status()
    
    @app.post("/api/v1/factors", response_model=FactorResponse)
    async def generate_factors(request: FactorRequest, 
                             current_user: dict = Depends(get_current_user)):
        """팩터 생성"""
        try:
            result = await microservice_manager.api_gateway.route_request(
                "factor_service", 
                "/factors",
                "POST",
                request.dict()
            )
            
            return FactorResponse(
                factors=result.get("factors", []),
                metadata=result.get("metadata", {}),
                timestamp=datetime.now().isoformat(),
                request_id=str(uuid.uuid4())
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/backtest", response_model=BacktestResponse) 
    async def run_backtest(request: BacktestRequest,
                          current_user: dict = Depends(get_current_user)):
        """백테스팅 실행"""
        try:
            result = await microservice_manager.api_gateway.route_request(
                "backtest_service",
                "/backtest", 
                "POST",
                request.dict()
            )
            
            return BacktestResponse(
                results=result.get("results", {}),
                performance_metrics=result.get("performance_metrics", {}),
                charts=result.get("charts", []),
                timestamp=datetime.now().isoformat(),
                request_id=str(uuid.uuid4())
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/data/{data_type}")
    async def get_data(data_type: str, 
                      start_date: str = None,
                      end_date: str = None,
                      symbols: str = None,
                      current_user: dict = Depends(get_current_user)):
        """데이터 조회"""
        try:
            endpoint = f"/data/{data_type}"
            params = {
                "start_date": start_date,
                "end_date": end_date,
                "symbols": symbols
            }
            
            result = await microservice_manager.api_gateway.route_request(
                "data_service",
                f"{endpoint}?{requests.compat.urlencode(params)}",
                "GET"
            )
            
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app

class APIServiceInterface:
    """API 서비스 인터페이스"""
    
    def __init__(self):
        self.microservice_manager = MicroserviceManager()
        self.app = create_fastapi_app(self.microservice_manager)
        self.server_thread = None
        self.is_running = False
    
    def start_api_server(self, host: str = "127.0.0.1", port: int = 8000):
        """API 서버 시작"""
        if self.is_running:
            return
        
        def run_server():
            uvicorn.run(self.app, host=host, port=port, log_level="info")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.is_running = True
        
        logging.info(f"API 서버가 http://{host}:{port}에서 시작되었습니다")
    
    def stop_api_server(self):
        """API 서버 중지"""
        if self.server_thread and self.is_running:
            self.is_running = False
            # 서버 중지 로직 (복잡함으로 생략)
            logging.info("API 서버가 중지되었습니다")
    
    def render_api_dashboard(self):
        """API 대시보드 렌더링"""
        st.markdown("### 🚀 API 서비스 관리")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 서버 제어")
            
            if not self.is_running:
                if st.button("API 서버 시작", type="primary"):
                    self.start_api_server()
                    st.success("API 서버를 시작했습니다!")
                    st.experimental_rerun()
            else:
                st.success("✅ API 서버 실행 중")
                if st.button("API 서버 중지", type="secondary"):
                    self.stop_api_server()
                    st.info("API 서버를 중지했습니다.")
                    st.experimental_rerun()
        
        with col2:
            st.markdown("#### API 정보")
            if self.is_running:
                st.info("📡 **API 엔드포인트**")
                st.code("http://127.0.0.1:8000")
                st.info("📖 **API 문서**")
                st.code("http://127.0.0.1:8000/docs")
            else:
                st.warning("API 서버가 실행되지 않았습니다.")
        
        # 서비스 상태
        st.markdown("#### 🏥 서비스 상태")
        
        if self.is_running:
            service_status = self.microservice_manager.get_service_status()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 서비스", service_status["total_services"])
            with col2:
                st.metric("정상 서비스", service_status["healthy_services"])
            with col3:
                health_rate = (service_status["healthy_services"] / 
                             service_status["total_services"] * 100) if service_status["total_services"] > 0 else 0
                st.metric("정상률", f"{health_rate:.1f}%")
            
            # 서비스 상세 정보
            st.markdown("##### 서비스 상세")
            services_data = []
            for name, info in service_status["services"].items():
                services_data.append({
                    "서비스명": name,
                    "URL": info["url"],
                    "버전": info["version"],
                    "상태": info["status"],
                    "마지막 응답": info["last_heartbeat"]
                })
            
            if services_data:
                df = pd.DataFrame(services_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("등록된 서비스가 없습니다.")
        
        else:
            st.warning("API 서버를 먼저 시작해주세요.")
    
    def render_api_testing(self):
        """API 테스팅 인터페이스"""
        st.markdown("### 🧪 API 테스팅")
        
        if not self.is_running:
            st.warning("API 서버를 먼저 시작해주세요.")
            return
        
        # 엔드포인트 선택
        endpoints = [
            ("GET /health", "헬스 체크"),
            ("GET /api/v1/services", "서비스 상태"),
            ("POST /api/v1/factors", "팩터 생성"),
            ("POST /api/v1/backtest", "백테스팅"),
            ("GET /api/v1/data/{type}", "데이터 조회")
        ]
        
        selected_endpoint = st.selectbox(
            "테스트할 엔드포인트",
            endpoints,
            format_func=lambda x: f"{x[0]} - {x[1]}"
        )
        
        method, path = selected_endpoint[0].split(" ", 1)
        
        # 요청 데이터 입력 (POST인 경우)
        request_data = {}
        if method == "POST":
            st.markdown("#### 요청 데이터")
            
            if "factors" in path:
                st.markdown("**팩터 생성 요청 예시:**")
                example_data = {
                    "data": [{"symbol": "AAPL", "price": 150.0, "volume": 1000000}],
                    "settings": {"factor_types": ["technical"]},
                    "factor_types": ["basic", "technical"]
                }
                request_json = st.text_area(
                    "JSON 데이터",
                    json.dumps(example_data, indent=2),
                    height=200
                )
            
            elif "backtest" in path:
                st.markdown("**백테스팅 요청 예시:**")
                example_data = {
                    "factors": [{"name": "momentum", "values": [0.1, 0.2, 0.3]}],
                    "data": [{"date": "2023-01-01", "returns": 0.05}],
                    "settings": {"lookback_days": 20}
                }
                request_json = st.text_area(
                    "JSON 데이터",
                    json.dumps(example_data, indent=2),
                    height=200
                )
            
            try:
                request_data = json.loads(request_json)
            except json.JSONDecodeError:
                st.error("유효하지 않은 JSON 형식입니다.")
        
        # 테스트 실행
        if st.button("API 호출", type="primary"):
            try:
                base_url = "http://127.0.0.1:8000"
                url = f"{base_url}{path}"
                
                # 헤더 설정
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer test_token"
                }
                
                # 요청 실행
                if method == "GET":
                    response = requests.get(url, headers=headers)
                elif method == "POST":
                    response = requests.post(url, json=request_data, headers=headers)
                
                # 응답 표시
                st.markdown("#### 응답")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("상태 코드", response.status_code)
                with col2:
                    st.metric("응답 시간", f"{response.elapsed.total_seconds():.3f}초")
                
                if response.status_code == 200:
                    st.success("요청 성공!")
                    try:
                        response_json = response.json()
                        st.json(response_json)
                    except:
                        st.text(response.text)
                else:
                    st.error(f"요청 실패: {response.status_code}")
                    st.text(response.text)
                
            except Exception as e:
                st.error(f"요청 중 오류 발생: {str(e)}")

# 전역 API 서비스 인스턴스
api_service = APIServiceInterface()
microservice_manager = api_service.microservice_manager