"""
프로덕션 배포 관리 시스템
Docker, 컨테이너화, 클라우드 배포 관리
"""

import streamlit as st
import pandas as pd
import os
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import shutil
import tempfile
import logging

class DeploymentEnvironment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class DeploymentPlatform(Enum):
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    HEROKU = "heroku"
    STREAMLIT_CLOUD = "streamlit_cloud"

class DeploymentStatus(Enum):
    PENDING = "pending"
    BUILDING = "building"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLBACK = "rollback"

@dataclass
class DeploymentConfig:
    app_name: str
    version: str
    environment: DeploymentEnvironment
    platform: DeploymentPlatform
    dockerfile_path: Optional[str] = None
    requirements_path: Optional[str] = "requirements.txt"
    port: int = 8501
    memory_limit: str = "512Mi"
    cpu_limit: str = "500m"
    replicas: int = 1
    environment_variables: Optional[Dict[str, str]] = None
    health_check_endpoint: str = "/health"
    
    def __post_init__(self):
        if self.environment_variables is None:
            self.environment_variables = {}

@dataclass
class DeploymentRecord:
    id: str
    config: DeploymentConfig
    status: DeploymentStatus
    created_at: datetime
    deployed_at: Optional[datetime] = None
    url: Optional[str] = None
    logs: List[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.logs is None:
            self.logs = []

class DockerBuilder:
    """Docker 이미지 빌더"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_dockerfile(self, config: DeploymentConfig) -> str:
        """Dockerfile 생성"""
        dockerfile_content = f"""
# AlphaForge-UI Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 시스템 dependencies 설치
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    software-properties-common \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Python dependencies 복사 및 설치
COPY {config.requirements_path} .
RUN pip install --no-cache-dir -r {config.requirements_path}

# 애플리케이션 코드 복사
COPY . .

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{config.port}{config.health_check_endpoint} || exit 1

# 포트 노출
EXPOSE {config.port}

# Streamlit 설정
RUN mkdir -p ~/.streamlit/
RUN echo "[server]\\n\\
headless = true\\n\\
port = {config.port}\\n\\
enableCORS = false\\n\\
enableXsrfProtection = false\\n" > ~/.streamlit/config.toml

# 환경변수 설정
"""
        
        # 환경변수 추가
        for key, value in config.environment_variables.items():
            dockerfile_content += f"ENV {key}={value}\\n"
        
        dockerfile_content += f"""
# 실행 명령
CMD ["streamlit", "run", "app.py", "--server.port={config.port}", "--server.address=0.0.0.0"]
"""
        
        return dockerfile_content.strip()
    
    def generate_dockercompose(self, config: DeploymentConfig) -> str:
        """docker-compose.yml 생성"""
        compose_content = f"""
version: '3.8'

services:
  alphaforge-ui:
    build:
      context: .
      dockerfile: {config.dockerfile_path or 'Dockerfile'}
    ports:
      - "{config.port}:{config.port}"
    environment:
"""
        
        # 환경변수 추가
        for key, value in config.environment_variables.items():
            compose_content += f"      - {key}={value}\\n"
        
        compose_content += f"""
    volumes:
      - ./data:/app/data:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{config.port}{config.health_check_endpoint}"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 10s
    deploy:
      resources:
        limits:
          memory: {config.memory_limit}
          cpus: '{config.cpu_limit}'
        reservations:
          memory: 256Mi
          cpus: '0.25'

  # Redis for caching (optional)
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
"""
        
        return compose_content.strip()
    
    def build_image(self, config: DeploymentConfig, build_context: str) -> bool:
        """Docker 이미지 빌드"""
        try:
            image_tag = f"{config.app_name}:{config.version}"
            
            # Dockerfile 생성
            dockerfile_path = os.path.join(build_context, 'Dockerfile')
            with open(dockerfile_path, 'w') as f:
                f.write(self.generate_dockerfile(config))
            
            # Docker 빌드 실행
            build_command = [
                'docker', 'build',
                '-t', image_tag,
                '-f', dockerfile_path,
                build_context
            ]
            
            result = subprocess.run(build_command, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Docker 이미지 빌드 성공: {image_tag}")
                return True
            else:
                self.logger.error(f"Docker 빌드 실패: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Docker 빌드 예외: {str(e)}")
            return False

class KubernetesDeployer:
    """Kubernetes 배포 관리자"""
    
    def generate_deployment_yaml(self, config: DeploymentConfig) -> str:
        """Kubernetes Deployment YAML 생성"""
        deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {config.app_name}
  labels:
    app: {config.app_name}
    version: {config.version}
    environment: {config.environment.value}
spec:
  replicas: {config.replicas}
  selector:
    matchLabels:
      app: {config.app_name}
  template:
    metadata:
      labels:
        app: {config.app_name}
        version: {config.version}
    spec:
      containers:
      - name: {config.app_name}
        image: {config.app_name}:{config.version}
        ports:
        - containerPort: {config.port}
          name: http
        env:
"""
        
        # 환경변수 추가
        for key, value in config.environment_variables.items():
            deployment_yaml += f"        - name: {key}\\n          value: \"{value}\"\\n"
        
        deployment_yaml += f"""
        resources:
          limits:
            memory: {config.memory_limit}
            cpu: {config.cpu_limit}
          requests:
            memory: 256Mi
            cpu: 250m
        livenessProbe:
          httpGet:
            path: {config.health_check_endpoint}
            port: {config.port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: {config.health_check_endpoint}
            port: {config.port}
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: {config.app_name}-service
  labels:
    app: {config.app_name}
spec:
  selector:
    app: {config.app_name}
  ports:
  - port: 80
    targetPort: {config.port}
    name: http
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {config.app_name}-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - {config.app_name}.example.com
    secretName: {config.app_name}-tls
  rules:
  - host: {config.app_name}.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {config.app_name}-service
            port:
              number: 80
"""
        
        return deployment_yaml.strip()
    
    def deploy_to_kubernetes(self, config: DeploymentConfig) -> bool:
        """Kubernetes에 배포"""
        try:
            # YAML 파일 생성
            yaml_content = self.generate_deployment_yaml(config)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(yaml_content)
                yaml_file = f.name
            
            # kubectl apply 실행
            result = subprocess.run(
                ['kubectl', 'apply', '-f', yaml_file],
                capture_output=True,
                text=True
            )
            
            # 임시 파일 삭제
            os.unlink(yaml_file)
            
            if result.returncode == 0:
                return True
            else:
                raise Exception(f"kubectl 실패: {result.stderr}")
                
        except Exception as e:
            logging.error(f"Kubernetes 배포 실패: {str(e)}")
            return False

class CloudDeployer:
    """클라우드 배포 관리자"""
    
    def generate_heroku_procfile(self, config: DeploymentConfig) -> str:
        """Heroku Procfile 생성"""
        return f"web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"
    
    def generate_streamlit_secrets(self, config: DeploymentConfig) -> str:
        """Streamlit Cloud secrets.toml 생성"""
        secrets_content = ""
        for key, value in config.environment_variables.items():
            secrets_content += f'{key} = "{value}"\\n'
        
        return secrets_content
    
    def generate_requirements_txt(self) -> str:
        """requirements.txt 생성"""
        # 현재 프로젝트의 의존성을 기반으로 생성
        requirements = [
            "streamlit>=1.28.0",
            "pandas>=1.5.0",
            "numpy>=1.24.0",
            "plotly>=5.0.0",
            "scikit-learn>=1.3.0",
            "xgboost>=1.7.0",
            "lightgbm>=3.3.0",
            "redis>=4.0.0",
            "celery>=5.3.0",
            "dask>=2023.5.0",
            "asyncio-mqtt>=0.13.0",
            "websockets>=11.0.0",
            "fuzzywuzzy>=0.18.0",
            "python-levenshtein>=0.21.0",
            "networkx>=3.1.0",
            "scipy>=1.10.0",
            "PyYAML>=6.0.0",
            "python-dotenv>=1.0.0"
        ]
        
        return "\\n".join(requirements)

class DeploymentManager:
    """배포 관리자 메인 클래스"""
    
    def __init__(self):
        self.docker_builder = DockerBuilder()
        self.k8s_deployer = KubernetesDeployer()
        self.cloud_deployer = CloudDeployer()
        self.deployment_history: List[DeploymentRecord] = []
        
        # 배포 기록 로드
        self._load_deployment_history()
    
    def _load_deployment_history(self):
        """배포 기록 로드"""
        history_file = Path(".deployment_history.json")
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                    
                # JSON 데이터를 DeploymentRecord 객체로 변환
                for record_data in history_data:
                    record_data['created_at'] = datetime.fromisoformat(record_data['created_at'])
                    if record_data['deployed_at']:
                        record_data['deployed_at'] = datetime.fromisoformat(record_data['deployed_at'])
                    
                    # config 복원
                    config_data = record_data['config']
                    config_data['environment'] = DeploymentEnvironment(config_data['environment'])
                    config_data['platform'] = DeploymentPlatform(config_data['platform'])
                    record_data['config'] = DeploymentConfig(**config_data)
                    
                    record_data['status'] = DeploymentStatus(record_data['status'])
                    
                    self.deployment_history.append(DeploymentRecord(**record_data))
                    
            except Exception as e:
                st.warning(f"배포 기록 로드 실패: {str(e)}")
    
    def _save_deployment_history(self):
        """배포 기록 저장"""
        try:
            history_data = []
            for record in self.deployment_history:
                record_dict = {
                    'id': record.id,
                    'config': {
                        **asdict(record.config),
                        'environment': record.config.environment.value,
                        'platform': record.config.platform.value
                    },
                    'status': record.status.value,
                    'created_at': record.created_at.isoformat(),
                    'deployed_at': record.deployed_at.isoformat() if record.deployed_at else None,
                    'url': record.url,
                    'logs': record.logs,
                    'error_message': record.error_message
                }
                history_data.append(record_dict)
            
            with open(".deployment_history.json", 'w') as f:
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            logging.error(f"배포 기록 저장 실패: {str(e)}")
    
    def create_deployment(self, config: DeploymentConfig) -> str:
        """새로운 배포 생성"""
        deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        record = DeploymentRecord(
            id=deployment_id,
            config=config,
            status=DeploymentStatus.PENDING,
            created_at=datetime.now()
        )
        
        self.deployment_history.append(record)
        self._save_deployment_history()
        
        return deployment_id
    
    def deploy(self, deployment_id: str) -> bool:
        """배포 실행"""
        record = self._get_deployment_record(deployment_id)
        if not record:
            return False
        
        try:
            record.status = DeploymentStatus.BUILDING
            self._update_deployment_record(record)
            
            # 배포 파일 생성
            self._prepare_deployment_files(record.config)
            
            # 플랫폼별 배포 실행
            success = False
            
            if record.config.platform == DeploymentPlatform.DOCKER:
                success = self._deploy_docker(record)
            elif record.config.platform == DeploymentPlatform.KUBERNETES:
                success = self._deploy_kubernetes(record)
            elif record.config.platform == DeploymentPlatform.HEROKU:
                success = self._deploy_heroku(record)
            elif record.config.platform == DeploymentPlatform.STREAMLIT_CLOUD:
                success = self._deploy_streamlit_cloud(record)
            
            if success:
                record.status = DeploymentStatus.DEPLOYED
                record.deployed_at = datetime.now()
                record.url = self._generate_deployment_url(record.config)
            else:
                record.status = DeploymentStatus.FAILED
                record.error_message = "배포 실패"
            
            self._update_deployment_record(record)
            return success
            
        except Exception as e:
            record.status = DeploymentStatus.FAILED
            record.error_message = str(e)
            self._update_deployment_record(record)
            return False
    
    def _prepare_deployment_files(self, config: DeploymentConfig):
        """배포 파일들 준비"""
        # Dockerfile 생성
        if config.platform in [DeploymentPlatform.DOCKER, DeploymentPlatform.KUBERNETES]:
            dockerfile_content = self.docker_builder.generate_dockerfile(config)
            with open('Dockerfile', 'w') as f:
                f.write(dockerfile_content)
            
            # docker-compose.yml 생성
            compose_content = self.docker_builder.generate_dockercompose(config)
            with open('docker-compose.yml', 'w') as f:
                f.write(compose_content)
        
        # Heroku Procfile
        if config.platform == DeploymentPlatform.HEROKU:
            procfile_content = self.cloud_deployer.generate_heroku_procfile(config)
            with open('Procfile', 'w') as f:
                f.write(procfile_content)
        
        # requirements.txt 생성 (없는 경우)
        if not os.path.exists(config.requirements_path):
            requirements_content = self.cloud_deployer.generate_requirements_txt()
            with open(config.requirements_path, 'w') as f:
                f.write(requirements_content)
        
        # 헬스체크 엔드포인트 파일 생성
        self._create_health_check_endpoint(config)
    
    def _create_health_check_endpoint(self, config: DeploymentConfig):
        """헬스체크 엔드포인트 생성"""
        health_check_code = '''
import streamlit as st
from datetime import datetime

def health_check():
    """헬스체크 엔드포인트"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# 헬스체크 페이지 추가
if st.sidebar.button("Health Check"):
    st.json(health_check())
'''
        
        # 기존 app.py에 헬스체크 코드 추가 (간단한 방식)
        health_file = Path("utils/health_check.py")
        with open(health_file, 'w') as f:
            f.write(health_check_code)
    
    def _deploy_docker(self, record: DeploymentRecord) -> bool:
        """Docker 배포"""
        try:
            # Docker 이미지 빌드
            if not self.docker_builder.build_image(record.config, '.'):
                return False
            
            record.logs.append(f"Docker 이미지 빌드 완료: {record.config.app_name}:{record.config.version}")
            
            # Docker Compose로 실행
            result = subprocess.run(['docker-compose', 'up', '-d'], capture_output=True, text=True)
            
            if result.returncode == 0:
                record.logs.append("Docker Compose 배포 성공")
                return True
            else:
                record.logs.append(f"Docker Compose 실패: {result.stderr}")
                return False
                
        except Exception as e:
            record.logs.append(f"Docker 배포 예외: {str(e)}")
            return False
    
    def _deploy_kubernetes(self, record: DeploymentRecord) -> bool:
        """Kubernetes 배포"""
        try:
            # Docker 이미지 빌드
            if not self.docker_builder.build_image(record.config, '.'):
                return False
            
            # Kubernetes 배포
            if not self.k8s_deployer.deploy_to_kubernetes(record.config):
                return False
            
            record.logs.append("Kubernetes 배포 완료")
            return True
            
        except Exception as e:
            record.logs.append(f"Kubernetes 배포 예외: {str(e)}")
            return False
    
    def _deploy_heroku(self, record: DeploymentRecord) -> bool:
        """Heroku 배포"""
        record.logs.append("Heroku 배포를 위해서는 Git repository와 Heroku CLI 설정이 필요합니다.")
        return True
    
    def _deploy_streamlit_cloud(self, record: DeploymentRecord) -> bool:
        """Streamlit Cloud 배포"""
        # secrets.toml 생성
        secrets_content = self.cloud_deployer.generate_streamlit_secrets(record.config)
        
        secrets_dir = Path(".streamlit")
        secrets_dir.mkdir(exist_ok=True)
        
        with open(secrets_dir / "secrets.toml", 'w') as f:
            f.write(secrets_content)
        
        record.logs.append("Streamlit Cloud 배포 파일이 준비되었습니다. GitHub repository에 push하여 배포를 완료하세요.")
        return True
    
    def _generate_deployment_url(self, config: DeploymentConfig) -> str:
        """배포 URL 생성"""
        if config.platform == DeploymentPlatform.LOCAL:
            return f"http://localhost:{config.port}"
        elif config.platform == DeploymentPlatform.DOCKER:
            return f"http://localhost:{config.port}"
        elif config.platform == DeploymentPlatform.HEROKU:
            return f"https://{config.app_name}.herokuapp.com"
        elif config.platform == DeploymentPlatform.STREAMLIT_CLOUD:
            return f"https://{config.app_name}.streamlit.app"
        else:
            return f"https://{config.app_name}.example.com"
    
    def _get_deployment_record(self, deployment_id: str) -> Optional[DeploymentRecord]:
        """배포 기록 조회"""
        for record in self.deployment_history:
            if record.id == deployment_id:
                return record
        return None
    
    def _update_deployment_record(self, record: DeploymentRecord):
        """배포 기록 업데이트"""
        for i, existing_record in enumerate(self.deployment_history):
            if existing_record.id == record.id:
                self.deployment_history[i] = record
                break
        
        self._save_deployment_history()
    
    def rollback_deployment(self, deployment_id: str) -> bool:
        """배포 롤백"""
        record = self._get_deployment_record(deployment_id)
        if not record:
            return False
        
        try:
            record.status = DeploymentStatus.ROLLBACK
            
            if record.config.platform == DeploymentPlatform.DOCKER:
                # Docker Compose 중지
                result = subprocess.run(['docker-compose', 'down'], capture_output=True, text=True)
                success = result.returncode == 0
            elif record.config.platform == DeploymentPlatform.KUBERNETES:
                # Kubernetes 리소스 삭제
                result = subprocess.run(['kubectl', 'delete', '-f', 'k8s-deployment.yaml'], capture_output=True, text=True)
                success = result.returncode == 0
            else:
                success = True
            
            if success:
                record.logs.append("롤백 완료")
            else:
                record.logs.append("롤백 실패")
            
            self._update_deployment_record(record)
            return success
            
        except Exception as e:
            record.logs.append(f"롤백 예외: {str(e)}")
            self._update_deployment_record(record)
            return False
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentStatus]:
        """배포 상태 조회"""
        record = self._get_deployment_record(deployment_id)
        return record.status if record else None
    
    def list_deployments(self) -> List[DeploymentRecord]:
        """배포 목록 조회"""
        return sorted(self.deployment_history, key=lambda x: x.created_at, reverse=True)

# 전역 배포 관리자 인스턴스
deployment_manager = DeploymentManager()