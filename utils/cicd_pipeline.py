"""
CI/CD 파이프라인 관리 시스템
지속적 통합 및 배포 자동화
"""

import streamlit as st
import pandas as pd
import os
import json
import yaml
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

class PipelineStage(Enum):
    BUILD = "build"
    TEST = "test"
    LINT = "lint"
    SECURITY_SCAN = "security_scan"
    DEPLOY_STAGING = "deploy_staging"
    INTEGRATION_TEST = "integration_test"
    DEPLOY_PRODUCTION = "deploy_production"

class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

class TriggerType(Enum):
    PUSH = "push"
    PULL_REQUEST = "pull_request"
    MANUAL = "manual"
    SCHEDULED = "scheduled"

@dataclass
class StageResult:
    stage: PipelineStage
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    logs: List[str] = None
    artifacts: List[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.logs is None:
            self.logs = []
        if self.artifacts is None:
            self.artifacts = []

@dataclass
class PipelineRun:
    id: str
    trigger: TriggerType
    branch: str
    commit_hash: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    stages: List[StageResult] = None
    total_duration: Optional[float] = None
    triggered_by: Optional[str] = None
    
    def __post_init__(self):
        if self.stages is None:
            self.stages = []

@dataclass
class PipelineConfig:
    name: str
    stages: List[PipelineStage]
    parallel_stages: List[List[PipelineStage]] = None
    environment_variables: Dict[str, str] = None
    notifications: Dict[str, Any] = None
    artifact_retention_days: int = 30
    timeout_minutes: int = 60
    
    def __post_init__(self):
        if self.parallel_stages is None:
            self.parallel_stages = []
        if self.environment_variables is None:
            self.environment_variables = {}
        if self.notifications is None:
            self.notifications = {}

class TestRunner:
    """테스트 실행기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def run_unit_tests(self) -> tuple[bool, List[str]]:
        """유닛 테스트 실행"""
        logs = []
        try:
            # pytest 실행
            result = subprocess.run(
                ['python', '-m', 'pytest', 'tests/', '-v', '--tb=short'],
                capture_output=True,
                text=True,
                timeout=300  # 5분 타임아웃
            )
            
            logs.extend(result.stdout.split('\\n'))
            logs.extend(result.stderr.split('\\n'))
            
            return result.returncode == 0, logs
            
        except subprocess.TimeoutExpired:
            logs.append("테스트 타임아웃")
            return False, logs
        except Exception as e:
            logs.append(f"테스트 실행 예외: {str(e)}")
            return False, logs
    
    def run_integration_tests(self) -> tuple[bool, List[str]]:
        """통합 테스트 실행"""
        logs = []
        try:
            # 통합 테스트 실행
            result = subprocess.run(
                ['python', '-m', 'pytest', 'tests/integration/', '-v'],
                capture_output=True,
                text=True,
                timeout=600  # 10분 타임아웃
            )
            
            logs.extend(result.stdout.split('\\n'))
            logs.extend(result.stderr.split('\\n'))
            
            return result.returncode == 0, logs
            
        except Exception as e:
            logs.append(f"통합 테스트 실행 예외: {str(e)}")
            return False, logs
    
    def run_performance_tests(self) -> tuple[bool, List[str]]:
        """성능 테스트 실행"""
        logs = []
        try:
            # 성능 테스트 (예시)
            result = subprocess.run(
                ['python', '-m', 'pytest', 'tests/performance/', '-v', '--benchmark-only'],
                capture_output=True,
                text=True,
                timeout=1200  # 20분 타임아웃
            )
            
            logs.extend(result.stdout.split('\\n'))
            logs.extend(result.stderr.split('\\n'))
            
            return result.returncode == 0, logs
            
        except Exception as e:
            logs.append(f"성능 테스트 실행 예외: {str(e)}")
            return False, logs

class LintRunner:
    """코드 품질 검사기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def run_flake8(self) -> tuple[bool, List[str]]:
        """Flake8 린팅"""
        logs = []
        try:
            result = subprocess.run(
                ['flake8', '.', '--count', '--statistics', '--max-line-length=88'],
                capture_output=True,
                text=True
            )
            
            logs.extend(result.stdout.split('\\n'))
            logs.extend(result.stderr.split('\\n'))
            
            return result.returncode == 0, logs
            
        except Exception as e:
            logs.append(f"Flake8 실행 예외: {str(e)}")
            return False, logs
    
    def run_black(self) -> tuple[bool, List[str]]:
        """Black 코드 포매팅 검사"""
        logs = []
        try:
            result = subprocess.run(
                ['black', '--check', '--diff', '.'],
                capture_output=True,
                text=True
            )
            
            logs.extend(result.stdout.split('\\n'))
            logs.extend(result.stderr.split('\\n'))
            
            return result.returncode == 0, logs
            
        except Exception as e:
            logs.append(f"Black 실행 예외: {str(e)}")
            return False, logs
    
    def run_mypy(self) -> tuple[bool, List[str]]:
        """MyPy 타입 검사"""
        logs = []
        try:
            result = subprocess.run(
                ['mypy', '.', '--ignore-missing-imports'],
                capture_output=True,
                text=True
            )
            
            logs.extend(result.stdout.split('\\n'))
            logs.extend(result.stderr.split('\\n'))
            
            return result.returncode == 0, logs
            
        except Exception as e:
            logs.append(f"MyPy 실행 예외: {str(e)}")
            return False, logs

class SecurityScanner:
    """보안 검사기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def run_bandit(self) -> tuple[bool, List[str]]:
        """Bandit 보안 검사"""
        logs = []
        try:
            result = subprocess.run(
                ['bandit', '-r', '.', '-f', 'json'],
                capture_output=True,
                text=True
            )
            
            logs.append(f"Bandit 종료 코드: {result.returncode}")
            if result.stdout:
                logs.extend(result.stdout.split('\\n'))
            if result.stderr:
                logs.extend(result.stderr.split('\\n'))
            
            # Bandit은 이슈가 발견되면 1을 반환하므로, 실제 오류와 구분
            return result.returncode in [0, 1], logs
            
        except Exception as e:
            logs.append(f"Bandit 실행 예외: {str(e)}")
            return False, logs
    
    def run_safety(self) -> tuple[bool, List[str]]:
        """Safety 의존성 취약점 검사"""
        logs = []
        try:
            result = subprocess.run(
                ['safety', 'check', '--json'],
                capture_output=True,
                text=True
            )
            
            logs.append(f"Safety 종료 코드: {result.returncode}")
            if result.stdout:
                logs.extend(result.stdout.split('\\n'))
            if result.stderr:
                logs.extend(result.stderr.split('\\n'))
            
            return result.returncode == 0, logs
            
        except Exception as e:
            logs.append(f"Safety 실행 예외: {str(e)}")
            return False, logs

class ArtifactManager:
    """아티팩트 관리자"""
    
    def __init__(self, base_path: str = ".artifacts"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def store_artifact(self, pipeline_id: str, stage: PipelineStage, 
                      artifact_name: str, content: Union[str, bytes]) -> str:
        """아티팩트 저장"""
        stage_dir = self.base_path / pipeline_id / stage.value
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        artifact_path = stage_dir / artifact_name
        
        if isinstance(content, str):
            with open(artifact_path, 'w') as f:
                f.write(content)
        else:
            with open(artifact_path, 'wb') as f:
                f.write(content)
        
        return str(artifact_path)
    
    def get_artifact_path(self, pipeline_id: str, stage: PipelineStage, 
                         artifact_name: str) -> Optional[Path]:
        """아티팩트 경로 조회"""
        artifact_path = self.base_path / pipeline_id / stage.value / artifact_name
        return artifact_path if artifact_path.exists() else None
    
    def cleanup_old_artifacts(self, retention_days: int = 30):
        """오래된 아티팩트 정리"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        for pipeline_dir in self.base_path.iterdir():
            if pipeline_dir.is_dir():
                # 디렉토리 생성 시간 확인
                create_time = datetime.fromtimestamp(pipeline_dir.stat().st_ctime)
                if create_time < cutoff_date:
                    import shutil
                    shutil.rmtree(pipeline_dir)

class PipelineExecutor:
    """파이프라인 실행기"""
    
    def __init__(self):
        self.test_runner = TestRunner()
        self.lint_runner = LintRunner()
        self.security_scanner = SecurityScanner()
        self.artifact_manager = ArtifactManager()
        self.logger = logging.getLogger(__name__)
    
    def execute_stage(self, stage: PipelineStage, pipeline_id: str) -> StageResult:
        """개별 스테이지 실행"""
        result = StageResult(
            stage=stage,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            success = False
            logs = []
            
            if stage == PipelineStage.BUILD:
                success, logs = self._run_build_stage()
            elif stage == PipelineStage.TEST:
                success, logs = self.test_runner.run_unit_tests()
            elif stage == PipelineStage.LINT:
                success, logs = self._run_lint_stage()
            elif stage == PipelineStage.SECURITY_SCAN:
                success, logs = self._run_security_stage()
            elif stage == PipelineStage.INTEGRATION_TEST:
                success, logs = self.test_runner.run_integration_tests()
            elif stage == PipelineStage.DEPLOY_STAGING:
                success, logs = self._run_deploy_staging()
            elif stage == PipelineStage.DEPLOY_PRODUCTION:
                success, logs = self._run_deploy_production()
            else:
                logs = [f"알 수 없는 스테이지: {stage.value}"]
                success = False
            
            result.status = PipelineStatus.SUCCESS if success else PipelineStatus.FAILED
            result.logs = logs
            
            # 로그를 아티팩트로 저장
            log_content = "\\n".join(logs)
            artifact_path = self.artifact_manager.store_artifact(
                pipeline_id, stage, f"{stage.value}_logs.txt", log_content
            )
            result.artifacts.append(artifact_path)
            
        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error_message = str(e)
            result.logs = [f"스테이지 실행 예외: {str(e)}"]
        
        result.end_time = datetime.now()
        result.duration = (result.end_time - result.start_time).total_seconds()
        
        return result
    
    def _run_build_stage(self) -> tuple[bool, List[str]]:
        """빌드 스테이지 실행"""
        logs = []
        try:
            # 의존성 설치
            logs.append("의존성 설치 중...")
            result = subprocess.run(
                ['pip', 'install', '-r', 'requirements.txt'],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            logs.extend(result.stdout.split('\\n'))
            if result.stderr:
                logs.extend(result.stderr.split('\\n'))
            
            if result.returncode != 0:
                return False, logs
            
            # Python 바이트코드 컴파일
            logs.append("Python 코드 컴파일 중...")
            result = subprocess.run(
                ['python', '-m', 'compileall', '.'],
                capture_output=True,
                text=True
            )
            
            logs.extend(result.stdout.split('\\n'))
            
            return result.returncode == 0, logs
            
        except Exception as e:
            logs.append(f"빌드 예외: {str(e)}")
            return False, logs
    
    def _run_lint_stage(self) -> tuple[bool, List[str]]:
        """린트 스테이지 실행"""
        all_logs = []
        all_success = True
        
        # Flake8 실행
        success, logs = self.lint_runner.run_flake8()
        all_logs.extend(["=== Flake8 Results ==="] + logs)
        all_success &= success
        
        # Black 실행
        success, logs = self.lint_runner.run_black()
        all_logs.extend(["=== Black Results ==="] + logs)
        all_success &= success
        
        # MyPy 실행 (선택사항 - 실패해도 전체 실패하지 않음)
        success, logs = self.lint_runner.run_mypy()
        all_logs.extend(["=== MyPy Results ==="] + logs)
        if not success:
            all_logs.append("MyPy 검사에서 이슈가 발견되었지만 진행합니다.")
        
        return all_success, all_logs
    
    def _run_security_stage(self) -> tuple[bool, List[str]]:
        """보안 검사 스테이지 실행"""
        all_logs = []
        all_success = True
        
        # Bandit 실행
        success, logs = self.security_scanner.run_bandit()
        all_logs.extend(["=== Bandit Results ==="] + logs)
        all_success &= success
        
        # Safety 실행
        success, logs = self.security_scanner.run_safety()
        all_logs.extend(["=== Safety Results ==="] + logs)
        all_success &= success
        
        return all_success, all_logs
    
    def _run_deploy_staging(self) -> tuple[bool, List[str]]:
        """스테이징 배포"""
        logs = ["스테이징 환경 배포 시뮬레이션"]
        # 실제로는 deployment_manager를 사용하여 스테이징 환경에 배포
        logs.append("스테이징 배포 완료")
        return True, logs
    
    def _run_deploy_production(self) -> tuple[bool, List[str]]:
        """프로덕션 배포"""
        logs = ["프로덕션 환경 배포 시뮬레이션"]
        # 실제로는 deployment_manager를 사용하여 프로덕션 환경에 배포
        logs.append("프로덕션 배포 완료")
        return True, logs

class CICDPipeline:
    """CI/CD 파이프라인 메인 클래스"""
    
    def __init__(self):
        self.executor = PipelineExecutor()
        self.runs: List[PipelineRun] = []
        self._load_pipeline_history()
    
    def _load_pipeline_history(self):
        """파이프라인 실행 기록 로드"""
        history_file = Path(".pipeline_history.json")
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                for run_data in history_data:
                    # 날짜 변환
                    run_data['start_time'] = datetime.fromisoformat(run_data['start_time'])
                    if run_data['end_time']:
                        run_data['end_time'] = datetime.fromisoformat(run_data['end_time'])
                    
                    # Enum 변환
                    run_data['trigger'] = TriggerType(run_data['trigger'])
                    run_data['status'] = PipelineStatus(run_data['status'])
                    
                    # 스테이지 결과 변환
                    stages = []
                    for stage_data in run_data.get('stages', []):
                        stage_data['stage'] = PipelineStage(stage_data['stage'])
                        stage_data['status'] = PipelineStatus(stage_data['status'])
                        stage_data['start_time'] = datetime.fromisoformat(stage_data['start_time'])
                        if stage_data['end_time']:
                            stage_data['end_time'] = datetime.fromisoformat(stage_data['end_time'])
                        
                        stages.append(StageResult(**stage_data))
                    
                    run_data['stages'] = stages
                    self.runs.append(PipelineRun(**run_data))
                    
            except Exception as e:
                logging.error(f"파이프라인 기록 로드 실패: {str(e)}")
    
    def _save_pipeline_history(self):
        """파이프라인 실행 기록 저장"""
        try:
            history_data = []
            for run in self.runs:
                stages_data = []
                for stage in run.stages:
                    stage_dict = {
                        'stage': stage.stage.value,
                        'status': stage.status.value,
                        'start_time': stage.start_time.isoformat(),
                        'end_time': stage.end_time.isoformat() if stage.end_time else None,
                        'duration': stage.duration,
                        'logs': stage.logs,
                        'artifacts': stage.artifacts,
                        'error_message': stage.error_message
                    }
                    stages_data.append(stage_dict)
                
                run_dict = {
                    'id': run.id,
                    'trigger': run.trigger.value,
                    'branch': run.branch,
                    'commit_hash': run.commit_hash,
                    'status': run.status.value,
                    'start_time': run.start_time.isoformat(),
                    'end_time': run.end_time.isoformat() if run.end_time else None,
                    'stages': stages_data,
                    'total_duration': run.total_duration,
                    'triggered_by': run.triggered_by
                }
                history_data.append(run_dict)
            
            with open(".pipeline_history.json", 'w') as f:
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            logging.error(f"파이프라인 기록 저장 실패: {str(e)}")
    
    def create_default_config(self) -> PipelineConfig:
        """기본 파이프라인 설정 생성"""
        return PipelineConfig(
            name="alphaforge-ci-cd",
            stages=[
                PipelineStage.BUILD,
                PipelineStage.LINT,
                PipelineStage.TEST,
                PipelineStage.SECURITY_SCAN,
                PipelineStage.DEPLOY_STAGING,
                PipelineStage.INTEGRATION_TEST,
                PipelineStage.DEPLOY_PRODUCTION
            ],
            parallel_stages=[
                [PipelineStage.LINT, PipelineStage.SECURITY_SCAN]  # 병렬 실행 가능
            ],
            environment_variables={
                "PYTHONPATH": ".",
                "ENV": "ci",
                "PYTEST_TIMEOUT": "300"
            },
            notifications={
                "slack": {
                    "webhook_url": "",
                    "channel": "#ci-cd",
                    "on_success": True,
                    "on_failure": True
                },
                "email": {
                    "recipients": [],
                    "on_success": False,
                    "on_failure": True
                }
            }
        )
    
    def trigger_pipeline(self, config: PipelineConfig, trigger: TriggerType,
                        branch: str = "main", commit_hash: str = "HEAD",
                        triggered_by: str = "system") -> str:
        """파이프라인 트리거"""
        
        # 파이프라인 실행 ID 생성
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(f'{branch}{commit_hash}'.encode()).hexdigest()[:8]}"
        
        # 파이프라인 실행 객체 생성
        pipeline_run = PipelineRun(
            id=run_id,
            trigger=trigger,
            branch=branch,
            commit_hash=commit_hash,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now(),
            triggered_by=triggered_by
        )
        
        self.runs.append(pipeline_run)
        
        # 비동기로 파이프라인 실행
        self._execute_pipeline(pipeline_run, config)
        
        return run_id
    
    def _execute_pipeline(self, pipeline_run: PipelineRun, config: PipelineConfig):
        """파이프라인 실행"""
        try:
            # 순차적으로 스테이지 실행
            for stage in config.stages:
                if pipeline_run.status == PipelineStatus.FAILED:
                    # 이전 스테이지가 실패하면 나머지 스킵
                    remaining_stages = config.stages[config.stages.index(stage):]
                    for remaining_stage in remaining_stages:
                        skipped_result = StageResult(
                            stage=remaining_stage,
                            status=PipelineStatus.SKIPPED,
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            duration=0,
                            logs=["이전 스테이지 실패로 인해 스킵됨"]
                        )
                        pipeline_run.stages.append(skipped_result)
                    break
                
                # 스테이지 실행
                stage_result = self.executor.execute_stage(stage, pipeline_run.id)
                pipeline_run.stages.append(stage_result)
                
                # 스테이지 실패 시 파이프라인 실패 처리
                if stage_result.status == PipelineStatus.FAILED:
                    pipeline_run.status = PipelineStatus.FAILED
                    break
            
            # 모든 스테이지가 성공하면 성공 처리
            if pipeline_run.status == PipelineStatus.RUNNING:
                pipeline_run.status = PipelineStatus.SUCCESS
            
            # 완료 시간 및 총 소요 시간 설정
            pipeline_run.end_time = datetime.now()
            pipeline_run.total_duration = (pipeline_run.end_time - pipeline_run.start_time).total_seconds()
            
        except Exception as e:
            pipeline_run.status = PipelineStatus.FAILED
            pipeline_run.end_time = datetime.now()
            pipeline_run.total_duration = (pipeline_run.end_time - pipeline_run.start_time).total_seconds()
            
            # 오류 로그 추가
            error_stage = StageResult(
                stage=PipelineStage.BUILD,  # 기본값
                status=PipelineStatus.FAILED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                logs=[f"파이프라인 실행 예외: {str(e)}"],
                error_message=str(e)
            )
            pipeline_run.stages.append(error_stage)
        
        finally:
            # 기록 저장
            self._save_pipeline_history()
    
    def get_pipeline_run(self, run_id: str) -> Optional[PipelineRun]:
        """파이프라인 실행 조회"""
        for run in self.runs:
            if run.id == run_id:
                return run
        return None
    
    def get_latest_runs(self, limit: int = 10) -> List[PipelineRun]:
        """최근 파이프라인 실행 조회"""
        sorted_runs = sorted(self.runs, key=lambda x: x.start_time, reverse=True)
        return sorted_runs[:limit]
    
    def cancel_pipeline(self, run_id: str) -> bool:
        """파이프라인 취소"""
        pipeline_run = self.get_pipeline_run(run_id)
        if pipeline_run and pipeline_run.status == PipelineStatus.RUNNING:
            pipeline_run.status = PipelineStatus.CANCELLED
            pipeline_run.end_time = datetime.now()
            pipeline_run.total_duration = (pipeline_run.end_time - pipeline_run.start_time).total_seconds()
            self._save_pipeline_history()
            return True
        return False
    
    def generate_github_workflow(self, config: PipelineConfig) -> str:
        """GitHub Actions 워크플로우 생성"""
        workflow_yaml = f"""
name: {config.name}

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    timeout-minutes: {config.timeout_minutes}
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{{{ runner.os }}}}-pip-${{{{ hashFiles('**/requirements.txt') }}}}
        restore-keys: |
          ${{{{ runner.os }}}}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest flake8 black mypy bandit safety
    
    - name: Run linting
      run: |
        flake8 . --count --statistics --max-line-length=88
        black --check --diff .
        mypy . --ignore-missing-imports
    
    - name: Run security checks
      run: |
        bandit -r . -f json
        safety check --json
    
    - name: Run tests
      run: |
        pytest tests/ -v --tb=short --cov=./ --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        
    - name: Build Docker image
      if: success() && github.ref == 'refs/heads/main'
      run: |
        docker build -t alphaforge-ui:${{{{ github.sha }}}} .
    
    - name: Deploy to staging
      if: success() && github.ref == 'refs/heads/main'
      run: |
        echo "Deploy to staging environment"
        # 실제 배포 스크립트 호출
        
    - name: Run integration tests
      if: success() && github.ref == 'refs/heads/main'
      run: |
        pytest tests/integration/ -v
        
    - name: Deploy to production
      if: success() && github.ref == 'refs/heads/main'
      run: |
        echo "Deploy to production environment"
        # 실제 배포 스크립트 호출
        
    - name: Notify on failure
      if: failure()
      run: |
        echo "Pipeline failed - send notification"
        # 실제 알림 스크립트 호출
"""
        
        return workflow_yaml.strip()

# 전역 CI/CD 파이프라인 인스턴스
cicd_pipeline = CICDPipeline()