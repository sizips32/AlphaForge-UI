"""
AutoML 엔진 및 자동 모델 선택 시스템
자동화된 머신러닝 파이프라인과 모델 최적화
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import json
import pickle
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 라이브러리
try:
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
    from sklearn.svm import SVR, SVC
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import SelectKBest, f_regression, f_classif
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("Scikit-learn이 설치되지 않았습니다. AutoML 기능이 제한됩니다.")

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class ProblemType(Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    TIME_SERIES = "time_series"

class ModelType(Enum):
    LINEAR = "linear"
    TREE_BASED = "tree_based"
    ENSEMBLE = "ensemble"
    BOOSTING = "boosting"
    NEURAL_NETWORK = "neural_network"

@dataclass
class ModelResult:
    model_name: str
    model_type: ModelType
    score: float
    cv_score: float
    cv_std: float
    training_time: float
    parameters: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
    model_object: Optional[Any] = None
    preprocessing_pipeline: Optional[Any] = None

@dataclass
class AutoMLConfig:
    problem_type: ProblemType
    target_column: str
    feature_columns: List[str]
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    max_time_minutes: int = 30
    n_jobs: int = -1
    scoring_metric: Optional[str] = None
    optimize_for: str = "accuracy"  # accuracy, speed, interpretability

class AutoMLEngine:
    """자동 머신러닝 엔진"""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.results: List[ModelResult] = []
        self.best_model: Optional[ModelResult] = None
        self.preprocessor = None
        self.is_trained = False
        
        # 사용 가능한 모델들 초기화
        self.available_models = self._initialize_models()
    
    def _initialize_models(self) -> Dict[str, Dict]:
        """사용 가능한 모델들 초기화"""
        models = {}
        
        if not SKLEARN_AVAILABLE:
            return models
        
        if self.config.problem_type == ProblemType.REGRESSION:
            models.update({
                'linear_regression': {
                    'model': LinearRegression,
                    'params': {},
                    'type': ModelType.LINEAR
                },
                'ridge_regression': {
                    'model': Ridge,
                    'params': {'alpha': [0.1, 1.0, 10.0, 100.0]},
                    'type': ModelType.LINEAR
                },
                'lasso_regression': {
                    'model': Lasso,
                    'params': {'alpha': [0.01, 0.1, 1.0, 10.0]},
                    'type': ModelType.LINEAR
                },
                'random_forest': {
                    'model': RandomForestRegressor,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, 20, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    'type': ModelType.ENSEMBLE
                },
                'gradient_boosting': {
                    'model': GradientBoostingRegressor,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    },
                    'type': ModelType.BOOSTING
                }
            })
            
            # XGBoost 추가
            if XGBOOST_AVAILABLE:
                models['xgboost'] = {
                    'model': XGBRegressor,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 0.9, 1.0]
                    },
                    'type': ModelType.BOOSTING
                }
            
            # LightGBM 추가
            if LIGHTGBM_AVAILABLE:
                models['lightgbm'] = {
                    'model': LGBMRegressor,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'num_leaves': [31, 50, 100]
                    },
                    'type': ModelType.BOOSTING
                }
        
        else:  # Classification
            models.update({
                'logistic_regression': {
                    'model': LogisticRegression,
                    'params': {
                        'C': [0.1, 1.0, 10.0, 100.0],
                        'solver': ['liblinear', 'lbfgs']
                    },
                    'type': ModelType.LINEAR
                },
                'random_forest': {
                    'model': RandomForestClassifier,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, 20, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    'type': ModelType.ENSEMBLE
                },
                'gradient_boosting': {
                    'model': GradientBoostingClassifier,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    },
                    'type': ModelType.BOOSTING
                }
            })
            
            # XGBoost 분류
            if XGBOOST_AVAILABLE:
                models['xgboost'] = {
                    'model': XGBClassifier,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 0.9, 1.0]
                    },
                    'type': ModelType.BOOSTING
                }
            
            # LightGBM 분류
            if LIGHTGBM_AVAILABLE:
                models['lightgbm'] = {
                    'model': LGBMClassifier,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'num_leaves': [31, 50, 100]
                    },
                    'type': ModelType.BOOSTING
                }
        
        return models
    
    def detect_problem_type(self, data: pd.DataFrame, target_column: str) -> ProblemType:
        """문제 유형 자동 감지"""
        target_series = data[target_column]
        
        # 수치형 데이터인지 확인
        if pd.api.types.is_numeric_dtype(target_series):
            unique_values = target_series.nunique()
            total_values = len(target_series)
            
            # 고유값의 비율이 낮으면 분류, 높으면 회귀
            unique_ratio = unique_values / total_values
            
            if unique_ratio < 0.05 or unique_values <= 10:
                if unique_values == 2:
                    return ProblemType.BINARY_CLASSIFICATION
                else:
                    return ProblemType.MULTICLASS_CLASSIFICATION
            else:
                return ProblemType.REGRESSION
        else:
            # 문자열이나 카테고리 데이터
            unique_values = target_series.nunique()
            if unique_values == 2:
                return ProblemType.BINARY_CLASSIFICATION
            else:
                return ProblemType.MULTICLASS_CLASSIFICATION
    
    def create_preprocessing_pipeline(self, data: pd.DataFrame) -> ColumnTransformer:
        """전처리 파이프라인 생성"""
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 타겟 컬럼 제거
        if self.config.target_column in numeric_features:
            numeric_features.remove(self.config.target_column)
        if self.config.target_column in categorical_features:
            categorical_features.remove(self.config.target_column)
        
        transformers = []
        
        # 수치형 데이터 전처리
        if numeric_features:
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_pipeline, numeric_features))
        
        # 카테고리형 데이터 전처리
        if categorical_features:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_pipeline, categorical_features))
        
        return ColumnTransformer(transformers=transformers, remainder='drop')
    
    def train_model(self, data: pd.DataFrame) -> List[ModelResult]:
        """모델 훈련"""
        if not SKLEARN_AVAILABLE:
            st.error("Scikit-learn이 필요합니다.")
            return []
        
        st.info("AutoML 훈련을 시작합니다...")
        
        # 데이터 준비
        X = data[self.config.feature_columns]
        y = data[self.config.target_column]
        
        # 라벨 인코딩 (분류 문제의 경우)
        label_encoder = None
        if self.config.problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        
        # 전처리 파이프라인 생성
        self.preprocessor = self.create_preprocessing_pipeline(X)
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # 전처리 적용
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # 스코어링 메트릭 설정
        if self.config.scoring_metric is None:
            if self.config.problem_type == ProblemType.REGRESSION:
                scoring = 'r2'
            else:
                scoring = 'accuracy'
        else:
            scoring = self.config.scoring_metric
        
        results = []
        
        # 각 모델 훈련
        progress_bar = st.progress(0)
        total_models = len(self.available_models)
        
        for i, (model_name, model_config) in enumerate(self.available_models.items()):
            try:
                start_time = datetime.now()
                
                st.write(f"훈련 중: {model_name}")
                
                # 모델 생성
                model = model_config['model']()
                
                # 하이퍼파라미터 튜닝
                if model_config['params']:
                    # 간단한 랜덤 서치 사용 (시간 제한을 위해)
                    search = RandomizedSearchCV(
                        model, 
                        model_config['params'],
                        n_iter=10,  # 제한된 반복
                        cv=min(self.config.cv_folds, 3),  # 더 적은 폴드
                        scoring=scoring,
                        random_state=self.config.random_state,
                        n_jobs=1  # 안정성을 위해 1로 설정
                    )
                    search.fit(X_train_processed, y_train)
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                else:
                    best_model = model
                    best_params = {}
                    best_model.fit(X_train_processed, y_train)
                
                # 교차 검증 점수
                cv_scores = cross_val_score(
                    best_model, X_train_processed, y_train, 
                    cv=min(self.config.cv_folds, 3), scoring=scoring
                )
                
                # 테스트 점수
                test_score = best_model.score(X_test_processed, y_test)
                
                # 특성 중요도 (가능한 경우)
                feature_importance = None
                if hasattr(best_model, 'feature_importances_'):
                    # 특성 이름 생성
                    feature_names = self.preprocessor.get_feature_names_out()
                    importance_dict = dict(zip(feature_names, best_model.feature_importances_))
                    # 상위 10개만 저장
                    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                    feature_importance = dict(sorted_importance[:10])
                
                # 훈련 시간 계산
                training_time = (datetime.now() - start_time).total_seconds()
                
                # 결과 저장
                result = ModelResult(
                    model_name=model_name,
                    model_type=model_config['type'],
                    score=test_score,
                    cv_score=cv_scores.mean(),
                    cv_std=cv_scores.std(),
                    training_time=training_time,
                    parameters=best_params,
                    feature_importance=feature_importance,
                    model_object=best_model
                )
                
                results.append(result)
                
                st.success(f"✅ {model_name}: {test_score:.4f}")
                
            except Exception as e:
                st.warning(f"⚠️ {model_name} 훈련 실패: {str(e)}")
            
            # 진행률 업데이트
            progress_bar.progress((i + 1) / total_models)
        
        # 결과 정렬 (점수 기준 내림차순)
        results.sort(key=lambda x: x.score, reverse=True)
        
        self.results = results
        if results:
            self.best_model = results[0]
            self.is_trained = True
        
        return results
    
    def get_model_comparison(self) -> pd.DataFrame:
        """모델 비교 테이블 생성"""
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        for result in self.results:
            comparison_data.append({
                'Model': result.model_name.replace('_', ' ').title(),
                'Type': result.model_type.value.replace('_', ' ').title(),
                'Test Score': f"{result.score:.4f}",
                'CV Score': f"{result.cv_score:.4f}",
                'CV Std': f"{result.cv_std:.4f}",
                'Training Time (s)': f"{result.training_time:.2f}",
                'Parameters': len(result.parameters)
            })
        
        return pd.DataFrame(comparison_data)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """예측 수행"""
        if not self.is_trained or not self.best_model:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 전처리 적용
        X_processed = self.preprocessor.transform(data[self.config.feature_columns])
        
        # 예측
        predictions = self.best_model.model_object.predict(X_processed)
        
        return predictions
    
    def save_model(self, filepath: str):
        """모델 저장"""
        if not self.is_trained:
            raise ValueError("훈련된 모델이 없습니다.")
        
        model_data = {
            'config': asdict(self.config),
            'best_model': self.best_model,
            'preprocessor': self.preprocessor,
            'results': self.results
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """모델 로드"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = AutoMLConfig(**model_data['config'])
        self.best_model = model_data['best_model']
        self.preprocessor = model_data['preprocessor']
        self.results = model_data['results']
        self.is_trained = True

class ModelExplainer:
    """모델 설명 시스템"""
    
    def __init__(self, automl_engine: AutoMLEngine):
        self.engine = automl_engine
    
    def explain_model(self, model_result: ModelResult) -> Dict[str, Any]:
        """모델 설명 생성"""
        explanation = {
            'model_name': model_result.model_name,
            'model_type': model_result.model_type.value,
            'performance_summary': self._create_performance_summary(model_result),
            'feature_importance': model_result.feature_importance,
            'model_characteristics': self._get_model_characteristics(model_result),
            'recommendations': self._get_recommendations(model_result)
        }
        
        return explanation
    
    def _create_performance_summary(self, model_result: ModelResult) -> str:
        """성능 요약 생성"""
        if self.engine.config.problem_type == ProblemType.REGRESSION:
            r2_percentage = model_result.score * 100
            summary = f"""
            이 모델은 테스트 데이터에서 {r2_percentage:.1f}%의 변동성을 설명합니다.
            교차 검증 점수는 {model_result.cv_score:.4f} (±{model_result.cv_std:.4f})입니다.
            """
        else:
            accuracy_percentage = model_result.score * 100
            summary = f"""
            이 모델의 테스트 정확도는 {accuracy_percentage:.1f}%입니다.
            교차 검증 정확도는 {model_result.cv_score:.4f} (±{model_result.cv_std:.4f})입니다.
            """
        
        return summary.strip()
    
    def _get_model_characteristics(self, model_result: ModelResult) -> Dict[str, str]:
        """모델 특성 설명"""
        characteristics = {
            ModelType.LINEAR: {
                'interpretability': '높음',
                'training_speed': '빠름',
                'prediction_speed': '매우 빠름',
                'overfitting_risk': '낮음',
                'data_requirement': '적음'
            },
            ModelType.TREE_BASED: {
                'interpretability': '높음',
                'training_speed': '빠름',
                'prediction_speed': '빠름',
                'overfitting_risk': '보통',
                'data_requirement': '보통'
            },
            ModelType.ENSEMBLE: {
                'interpretability': '보통',
                'training_speed': '보통',
                'prediction_speed': '보통',
                'overfitting_risk': '낮음',
                'data_requirement': '보통'
            },
            ModelType.BOOSTING: {
                'interpretability': '보통',
                'training_speed': '느림',
                'prediction_speed': '빠름',
                'overfitting_risk': '보통',
                'data_requirement': '많음'
            }
        }
        
        return characteristics.get(model_result.model_type, {})
    
    def _get_recommendations(self, model_result: ModelResult) -> List[str]:
        """개선 권장사항"""
        recommendations = []
        
        # 성능 기반 권장사항
        if model_result.score < 0.7:
            recommendations.append("모델 성능이 낮습니다. 더 많은 데이터나 특성 엔지니어링을 고려해보세요.")
        
        # CV 표준편차 기반 권장사항
        if model_result.cv_std > 0.1:
            recommendations.append("교차 검증 점수의 변동성이 큽니다. 모델의 안정성을 개선해보세요.")
        
        # 모델 타입별 권장사항
        if model_result.model_type == ModelType.LINEAR and model_result.score < 0.6:
            recommendations.append("선형 모델의 성능이 낮습니다. 비선형 모델을 시도해보세요.")
        
        if not recommendations:
            recommendations.append("모델 성능이 양호합니다. 추가 하이퍼파라미터 튜닝을 고려해보세요.")
        
        return recommendations

class AutoMLInterface:
    """AutoML 사용자 인터페이스"""
    
    def __init__(self):
        self.engine = None
        self.explainer = None
    
    def render_configuration(self) -> Optional[AutoMLConfig]:
        """설정 인터페이스 렌더링"""
        st.markdown("### 🤖 AutoML 설정")
        
        # 데이터 업로드 확인
        if 'data' not in st.session_state or st.session_state.data is None:
            st.warning("먼저 데이터를 업로드해주세요.")
            return None
        
        data = st.session_state.data
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 타겟 컬럼 선택
            target_column = st.selectbox(
                "타겟 컬럼 선택",
                options=data.columns.tolist(),
                help="예측하고자 하는 목표 변수를 선택하세요"
            )
            
            # 문제 유형 자동 감지
            if target_column:
                temp_engine = AutoMLEngine(AutoMLConfig(ProblemType.REGRESSION, target_column, []))
                detected_type = temp_engine.detect_problem_type(data, target_column)
                
                problem_type = st.selectbox(
                    "문제 유형",
                    options=[e.value for e in ProblemType],
                    index=[e.value for e in ProblemType].index(detected_type.value),
                    format_func=lambda x: x.replace('_', ' ').title(),
                    help="자동으로 감지된 문제 유형입니다"
                )
            
            # 특성 컬럼 선택
            available_features = [col for col in data.columns if col != target_column]
            feature_columns = st.multiselect(
                "특성 컬럼 선택",
                options=available_features,
                default=available_features[:10],  # 처음 10개만 기본 선택
                help="모델 학습에 사용할 특성들을 선택하세요"
            )
        
        with col2:
            # 고급 설정
            st.markdown("#### 고급 설정")
            
            test_size = st.slider(
                "테스트 데이터 비율",
                min_value=0.1, max_value=0.5, value=0.2, step=0.05,
                help="전체 데이터 중 테스트용으로 사용할 비율"
            )
            
            cv_folds = st.slider(
                "교차 검증 폴드 수",
                min_value=3, max_value=10, value=5,
                help="교차 검증에 사용할 폴드 수"
            )
            
            max_time = st.slider(
                "최대 훈련 시간 (분)",
                min_value=5, max_value=60, value=30,
                help="AutoML 훈련에 사용할 최대 시간"
            )
            
            optimize_for = st.selectbox(
                "최적화 목표",
                options=["accuracy", "speed", "interpretability"],
                help="모델 선택 시 우선순위를 둘 기준"
            )
        
        if not feature_columns:
            st.warning("최소 하나의 특성 컬럼을 선택해야 합니다.")
            return None
        
        # 설정 생성
        config = AutoMLConfig(
            problem_type=ProblemType(problem_type),
            target_column=target_column,
            feature_columns=feature_columns,
            test_size=test_size,
            cv_folds=cv_folds,
            max_time_minutes=max_time,
            optimize_for=optimize_for
        )
        
        return config
    
    def render_training(self, config: AutoMLConfig):
        """훈련 인터페이스 렌더링"""
        st.markdown("### 🚂 모델 훈련")
        
        if st.button("AutoML 훈련 시작", type="primary"):
            self.engine = AutoMLEngine(config)
            
            with st.spinner("모델들을 훈련하고 있습니다..."):
                results = self.engine.train_model(st.session_state.data)
            
            if results:
                st.success(f"✅ {len(results)}개의 모델이 성공적으로 훈련되었습니다!")
                st.session_state.automl_engine = self.engine
                st.session_state.automl_results = results
                
                # 최고 모델 정보 표시
                best_model = results[0]
                st.info(f"🏆 최고 성능 모델: {best_model.model_name} (점수: {best_model.score:.4f})")
            else:
                st.error("모델 훈련에 실패했습니다.")
    
    def render_results(self):
        """결과 인터페이스 렌더링"""
        if 'automl_engine' not in st.session_state or 'automl_results' not in st.session_state:
            st.info("먼저 AutoML 훈련을 완료해주세요.")
            return
        
        engine = st.session_state.automl_engine
        results = st.session_state.automl_results
        
        st.markdown("### 📊 모델 비교 결과")
        
        # 모델 비교 테이블
        comparison_df = engine.get_model_comparison()
        st.dataframe(comparison_df, use_container_width=True)
        
        # 모델 선택
        selected_model_name = st.selectbox(
            "자세히 볼 모델 선택",
            options=[result.model_name for result in results],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # 선택된 모델 상세 정보
        selected_result = next(r for r in results if r.model_name == selected_model_name)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"#### {selected_result.model_name.replace('_', ' ').title()} 상세 정보")
            
            # 모델 설명
            self.explainer = ModelExplainer(engine)
            explanation = self.explainer.explain_model(selected_result)
            
            st.write("**성능 요약:**")
            st.write(explanation['performance_summary'])
            
            st.write("**모델 특성:**")
            characteristics = explanation['model_characteristics']
            if characteristics:
                for key, value in characteristics.items():
                    st.write(f"- {key.replace('_', ' ').title()}: {value}")
            
            st.write("**권장사항:**")
            for recommendation in explanation['recommendations']:
                st.write(f"- {recommendation}")
        
        with col2:
            # 특성 중요도
            if selected_result.feature_importance:
                st.markdown("#### 특성 중요도")
                
                importance_df = pd.DataFrame(
                    list(selected_result.feature_importance.items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                
                import plotly.express as px
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 10 특성 중요도"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # 모델 저장
        st.markdown("#### 💾 모델 저장")
        if st.button("최고 성능 모델 저장"):
            try:
                filename = f"automl_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                engine.save_model(filename)
                st.success(f"모델이 '{filename}'로 저장되었습니다!")
            except Exception as e:
                st.error(f"모델 저장 실패: {str(e)}")

# 전역 인터페이스 인스턴스
automl_interface = AutoMLInterface()