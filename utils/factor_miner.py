"""
팩터 마이닝 모듈
AlphaForge 생성-예측 신경망을 사용한 알파 팩터 자동 발굴 기능을 제공합니다.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class FactorMiner:
    """팩터 마이닝 클래스"""
    
    def __init__(self, settings):
        self.settings = settings
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def generate_basic_factors(self, data):
        """기본 팩터들을 생성합니다."""
        try:
            factors = []
            
            # 데이터 검증
            if data is None or data.empty:
                raise ValueError("입력 데이터가 비어있습니다")
            
            if 'Close' not in data.columns:
                raise ValueError("Close 컬럼이 없습니다")
            
            # 모멘텀 팩터
            if 'Momentum' in self.settings['factor_types']:
                momentum_factors = self._generate_momentum_factors(data)
                factors.extend(momentum_factors)
            
            # 밸류 팩터
            if 'Value' in self.settings['factor_types']:
                value_factors = self._generate_value_factors(data)
                factors.extend(value_factors)
            
            # 퀄리티 팩터
            if 'Quality' in self.settings['factor_types']:
                quality_factors = self._generate_quality_factors(data)
                factors.extend(quality_factors)
            
            # 사이즈 팩터
            if 'Size' in self.settings['factor_types']:
                size_factors = self._generate_size_factors(data)
                factors.extend(size_factors)
            
            # 저변동성 팩터
            if 'Low Volatility' in self.settings['factor_types']:
                low_vol_factors = self._generate_low_volatility_factors(data)
                factors.extend(low_vol_factors)
            
            # 결과 검증
            if not factors:
                raise ValueError("생성된 팩터가 없습니다")
            
            return factors[:self.settings['factor_pool_size']]
            
        except Exception as e:
            print(f"ERROR in generate_basic_factors: {str(e)}")
            raise Exception(f"기본 팩터 생성 실패: {str(e)}")
    
    def generate_ai_factors(self, data, basic_factors):
        """AI 기반 팩터를 생성합니다."""
        try:
            # 입력 검증
            if not basic_factors:
                raise ValueError("기본 팩터가 없습니다")
            
            # 데이터 준비
            X, y = self._prepare_training_data(data, basic_factors)
            
            # 데이터 검증
            if X is None or len(X) == 0:
                raise ValueError("훈련 데이터 준비 실패")
            
            # 입력 크기 계산
            input_size = X.shape[1] if len(X.shape) > 1 else 1
            
            # 신경망 모델 생성
            model = self._create_neural_network(input_size)
            
            # 모델 학습
            trained_model = self._train_model(model, X, y)
            
            # 새로운 팩터 생성
            ai_factors = self._generate_ai_factors_from_model(trained_model, data, basic_factors)
            
            # 결과 검증
            if not ai_factors:
                raise ValueError("AI 팩터 생성 실패")
            
            return ai_factors
            
        except Exception as e:
            print(f"ERROR in generate_ai_factors: {str(e)}")
            raise Exception(f"AI 팩터 생성 실패: {str(e)}")
    
    def _generate_momentum_factors(self, data):
        """모멘텀 팩터를 생성합니다."""
        factors = []
        
        # 가격 모멘텀
        for period in [5, 10, 20, 60]:
            factor_name = f"Momentum_{period}d"
            factor_values = data.groupby('Ticker')['Close'].pct_change(period)
            factors.append({
                'name': factor_name,
                'values': factor_values,
                'type': 'momentum',
                'formula': f'pct_change(Close, {period})'
            })
        
        # 거래량 모멘텀
        if 'Volume' in data.columns:
            for period in [5, 10, 20]:
                factor_name = f"Volume_Momentum_{period}d"
                factor_values = data.groupby('Ticker')['Volume'].pct_change(period)
                factors.append({
                    'name': factor_name,
                    'values': factor_values,
                    'type': 'momentum',
                    'formula': f'pct_change(Volume, {period})'
                })
        
        return factors
    
    def _generate_value_factors(self, data):
        """밸류 팩터를 생성합니다."""
        factors = []
        
        # 이동평균 대비 가격
        for ma_period in [20, 50, 200]:
            if f'MA{ma_period}' in data.columns:
                factor_name = f"Value_MA{ma_period}_Ratio"
                factor_values = data['Close'] / data[f'MA{ma_period}']
                factors.append({
                    'name': factor_name,
                    'values': factor_values,
                    'type': 'value',
                    'formula': f'Close / MA{ma_period}'
                })
        
        # 볼린저 밴드 위치
        if 'BB_Position' in data.columns:
            factor_name = "Value_BB_Position"
            factor_values = data['BB_Position']
            factors.append({
                'name': factor_name,
                'values': factor_values,
                'type': 'value',
                'formula': 'BB_Position'
            })
        
        return factors
    
    def _generate_quality_factors(self, data):
        """퀄리티 팩터를 생성합니다."""
        factors = []
        
        # 변동성 기반 퀄리티
        if 'Volatility_20d' in data.columns:
            factor_name = "Quality_Low_Volatility"
            factor_values = 1 / (data['Volatility_20d'] + 1e-8)
            factors.append({
                'name': factor_name,
                'values': factor_values,
                'type': 'quality',
                'formula': '1 / Volatility_20d'
            })
        
        # RSI 기반 퀄리티
        if 'RSI' in data.columns:
            factor_name = "Quality_RSI_Stability"
            factor_values = 1 / (abs(data['RSI'] - 50) + 1e-8)
            factors.append({
                'name': factor_name,
                'values': factor_values,
                'type': 'quality',
                'formula': '1 / abs(RSI - 50)'
            })
        
        return factors
    
    def _generate_size_factors(self, data):
        """사이즈 팩터를 생성합니다."""
        factors = []
        
        # 거래량 기반 사이즈
        if 'Volume' in data.columns:
            factor_name = "Size_Volume_Rank"
            factor_values = data.groupby('Date')['Volume'].rank(pct=True)
            factors.append({
                'name': factor_name,
                'values': factor_values,
                'type': 'size',
                'formula': 'rank(Volume) / count(Volume)'
            })
        
        # 시가총액 추정 (거래량 * 가격)
        if 'Volume' in data.columns:
            factor_name = "Size_Market_Cap_Est"
            factor_values = data['Volume'] * data['Close']
            factors.append({
                'name': factor_name,
                'values': factor_values,
                'type': 'size',
                'formula': 'Volume * Close'
            })
        
        return factors
    
    def _generate_low_volatility_factors(self, data):
        """저변동성 팩터를 생성합니다."""
        factors = []
        
        # 변동성 역수
        if 'Volatility_20d' in data.columns:
            factor_name = "Low_Volatility_20d"
            factor_values = -data['Volatility_20d']
            factors.append({
                'name': factor_name,
                'values': factor_values,
                'type': 'low_volatility',
                'formula': '-Volatility_20d'
            })
        
        # 볼린저 밴드 폭
        if 'BB_Width' in data.columns:
            factor_name = "Low_Volatility_BB_Width"
            factor_values = -data['BB_Width']
            factors.append({
                'name': factor_name,
                'values': factor_values,
                'type': 'low_volatility',
                'formula': '-BB_Width'
            })
        
        return factors
    
    def _create_neural_network(self, input_size):
        """신경망 모델을 생성합니다."""
        hidden_size = self.settings['neurons_per_layer']
        num_layers = self.settings['hidden_layers']
        
        layers = []
        
        # 입력층
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.settings['dropout_rate']))
        
        # 은닉층
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.settings['dropout_rate']))
        
        # 출력층
        layers.append(nn.Linear(hidden_size, 1))
        
        model = nn.Sequential(*layers)
        return model.to(self.device)
    
    def _prepare_training_data(self, data, basic_factors):
        """학습 데이터를 준비합니다."""
        # 기본 특성 선택
        feature_columns = ['Returns', 'Volatility_20d', 'RSI', 'MACD', 'BB_Position']
        available_features = [col for col in feature_columns if col in data.columns]
        
        # 특성 데이터
        X = data[available_features].fillna(0)
        
        # 타겟 데이터 (다음 기 수익률)
        y = data.groupby('Ticker')['Returns'].shift(-1).fillna(0)
        
        # 데이터 정규화
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y.values
    
    def _train_model(self, model, X, y):
        """모델을 학습합니다."""
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 텐서 변환
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # 손실 함수와 옵티마이저
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.settings['learning_rate'])
        
        # 학습
        for epoch in range(self.settings['epochs']):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(X_train_tensor).squeeze()
            loss = criterion(outputs, y_train_tensor)
            
            loss.backward()
            optimizer.step()
        
        return model
    
    def _generate_ai_factors_from_model(self, model, data, basic_factors):
        """학습된 모델로부터 AI 팩터를 생성합니다."""
        ai_factors = []
        
        # 기본 특성 선택
        feature_columns = ['Returns', 'Volatility_20d', 'RSI', 'MACD', 'BB_Position']
        available_features = [col for col in feature_columns if col in data.columns]
        
        # 특성 데이터
        X = data[available_features].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # 모델 예측
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            predictions = model(X_tensor).cpu().numpy().flatten()
        
        # AI 팩터 생성
        for i in range(min(5, self.settings['factor_pool_size'] - len(basic_factors))):
            factor_name = f"AI_Factor_{i+1}"
            
            # 예측값을 기반으로 한 복합 팩터
            if i == 0:
                factor_values = predictions
            elif i == 1:
                factor_values = predictions * data['Returns'].fillna(0)
            elif i == 2:
                factor_values = predictions * (1 / (data['Volatility_20d'].fillna(1) + 1e-8))
            elif i == 3:
                factor_values = predictions * data['RSI'].fillna(50) / 100
            else:
                factor_values = predictions * data['BB_Position'].fillna(0.5)
            
            ai_factors.append({
                'name': factor_name,
                'values': pd.Series(factor_values, index=data.index),
                'type': 'ai_generated',
                'formula': f'AI_Model_Output_{i+1}'
            })
        
        return basic_factors + ai_factors 
