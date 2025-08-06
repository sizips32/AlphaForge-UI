"""
데이터 처리 모듈
업로드된 데이터의 전처리 및 변환을 담당합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """데이터 처리 클래스"""
    
    def __init__(self):
        self.processed_data = None
        self.features = {}
    
    def process_data(self, data):
        """데이터 전처리를 수행합니다."""
        try:
            # 입력 데이터 검증
            if data is None or data.empty:
                raise ValueError("입력 데이터가 비어있습니다.")
            
            # 필수 컬럼 확인
            required_columns = ['Date', 'Ticker', 'Close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_columns}")
            
            # 데이터 복사본 생성
            processed = data.copy()
            
            # 1. 기본 정리
            processed = self._clean_data(processed)
            
            # 2. 날짜 처리
            processed = self._process_dates(processed)
            
            # 3. 가격 데이터 처리
            processed = self._process_prices(processed)
            
            # 4. 거래량 처리
            processed = self._process_volume(processed)
            
            # 5. 기술적 지표 계산
            processed = self._calculate_technical_indicators(processed)
            
            # 6. 팩터 계산
            processed = self._calculate_factors(processed)
            
            # 최종 검증
            if processed is None or processed.empty:
                raise ValueError("데이터 처리 후 결과가 비어있습니다.")
            
            self.processed_data = processed
            return processed
            
        except Exception as e:
            error_msg = f"데이터 처리 중 오류 발생: {str(e)}"
            print(f"ERROR: {error_msg}")  # 콘솔 로깅
            raise Exception(error_msg)
    
    def _clean_data(self, data):
        """데이터 기본 정리를 수행합니다."""
        try:
            # 복사본 생성
            cleaned = data.copy()
            
            # 컬럼명 정규화
            cleaned.columns = cleaned.columns.str.strip()
            
            # 공백 제거
            for col in cleaned.columns:
                if cleaned[col].dtype == 'object':
                    cleaned[col] = cleaned[col].str.strip()
            
            # 중복 제거
            cleaned = cleaned.drop_duplicates()
            
            # 정렬
            if 'Date' in cleaned.columns and 'Ticker' in cleaned.columns:
                cleaned = cleaned.sort_values(['Date', 'Ticker']).reset_index(drop=True)
            
            # 결측치 확인
            missing_counts = cleaned.isnull().sum()
            if missing_counts.sum() > 0:
                print(f"WARNING: 결측치 발견 - {dict(missing_counts)}")
            
            return cleaned
            
        except Exception as e:
            print(f"ERROR in _clean_data: {str(e)}")
            raise Exception(f"데이터 정리 중 오류: {str(e)}")
    
    def _process_dates(self, data):
        """날짜 데이터를 처리합니다."""
        try:
            if 'Date' in data.columns:
                # 날짜 형식 변환
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                
                # 변환 실패한 날짜 확인
                invalid_dates = data['Date'].isnull().sum()
                if invalid_dates > 0:
                    print(f"WARNING: {invalid_dates}개의 잘못된 날짜 형식 발견")
                    # 잘못된 날짜 행 제거
                    data = data.dropna(subset=['Date'])
                
                # 날짜 정렬
                data = data.sort_values(['Date', 'Ticker']).reset_index(drop=True)
                
                # 미래 날짜 제거
                current_date = pd.Timestamp.now()
                future_dates = data[data['Date'] > current_date]
                if len(future_dates) > 0:
                    print(f"WARNING: {len(future_dates)}개의 미래 날짜 제거")
                    data = data[data['Date'] <= current_date]
                
                # 주말 데이터 제거 (선택사항)
                weekend_data = data[data['Date'].dt.weekday >= 5]
                if len(weekend_data) > 0:
                    print(f"WARNING: {len(weekend_data)}개의 주말 데이터 제거")
                    data = data[data['Date'].dt.weekday < 5]
                
                # 날짜 인덱스 생성
                data['year'] = data['Date'].dt.year
                data['month'] = data['Date'].dt.month
                data['quarter'] = data['Date'].dt.quarter
                data['week'] = data['Date'].dt.isocalendar().week
                
                return data
            else:
                raise ValueError("Date 컬럼이 없습니다")
                
        except Exception as e:
            print(f"ERROR in _process_dates: {str(e)}")
            raise Exception(f"날짜 처리 중 오류: {str(e)}")
    
    def _process_prices(self, data):
        """가격 데이터를 처리합니다."""
        price_columns = ['Open', 'High', 'Low', 'Close']
        
        for col in price_columns:
            if col in data.columns:
                # 음수 값 처리
                data[col] = data[col].abs()
                
                # 0 값 처리 (전일 종가로 대체)
                if col != 'Open':
                    data[col] = data.groupby('Ticker')[col].ffill()
        
        # OHLC 관계 검증 및 수정
        if all(col in data.columns for col in price_columns):
            data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
            data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
        
        return data
    
    def _process_volume(self, data):
        """거래량 데이터를 처리합니다."""
        if 'Volume' in data.columns:
            # 음수 거래량 처리
            data['Volume'] = data['Volume'].abs()
            
            # 0 거래량 처리
            data['Volume'] = data.groupby('Ticker')['Volume'].ffill()
            
            # 거래량 이동평균
            data['Volume_MA5'] = data.groupby('Ticker')['Volume'].rolling(5).mean().reset_index(0, drop=True)
            data['Volume_MA20'] = data.groupby('Ticker')['Volume'].rolling(20).mean().reset_index(0, drop=True)
        
        return data
    
    def _calculate_technical_indicators(self, data):
        """기술적 지표를 계산합니다."""
        if 'Close' in data.columns:
            # 이동평균
            data['MA5'] = data.groupby('Ticker')['Close'].rolling(5).mean().reset_index(0, drop=True)
            data['MA10'] = data.groupby('Ticker')['Close'].rolling(10).mean().reset_index(0, drop=True)
            data['MA20'] = data.groupby('Ticker')['Close'].rolling(20).mean().reset_index(0, drop=True)
            data['MA50'] = data.groupby('Ticker')['Close'].rolling(50).mean().reset_index(0, drop=True)
            data['MA200'] = data.groupby('Ticker')['Close'].rolling(200).mean().reset_index(0, drop=True)
            
            # 수익률
            data['Returns'] = data.groupby('Ticker')['Close'].pct_change()
            data['Returns_5d'] = data.groupby('Ticker')['Close'].pct_change(5)
            data['Returns_20d'] = data.groupby('Ticker')['Close'].pct_change(20)
            
            # 변동성
            data['Volatility_20d'] = data.groupby('Ticker')['Returns'].rolling(20).std().reset_index(0, drop=True)
            data['Volatility_60d'] = data.groupby('Ticker')['Returns'].rolling(60).std().reset_index(0, drop=True)
            
            # RSI
            data['RSI'] = self._calculate_rsi(data)
            
            # MACD
            macd_data = self._calculate_macd(data)
            data['MACD'] = macd_data['MACD']
            data['MACD_Signal'] = macd_data['Signal']
            data['MACD_Histogram'] = macd_data['Histogram']
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(data)
            data['BB_Upper'] = bb_data['Upper']
            data['BB_Lower'] = bb_data['Lower']
            data['BB_Middle'] = bb_data['Middle']
            data['BB_Width'] = bb_data['Width']
            data['BB_Position'] = bb_data['Position']
        
        return data
    
    def _calculate_rsi(self, data, period=14):
        """RSI를 계산합니다."""
        rsi_values = []
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('Date')
            
            # 가격 변화
            delta = ticker_data['Close'].diff()
            
            # 상승/하락 분리
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # 평균 계산
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # RSI 계산 (0으로 나누기 방지)
            rs = avg_gain / (avg_loss + 1e-8)  # 작은 값 추가하여 0으로 나누기 방지
            rsi = 100 - (100 / (1 + rs))
            
            rsi_values.extend(rsi.values)
        
        return pd.Series(rsi_values, index=data.index)
    
    def _calculate_macd(self, data, fast=12, slow=26, signal=9):
        """MACD를 계산합니다."""
        macd_values = []
        signal_values = []
        histogram_values = []
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('Date')
            
            # EMA 계산
            ema_fast = ticker_data['Close'].ewm(span=fast).mean()
            ema_slow = ticker_data['Close'].ewm(span=slow).mean()
            
            # MACD 라인
            macd_line = ema_fast - ema_slow
            
            # 시그널 라인
            signal_line = macd_line.ewm(span=signal).mean()
            
            # 히스토그램
            histogram = macd_line - signal_line
            
            macd_values.extend(macd_line.values)
            signal_values.extend(signal_line.values)
            histogram_values.extend(histogram.values)
        
        return {
            'MACD': pd.Series(macd_values, index=data.index),
            'Signal': pd.Series(signal_values, index=data.index),
            'Histogram': pd.Series(histogram_values, index=data.index)
        }
    
    def _calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """볼린저 밴드를 계산합니다."""
        upper_values = []
        lower_values = []
        middle_values = []
        width_values = []
        position_values = []
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('Date')
            
            # 중간선 (이동평균)
            middle = ticker_data['Close'].rolling(window=period).mean()
            
            # 표준편차
            std = ticker_data['Close'].rolling(window=period).std()
            
            # 상단/하단 밴드
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            # 밴드 폭 (0으로 나누기 방지)
            width = (upper - lower) / (middle + 1e-8)
            
            # 밴드 내 위치 (0으로 나누기 방지)
            position = (ticker_data['Close'] - lower) / (upper - lower + 1e-8)
            
            upper_values.extend(upper.values)
            lower_values.extend(lower.values)
            middle_values.extend(middle.values)
            width_values.extend(width.values)
            position_values.extend(position.values)
        
        return {
            'Upper': pd.Series(upper_values, index=data.index),
            'Lower': pd.Series(lower_values, index=data.index),
            'Middle': pd.Series(middle_values, index=data.index),
            'Width': pd.Series(width_values, index=data.index),
            'Position': pd.Series(position_values, index=data.index)
        }
    
    def _calculate_factors(self, data):
        """기본 팩터들을 계산합니다."""
        # 모멘텀 팩터
        data['Momentum_1M'] = data.groupby('Ticker')['Close'].pct_change(20)
        data['Momentum_3M'] = data.groupby('Ticker')['Close'].pct_change(60)
        data['Momentum_6M'] = data.groupby('Ticker')['Close'].pct_change(120)
        
        # 밸류 팩터 (P/E 대신 가격 대비 이동평균 비율 사용)
        data['Value_MA20'] = data['Close'] / data['MA20']
        data['Value_MA50'] = data['Close'] / data['MA50']
        data['Value_MA200'] = data['Close'] / data['MA200']
        
        # 퀄리티 팩터 (변동성의 역수)
        data['Quality_LowVol'] = 1 / (data['Volatility_20d'] + 1e-8)
        data['Quality_Stability'] = 1 / (data['Volatility_60d'] + 1e-8)
        
        # 사이즈 팩터 (거래량 기반)
        if 'Volume' in data.columns:
            data['Size_Volume'] = data.groupby('Date')['Volume'].rank(pct=True)
        
        # 저변동성 팩터
        data['LowVolatility'] = -data['Volatility_20d']
        
        return data
    
    def get_factor_data(self, factor_name):
        """특정 팩터 데이터를 반환합니다."""
        if self.processed_data is None:
            return None
        
        factor_columns = [col for col in self.processed_data.columns if factor_name.lower() in col.lower()]
        
        if factor_columns:
            return self.processed_data[['Date', 'Ticker'] + factor_columns].copy()
        else:
            return None
    
    def get_all_factors(self):
        """모든 팩터 데이터를 반환합니다."""
        if self.processed_data is None:
            return None
        
        factor_columns = [
            'Momentum_1M', 'Momentum_3M', 'Momentum_6M',
            'Value_MA20', 'Value_MA50', 'Value_MA200',
            'Quality_LowVol', 'Quality_Stability',
            'Size_Volume', 'LowVolatility'
        ]
        
        available_factors = [col for col in factor_columns if col in self.processed_data.columns]
        
        if available_factors:
            return self.processed_data[['Date', 'Ticker'] + available_factors].copy()
        else:
            return None
    
    def get_summary_stats(self):
        """처리된 데이터의 요약 통계를 반환합니다."""
        if self.processed_data is None:
            return None
        
        summary = {
            'total_rows': len(self.processed_data),
            'unique_tickers': self.processed_data['Ticker'].nunique(),
            'date_range': f"{self.processed_data['Date'].min().strftime('%Y-%m-%d')} ~ {self.processed_data['Date'].max().strftime('%Y-%m-%d')}",
            'total_columns': len(self.processed_data.columns),
            'factor_columns': len([col for col in self.processed_data.columns if any(factor in col for factor in ['Momentum', 'Value', 'Quality', 'Size', 'LowVolatility'])])
        }
        
        return summary 
