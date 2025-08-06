"""
데이터 검증 모듈
업로드된 데이터의 품질과 형식을 검증합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from utils.config import DATA_VALIDATION

class DataValidator:
    """데이터 검증 클래스"""
    
    def __init__(self):
        self.required_columns = DATA_VALIDATION['required_columns']
        self.optional_columns = DATA_VALIDATION['optional_columns']
        self.date_format = DATA_VALIDATION['date_format']
        self.min_history_days = DATA_VALIDATION['min_history_days']
        self.max_missing_ratio = DATA_VALIDATION['max_missing_ratio']
        self.price_positive = DATA_VALIDATION['price_positive']
    
    def validate(self, data):
        """데이터 검증을 수행합니다."""
        issues = []
        is_valid = True
        
        # 1. 필수 컬럼 존재 여부 확인
        missing_required = [col for col in self.required_columns if col not in data.columns]
        if missing_required:
            issues.append(f"필수 컬럼이 누락되었습니다: {', '.join(missing_required)}")
            is_valid = False
        
        # 2. 데이터 타입 검증
        type_issues = self._validate_data_types(data)
        issues.extend(type_issues)
        if type_issues:
            is_valid = False
        
        # 3. 결측치 검증
        missing_issues = self._validate_missing_data(data)
        issues.extend(missing_issues)
        if missing_issues:
            is_valid = False
        
        # 4. 날짜 형식 및 범위 검증
        date_issues = self._validate_dates(data)
        issues.extend(date_issues)
        if date_issues:
            is_valid = False
        
        # 5. 가격 데이터 검증
        price_issues = self._validate_prices(data)
        issues.extend(price_issues)
        if price_issues:
            is_valid = False
        
        # 6. 중복 데이터 검증
        duplicate_issues = self._validate_duplicates(data)
        issues.extend(duplicate_issues)
        if duplicate_issues:
            is_valid = False
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'warnings': self._generate_warnings(data)
        }
    
    def _validate_data_types(self, data):
        """데이터 타입을 검증합니다."""
        issues = []
        
        # Date 컬럼 검증
        if 'Date' in data.columns:
            try:
                pd.to_datetime(data['Date'])
            except:
                issues.append("Date 컬럼의 날짜 형식이 올바르지 않습니다.")
        
        # Ticker 컬럼 검증
        if 'Ticker' in data.columns:
            if not data['Ticker'].dtype == 'object':
                issues.append("Ticker 컬럼은 문자열 형식이어야 합니다.")
        
        # 가격 컬럼들 검증
        price_columns = ['Close', 'Open', 'High', 'Low']
        for col in price_columns:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    issues.append(f"{col} 컬럼은 숫자 형식이어야 합니다.")
        
        # Volume 컬럼 검증
        if 'Volume' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['Volume']):
                issues.append("Volume 컬럼은 숫자 형식이어야 합니다.")
        
        return issues
    
    def _validate_missing_data(self, data):
        """결측치를 검증합니다."""
        issues = []
        
        # 필수 컬럼의 결측치 검증
        for col in self.required_columns:
            if col in data.columns:
                missing_ratio = data[col].isnull().sum() / len(data)
                if missing_ratio > self.max_missing_ratio:
                    issues.append(f"{col} 컬럼의 결측치 비율이 {missing_ratio:.1%}로 허용 범위({self.max_missing_ratio:.1%})를 초과합니다.")
        
        return issues
    
    def _validate_dates(self, data):
        """날짜 데이터를 검증합니다."""
        issues = []
        
        if 'Date' in data.columns:
            try:
                dates = pd.to_datetime(data['Date'])
                
                # 날짜 범위 검증
                date_range = dates.max() - dates.min()
                if date_range.days < self.min_history_days:
                    issues.append(f"데이터 기간이 {date_range.days}일로 최소 요구사항({self.min_history_days}일)을 충족하지 않습니다.")
                
                # 미래 날짜 검증
                future_dates = dates[dates > pd.Timestamp.now()]
                if len(future_dates) > 0:
                    issues.append(f"미래 날짜가 포함되어 있습니다: {len(future_dates)}개")
                
                # 주말 데이터 검증 (경고)
                weekend_dates = dates[dates.dt.weekday >= 5]
                if len(weekend_dates) > 0:
                    issues.append(f"주말 데이터가 포함되어 있습니다: {len(weekend_dates)}개")
            
            except Exception as e:
                issues.append(f"날짜 데이터 처리 중 오류: {str(e)}")
        
        return issues
    
    def _validate_prices(self, data):
        """가격 데이터를 검증합니다."""
        issues = []
        
        price_columns = ['Close', 'Open', 'High', 'Low']
        
        for col in price_columns:
            if col in data.columns:
                # 음수 가격 검증
                if self.price_positive and (data[col] <= 0).any():
                    negative_count = (data[col] <= 0).sum()
                    issues.append(f"{col} 컬럼에 {negative_count}개의 음수 또는 0 값이 있습니다.")
                
                # 이상치 검증 (3시그마 규칙)
                mean_price = data[col].mean()
                std_price = data[col].std()
                outliers = data[col][(data[col] < mean_price - 3*std_price) | (data[col] > mean_price + 3*std_price)]
                if len(outliers) > 0:
                    issues.append(f"{col} 컬럼에 {len(outliers)}개의 이상치가 있습니다.")
        
        # OHLC 관계 검증
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            invalid_ohlc = (
                (data['High'] < data['Low']) |
                (data['High'] < data['Open']) |
                (data['High'] < data['Close']) |
                (data['Low'] > data['Open']) |
                (data['Low'] > data['Close'])
            )
            
            if invalid_ohlc.any():
                invalid_count = invalid_ohlc.sum()
                issues.append(f"OHLC 데이터 관계가 올바르지 않은 행이 {invalid_count}개 있습니다.")
        
        return issues
    
    def _validate_duplicates(self, data):
        """중복 데이터를 검증합니다."""
        issues = []
        
        # 완전 중복 행 검증
        duplicate_rows = data.duplicated().sum()
        if duplicate_rows > 0:
            issues.append(f"완전히 중복된 행이 {duplicate_rows}개 있습니다.")
        
        # 날짜-종목 조합 중복 검증
        if 'Date' in data.columns and 'Ticker' in data.columns:
            duplicate_combinations = data.duplicated(subset=['Date', 'Ticker']).sum()
            if duplicate_combinations > 0:
                issues.append(f"동일한 날짜-종목 조합이 {duplicate_combinations}개 있습니다.")
        
        return issues
    
    def _generate_warnings(self, data):
        """경고 메시지를 생성합니다."""
        warnings = []
        
        # 선택적 컬럼 누락 경고
        missing_optional = [col for col in self.optional_columns if col not in data.columns]
        if missing_optional:
            warnings.append(f"선택적 컬럼이 누락되었습니다: {', '.join(missing_optional)}")
        
        # 데이터 크기 경고
        if len(data) < 1000:
            warnings.append("데이터 크기가 작습니다. 더 많은 데이터를 사용하는 것을 권장합니다.")
        
        # 종목 수 경고
        if 'Ticker' in data.columns:
            unique_tickers = data['Ticker'].nunique()
            if unique_tickers < 10:
                warnings.append(f"종목 수가 적습니다 ({unique_tickers}개). 더 다양한 종목을 포함하는 것을 권장합니다.")
        
        return warnings
    
    def get_validation_summary(self, data):
        """검증 결과 요약을 반환합니다."""
        validation_result = self.validate(data)
        
        summary = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'required_columns_present': all(col in data.columns for col in self.required_columns),
            'missing_ratio': data.isnull().sum().sum() / (len(data) * len(data.columns)),
            'duplicate_rows': data.duplicated().sum(),
            'validation_passed': validation_result['is_valid'],
            'issue_count': len(validation_result['issues']),
            'warning_count': len(validation_result['warnings'])
        }
        
        if 'Date' in data.columns and 'Ticker' in data.columns:
            try:
                dates = pd.to_datetime(data['Date'])
                summary['date_range_days'] = (dates.max() - dates.min()).days
                summary['unique_tickers'] = data['Ticker'].nunique()
            except:
                summary['date_range_days'] = 0
                summary['unique_tickers'] = 0
        
        return summary 
