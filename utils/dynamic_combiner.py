"""
동적 결합 모듈
시점별 팩터 성과 기반 메가-알파 생성 및 동적 가중치 조정 기능을 제공합니다.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class DynamicCombiner:
    """동적 결합 클래스"""
    
    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings = settings
        self.weight_history = []
        self.factor_performance_history = []
    
    def create_mega_alpha(self, data: pd.DataFrame, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """메가-알파를 생성합니다."""
        try:
            # 입력 검증
            if data is None or data.empty:
                raise ValueError("입력 데이터가 비어있습니다")
            
            if not factors:
                raise ValueError("팩터가 없습니다")
            
            # 팩터 값들을 데이터프레임으로 변환
            factor_data = {}
            for factor in factors:
                factor_name = factor.get('name', f'Factor_{len(factor_data)}')
                factor_values = factor.get('values', pd.Series(0, index=data.index))
                
                # 팩터 값이 Series가 아닌 경우 처리
                if not isinstance(factor_values, pd.Series):
                    if isinstance(factor_values, (list, np.ndarray)):
                        # 길이가 맞지 않으면 0으로 채움
                        if len(factor_values) != len(data):
                            factor_values = np.zeros(len(data))
                        factor_values = pd.Series(factor_values, index=data.index)
                    else:
                        # 스칼라 값인 경우 모든 날짜에 동일하게 적용
                        factor_values = pd.Series(float(factor_values), index=data.index)
                
                factor_data[factor_name] = factor_values
            
            factor_df = pd.DataFrame(factor_data)
            
            # 날짜별로 그룹화
            if 'Date' in data.columns:
                factor_df['Date'] = data['Date']
                factor_df = factor_df.set_index('Date')
            
            # 결측치 처리
            factor_df = factor_df.fillna(0)
            
            # 동적 가중치 계산
            weights = self._calculate_dynamic_weights(factor_df, data)
            
            # 메가-알파 생성
            mega_alpha_values = self._combine_factors(factor_df, weights)
            
            # 성과 분석
            performance_metrics = self._calculate_performance_metrics(mega_alpha_values, data)
            
            return {
                'values': mega_alpha_values,
                'weights': weights,
                'weight_history': self.weight_history,
                'factor_performance_history': self.factor_performance_history,
                'performance_metrics': performance_metrics,
                'cumulative_returns': self._calculate_cumulative_returns(mega_alpha_values, data),
                'monthly_returns': self._calculate_monthly_returns(mega_alpha_values, data),
                'factor_contribution': self._calculate_factor_contribution(factor_df, weights)
            }
            
        except Exception as e:
            print(f"ERROR in create_mega_alpha: {str(e)}")
            raise Exception(f"메가-알파 생성 실패: {str(e)}")
    
    def _calculate_dynamic_weights(self, factor_df, data):
        """동적 가중치를 계산합니다."""
        weights = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
        
        # 리밸런싱 주기에 따른 가중치 계산
        rebalancing_freq = self.settings['rebalancing_freq']
        
        if rebalancing_freq == "일간":
            freq_days = 1
        elif rebalancing_freq == "주간":
            freq_days = 7
        elif rebalancing_freq == "월간":
            freq_days = 30
        elif rebalancing_freq == "분기":
            freq_days = 90
        else:
            freq_days = 30  # 기본값
        
        # 날짜별 가중치 계산
        for i, date in enumerate(factor_df.index):
            if i % freq_days == 0 or i == 0:
                # 해당 시점까지의 팩터 성과 계산
                historical_performance = self._calculate_historical_performance(
                    factor_df.iloc[:i+1], data, date
                )
                
                # 가중치 최적화
                optimal_weights = self._optimize_weights(historical_performance)
                
                # 가중치 히스토리 저장
                self.weight_history.append({
                    'date': date,
                    'weights': optimal_weights.copy()
                })
            
            # 현재 가중치 적용
            weights.loc[date] = optimal_weights
        
        return weights
    
    def _calculate_historical_performance(self, factor_data, price_data, current_date):
        """과거 팩터 성과를 계산합니다."""
        performance = {}
        
        # 수익률 계산
        returns = price_data.groupby('Ticker')['Close'].pct_change()
        
        for factor_name in factor_data.columns:
            factor_values = factor_data[factor_name]
            
            # IC 계산
            ic = self._calculate_rolling_ic(factor_values, returns, current_date)
            
            # ICIR 계산
            icir = self._calculate_rolling_icir(factor_values, returns, current_date)
            
            # 승률 계산
            win_rate = self._calculate_rolling_win_rate(factor_values, returns, current_date)
            
            performance[factor_name] = {
                'ic': ic,
                'icir': icir,
                'win_rate': win_rate
            }
        
        return performance
    
    def _calculate_rolling_ic(self, factor_values, returns, current_date, window=60):
        """롤링 IC를 계산합니다."""
        try:
            # 현재 날짜까지의 데이터
            mask = factor_values.index <= current_date
            factor_subset = factor_values[mask]
            returns_subset = returns[mask]
            
            if len(factor_subset) < window:
                return 0.0
            
            # 최근 window 기간의 IC 계산
            recent_factor = factor_subset.tail(window)
            recent_returns = returns_subset.tail(window)
            
            # 결측치 제거
            valid_mask = ~(recent_factor.isna() | recent_returns.isna())
            factor_clean = recent_factor[valid_mask]
            returns_clean = recent_returns[valid_mask]
            
            if len(factor_clean) < 10:
                return 0.0
            
            # 순위 상관계수 계산
            ic = factor_clean.corr(returns_clean, method='spearman')
            
            return ic if not np.isnan(ic) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_rolling_icir(self, factor_values, returns, current_date, window=60):
        """롤링 ICIR을 계산합니다."""
        try:
            # 현재 날짜까지의 데이터
            mask = factor_values.index <= current_date
            factor_subset = factor_values[mask]
            returns_subset = returns[mask]
            
            if len(factor_subset) < window * 2:
                return 0.0
            
            # 월별 IC 계산
            ic_series = []
            for i in range(window, len(factor_subset), window//4):
                start_idx = max(0, i - window)
                end_idx = i
                
                period_factor = factor_subset.iloc[start_idx:end_idx]
                period_returns = returns_subset.iloc[start_idx:end_idx]
                
                ic = self._calculate_rolling_ic(period_factor, period_returns, current_date, window//4)
                ic_series.append(ic)
            
            if len(ic_series) < 3:
                return 0.0
            
            # ICIR 계산
            mean_ic = np.mean(ic_series)
            std_ic = np.std(ic_series)
            
            icir = mean_ic / std_ic if std_ic > 0 else 0.0
            
            return icir if not np.isnan(icir) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_rolling_win_rate(self, factor_values, returns, current_date, window=60):
        """롤링 승률을 계산합니다."""
        try:
            # 현재 날짜까지의 데이터
            mask = factor_values.index <= current_date
            factor_subset = factor_values[mask]
            returns_subset = returns[mask]
            
            if len(factor_subset) < window:
                return 0.5
            
            # 최근 window 기간의 승률 계산
            recent_factor = factor_subset.tail(window)
            recent_returns = returns_subset.tail(window)
            
            # 결측치 제거
            valid_mask = ~(recent_factor.isna() | recent_returns.isna())
            factor_clean = recent_factor[valid_mask]
            returns_clean = recent_returns[valid_mask]
            
            if len(factor_clean) < 10:
                return 0.5
            
            # 팩터 기반 포트폴리오 수익률
            factor_ranks = factor_clean.rank(pct=True)
            positions = np.where(factor_ranks > 0.7, 1, np.where(factor_ranks < 0.3, -1, 0))
            portfolio_returns = positions * returns_clean
            
            # 승률 계산
            win_rate = (portfolio_returns > 0).mean()
            
            return win_rate if not np.isnan(win_rate) else 0.5
            
        except Exception:
            return 0.5
    
    def _optimize_weights(self, performance):
        """가중치를 최적화합니다."""
        method = self.settings['combination_method']
        
        if method == "등가중치":
            return self._equal_weight_optimization(performance)
        elif method == "동적 가중치":
            return self._dynamic_weight_optimization(performance)
        elif method == "최적화 가중치":
            return self._optimization_weight_optimization(performance)
        elif method == "적응형 가중치":
            return self._adaptive_weight_optimization(performance)
        else:
            return self._equal_weight_optimization(performance)
    
    def _equal_weight_optimization(self, performance):
        """등가중치 최적화"""
        n_factors = len(performance)
        if n_factors == 0:
            return {}
        
        equal_weight = 1.0 / n_factors
        weights = {factor: equal_weight for factor in performance.keys()}
        
        return weights
    
    def _dynamic_weight_optimization(self, performance):
        """동적 가중치 최적화"""
        if not performance:
            return {}
        
        # IC와 ICIR 기반 가중치 계산
        weights = {}
        total_score = 0
        
        for factor, metrics in performance.items():
            ic = metrics.get('ic', 0)
            icir = metrics.get('icir', 0)
            win_rate = metrics.get('win_rate', 0.5)
            
            # 종합 점수 계산
            score = (ic * 0.4 + icir * 0.4 + win_rate * 0.2)
            score = max(0, score)  # 음수 가중치 방지
            
            weights[factor] = score
            total_score += score
        
        # 정규화
        if total_score > 0:
            for factor in weights:
                weights[factor] /= total_score
        else:
            # 모든 점수가 0인 경우 등가중치
            n_factors = len(weights)
            for factor in weights:
                weights[factor] = 1.0 / n_factors
        
        return weights
    
    def _optimization_weight_optimization(self, performance):
        """최적화 가중치 계산"""
        if not performance:
            return {}
        
        # 목표 함수 정의
        def objective(weights):
            # 샤프 비율 최대화 (예시)
            portfolio_ic = sum(weights[i] * list(performance.values())[i]['ic'] 
                             for i in range(len(weights)))
            portfolio_icir = sum(weights[i] * list(performance.values())[i]['icir'] 
                               for i in range(len(weights)))
            
            # 샤프 비율 근사
            sharpe = portfolio_icir if portfolio_icir > 0 else 0
            return -sharpe  # 최소화를 위해 음수
        
        # 제약 조건
        n_factors = len(performance)
        constraints = [
            {'type': 'eq', 'fun': lambda x: sum(x) - 1},  # 가중치 합 = 1
        ]
        
        # 경계 조건
        bounds = [(0, self.settings['max_weight']) for _ in range(n_factors)]
        
        # 초기값
        x0 = [1.0 / n_factors] * n_factors
        
        # 최적화 실행
        try:
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = {factor: weight for factor, weight in 
                          zip(performance.keys(), result.x)}
                return weights
            else:
                return self._equal_weight_optimization(performance)
                
        except Exception:
            return self._equal_weight_optimization(performance)
    
    def _adaptive_weight_optimization(self, performance):
        """적응형 가중치 최적화"""
        if not performance:
            return {}
        
        # 시장 상황에 따른 적응형 가중치
        weights = {}
        total_score = 0
        
        for factor, metrics in performance.items():
            ic = metrics.get('ic', 0)
            icir = metrics.get('icir', 0)
            win_rate = metrics.get('win_rate', 0.5)
            
            # 적응형 점수 계산
            # IC가 높을 때 더 큰 가중치
            ic_weight = np.tanh(ic * 10)  # tanh 함수로 정규화
            
            # ICIR이 높을 때 더 큰 가중치
            icir_weight = np.tanh(icir * 2)
            
            # 승률이 극단적일 때 가중치 감소 (과적합 방지)
            win_rate_weight = 1 - abs(win_rate - 0.5) * 2
            
            # 종합 점수
            score = ic_weight * 0.5 + icir_weight * 0.3 + win_rate_weight * 0.2
            score = max(0, score)
            
            weights[factor] = score
            total_score += score
        
        # 정규화
        if total_score > 0:
            for factor in weights:
                weights[factor] /= total_score
        else:
            n_factors = len(weights)
            for factor in weights:
                weights[factor] = 1.0 / n_factors
        
        return weights
    
    def _combine_factors(self, factor_df, weights):
        """팩터들을 결합하여 메가-알파를 생성합니다."""
        try:
            # 결과를 저장할 Series 초기화
            mega_alpha = pd.Series(0.0, index=factor_df.index, dtype=float)
            
            for date in factor_df.index:
                current_value = 0.0
                
                if date in weights.index:
                    date_weights = weights.loc[date]
                    
                    for factor_name in factor_df.columns:
                        if factor_name in date_weights and factor_name in factor_df.columns:
                            weight = date_weights[factor_name]
                            factor_value = factor_df.loc[date, factor_name]
                            
                            # 스칼라 값으로 변환하여 계산
                            if not pd.isna(factor_value).any() and not pd.isna(weight).any():
                                if isinstance(factor_value, (list, np.ndarray)):
                                    # 배열인 경우 평균값 사용
                                    factor_value = np.mean(factor_value) if len(factor_value) > 0 else 0.0
                                elif isinstance(factor_value, pd.Series):
                                    # Series인 경우 평균값 사용
                                    factor_value = factor_value.mean() if len(factor_value) > 0 else 0.0
                                
                                # 스칼라 값으로 변환
                                weight = float(weight)
                                factor_value = float(factor_value)
                                
                                current_value += weight * factor_value
                else:
                    # 가중치가 없는 날짜는 등가중치 사용
                    n_factors = len(factor_df.columns)
                    equal_weight = 1.0 / n_factors if n_factors > 0 else 0.0
                    
                    for factor_name in factor_df.columns:
                        factor_value = factor_df.loc[date, factor_name]
                        if not pd.isna(factor_value).any():
                            if isinstance(factor_value, (list, np.ndarray)):
                                factor_value = np.mean(factor_value) if len(factor_value) > 0 else 0.0
                            elif isinstance(factor_value, pd.Series):
                                factor_value = factor_value.mean() if len(factor_value) > 0 else 0.0
                            
                            factor_value = float(factor_value)
                            current_value += equal_weight * factor_value
                
                # 최종 값을 할당
                mega_alpha.loc[date] = current_value
            
            return mega_alpha
            
        except Exception as e:
            print(f"ERROR in _combine_factors: {str(e)}")
            # 에러 발생 시 기본값 반환
            return pd.Series(0.0, index=factor_df.index)
    
    def _calculate_performance_metrics(self, mega_alpha, data):
        """메가-알파의 성과 지표를 계산합니다."""
        # 수익률 계산
        returns = data.groupby('Ticker')['Close'].pct_change()
        
        # IC 계산
        ic = self._calculate_rolling_ic(mega_alpha, returns, mega_alpha.index[-1])
        
        # ICIR 계산
        icir = self._calculate_rolling_icir(mega_alpha, returns, mega_alpha.index[-1])
        
        # 샤프 비율 계산
        sharpe = self._calculate_sharpe_ratio(mega_alpha, returns)
        
        # 최대 낙폭 계산
        max_drawdown = self._calculate_max_drawdown(mega_alpha, returns)
        
        return {
            'ic': ic,
            'icir': icir,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'volatility': self._calculate_volatility(mega_alpha, returns)
        }
    
    def _calculate_sharpe_ratio(self, mega_alpha, returns):
        """샤프 비율을 계산합니다."""
        try:
            # 메가-알파 기반 포트폴리오 수익률
            alpha_ranks = mega_alpha.rank(pct=True)
            positions = np.where(alpha_ranks > 0.7, 1, np.where(alpha_ranks < 0.3, -1, 0))
            portfolio_returns = positions * returns
            
            # 샤프 비율 계산
            mean_return = np.mean(portfolio_returns) * 252
            std_return = np.std(portfolio_returns) * np.sqrt(252)
            
            sharpe = mean_return / std_return if std_return > 0 else 0.0
            
            return sharpe if not np.isnan(sharpe) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_max_drawdown(self, mega_alpha, returns):
        """최대 낙폭을 계산합니다."""
        try:
            # 누적 수익률 계산
            alpha_ranks = mega_alpha.rank(pct=True)
            positions = np.where(alpha_ranks > 0.7, 1, np.where(alpha_ranks < 0.3, -1, 0))
            portfolio_returns = positions * returns
            
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            # 최대 낙폭 계산
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            
            max_drawdown = drawdown.min()
            
            return abs(max_drawdown) if not np.isnan(max_drawdown) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_volatility(self, mega_alpha, returns):
        """변동성을 계산합니다."""
        try:
            alpha_ranks = mega_alpha.rank(pct=True)
            positions = np.where(alpha_ranks > 0.7, 1, np.where(alpha_ranks < 0.3, -1, 0))
            portfolio_returns = positions * returns
            
            volatility = np.std(portfolio_returns) * np.sqrt(252)
            
            return volatility if not np.isnan(volatility) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_cumulative_returns(self, mega_alpha, data):
        """누적 수익률을 계산합니다."""
        try:
            returns = data.groupby('Ticker')['Close'].pct_change()
            alpha_ranks = mega_alpha.rank(pct=True)
            positions = np.where(alpha_ranks > 0.7, 1, np.where(alpha_ranks < 0.3, -1, 0))
            portfolio_returns = positions * returns
            
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            return cumulative_returns
            
        except Exception:
            return pd.Series(1, index=mega_alpha.index)
    
    def _calculate_monthly_returns(self, mega_alpha, data):
        """월별 수익률을 계산합니다."""
        try:
            returns = data.groupby('Ticker')['Close'].pct_change()
            alpha_ranks = mega_alpha.rank(pct=True)
            positions = np.where(alpha_ranks > 0.7, 1, np.where(alpha_ranks < 0.3, -1, 0))
            portfolio_returns = positions * returns
            
            # 월별 수익률
            monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            return monthly_returns
            
        except Exception:
            return pd.Series(0, index=mega_alpha.index)
    
    def _calculate_factor_contribution(self, factor_df, weights):
        """팩터별 기여도를 계산합니다."""
        try:
            contribution = []
            
            for factor_name in factor_df.columns:
                # 가중치 평균
                avg_weight = weights[factor_name].mean()
                
                # 팩터 값의 표준편차
                factor_std = factor_df[factor_name].std()
                
                # 기여도 = 가중치 * 표준편차
                factor_contribution = avg_weight * factor_std
                
                contribution.append({
                    'factor': factor_name,
                    'contribution': factor_contribution
                })
            
            return contribution
            
        except Exception:
            return [] 
