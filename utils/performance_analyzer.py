"""
성과 분석 모듈
팩터의 성과 지표를 계산하고 분석합니다.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PerformanceAnalyzer:
    """성과 분석 클래스"""
    
    def __init__(self):
        pass
    
    def analyze_factors(self, data, factors):
        """팩터들의 성과를 분석합니다."""
        try:
            # 입력 검증
            if data is None or data.empty:
                raise ValueError("분석할 데이터가 없습니다")
            
            if not factors:
                raise ValueError("분석할 팩터가 없습니다")
            
            results = {
                'factor_performance': [],
                'avg_ic': 0,
                'avg_icir': 0,
                'win_rate': 0,
                'total_factors': len(factors)
            }
            
            # 각 팩터별 성과 분석
            for i, factor in enumerate(factors):
                try:
                    performance = self._analyze_single_factor(data, factor)
                    results['factor_performance'].append(performance)
                except Exception as e:
                    print(f"WARNING: 팩터 {i} 분석 실패: {str(e)}")
                    # 기본 성과 정보 추가
                    results['factor_performance'].append({
                        'Factor': factor.get('name', f'Factor_{i}'),
                        'IC': 0.0,
                        'ICIR': 0.0,
                        'Win_Rate': 0.0,
                        'Sharpe': 0.0,
                        'formula': factor.get('formula', 'N/A'),
                        'type': factor.get('type', 'unknown')
                    })
            
            # 전체 통계 계산
            if results['factor_performance']:
                ics = [p['IC'] for p in results['factor_performance'] if not np.isnan(p['IC'])]
                icirs = [p['ICIR'] for p in results['factor_performance'] if not np.isnan(p['ICIR'])]
                win_rates = [p['Win_Rate'] for p in results['factor_performance'] if not np.isnan(p['Win_Rate'])]
                
                results['avg_ic'] = np.mean(ics) if ics else 0.0
                results['avg_icir'] = np.mean(icirs) if icirs else 0.0
                results['win_rate'] = np.mean(win_rates) if win_rates else 0.0
            
            return results
            
        except Exception as e:
            print(f"ERROR in analyze_factors: {str(e)}")
            raise Exception(f"팩터 성과 분석 실패: {str(e)}")
    
    def _analyze_single_factor(self, data, factor):
        """단일 팩터의 성과를 분석합니다."""
        factor_values = factor['values']
        factor_name = factor['name']
        
        # 수익률 계산
        returns = data.groupby('Ticker')['Close'].pct_change()
        
        # IC (Information Coefficient) 계산
        ic = self._calculate_ic(factor_values, returns)
        
        # ICIR (IC Information Ratio) 계산
        icir = self._calculate_icir(factor_values, returns)
        
        # 승률 계산
        win_rate = self._calculate_win_rate(factor_values, returns)
        
        # 샤프 비율 계산
        sharpe = self._calculate_sharpe_ratio(factor_values, returns)
        
        # 팩터 수식
        formula = factor.get('formula', 'N/A')
        
        return {
            'Factor': factor_name,
            'IC': ic,
            'ICIR': icir,
            'Win_Rate': win_rate,
            'Sharpe': sharpe,
            'formula': formula,
            'type': factor.get('type', 'unknown')
        }
    
    def _calculate_ic(self, factor_values, returns):
        """Information Coefficient를 계산합니다."""
        try:
            # 결측치 제거
            valid_mask = ~(factor_values.isna() | returns.isna())
            factor_clean = factor_values[valid_mask]
            returns_clean = returns[valid_mask]
            
            if len(factor_clean) < 10:
                return 0.0
            
            # 순위 상관계수 계산
            ic = factor_clean.corr(returns_clean, method='spearman')
            
            return ic if not np.isnan(ic) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_icir(self, factor_values, returns):
        """IC Information Ratio를 계산합니다."""
        try:
            # 시계열 IC 계산
            ic_series = self._calculate_rolling_ic(factor_values, returns)
            
            if len(ic_series) < 10:
                return 0.0
            
            # ICIR = 평균 IC / IC 표준편차
            mean_ic = np.mean(ic_series)
            std_ic = np.std(ic_series)
            
            icir = mean_ic / std_ic if std_ic > 0 else 0.0
            
            return icir if not np.isnan(icir) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_rolling_ic(self, factor_values, returns, window=20):
        """롤링 IC를 계산합니다."""
        ic_series = []
        
        # 날짜별로 그룹화
        if 'Date' in factor_values.index.names or 'Date' in factor_values.index:
            dates = factor_values.index.get_level_values('Date') if factor_values.index.nlevels > 1 else factor_values.index
            unique_dates = sorted(dates.unique())
            
            for i in range(window, len(unique_dates)):
                start_date = unique_dates[i-window]
                end_date = unique_dates[i]
                
                # 기간별 데이터 추출
                mask = (dates >= start_date) & (dates <= end_date)
                period_factor = factor_values[mask]
                period_returns = returns[mask]
                
                # 기간별 IC 계산
                ic = self._calculate_ic(period_factor, period_returns)
                ic_series.append(ic)
        
        return np.array(ic_series)
    
    def _calculate_win_rate(self, factor_values, returns):
        """승률을 계산합니다."""
        try:
            # 결측치 제거
            valid_mask = ~(factor_values.isna() | returns.isna())
            factor_clean = factor_values[valid_mask]
            returns_clean = returns[valid_mask]
            
            if len(factor_clean) < 10:
                return 0.0
            
            # 팩터 값으로 포트폴리오 구성
            # 상위 30% 종목 매수, 하위 30% 종목 매도
            factor_ranks = factor_clean.groupby(factor_clean.index.get_level_values('Date') if factor_clean.index.nlevels > 1 else factor_clean.index).rank(pct=True)
            
            # 포지션 생성
            positions = np.where(factor_ranks > 0.7, 1, np.where(factor_ranks < 0.3, -1, 0))
            
            # 수익률 계산
            portfolio_returns = positions * returns_clean
            
            # 승률 계산 (양의 수익률 비율)
            win_rate = (portfolio_returns > 0).mean()
            
            return win_rate if not np.isnan(win_rate) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_sharpe_ratio(self, factor_values, returns):
        """샤프 비율을 계산합니다."""
        try:
            # 결측치 제거
            valid_mask = ~(factor_values.isna() | returns.isna())
            factor_clean = factor_values[valid_mask]
            returns_clean = returns[valid_mask]
            
            if len(factor_clean) < 10:
                return 0.0
            
            # 팩터 기반 포트폴리오 수익률
            factor_ranks = factor_clean.groupby(factor_clean.index.get_level_values('Date') if factor_clean.index.nlevels > 1 else factor_clean.index).rank(pct=True)
            positions = np.where(factor_ranks > 0.7, 1, np.where(factor_ranks < 0.3, -1, 0))
            portfolio_returns = positions * returns_clean
            
            # 샤프 비율 계산 (연율화)
            mean_return = np.mean(portfolio_returns) * 252  # 연율화
            std_return = np.std(portfolio_returns) * np.sqrt(252)  # 연율화
            
            sharpe = mean_return / std_return if std_return > 0 else 0.0
            
            return sharpe if not np.isnan(sharpe) else 0.0
            
        except Exception:
            return 0.0
    
    def calculate_factor_correlation(self, factors):
        """팩터 간 상관관계를 계산합니다."""
        if len(factors) < 2:
            return None
        
        # 팩터 값들을 데이터프레임으로 변환
        factor_data = {}
        for factor in factors:
            factor_data[factor['name']] = factor['values']
        
        factor_df = pd.DataFrame(factor_data)
        
        # 상관관계 매트릭스 계산
        correlation_matrix = factor_df.corr()
        
        return correlation_matrix
    
    def calculate_factor_turnover(self, factors, data):
        """팩터 턴오버를 계산합니다."""
        turnover_rates = {}
        
        for factor in factors:
            factor_values = factor['values']
            factor_name = factor['name']
            
            try:
                # 날짜별 순위 변화 계산
                if factor_values.index.nlevels > 1:
                    dates = factor_values.index.get_level_values('Date')
                else:
                    dates = factor_values.index
                
                unique_dates = sorted(dates.unique())
                
                if len(unique_dates) < 2:
                    turnover_rates[factor_name] = 0.0
                    continue
                
                turnover_sum = 0
                count = 0
                
                for i in range(1, len(unique_dates)):
                    prev_date = unique_dates[i-1]
                    curr_date = unique_dates[i]
                    
                    # 이전 날짜와 현재 날짜의 순위
                    prev_mask = dates == prev_date
                    curr_mask = dates == curr_date
                    
                    prev_ranks = factor_values[prev_mask].rank(pct=True)
                    curr_ranks = factor_values[curr_mask].rank(pct=True)
                    
                    # 공통 종목 찾기
                    common_tickers = set(prev_ranks.index) & set(curr_ranks.index)
                    
                    if len(common_tickers) > 0:
                        # 순위 변화의 절대값 평균
                        rank_changes = []
                        for ticker in common_tickers:
                            if ticker in prev_ranks.index and ticker in curr_ranks.index:
                                change = abs(prev_ranks[ticker] - curr_ranks[ticker])
                                rank_changes.append(change)
                        
                        if rank_changes:
                            turnover_sum += np.mean(rank_changes)
                            count += 1
                
                avg_turnover = turnover_sum / count if count > 0 else 0.0
                turnover_rates[factor_name] = avg_turnover
                
            except Exception:
                turnover_rates[factor_name] = 0.0
        
        return turnover_rates
    
    def generate_performance_report(self, data, factors):
        """종합 성과 리포트를 생성합니다."""
        # 기본 성과 분석
        performance_results = self.analyze_factors(data, factors)
        
        # 상관관계 분석
        correlation_matrix = self.calculate_factor_correlation(factors)
        
        # 턴오버 분석
        turnover_rates = self.calculate_factor_turnover(factors, data)
        
        # 리포트 구성
        report = {
            'summary': {
                'total_factors': len(factors),
                'avg_ic': performance_results['avg_ic'],
                'avg_icir': performance_results['avg_icir'],
                'avg_win_rate': performance_results['win_rate'],
                'avg_turnover': np.mean(list(turnover_rates.values())) if turnover_rates else 0.0
            },
            'factor_performance': performance_results['factor_performance'],
            'correlation_matrix': correlation_matrix,
            'turnover_rates': turnover_rates,
            'recommendations': self._generate_recommendations(performance_results, turnover_rates)
        }
        
        return report
    
    def analyze_mega_alpha(self, data, mega_alpha):
        """메가-알파의 성과를 분석합니다."""
        try:
            # 메가-알파 값 추출
            if isinstance(mega_alpha, dict) and 'values' in mega_alpha:
                alpha_values = mega_alpha['values']
            else:
                alpha_values = mega_alpha
            
            # 수익률 계산
            returns = data.groupby('Ticker')['Close'].pct_change()
            
            # IC 계산
            ic = self._calculate_ic(alpha_values, returns)
            
            # ICIR 계산
            icir = self._calculate_icir(alpha_values, returns)
            
            # 승률 계산
            win_rate = self._calculate_win_rate(alpha_values, returns)
            
            # 샤프 비율 계산
            sharpe = self._calculate_sharpe_ratio(alpha_values, returns)
            
            # 변동성 계산
            volatility = alpha_values.std() * np.sqrt(252) if len(alpha_values) > 0 else 0
            
            # 최대 낙폭 계산
            max_drawdown = self._calculate_max_drawdown(alpha_values, returns)
            
            return {
                'ic': ic,
                'icir': icir,
                'win_rate': win_rate,
                'sharpe': sharpe,
                'volatility': volatility,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            # 오류 발생 시 기본값 반환
            return {
                'ic': 0.0,
                'icir': 0.0,
                'win_rate': 0.0,
                'sharpe': 0.0,
                'volatility': 0.0,
                'max_drawdown': 0.0
            }
    
    def _calculate_max_drawdown(self, alpha_values, returns):
        """최대 낙폭을 계산합니다."""
        try:
            # 팩터 기반 포트폴리오 수익률
            factor_ranks = alpha_values.groupby(alpha_values.index.get_level_values('Date') if alpha_values.index.nlevels > 1 else alpha_values.index).rank(pct=True)
            positions = np.where(factor_ranks > 0.7, 1, np.where(factor_ranks < 0.3, -1, 0))
            portfolio_returns = positions * returns
            
            # 누적 수익률 계산
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            # 최대 낙폭 계산
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            return abs(max_drawdown) if not np.isnan(max_drawdown) else 0.0
            
        except Exception:
            return 0.0
    
    def _generate_recommendations(self, performance_results, turnover_rates):
        """성과 분석 결과를 바탕으로 권장사항을 생성합니다."""
        recommendations = []
        
        # IC 기반 권장사항
        avg_ic = performance_results['avg_ic']
        if avg_ic > 0.05:
            recommendations.append("🎯 평균 IC가 매우 높습니다. 팩터 품질이 우수합니다.")
        elif avg_ic > 0.02:
            recommendations.append("✅ 평균 IC가 양호합니다. 팩터가 유효합니다.")
        else:
            recommendations.append("⚠️ 평균 IC가 낮습니다. 팩터 선택을 재검토하세요.")
        
        # ICIR 기반 권장사항
        avg_icir = performance_results['avg_icir']
        if avg_icir > 1.0:
            recommendations.append("🚀 평균 ICIR이 매우 높습니다. 안정적인 팩터입니다.")
        elif avg_icir > 0.5:
            recommendations.append("✅ 평균 ICIR이 양호합니다. 팩터가 안정적입니다.")
        else:
            recommendations.append("⚠️ 평균 ICIR이 낮습니다. 팩터 안정성을 개선하세요.")
        
        # 턴오버 기반 권장사항
        if turnover_rates:
            avg_turnover = np.mean(list(turnover_rates.values()))
            if avg_turnover < 0.1:
                recommendations.append("💰 턴오버가 낮습니다. 거래 비용이 절약됩니다.")
            elif avg_turnover > 0.3:
                recommendations.append("⚠️ 턴오버가 높습니다. 거래 비용을 고려하세요.")
        
        # 승률 기반 권장사항
        win_rate = performance_results['win_rate']
        if win_rate > 0.6:
            recommendations.append("🏆 승률이 높습니다. 일관된 성과를 보입니다.")
        elif win_rate < 0.4:
            recommendations.append("⚠️ 승률이 낮습니다. 팩터 선택을 재검토하세요.")
        
        return recommendations 
