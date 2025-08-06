"""
백테스팅 모듈
포트폴리오 백테스팅 및 성과 분석 기능을 제공합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class Backtester:
    """백테스팅 클래스"""
    
    def __init__(self, settings):
        self.settings = settings
        self.portfolio_history = []
        self.trade_history = []
    
    def construct_portfolio(self, data, mega_alpha):
        """메가-알파를 기반으로 포트폴리오를 구성합니다."""
        portfolio = {}
        
        # 날짜별로 포트폴리오 구성
        for date in mega_alpha.index:
            # 해당 날짜의 알파 값으로 종목 순위 결정
            alpha_values = mega_alpha.loc[date]
            
            if pd.isna(alpha_values).any() or alpha_values.empty:
                continue
            
            # 상위 N개 종목 선택
            top_stocks = self._select_top_stocks(data, alpha_values, date)
            
            # 가중치 계산
            weights = self._calculate_weights(top_stocks, alpha_values)
            
            portfolio[date] = {
                'stocks': top_stocks,
                'weights': weights,
                'alpha_value': alpha_values
            }
        
        return portfolio
    
    def run_backtest(self, data, portfolio):
        """백테스팅을 실행합니다."""
        results = {
            'dates': [],
            'portfolio_values': [],
            'returns': [],
            'positions': [],
            'trades': []
        }
        
        initial_capital = self.settings['initial_capital']
        current_capital = initial_capital
        current_positions = {}
        
        # 날짜별 백테스팅
        dates = sorted(portfolio.keys())
        
        for i, date in enumerate(dates):
            # 포트폴리오 업데이트
            if date in portfolio:
                new_positions = portfolio[date]
                
                # 거래 실행
                trades = self._execute_trades(
                    current_positions, new_positions, current_capital, date
                )
                
                # 포지션 업데이트
                current_positions = new_positions.copy()
                
                # 거래 비용 계산
                transaction_cost = self._calculate_transaction_cost(trades)
                current_capital -= transaction_cost
                
                # 수익률 계산
                if i > 0:
                    prev_date = dates[i-1]
                    returns = self._calculate_returns(
                        data, current_positions, prev_date, date
                    )
                    current_capital *= (1 + returns)
                
                # 결과 저장
                results['dates'].append(date)
                results['portfolio_values'].append(current_capital)
                results['returns'].append(returns if i > 0 else 0)
                results['positions'].append(current_positions.copy())
                results['trades'].extend(trades)
        
        return results
    
    def _select_top_stocks(self, data, alpha_values, date):
        """상위 종목을 선택합니다."""
        portfolio_size = self.settings['portfolio_size']
        
        # 해당 날짜의 데이터 필터링
        date_data = data[data['Date'] == date].copy()
        
        if len(date_data) == 0:
            return []
        
        # 알파 값을 데이터에 추가
        if isinstance(alpha_values, pd.Series):
            # Series인 경우 인덱스로 매칭 (중복 제거)
            alpha_series = alpha_values.drop_duplicates()
            date_data['alpha'] = date_data['Ticker'].map(alpha_series)
        else:
            # 스칼라 값인 경우 모든 종목에 동일하게 적용
            date_data['alpha'] = alpha_values
        
        # 결측치 처리
        date_data['alpha'] = date_data['alpha'].fillna(0)
        
        # 알파 값으로 정렬
        date_data = date_data.sort_values('alpha', ascending=False)
        
        # 상위 N개 종목 선택
        top_stocks = date_data.head(portfolio_size)['Ticker'].tolist()
        
        return top_stocks
    
    def _calculate_weights(self, stocks, alpha_values):
        """포트폴리오 가중치를 계산합니다."""
        if not stocks:
            return {}
        
        # 등가중치 또는 알파 기반 가중치
        if len(stocks) == 1:
            weights = {stocks[0]: 1.0}
        else:
            # 알파 값 기반 가중치 (소프트맥스)
            alpha_scores = [alpha_values.get(stock, 0) for stock in stocks]
            alpha_scores = np.array(alpha_scores)
            
            # 소프트맥스 정규화
            exp_scores = np.exp(alpha_scores - np.max(alpha_scores))
            weights_array = exp_scores / np.sum(exp_scores)
            
            # 최대 포지션 크기 제한
            max_weight = self.settings['max_position_size']
            weights_array = np.minimum(weights_array, max_weight)
            
            # 재정규화
            weights_array = weights_array / np.sum(weights_array)
            
            weights = {stock: weight for stock, weight in zip(stocks, weights_array)}
        
        return weights
    
    def _execute_trades(self, current_positions, new_positions, capital, date):
        """거래를 실행합니다."""
        trades = []
        
        current_stocks = set(current_positions.get('stocks', []))
        new_stocks = set(new_positions.get('stocks', []))
        
        # 매도할 종목
        sell_stocks = current_stocks - new_stocks
        
        # 매수할 종목
        buy_stocks = new_stocks - current_stocks
        
        # 보유 종목의 가중치 변경
        hold_stocks = current_stocks & new_stocks
        
        # 매도 거래
        for stock in sell_stocks:
            trades.append({
                'date': date,
                'stock': stock,
                'action': 'SELL',
                'quantity': current_positions.get('weights', {}).get(stock, 0) * capital,
                'price': self._get_stock_price(stock, date)
            })
        
        # 매수 거래
        for stock in buy_stocks:
            quantity = new_positions.get('weights', {}).get(stock, 0) * capital
            trades.append({
                'date': date,
                'stock': stock,
                'action': 'BUY',
                'quantity': quantity,
                'price': self._get_stock_price(stock, date)
            })
        
        # 가중치 조정 거래
        for stock in hold_stocks:
            old_weight = current_positions.get('weights', {}).get(stock, 0)
            new_weight = new_positions.get('weights', {}).get(stock, 0)
            
            if abs(new_weight - old_weight) > 0.01:  # 1% 이상 차이
                quantity = (new_weight - old_weight) * capital
                action = 'BUY' if quantity > 0 else 'SELL'
                
                trades.append({
                    'date': date,
                    'stock': stock,
                    'action': action,
                    'quantity': abs(quantity),
                    'price': self._get_stock_price(stock, date)
                })
        
        return trades
    
    def _get_stock_price(self, stock, date):
        """주식 가격을 가져옵니다."""
        # 실제 구현에서는 데이터베이스나 API에서 가격 조회
        # 여기서는 예시로 랜덤 가격 반환
        return np.random.uniform(50, 200)
    
    def _calculate_transaction_cost(self, trades):
        """거래 비용을 계산합니다."""
        transaction_cost_rate = self.settings['transaction_cost']
        total_cost = 0
        
        for trade in trades:
            trade_value = trade['quantity'] * trade['price']
            cost = trade_value * transaction_cost_rate
            total_cost += cost
        
        return total_cost
    
    def _calculate_returns(self, data, positions, start_date, end_date):
        """수익률을 계산합니다."""
        stocks = positions.get('stocks', [])
        weights = positions.get('weights', {})
        
        if not stocks:
            return 0.0
        
        # 각 종목의 수익률 계산
        stock_returns = {}
        
        for stock in stocks:
            stock_data = data[
                (data['Ticker'] == stock) & 
                (data['Date'] >= start_date) & 
                (data['Date'] <= end_date)
            ]
            
            if len(stock_data) >= 2:
                start_price = stock_data.iloc[0]['Close']
                end_price = stock_data.iloc[-1]['Close']
                stock_return = (end_price - start_price) / start_price
                stock_returns[stock] = stock_return
            else:
                stock_returns[stock] = 0.0
        
        # 포트폴리오 수익률 계산
        portfolio_return = 0.0
        
        for stock, weight in weights.items():
            if stock in stock_returns:
                portfolio_return += weight * stock_returns[stock]
        
        return portfolio_return
    
    def calculate_performance_metrics(self, results):
        """성과 지표를 계산합니다."""
        portfolio_values = np.array(results['portfolio_values'])
        returns = np.array(results['returns'])
        
        # 기본 지표
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # 연간 수익률
        days = len(results['dates'])
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # 변동성
        volatility = np.std(returns) * np.sqrt(252)
        
        # 샤프 비율
        risk_free_rate = self.settings['risk_free_rate']
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
        
        # 최대 낙폭
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # 승률
        win_rate = np.mean(returns > 0)
        
        # 베타 (시장 대비)
        market_returns = self._get_market_returns(results['dates'])
        if len(market_returns) > 0:
            beta = np.cov(returns, market_returns)[0, 1] / np.var(market_returns)
            alpha = np.mean(returns) - beta * np.mean(market_returns)
        else:
            beta = 1.0
            alpha = 0.0
        
        # 정보 비율
        information_ratio = alpha / np.std(returns) * np.sqrt(252)
        
        # VaR (95%)
        var_95 = np.percentile(returns, 5)
        
        # CVaR (95%)
        cvar_95 = np.mean(returns[returns <= var_95])
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'cumulative_returns': cumulative_returns,
            'drawdown': drawdown,
            'monthly_returns': self._calculate_monthly_returns(returns, results['dates']),
            'trade_statistics': self._calculate_trade_statistics(results['trades'])
        }
    
    def _get_market_returns(self, dates):
        """시장 수익률을 가져옵니다."""
        # 실제 구현에서는 벤치마크 데이터 조회
        # 여기서는 예시로 랜덤 수익률 반환
        return np.random.normal(0.0005, 0.015, len(dates))
    
    def _calculate_monthly_returns(self, returns, dates):
        """월별 수익률을 계산합니다."""
        # 날짜를 월별로 그룹화
        monthly_data = pd.DataFrame({
            'date': dates,
            'return': returns
        })
        monthly_data['date'] = pd.to_datetime(monthly_data['date'])
        monthly_data['year_month'] = monthly_data['date'].dt.to_period('M')
        
        # 월별 수익률 계산
        monthly_returns = monthly_data.groupby('year_month').apply(
            lambda x: (1 + x['return']).prod() - 1
        )
        
        return monthly_returns
    
    def _calculate_trade_statistics(self, trades):
        """거래 통계를 계산합니다."""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_holding_period': 0.0,
                'total_cost': 0.0
            }
        
        # 거래 수
        total_trades = len(trades)
        
        # 승률 (실제로는 수익/손실 계산 필요)
        win_rate = 0.5  # 예시 값
        
        # 평균 보유 기간
        avg_holding_period = 30.0  # 예시 값
        
        # 총 거래 비용
        total_cost = sum(
            trade['quantity'] * trade['price'] * self.settings['transaction_cost']
            for trade in trades
        )
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_holding_period': avg_holding_period,
            'total_cost': total_cost
        } 
