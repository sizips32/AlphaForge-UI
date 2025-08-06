"""
야후 파이낸스 데이터 다운로더
yfinance 라이브러리를 사용하여 주가 데이터를 다운로드합니다.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from typing import List, Dict, Optional, Tuple
import time
import logging
from utils.env_manager import get_api_key

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YahooFinanceDownloader:
    """야후 파이낸스에서 주가 데이터를 다운로드하는 클래스"""
    
    def __init__(self) -> None:
        """초기화"""
        self.session = None
        self._setup_session()
    
    def _setup_session(self) -> None:
        """yfinance 세션 설정"""
        try:
            # API 키 확인
            self.api_key = get_api_key('yahoo_finance')
            if self.api_key:
                logger.info("Yahoo Finance API 키가 설정되었습니다.")
            
            # yfinance 세션 설정 (재시도 및 타임아웃 설정)
            import requests
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            # API 키가 있으면 헤더에 추가
            if self.api_key:
                session.headers.update({
                    'X-API-Key': self.api_key
                })
            
            # yfinance에 세션 적용
            self.session = session
            
        except Exception as e:
            logger.warning(f"세션 설정 실패: {e}")
            self.session = None
    
    def validate_ticker(self, ticker: str) -> bool:
        """
        티커 심볼의 유효성을 검증합니다.
        
        Args:
            ticker (str): 검증할 티커 심볼
            
        Returns:
            bool: 유효한 티커인지 여부
        """
        try:
            # 티커 심볼 정리
            ticker = ticker.strip().upper()
            
            # 기본 검증
            if not ticker or len(ticker) > 10:
                return False
            
            # yfinance로 티커 정보 확인
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            # 기본 정보가 있는지 확인 (더 안정적인 검증)
            if not info:
                return False
            
            # 시장 가격이나 종목명이 있는지 확인
            has_price = 'regularMarketPrice' in info and info['regularMarketPrice'] is not None
            has_name = 'longName' in info or 'shortName' in info
            
            return has_price or has_name
            
        except Exception as e:
            logger.warning(f"티커 검증 실패 {ticker}: {e}")
            return False
    
    def get_ticker_info(self, ticker: str) -> Optional[Dict]:
        """
        티커의 기본 정보를 가져옵니다.
        
        Args:
            ticker (str): 티커 심볼
            
        Returns:
            Optional[Dict]: 티커 정보 또는 None
        """
        try:
            ticker = ticker.strip().upper()
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            if not info:
                return None
            
            # 필요한 정보만 추출
            ticker_info = {
                'symbol': ticker,
                'name': info.get('longName', info.get('shortName', ticker)),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown'),
                'country': info.get('country', 'Unknown')
            }
            
            return ticker_info
            
        except Exception as e:
            logger.error(f"티커 정보 가져오기 실패 {ticker}: {e}")
            return None
    
    def download_single_ticker(self, 
                              ticker: str, 
                              start_date: str, 
                              end_date: str,
                              progress_bar=None) -> Optional[pd.DataFrame]:
        """
        단일 티커의 주가 데이터를 다운로드합니다.
        
        Args:
            ticker (str): 티커 심볼
            start_date (str): 시작 날짜 (YYYY-MM-DD)
            end_date (str): 종료 날짜 (YYYY-MM-DD)
            progress_bar: 진행률 표시용 (Streamlit progress bar)
            
        Returns:
            Optional[pd.DataFrame]: 주가 데이터 또는 None
        """
        try:
            ticker = ticker.strip().upper()
            
            # 진행률 업데이트
            if progress_bar:
                progress_bar.progress(0.1, text=f"{ticker} 데이터 다운로드 중...")
            
            # yfinance 티커 객체 생성
            ticker_obj = yf.Ticker(ticker)
            
            # 데이터 다운로드
            if progress_bar:
                progress_bar.progress(0.3, text=f"{ticker} 데이터 가져오는 중...")
            
            data = ticker_obj.history(
                start=start_date,
                end=end_date,
                interval='1d',
                auto_adjust=True,  # 배당금 및 분할 조정
                prepost=False      # 장 전후 거래 제외
            )
            
            if progress_bar:
                progress_bar.progress(0.7, text=f"{ticker} 데이터 처리 중...")
            
            # 데이터가 비어있는지 확인
            if data.empty:
                logger.warning(f"{ticker}: 데이터가 없습니다.")
                return None
            
            # 컬럼명 표준화
            data = data.reset_index()
            
            # yfinance에서 반환되는 컬럼명에 따라 매핑
            column_mapping = {
                'Date': 'Date',
                'Open': 'Open',
                'High': 'High', 
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            }
            
            # 실제 컬럼명에 맞게 매핑
            if 'Adj Close' in data.columns:
                column_mapping['Adj Close'] = 'Close'  # 조정된 종가를 사용
            
            # 컬럼명 변경
            data = data.rename(columns=column_mapping)
            
            # 필요한 컬럼만 선택
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            available_columns = [col for col in required_columns if col in data.columns]
            data = data[available_columns]
            
            # Ticker 컬럼 추가
            data['Ticker'] = ticker
            
            # 날짜 형식 통일
            data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
            
            # 데이터 타입 변환
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # 결측치 제거
            data = data.dropna()
            
            if progress_bar:
                progress_bar.progress(1.0, text=f"{ticker} 완료!")
            
            logger.info(f"{ticker}: {len(data)} 행 다운로드 완료")
            return data
            
        except Exception as e:
            logger.error(f"{ticker} 다운로드 실패: {e}")
            if progress_bar:
                progress_bar.progress(1.0, text=f"{ticker} 실패!")
            return None
    
    def download_multiple_tickers(self, 
                                 tickers: List[str], 
                                 start_date: str, 
                                 end_date: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        여러 티커의 주가 데이터를 다운로드합니다.
        
        Args:
            tickers (List[str]): 티커 심볼 리스트
            start_date (str): 시작 날짜 (YYYY-MM-DD)
            end_date (str): 종료 날짜 (YYYY-MM-DD)
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: (데이터프레임, 실패한 티커 리스트)
        """
        all_data = []
        failed_tickers = []
        
        # 진행률 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_tickers = len(tickers)
        
        for i, ticker in enumerate(tickers):
            try:
                # 진행률 업데이트
                progress = (i + 1) / total_tickers
                status_text.text(f"다운로드 중... ({i+1}/{total_tickers}) - {ticker}")
                progress_bar.progress(progress)
                
                # 단일 티커 다운로드
                data = self.download_single_ticker(ticker, start_date, end_date)
                
                if data is not None and not data.empty:
                    all_data.append(data)
                else:
                    failed_tickers.append(ticker)
                
                # API 호출 간격 조절 (야후 파이낸스 제한 방지)
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"{ticker} 처리 중 오류: {e}")
                failed_tickers.append(ticker)
        
        # 진행률 완료
        progress_bar.progress(1.0)
        status_text.text("다운로드 완료!")
        
        # 데이터 결합
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values(['Date', 'Ticker']).reset_index(drop=True)
        else:
            combined_data = pd.DataFrame()
        
        return combined_data, failed_tickers
    
    def get_popular_tickers(self) -> List[Dict]:
        """
        인기 있는 티커들의 목록을 반환합니다.
        
        Returns:
            List[Dict]: 인기 티커 정보 리스트
        """
        popular_tickers = [
            # 미국 대형주
            {'symbol': 'AAPL', 'name': 'Apple Inc.', 'category': 'Technology'},
            {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'category': 'Technology'},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'category': 'Technology'},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'category': 'Consumer Discretionary'},
            {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'category': 'Automotive'},
            {'symbol': 'META', 'name': 'Meta Platforms Inc.', 'category': 'Technology'},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'category': 'Technology'},
            {'symbol': 'NFLX', 'name': 'Netflix Inc.', 'category': 'Communication Services'},
            {'symbol': 'ADBE', 'name': 'Adobe Inc.', 'category': 'Technology'},
            {'symbol': 'CRM', 'name': 'Salesforce Inc.', 'category': 'Technology'},
            
            # 한국 주식 (ADR)
            {'symbol': '005930.KS', 'name': 'Samsung Electronics', 'category': 'Technology'},
            {'symbol': '000660.KS', 'name': 'SK Hynix', 'category': 'Technology'},
            {'symbol': '035420.KS', 'name': 'NAVER', 'category': 'Technology'},
            {'symbol': '051910.KS', 'name': 'LG Chem', 'category': 'Materials'},
            {'symbol': '006400.KS', 'name': 'Samsung SDI', 'category': 'Technology'},
            
            # ETF
            {'symbol': 'SPY', 'name': 'SPDR S&P 500 ETF', 'category': 'ETF'},
            {'symbol': 'QQQ', 'name': 'Invesco QQQ Trust', 'category': 'ETF'},
            {'symbol': 'IWM', 'name': 'iShares Russell 2000 ETF', 'category': 'ETF'},
            {'symbol': 'VTI', 'name': 'Vanguard Total Stock Market ETF', 'category': 'ETF'},
            {'symbol': 'VEA', 'name': 'Vanguard FTSE Developed Markets ETF', 'category': 'ETF'}
        ]
        
        return popular_tickers
    
    def get_market_indices(self) -> List[Dict]:
        """
        주요 시장 지수들의 목록을 반환합니다.
        
        Returns:
            List[Dict]: 시장 지수 정보 리스트
        """
        indices = [
            {'symbol': '^GSPC', 'name': 'S&P 500', 'category': 'Index'},
            {'symbol': '^DJI', 'name': 'Dow Jones Industrial Average', 'category': 'Index'},
            {'symbol': '^IXIC', 'name': 'NASDAQ Composite', 'category': 'Index'},
            {'symbol': '^RUT', 'name': 'Russell 2000', 'category': 'Index'},
            {'symbol': '^VIX', 'name': 'CBOE Volatility Index', 'category': 'Index'},
            {'symbol': '^KS11', 'name': 'KOSPI', 'category': 'Index'},
            {'symbol': '^KQ11', 'name': 'KOSDAQ', 'category': 'Index'},
            {'symbol': '^N225', 'name': 'Nikkei 225', 'category': 'Index'},
            {'symbol': '^GDAXI', 'name': 'DAX', 'category': 'Index'},
            {'symbol': '^FTSE', 'name': 'FTSE 100', 'category': 'Index'}
        ]
        
        return indices
    
    def validate_date_range(self, start_date: str, end_date: str) -> Tuple[bool, str]:
        """
        날짜 범위의 유효성을 검증합니다.
        
        Args:
            start_date (str): 시작 날짜
            end_date (str): 종료 날짜
            
        Returns:
            Tuple[bool, str]: (유효성, 오류 메시지)
        """
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            if start > end:
                return False, "시작 날짜가 종료 날짜보다 늦습니다."
            
            if start > datetime.now():
                return False, "시작 날짜가 미래입니다."
            
            # 최대 10년 데이터 제한
            if (end - start).days > 3650:
                return False, "최대 10년 데이터만 다운로드 가능합니다."
            
            return True, ""
            
        except ValueError:
            return False, "날짜 형식이 올바르지 않습니다. (YYYY-MM-DD)"
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict:
        """
        다운로드된 데이터의 요약 정보를 반환합니다.
        
        Args:
            data (pd.DataFrame): 주가 데이터
            
        Returns:
            Dict: 데이터 요약 정보
        """
        if data.empty:
            return {}
        
        summary = {
            'total_rows': len(data),
            'unique_tickers': data['Ticker'].nunique(),
            'date_range': f"{data['Date'].min()} ~ {data['Date'].max()}",
            'avg_close': data['Close'].mean(),
            'total_volume': data['Volume'].sum(),
            'data_completeness': (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        }
        
        return summary 
