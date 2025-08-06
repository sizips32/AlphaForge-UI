"""
리포트 페이지
종합 분석 결과 및 투자 전략 가이드를 제공합니다.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def show_page():
    """리포트 페이지를 표시합니다."""
    st.title("📋 리포트")
    st.markdown("종합 분석 결과 및 투자 전략 가이드를 제공합니다.")
    
    # 데이터 확인
    if 'backtest_results' not in st.session_state:
        st.error("❌ 먼저 백테스팅을 완료해주세요.")
        st.info("📊 백테스팅 페이지에서 성과 분석을 완료하세요.")
        return
    
    # 탭 인터페이스
    tab1, tab2, tab3 = st.tabs(["📊 종합 분석", "📈 전략 가이드", "💾 리포트 다운로드"])
    
    with tab1:
        show_comprehensive_analysis()
    
    with tab2:
        show_strategy_guide()
    
    with tab3:
        show_report_download()

def show_comprehensive_analysis():
    """종합 분석 탭을 표시합니다."""
    st.subheader("📊 종합 분석 결과")
    
    # 결과 데이터 가져오기
    backtest_results = st.session_state['backtest_results']
    combination_results = st.session_state['combination_results']
    mining_results = st.session_state['mining_results']
    
    # 분석 요약
    st.markdown("### 🎯 분석 요약")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("생성된 팩터", f"{len(mining_results['factors'])}개")
    
    with col2:
        st.metric("메가-알파 IC", f"{combination_results['performance'].get('ic', 0):.3f}")
    
    with col3:
        st.metric("포트폴리오 수익률", f"{backtest_results['performance'].get('total_return', 0):.1%}")
    
    with col4:
        st.metric("샤프 비율", f"{backtest_results['performance'].get('sharpe_ratio', 0):.2f}")
    
    # 팩터 분석
    st.markdown("### 🧠 팩터 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 팩터 성과 분포
        factor_performance = pd.DataFrame(mining_results['performance']['factor_performance'])
        
        fig_factor_dist = px.histogram(
            factor_performance,
            x='IC',
            title="팩터 IC 분포",
            nbins=20
        )
        fig_factor_dist.update_layout(height=400)
        st.plotly_chart(fig_factor_dist, use_container_width=True, key="report_factor_dist_chart")
    
    with col2:
        # 팩터 유형별 성과
        factor_performance['Type'] = factor_performance['type'].str.capitalize()
        
        fig_factor_type = px.box(
            factor_performance,
            x='Type',
            y='IC',
            title="팩터 유형별 IC 분포"
        )
        fig_factor_type.update_layout(height=400)
        st.plotly_chart(fig_factor_type, use_container_width=True, key="report_factor_type_chart")
    
    # 메가-알파 분석
    st.markdown("### ⚖️ 메가-알파 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 가중치 변화
        if 'weight_history' in combination_results['mega_alpha']:
            weight_history = combination_results['mega_alpha']['weight_history']
            if weight_history:
                weight_df = pd.DataFrame(weight_history)
                
                fig_weights = px.line(
                    weight_df,
                    x='date',
                    y='weight',
                    color='factor',
                    title="시점별 팩터 가중치 변화"
                )
                fig_weights.update_layout(height=400)
                st.plotly_chart(fig_weights, use_container_width=True, key="report_weights_chart")
    
    with col2:
        # 팩터 기여도
        if 'factor_contribution' in combination_results['mega_alpha']:
            contribution_data = combination_results['mega_alpha']['factor_contribution']
            if contribution_data:
                contribution_df = pd.DataFrame(contribution_data)
                
                fig_contribution = px.bar(
                    contribution_df,
                    x='factor',
                    y='contribution',
                    title="팩터별 기여도",
                    color='contribution'
                )
                fig_contribution.update_layout(height=400)
                st.plotly_chart(fig_contribution, use_container_width=True, key="report_contribution_chart")
    
    # 포트폴리오 성과 분석
    st.markdown("### 📈 포트폴리오 성과 분석")
    
    # 성과 지표 테이블
    performance_metrics = backtest_results['performance']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 수익률 지표")
        st.metric("총 수익률", f"{performance_metrics.get('total_return', 0):.1%}")
        st.metric("연간 수익률", f"{performance_metrics.get('annual_return', 0):.1%}")
        st.metric("월평균 수익률", f"{performance_metrics.get('annual_return', 0)/12:.1%}")
    
    with col2:
        st.markdown("#### 🛡️ 리스크 지표")
        st.metric("변동성", f"{performance_metrics.get('volatility', 0):.1%}")
        st.metric("최대 낙폭", f"{performance_metrics.get('max_drawdown', 0):.1%}")
        st.metric("VaR (95%)", f"{performance_metrics.get('var_95', 0):.1%}")
    
    with col3:
        st.markdown("#### 📊 위험조정 지표")
        st.metric("샤프 비율", f"{performance_metrics.get('sharpe_ratio', 0):.2f}")
        st.metric("정보 비율", f"{performance_metrics.get('information_ratio', 0):.2f}")
        st.metric("승률", f"{performance_metrics.get('win_rate', 0):.1%}")
    
    # 성과 차트
    col1, col2 = st.columns(2)
    
    with col1:
        # 누적 수익률
        if 'cumulative_returns' in performance_metrics:
            cumulative_returns = performance_metrics['cumulative_returns']
            
            fig_cumulative = px.line(
                x=range(len(cumulative_returns)),
                y=cumulative_returns,
                title="누적 수익률",
                labels={'x': '거래일', 'y': '누적 수익률'}
            )
            fig_cumulative.update_layout(height=400)
            st.plotly_chart(fig_cumulative, use_container_width=True, key="report_cumulative_chart")
    
    with col2:
        # 낙폭 추이
        if 'drawdown' in performance_metrics:
            drawdown = performance_metrics['drawdown']
            
            fig_drawdown = px.line(
                x=range(len(drawdown)),
                y=drawdown * 100,
                title="낙폭 추이",
                labels={'x': '거래일', 'y': '낙폭 (%)'}
            )
            fig_drawdown.update_layout(height=400)
            fig_drawdown.update_traces(line_color='red')
            st.plotly_chart(fig_drawdown, use_container_width=True, key="report_drawdown_chart")

def show_strategy_guide():
    """전략 가이드 탭을 표시합니다."""
    st.subheader("📈 투자 전략 가이드")
    
    # 전략 요약
    st.markdown("### 🎯 전략 요약")
    
    # 백테스팅 결과 기반 전략 평가
    backtest_results = st.session_state['backtest_results']
    performance = backtest_results['performance']
    
    # 전략 등급 평가
    strategy_grade = evaluate_strategy_grade(performance)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### 📊 전략 등급: **{strategy_grade['grade']}**")
        st.markdown(f"**종합 점수**: {strategy_grade['score']:.1f}/100")
        st.markdown(f"**평가**: {strategy_grade['description']}")
    
    with col2:
        st.markdown("#### 🏆 핵심 강점")
        for strength in strategy_grade['strengths']:
            st.markdown(f"✅ {strength}")
        
        st.markdown("#### ⚠️ 개선 필요")
        for weakness in strategy_grade['weaknesses']:
            st.markdown(f"🔧 {weakness}")
    
    # 투자 권장사항
    st.markdown("### 💡 투자 권장사항")
    
    recommendations = generate_investment_recommendations(performance)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 포트폴리오 구성")
        for rec in recommendations['portfolio']:
            st.markdown(f"• {rec}")
    
    with col2:
        st.markdown("#### 🛡️ 리스크 관리")
        for rec in recommendations['risk_management']:
            st.markdown(f"• {rec}")
    
    # 시장 상황별 대응
    st.markdown("### 🌍 시장 상황별 대응")
    
    market_scenarios = {
        "📈 상승장": "팩터 가중치를 모멘텀 중심으로 조정하고, 리스크 허용도를 높입니다.",
        "📉 하락장": "저변동성 팩터 비중을 높이고, 포지션 크기를 줄입니다.",
        "🔄 횡보장": "밸류 팩터 비중을 높이고, 단기 모멘텀을 활용합니다.",
        "⚡ 변동성 확대": "리스크 관리에 집중하고, 포지션 크기를 대폭 줄입니다."
    }
    
    for scenario, strategy in market_scenarios.items():
        with st.expander(scenario):
            st.markdown(strategy)
    
    # 리밸런싱 전략
    st.markdown("### 🔄 리밸런싱 전략")
    
    rebalancing_strategy = {
        "📅 리밸런싱 주기": "월간 리밸런싱을 권장합니다. 시장 변동성이 클 때는 주간 리밸런싱을 고려하세요.",
        "⚖️ 가중치 조정": "팩터 성과 변화에 따라 동적으로 가중치를 조정합니다.",
        "💰 거래 비용": "거래 비용을 고려하여 최소 1% 이상의 가중치 변화가 있을 때만 리밸런싱합니다.",
        "📊 성과 모니터링": "월별 성과를 모니터링하고, 지속적인 성과 저하 시 전략을 재검토합니다."
    }
    
    for item, description in rebalancing_strategy.items():
        with st.expander(item):
            st.markdown(description)

def show_report_download():
    """리포트 다운로드 탭을 표시합니다."""
    st.subheader("💾 리포트 다운로드")
    
    # 리포트 생성
    report_data = generate_comprehensive_report()
    
    # 리포트 미리보기
    st.markdown("### 📄 리포트 미리보기")
    
    with st.expander("📋 리포트 내용 미리보기"):
        st.markdown(report_data['content'])
    
    # 다운로드 옵션
    st.markdown("### 💾 다운로드 옵션")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # PDF 리포트
        st.markdown("#### 📄 PDF 리포트")
        st.markdown("종합 분석 결과를 포함한 전문 리포트")
        if st.button("📄 PDF 다운로드", use_container_width=True):
            st.info("PDF 생성 기능은 추가 구현이 필요합니다.")
    
    with col2:
        # Excel 리포트
        st.markdown("#### 📊 Excel 리포트")
        st.markdown("상세 데이터와 차트를 포함한 Excel 파일")
        if st.button("📊 Excel 다운로드", use_container_width=True):
            st.info("Excel 생성 기능은 추가 구현이 필요합니다.")
    
    with col3:
        # 요약 리포트
        st.markdown("#### 📋 요약 리포트")
        st.markdown("핵심 성과 지표 요약")
        if st.button("📋 요약 다운로드", use_container_width=True):
            st.info("요약 리포트 생성 기능은 추가 구현이 필요합니다.")
    
    # 데이터 내보내기
    st.markdown("### 📊 데이터 내보내기")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 성과 데이터
        if 'backtest_results' in st.session_state:
            performance_data = st.session_state['backtest_results']['performance']
            performance_df = pd.DataFrame([performance_data])
            
            csv_performance = performance_df.to_csv(index=False)
            st.download_button(
                label="📈 성과 데이터 다운로드 (CSV)",
                data=csv_performance,
                file_name="performance_data.csv",
                mime="text/csv"
            )
    
    with col2:
        # 팩터 데이터
        if 'mining_results' in st.session_state:
            factor_data = st.session_state['mining_results']['performance']['factor_performance']
            factor_df = pd.DataFrame(factor_data)
            
            csv_factors = factor_df.to_csv(index=False)
            st.download_button(
                label="🧠 팩터 데이터 다운로드 (CSV)",
                data=csv_factors,
                file_name="factor_data.csv",
                mime="text/csv"
            )

def evaluate_strategy_grade(performance):
    """전략 등급을 평가합니다."""
    # 점수 계산
    score = 0
    strengths = []
    weaknesses = []
    
    # 수익률 평가 (30점)
    total_return = performance.get('total_return', 0)
    if total_return > 0.2:
        score += 30
        strengths.append("높은 수익률 달성")
    elif total_return > 0.1:
        score += 20
        strengths.append("양호한 수익률")
    elif total_return > 0.05:
        score += 10
    else:
        weaknesses.append("수익률 개선 필요")
    
    # 샤프 비율 평가 (25점)
    sharpe = performance.get('sharpe_ratio', 0)
    if sharpe > 1.5:
        score += 25
        strengths.append("우수한 위험조정 수익률")
    elif sharpe > 1.0:
        score += 20
        strengths.append("양호한 위험조정 수익률")
    elif sharpe > 0.5:
        score += 10
    else:
        weaknesses.append("위험조정 수익률 개선 필요")
    
    # 최대 낙폭 평가 (20점)
    max_dd = abs(performance.get('max_drawdown', 0))
    if max_dd < 0.1:
        score += 20
        strengths.append("낮은 최대 낙폭")
    elif max_dd < 0.15:
        score += 15
        strengths.append("적정 수준의 낙폭")
    elif max_dd < 0.2:
        score += 10
    else:
        weaknesses.append("낙폭 관리 개선 필요")
    
    # 승률 평가 (15점)
    win_rate = performance.get('win_rate', 0)
    if win_rate > 0.6:
        score += 15
        strengths.append("높은 승률")
    elif win_rate > 0.55:
        score += 10
        strengths.append("양호한 승률")
    elif win_rate > 0.5:
        score += 5
    else:
        weaknesses.append("승률 개선 필요")
    
    # 변동성 평가 (10점)
    volatility = performance.get('volatility', 0)
    if volatility < 0.15:
        score += 10
        strengths.append("낮은 변동성")
    elif volatility < 0.2:
        score += 5
    else:
        weaknesses.append("변동성 관리 필요")
    
    # 등급 결정
    if score >= 85:
        grade = "A+"
        description = "매우 우수한 전략"
    elif score >= 75:
        grade = "A"
        description = "우수한 전략"
    elif score >= 65:
        grade = "B+"
        description = "양호한 전략"
    elif score >= 55:
        grade = "B"
        description = "보통 전략"
    elif score >= 45:
        grade = "C+"
        description = "개선 필요"
    else:
        grade = "C"
        description = "대폭 개선 필요"
    
    return {
        'grade': grade,
        'score': score,
        'description': description,
        'strengths': strengths,
        'weaknesses': weaknesses
    }

def generate_investment_recommendations(performance):
    """투자 권장사항을 생성합니다."""
    recommendations = {
        'portfolio': [],
        'risk_management': []
    }
    
    # 포트폴리오 구성 권장사항
    total_return = performance.get('total_return', 0)
    sharpe = performance.get('sharpe_ratio', 0)
    max_dd = abs(performance.get('max_drawdown', 0))
    
    if total_return > 0.15:
        recommendations['portfolio'].append("공격적 포트폴리오 구성 권장")
    elif total_return > 0.1:
        recommendations['portfolio'].append("균형잡힌 포트폴리오 구성 권장")
    else:
        recommendations['portfolio'].append("보수적 포트폴리오 구성 권장")
    
    if sharpe > 1.0:
        recommendations['portfolio'].append("레버리지 활용 고려")
    else:
        recommendations['portfolio'].append("현금 비중 확대 고려")
    
    # 리스크 관리 권장사항
    if max_dd > 0.15:
        recommendations['risk_management'].append("스톱로스 강화 필요")
    else:
        recommendations['risk_management'].append("현재 리스크 관리 적정")
    
    if performance.get('volatility', 0) > 0.2:
        recommendations['risk_management'].append("변동성 헤징 전략 고려")
    else:
        recommendations['risk_management'].append("현재 변동성 수준 적정")
    
    return recommendations

def generate_comprehensive_report():
    """종합 리포트를 생성합니다."""
    # 현재 날짜
    current_date = datetime.now().strftime("%Y년 %m월 %d일")
    
    # 리포트 내용
    content = f"""
# AlphaForge-UI 종합 분석 리포트

**생성일**: {current_date}

## 📊 분석 요약

이 리포트는 AlphaForge 프레임워크를 사용한 알파 팩터 발굴 및 동적 포트폴리오 구축 결과를 종합적으로 분석합니다.

## 🧠 팩터 분석 결과

- **생성된 팩터 수**: {len(st.session_state['mining_results']['factors'])}개
- **평균 IC**: {st.session_state['mining_results']['performance']['avg_ic']:.3f}
- **평균 ICIR**: {st.session_state['mining_results']['performance']['avg_icir']:.2f}

## ⚖️ 메가-알파 성과

- **메가-알파 IC**: {st.session_state['combination_results']['performance'].get('ic', 0):.3f}
- **메가-알파 ICIR**: {st.session_state['combination_results']['performance'].get('icir', 0):.2f}

## 📈 포트폴리오 성과

- **총 수익률**: {st.session_state['backtest_results']['performance'].get('total_return', 0):.1%}
- **연간 수익률**: {st.session_state['backtest_results']['performance'].get('annual_return', 0):.1%}
- **샤프 비율**: {st.session_state['backtest_results']['performance'].get('sharpe_ratio', 0):.2f}
- **최대 낙폭**: {st.session_state['backtest_results']['performance'].get('max_drawdown', 0):.1%}

## 💡 투자 권장사항

1. **포트폴리오 구성**: 시장 상황에 따른 동적 가중치 조정
2. **리스크 관리**: 정기적인 리밸런싱 및 스톱로스 설정
3. **성과 모니터링**: 월별 성과 분석 및 전략 재검토

## ⚠️ 주의사항

- 과거 성과가 미래 성과를 보장하지 않습니다.
- 투자 결정 전 충분한 검토가 필요합니다.
- 시장 상황 변화에 따른 전략 조정이 필요할 수 있습니다.
"""
    
    return {
        'content': content,
        'date': current_date
    }

if __name__ == "__main__":
    show_page() 
