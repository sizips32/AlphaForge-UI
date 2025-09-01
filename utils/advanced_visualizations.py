"""
고급 시각화 및 3D 차트 시스템
인터랙티브 차트, 3D 플롯, 애니메이션 등 제공
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import json
import colorsys
from scipy import stats
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

class ColorPalette:
    """색상 팔레트 관리"""
    
    THEMES = {
        'default': ['#2E86C1', '#E74C3C', '#F39C12', '#27AE60', '#8E44AD', '#F1C40F', '#E67E22', '#34495E'],
        'corporate': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
        'vibrant': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#FFD93D'],
        'dark': ['#00D4AA', '#FF6B9D', '#FFD93D', '#6BCF7F', '#4D96FF', '#FF8C42', '#9B59B6', '#1ABC9C'],
        'gradient': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7']
    }
    
    @classmethod
    def get_colors(cls, theme: str = 'default', n_colors: int = 8) -> List[str]:
        """테마별 색상 반환"""
        colors = cls.THEMES.get(theme, cls.THEMES['default'])
        if n_colors <= len(colors):
            return colors[:n_colors]
        
        # 색상 부족시 그라데이션 생성
        extended_colors = colors.copy()
        base_colors = len(colors)
        for i in range(n_colors - base_colors):
            hue = (i / (n_colors - base_colors)) * 360
            rgb = colorsys.hsv_to_rgb(hue/360, 0.7, 0.9)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
            )
            extended_colors.append(hex_color)
        
        return extended_colors

class Chart3D:
    """3D 차트 생성기"""
    
    def __init__(self, theme: str = 'default'):
        self.theme = theme
        self.colors = ColorPalette.get_colors(theme)
    
    def surface_plot(self, 
                    data: pd.DataFrame, 
                    x_col: str, 
                    y_col: str, 
                    z_col: str,
                    title: str = "3D Surface Plot") -> go.Figure:
        """3D 표면 플롯"""
        
        # 데이터를 격자로 변환
        x_unique = sorted(data[x_col].unique())
        y_unique = sorted(data[y_col].unique())
        
        # 피벗 테이블 생성
        pivot_data = data.pivot_table(values=z_col, index=y_col, columns=x_col, fill_value=0)
        
        fig = go.Figure(data=[go.Surface(
            z=pivot_data.values,
            x=x_unique,
            y=y_unique,
            colorscale='Viridis',
            showscale=True
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600,
            template='plotly_dark' if 'dark' in self.theme else 'plotly_white'
        )
        
        return fig
    
    def scatter_3d(self, 
                   data: pd.DataFrame,
                   x_col: str, 
                   y_col: str, 
                   z_col: str,
                   color_col: str = None,
                   size_col: str = None,
                   title: str = "3D Scatter Plot") -> go.Figure:
        """3D 산점도"""
        
        fig = px.scatter_3d(
            data, 
            x=x_col, 
            y=y_col, 
            z=z_col,
            color=color_col,
            size=size_col,
            title=title,
            color_continuous_scale='Viridis' if color_col else None,
            color_discrete_sequence=self.colors
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2)
                )
            ),
            width=800,
            height=600,
            template='plotly_dark' if 'dark' in self.theme else 'plotly_white'
        )
        
        return fig
    
    def network_3d(self, 
                   nodes: pd.DataFrame,
                   edges: pd.DataFrame,
                   node_id_col: str = 'id',
                   node_label_col: str = 'label',
                   edge_source_col: str = 'source',
                   edge_target_col: str = 'target',
                   title: str = "3D Network Graph") -> go.Figure:
        """3D 네트워크 그래프"""
        
        # NetworkX 그래프 생성
        G = nx.Graph()
        
        # 노드 추가
        for _, node in nodes.iterrows():
            G.add_node(node[node_id_col], label=node.get(node_label_col, ''))
        
        # 엣지 추가
        for _, edge in edges.iterrows():
            G.add_edge(edge[edge_source_col], edge[edge_target_col])
        
        # 3D 레이아웃
        pos_3d = nx.spring_layout(G, dim=3, iterations=50)
        
        # 노드 좌표 추출
        node_trace = go.Scatter3d(
            x=[pos_3d[node][0] for node in G.nodes()],
            y=[pos_3d[node][1] for node in G.nodes()],
            z=[pos_3d[node][2] for node in G.nodes()],
            mode='markers+text',
            text=[G.nodes[node].get('label', str(node)) for node in G.nodes()],
            textposition="middle center",
            marker=dict(
                size=10,
                color=self.colors[0],
                opacity=0.8
            )
        )
        
        # 엣지 좌표
        edge_x, edge_y, edge_z = [], [], []
        for edge in G.edges():
            x0, y0, z0 = pos_3d[edge[0]]
            x1, y1, z1 = pos_3d[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        
        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='rgba(125,125,125,0.5)', width=2),
            hoverinfo='none'
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=title,
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
            width=800,
            height=600,
            template='plotly_dark' if 'dark' in self.theme else 'plotly_white'
        )
        
        return fig

class InteractiveCharts:
    """인터랙티브 차트 생성기"""
    
    def __init__(self, theme: str = 'default'):
        self.theme = theme
        self.colors = ColorPalette.get_colors(theme)
    
    def animated_line_chart(self, 
                           data: pd.DataFrame,
                           x_col: str,
                           y_col: str,
                           animation_frame: str,
                           color_col: str = None,
                           title: str = "Animated Line Chart") -> go.Figure:
        """애니메이션 라인 차트"""
        
        fig = px.line(
            data,
            x=x_col,
            y=y_col,
            animation_frame=animation_frame,
            color=color_col,
            title=title,
            color_discrete_sequence=self.colors
        )
        
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            template='plotly_dark' if 'dark' in self.theme else 'plotly_white',
            hovermode='x unified'
        )
        
        # 애니메이션 설정
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {"args": [None, {"frame": {"duration": 500, "redraw": True}}], "label": "Play", "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0, "redraw": True}}], "label": "Pause", "method": "animate"}
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }]
        )
        
        return fig
    
    def correlation_heatmap(self, 
                          data: pd.DataFrame,
                          title: str = "Correlation Matrix") -> go.Figure:
        """상관관계 히트맵"""
        
        # 숫자형 컬럼만 선택
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            annotation_text=corr_matrix.round(2).values,
            showscale=True,
            colorscale='RdYlBu_r'
        )
        
        fig.update_layout(
            title=title,
            width=600,
            height=500,
            template='plotly_dark' if 'dark' in self.theme else 'plotly_white'
        )
        
        return fig
    
    def parallel_coordinates(self, 
                           data: pd.DataFrame,
                           dimensions: List[str],
                           color_col: str = None,
                           title: str = "Parallel Coordinates") -> go.Figure:
        """평행 좌표계"""
        
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(color=data[color_col] if color_col else self.colors[0],
                         colorscale='Viridis' if color_col else None,
                         showscale=True if color_col else False),
                dimensions=list([
                    dict(range=[data[dim].min(), data[dim].max()],
                         constraintrange=[data[dim].min(), data[dim].max()],
                         label=dim, values=data[dim])
                    for dim in dimensions
                ])
            )
        )
        
        fig.update_layout(
            title=title,
            template='plotly_dark' if 'dark' in self.theme else 'plotly_white'
        )
        
        return fig
    
    def sunburst_chart(self, 
                      data: pd.DataFrame,
                      path_cols: List[str],
                      values_col: str,
                      title: str = "Sunburst Chart") -> go.Figure:
        """선버스트 차트"""
        
        fig = px.sunburst(
            data,
            path=path_cols,
            values=values_col,
            title=title,
            color_discrete_sequence=self.colors
        )
        
        fig.update_layout(
            template='plotly_dark' if 'dark' in self.theme else 'plotly_white'
        )
        
        return fig

class StatisticalCharts:
    """통계적 차트 생성기"""
    
    def __init__(self, theme: str = 'default'):
        self.theme = theme
        self.colors = ColorPalette.get_colors(theme)
    
    def distribution_plot(self, 
                         data: pd.DataFrame,
                         column: str,
                         title: str = "Distribution Plot") -> go.Figure:
        """분포 플롯"""
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('히스토그램', 'Box Plot'),
            vertical_spacing=0.12
        )
        
        # 히스토그램
        fig.add_trace(
            go.Histogram(
                x=data[column],
                name='분포',
                marker_color=self.colors[0],
                opacity=0.7,
                nbinsx=30
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=data[column],
                name='Box Plot',
                marker_color=self.colors[1]
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=title,
            template='plotly_dark' if 'dark' in self.theme else 'plotly_white',
            showlegend=False
        )
        
        return fig
    
    def qq_plot(self, 
               data: pd.DataFrame,
               column: str,
               title: str = "Q-Q Plot") -> go.Figure:
        """Q-Q 플롯"""
        
        # 정규분포와 비교
        (osm, osr), (slope, intercept, r) = stats.probplot(data[column].dropna(), dist="norm")
        
        fig = go.Figure()
        
        # 실제 데이터 포인트
        fig.add_trace(go.Scatter(
            x=osm, 
            y=osr,
            mode='markers',
            name='데이터 포인트',
            marker=dict(color=self.colors[0], size=8)
        ))
        
        # 이론적 라인
        line_x = np.array([osm.min(), osm.max()])
        line_y = slope * line_x + intercept
        fig.add_trace(go.Scatter(
            x=line_x,
            y=line_y,
            mode='lines',
            name=f'이론적 라인 (R²={r**2:.3f})',
            line=dict(color=self.colors[1], width=2)
        ))
        
        fig.update_layout(
            title=f"{title} - {column}",
            xaxis_title="이론적 분위수",
            yaxis_title="실제 분위수",
            template='plotly_dark' if 'dark' in self.theme else 'plotly_white'
        )
        
        return fig
    
    def regression_plot(self, 
                       data: pd.DataFrame,
                       x_col: str,
                       y_col: str,
                       title: str = "Regression Plot") -> go.Figure:
        """회귀 플롯"""
        
        fig = px.scatter(
            data, 
            x=x_col, 
            y=y_col,
            trendline="ols",
            title=title,
            color_discrete_sequence=[self.colors[0]]
        )
        
        # 회귀 통계 계산
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            data[x_col].dropna(), 
            data[y_col].dropna()
        )
        
        # 통계 정보 추가
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=f"R² = {r_value**2:.3f}<br>p = {p_value:.3f}<br>기울기 = {slope:.3f}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)" if 'dark' not in self.theme else "rgba(0,0,0,0.8)",
            bordercolor="black" if 'dark' not in self.theme else "white",
            borderwidth=1
        )
        
        fig.update_layout(
            template='plotly_dark' if 'dark' in self.theme else 'plotly_white'
        )
        
        return fig

class FinancialCharts:
    """금융 차트 생성기"""
    
    def __init__(self, theme: str = 'default'):
        self.theme = theme
        self.colors = ColorPalette.get_colors(theme)
    
    def candlestick_chart(self, 
                         data: pd.DataFrame,
                         date_col: str,
                         open_col: str,
                         high_col: str,
                         low_col: str,
                         close_col: str,
                         volume_col: str = None,
                         title: str = "Candlestick Chart") -> go.Figure:
        """캔들스틱 차트"""
        
        # 서브플롯 생성 (볼륨 포함 여부에 따라)
        if volume_col:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(title, '거래량'),
                row_width=[0.7, 0.3]
            )
        else:
            fig = make_subplots(rows=1, cols=1)
        
        # 캔들스틱
        fig.add_trace(
            go.Candlestick(
                x=data[date_col],
                open=data[open_col],
                high=data[high_col],
                low=data[low_col],
                close=data[close_col],
                name="OHLC",
                increasing_line_color=self.colors[2],  # 상승
                decreasing_line_color=self.colors[3]   # 하락
            ),
            row=1, col=1
        )
        
        # 볼륨 (있는 경우)
        if volume_col:
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(data[close_col], data[open_col])]
            
            fig.add_trace(
                go.Bar(
                    x=data[date_col],
                    y=data[volume_col],
                    name="Volume",
                    marker_color=colors
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            template='plotly_dark' if 'dark' in self.theme else 'plotly_white',
            xaxis_rangeslider_visible=False,
            showlegend=False if volume_col else True
        )
        
        return fig
    
    def portfolio_performance(self, 
                             data: pd.DataFrame,
                             date_col: str,
                             returns_col: str,
                             benchmark_col: str = None,
                             title: str = "Portfolio Performance") -> go.Figure:
        """포트폴리오 성과 차트"""
        
        # 누적 수익률 계산
        data['cumulative_returns'] = (1 + data[returns_col]).cumprod()
        
        fig = go.Figure()
        
        # 포트폴리오 수익률
        fig.add_trace(go.Scatter(
            x=data[date_col],
            y=data['cumulative_returns'],
            mode='lines',
            name='포트폴리오',
            line=dict(color=self.colors[0], width=2)
        ))
        
        # 벤치마크 (있는 경우)
        if benchmark_col and benchmark_col in data.columns:
            data['benchmark_cumulative'] = (1 + data[benchmark_col]).cumprod()
            fig.add_trace(go.Scatter(
                x=data[date_col],
                y=data['benchmark_cumulative'],
                mode='lines',
                name='벤치마크',
                line=dict(color=self.colors[1], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="날짜",
            yaxis_title="누적 수익률",
            template='plotly_dark' if 'dark' in self.theme else 'plotly_white',
            hovermode='x unified'
        )
        
        return fig

class DimensionalityReduction:
    """차원 축소 시각화"""
    
    def __init__(self, theme: str = 'default'):
        self.theme = theme
        self.colors = ColorPalette.get_colors(theme)
    
    def pca_plot(self, 
                data: pd.DataFrame,
                features: List[str],
                target_col: str = None,
                title: str = "PCA Plot") -> Tuple[go.Figure, pd.DataFrame]:
        """PCA 플롯"""
        
        # PCA 수행
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data[features])
        
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        pca_df['explained_variance'] = [
            f"PC1: {pca.explained_variance_ratio_[0]:.1%}", 
            f"PC2: {pca.explained_variance_ratio_[1]:.1%}"
        ]
        
        if target_col:
            pca_df[target_col] = data[target_col].values
        
        # 플롯 생성
        if target_col:
            fig = px.scatter(
                pca_df, x='PC1', y='PC2',
                color=target_col,
                title=f"{title} (총 분산 설명: {sum(pca.explained_variance_ratio_):.1%})",
                color_discrete_sequence=self.colors
            )
        else:
            fig = px.scatter(
                pca_df, x='PC1', y='PC2',
                title=f"{title} (총 분산 설명: {sum(pca.explained_variance_ratio_):.1%})",
                color_discrete_sequence=[self.colors[0]]
            )
        
        fig.update_layout(
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
            template='plotly_dark' if 'dark' in self.theme else 'plotly_white'
        )
        
        return fig, pca_df
    
    def tsne_plot(self, 
                 data: pd.DataFrame,
                 features: List[str],
                 target_col: str = None,
                 perplexity: int = 30,
                 title: str = "t-SNE Plot") -> Tuple[go.Figure, pd.DataFrame]:
        """t-SNE 플롯"""
        
        # t-SNE 수행
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_result = tsne.fit_transform(data[features])
        
        tsne_df = pd.DataFrame(data=tsne_result, columns=['t-SNE1', 't-SNE2'])
        
        if target_col:
            tsne_df[target_col] = data[target_col].values
        
        # 플롯 생성
        if target_col:
            fig = px.scatter(
                tsne_df, x='t-SNE1', y='t-SNE2',
                color=target_col,
                title=f"{title} (Perplexity: {perplexity})",
                color_discrete_sequence=self.colors
            )
        else:
            fig = px.scatter(
                tsne_df, x='t-SNE1', y='t-SNE2',
                title=f"{title} (Perplexity: {perplexity})",
                color_discrete_sequence=[self.colors[0]]
            )
        
        fig.update_layout(
            template='plotly_dark' if 'dark' in self.theme else 'plotly_white'
        )
        
        return fig, tsne_df