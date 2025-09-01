"""
고급 검색 및 필터링 시스템
다중 조건 필터링, 자연어 검색, 스마트 필터 제공
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import re
from dataclasses import dataclass
from enum import Enum
import json
from fuzzywuzzy import fuzz, process
import operator
from functools import reduce

class FilterOperator(Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    REGEX = "regex"
    FUZZY_MATCH = "fuzzy_match"

class LogicOperator(Enum):
    AND = "and"
    OR = "or"
    NOT = "not"

@dataclass
class FilterCondition:
    """필터 조건 클래스"""
    column: str
    operator: FilterOperator
    value: Any
    logic: LogicOperator = LogicOperator.AND
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'column': self.column,
            'operator': self.operator.value,
            'value': self.value,
            'logic': self.logic.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FilterCondition':
        return cls(
            column=data['column'],
            operator=FilterOperator(data['operator']),
            value=data['value'],
            logic=LogicOperator(data.get('logic', 'and'))
        )

class AdvancedFilter:
    """고급 필터링 시스템"""
    
    def __init__(self):
        self.conditions: List[FilterCondition] = []
        self.operator_functions = self._setup_operators()
    
    def _setup_operators(self) -> Dict[FilterOperator, Callable]:
        """연산자 함수 매핑"""
        return {
            FilterOperator.EQUALS: lambda col, val: col == val,
            FilterOperator.NOT_EQUALS: lambda col, val: col != val,
            FilterOperator.GREATER_THAN: lambda col, val: col > val,
            FilterOperator.LESS_THAN: lambda col, val: col < val,
            FilterOperator.GREATER_EQUAL: lambda col, val: col >= val,
            FilterOperator.LESS_EQUAL: lambda col, val: col <= val,
            FilterOperator.CONTAINS: lambda col, val: col.astype(str).str.contains(str(val), case=False, na=False),
            FilterOperator.NOT_CONTAINS: lambda col, val: ~col.astype(str).str.contains(str(val), case=False, na=False),
            FilterOperator.STARTS_WITH: lambda col, val: col.astype(str).str.startswith(str(val), na=False),
            FilterOperator.ENDS_WITH: lambda col, val: col.astype(str).str.endswith(str(val), na=False),
            FilterOperator.IN: lambda col, val: col.isin(val if isinstance(val, list) else [val]),
            FilterOperator.NOT_IN: lambda col, val: ~col.isin(val if isinstance(val, list) else [val]),
            FilterOperator.BETWEEN: lambda col, val: (col >= val[0]) & (col <= val[1]) if len(val) == 2 else col == col,
            FilterOperator.IS_NULL: lambda col, val: col.isna(),
            FilterOperator.IS_NOT_NULL: lambda col, val: col.notna(),
            FilterOperator.REGEX: lambda col, val: col.astype(str).str.match(str(val), na=False),
            FilterOperator.FUZZY_MATCH: lambda col, val: self._fuzzy_match(col, val)
        }
    
    def _fuzzy_match(self, column: pd.Series, value: str, threshold: int = 80) -> pd.Series:
        """퍼지 매칭"""
        def fuzzy_score(text):
            if pd.isna(text):
                return False
            return fuzz.partial_ratio(str(text).lower(), str(value).lower()) >= threshold
        
        return column.apply(fuzzy_score)
    
    def add_condition(self, condition: FilterCondition):
        """조건 추가"""
        self.conditions.append(condition)
    
    def remove_condition(self, index: int):
        """조건 제거"""
        if 0 <= index < len(self.conditions):
            self.conditions.pop(index)
    
    def clear_conditions(self):
        """모든 조건 제거"""
        self.conditions.clear()
    
    def apply_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """필터 적용"""
        if not self.conditions:
            return data
        
        masks = []
        current_logic = LogicOperator.AND
        
        for condition in self.conditions:
            if condition.column not in data.columns:
                continue
                
            column_data = data[condition.column]
            operator_func = self.operator_functions.get(condition.operator)
            
            if operator_func:
                try:
                    mask = operator_func(column_data, condition.value)
                    
                    if current_logic == LogicOperator.AND:
                        masks.append(('and', mask))
                    elif current_logic == LogicOperator.OR:
                        masks.append(('or', mask))
                    else:  # NOT
                        masks.append(('and', ~mask))
                    
                    current_logic = condition.logic
                
                except Exception as e:
                    st.warning(f"필터 조건 적용 실패: {condition.column} {condition.operator.value} {condition.value} - {e}")
        
        if not masks:
            return data
        
        # 마스크 결합
        combined_mask = masks[0][1]
        for logic_op, mask in masks[1:]:
            if logic_op == 'and':
                combined_mask = combined_mask & mask
            else:  # or
                combined_mask = combined_mask | mask
        
        return data[combined_mask].copy()

class SmartSearch:
    """스마트 검색 시스템"""
    
    def __init__(self):
        self.search_history: List[str] = []
        self.column_weights = {}
    
    def set_column_weights(self, weights: Dict[str, float]):
        """컬럼별 가중치 설정"""
        self.column_weights = weights
    
    def natural_language_search(self, data: pd.DataFrame, query: str) -> pd.DataFrame:
        """자연어 검색"""
        if not query.strip():
            return data
        
        # 검색 기록 저장
        if query not in self.search_history:
            self.search_history.append(query)
            if len(self.search_history) > 50:  # 최근 50개만 유지
                self.search_history.pop(0)
        
        # 숫자 패턴 감지
        number_patterns = re.findall(r'[><=]+\s*\d+\.?\d*', query)
        date_patterns = re.findall(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}', query)
        
        search_masks = []
        search_columns = data.select_dtypes(include=['object', 'string']).columns
        
        # 텍스트 검색
        query_terms = query.lower().split()
        for term in query_terms:
            if len(term) < 2:  # 너무 짧은 단어 제외
                continue
                
            column_masks = []
            for col in search_columns:
                try:
                    weight = self.column_weights.get(col, 1.0)
                    mask = data[col].astype(str).str.contains(term, case=False, na=False)
                    
                    if weight > 1.0:  # 가중치가 높은 컬럼은 여러 번 추가
                        column_masks.extend([mask] * int(weight))
                    else:
                        column_masks.append(mask)
                        
                except Exception:
                    continue
            
            if column_masks:
                # OR 연산으로 결합
                term_mask = reduce(operator.or_, column_masks)
                search_masks.append(term_mask)
        
        # 숫자 조건 처리
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for pattern in number_patterns:
            self._apply_numeric_pattern(data, pattern, search_masks, numeric_columns)
        
        # 최종 결합 (AND 연산)
        if search_masks:
            final_mask = reduce(operator.and_, search_masks)
            return data[final_mask].copy()
        
        return data
    
    def _apply_numeric_pattern(self, data: pd.DataFrame, pattern: str, 
                             search_masks: List[pd.Series], numeric_columns: pd.Index):
        """숫자 패턴 적용"""
        try:
            if '>=' in pattern:
                operator_str, value_str = pattern.split('>=')
                value = float(value_str.strip())
                for col in numeric_columns:
                    mask = data[col] >= value
                    search_masks.append(mask)
            elif '<=' in pattern:
                operator_str, value_str = pattern.split('<=')
                value = float(value_str.strip())
                for col in numeric_columns:
                    mask = data[col] <= value
                    search_masks.append(mask)
            elif '>' in pattern:
                operator_str, value_str = pattern.split('>')
                value = float(value_str.strip())
                for col in numeric_columns:
                    mask = data[col] > value
                    search_masks.append(mask)
            elif '<' in pattern:
                operator_str, value_str = pattern.split('<')
                value = float(value_str.strip())
                for col in numeric_columns:
                    mask = data[col] < value
                    search_masks.append(mask)
        except (ValueError, IndexError):
            pass
    
    def fuzzy_search(self, data: pd.DataFrame, query: str, 
                    columns: List[str] = None, threshold: int = 80) -> pd.DataFrame:
        """퍼지 검색"""
        if not query.strip():
            return data
        
        if columns is None:
            columns = data.select_dtypes(include=['object', 'string']).columns.tolist()
        
        masks = []
        for col in columns:
            if col in data.columns:
                try:
                    mask = data[col].apply(
                        lambda x: fuzz.partial_ratio(str(x).lower(), query.lower()) >= threshold
                        if pd.notna(x) else False
                    )
                    masks.append(mask)
                except Exception:
                    continue
        
        if masks:
            final_mask = reduce(operator.or_, masks)
            return data[final_mask].copy()
        
        return data

class FilterBuilder:
    """필터 빌더 UI 컴포넌트"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.filter_system = AdvancedFilter()
        self.search_system = SmartSearch()
    
    def create_filter_interface(self) -> pd.DataFrame:
        """필터 인터페이스 생성"""
        st.markdown("### 🔍 고급 검색 & 필터")
        
        # 탭 인터페이스
        tab1, tab2, tab3, tab4 = st.tabs(["🔎 스마트 검색", "🎯 필터 빌더", "📊 빠른 필터", "📈 저장된 필터"])
        
        filtered_data = self.data.copy()
        
        with tab1:
            filtered_data = self._smart_search_tab(filtered_data)
        
        with tab2:
            filtered_data = self._filter_builder_tab(filtered_data)
        
        with tab3:
            filtered_data = self._quick_filters_tab(filtered_data)
        
        with tab4:
            self._saved_filters_tab()
        
        return filtered_data
    
    def _smart_search_tab(self, data: pd.DataFrame) -> pd.DataFrame:
        """스마트 검색 탭"""
        st.markdown("#### 자연어 검색")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "검색어를 입력하세요",
                placeholder="예: 수익률 > 10, 2023년 데이터, high volatility",
                help="자연어로 검색하거나 조건을 입력하세요 (예: > 100, < 50, contains ABC)"
            )
        
        with col2:
            search_type = st.selectbox(
                "검색 유형",
                ["자연어", "퍼지 검색", "정확 검색"]
            )
        
        if search_query:
            if search_type == "자연어":
                filtered_data = self.search_system.natural_language_search(data, search_query)
            elif search_type == "퍼지 검색":
                threshold = st.slider("유사도 임계값", 50, 100, 80)
                filtered_data = self.search_system.fuzzy_search(data, search_query, threshold=threshold)
            else:  # 정확 검색
                filtered_data = self._exact_search(data, search_query)
            
            # 검색 결과 정보
            st.info(f"검색 결과: {len(filtered_data):,}개 (전체 {len(data):,}개 중)")
            
            return filtered_data
        
        return data
    
    def _filter_builder_tab(self, data: pd.DataFrame) -> pd.DataFrame:
        """필터 빌더 탭"""
        st.markdown("#### 조건별 필터")
        
        # 새 조건 추가
        with st.expander("➕ 새 조건 추가", expanded=len(self.filter_system.conditions) == 0):
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            
            with col1:
                column = st.selectbox("컬럼", data.columns.tolist(), key="filter_column")
            
            with col2:
                # 컬럼 타입에 따른 연산자 선택
                if data[column].dtype in ['object', 'string']:
                    available_ops = [
                        FilterOperator.CONTAINS, FilterOperator.NOT_CONTAINS,
                        FilterOperator.EQUALS, FilterOperator.NOT_EQUALS,
                        FilterOperator.STARTS_WITH, FilterOperator.ENDS_WITH,
                        FilterOperator.IN, FilterOperator.REGEX, FilterOperator.FUZZY_MATCH
                    ]
                else:
                    available_ops = [
                        FilterOperator.EQUALS, FilterOperator.NOT_EQUALS,
                        FilterOperator.GREATER_THAN, FilterOperator.LESS_THAN,
                        FilterOperator.GREATER_EQUAL, FilterOperator.LESS_EQUAL,
                        FilterOperator.BETWEEN, FilterOperator.IN
                    ]
                
                operator = st.selectbox(
                    "연산자",
                    available_ops,
                    format_func=lambda x: self._operator_display_name(x),
                    key="filter_operator"
                )
            
            with col3:
                value = self._create_value_input(data, column, operator)
            
            with col4:
                logic = st.selectbox(
                    "논리",
                    [LogicOperator.AND, LogicOperator.OR],
                    format_func=lambda x: x.value.upper(),
                    key="filter_logic"
                )
                
                if st.button("추가", key="add_condition"):
                    condition = FilterCondition(column, operator, value, logic)
                    self.filter_system.add_condition(condition)
                    st.experimental_rerun()
        
        # 현재 조건 표시
        if self.filter_system.conditions:
            st.markdown("#### 현재 필터 조건")
            
            for i, condition in enumerate(self.filter_system.conditions):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    logic_text = "" if i == 0 else f"{condition.logic.value.upper()} "
                    st.write(f"{logic_text}{condition.column} {self._operator_display_name(condition.operator)} {condition.value}")
                
                with col2:
                    if st.button("삭제", key=f"remove_{i}"):
                        self.filter_system.remove_condition(i)
                        st.experimental_rerun()
            
            # 필터 적용
            filtered_data = self.filter_system.apply_filter(data)
            st.success(f"필터 적용 결과: {len(filtered_data):,}개")
            
            # 조건 초기화 버튼
            if st.button("모든 조건 삭제"):
                self.filter_system.clear_conditions()
                st.experimental_rerun()
            
            return filtered_data
        
        return data
    
    def _quick_filters_tab(self, data: pd.DataFrame) -> pd.DataFrame:
        """빠른 필터 탭"""
        st.markdown("#### 빠른 필터")
        
        filtered_data = data.copy()
        
        # 숫자형 컬럼 필터
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.markdown("##### 📊 숫자형 데이터")
            
            cols = st.columns(min(3, len(numeric_cols)))
            for i, col in enumerate(numeric_cols[:6]):  # 최대 6개
                with cols[i % 3]:
                    min_val = float(data[col].min())
                    max_val = float(data[col].max())
                    
                    if min_val != max_val:
                        selected_range = st.slider(
                            col,
                            min_val, max_val, (min_val, max_val),
                            key=f"quick_numeric_{col}"
                        )
                        
                        if selected_range != (min_val, max_val):
                            filtered_data = filtered_data[
                                (filtered_data[col] >= selected_range[0]) & 
                                (filtered_data[col] <= selected_range[1])
                            ]
        
        # 카테고리형 컬럼 필터
        categorical_cols = data.select_dtypes(include=['object', 'string']).columns
        if len(categorical_cols) > 0:
            st.markdown("##### 📝 카테고리 데이터")
            
            for col in categorical_cols[:4]:  # 최대 4개
                unique_values = sorted(data[col].dropna().unique())
                if len(unique_values) <= 50:  # 너무 많은 값은 제외
                    selected_values = st.multiselect(
                        col,
                        unique_values,
                        default=unique_values,
                        key=f"quick_cat_{col}"
                    )
                    
                    if selected_values != unique_values:
                        filtered_data = filtered_data[filtered_data[col].isin(selected_values)]
        
        # 날짜 필터 (날짜형 컬럼이 있는 경우)
        date_cols = data.select_dtypes(include=['datetime64', 'datetime']).columns
        if len(date_cols) > 0:
            st.markdown("##### 📅 날짜 필터")
            
            date_col = st.selectbox("날짜 컬럼", date_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "시작 날짜",
                    value=data[date_col].min(),
                    key="quick_start_date"
                )
            
            with col2:
                end_date = st.date_input(
                    "종료 날짜",
                    value=data[date_col].max(),
                    key="quick_end_date"
                )
            
            filtered_data = filtered_data[
                (filtered_data[date_col] >= pd.Timestamp(start_date)) &
                (filtered_data[date_col] <= pd.Timestamp(end_date))
            ]
        
        if len(filtered_data) != len(data):
            st.success(f"빠른 필터 결과: {len(filtered_data):,}개")
        
        return filtered_data
    
    def _saved_filters_tab(self):
        """저장된 필터 탭"""
        st.markdown("#### 저장된 필터")
        
        # 현재 필터 저장
        if self.filter_system.conditions:
            filter_name = st.text_input("필터 이름", placeholder="내 필터")
            
            if st.button("현재 필터 저장") and filter_name:
                self._save_filter(filter_name)
                st.success(f"필터 '{filter_name}'가 저장되었습니다!")
        
        # 저장된 필터 목록
        saved_filters = self._load_saved_filters()
        if saved_filters:
            st.markdown("##### 저장된 필터 목록")
            
            for name, conditions_data in saved_filters.items():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**{name}** ({len(conditions_data)} 조건)")
                
                with col2:
                    if st.button("불러오기", key=f"load_{name}"):
                        self._load_filter(conditions_data)
                        st.success(f"필터 '{name}'를 불러왔습니다!")
                        st.experimental_rerun()
                
                with col3:
                    if st.button("삭제", key=f"delete_{name}"):
                        self._delete_filter(name)
                        st.success(f"필터 '{name}'를 삭제했습니다!")
                        st.experimental_rerun()
    
    def _operator_display_name(self, operator: FilterOperator) -> str:
        """연산자 표시 이름"""
        display_names = {
            FilterOperator.EQUALS: "같음",
            FilterOperator.NOT_EQUALS: "같지 않음",
            FilterOperator.GREATER_THAN: "보다 큼",
            FilterOperator.LESS_THAN: "보다 작음",
            FilterOperator.GREATER_EQUAL: "이상",
            FilterOperator.LESS_EQUAL: "이하",
            FilterOperator.CONTAINS: "포함",
            FilterOperator.NOT_CONTAINS: "포함하지 않음",
            FilterOperator.STARTS_WITH: "시작",
            FilterOperator.ENDS_WITH: "끝남",
            FilterOperator.IN: "포함됨",
            FilterOperator.NOT_IN: "포함되지 않음",
            FilterOperator.BETWEEN: "사이",
            FilterOperator.IS_NULL: "비어있음",
            FilterOperator.IS_NOT_NULL: "비어있지 않음",
            FilterOperator.REGEX: "정규식",
            FilterOperator.FUZZY_MATCH: "유사 매칭"
        }
        return display_names.get(operator, operator.value)
    
    def _create_value_input(self, data: pd.DataFrame, column: str, operator: FilterOperator):
        """값 입력 위젯 생성"""
        if operator in [FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL]:
            return None
        
        if operator == FilterOperator.BETWEEN:
            if data[column].dtype in [np.number]:
                min_val = float(data[column].min())
                max_val = float(data[column].max())
                return st.slider("범위", min_val, max_val, (min_val, max_val), key="filter_value")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    start = st.text_input("시작", key="filter_start")
                with col2:
                    end = st.text_input("끝", key="filter_end")
                return [start, end]
        
        elif operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
            unique_values = data[column].dropna().unique().tolist()[:100]  # 최대 100개
            return st.multiselect("값 선택", unique_values, key="filter_value")
        
        elif data[column].dtype in [np.number]:
            return st.number_input("값", value=float(data[column].median()), key="filter_value")
        
        else:
            return st.text_input("값", key="filter_value")
    
    def _exact_search(self, data: pd.DataFrame, query: str) -> pd.DataFrame:
        """정확 검색"""
        masks = []
        search_columns = data.select_dtypes(include=['object', 'string']).columns
        
        for col in search_columns:
            try:
                mask = data[col].astype(str).str.contains(query, case=False, na=False)
                masks.append(mask)
            except Exception:
                continue
        
        if masks:
            final_mask = reduce(operator.or_, masks)
            return data[final_mask].copy()
        
        return data
    
    def _save_filter(self, name: str):
        """필터 저장"""
        if 'saved_filters' not in st.session_state:
            st.session_state.saved_filters = {}
        
        conditions_data = [condition.to_dict() for condition in self.filter_system.conditions]
        st.session_state.saved_filters[name] = conditions_data
    
    def _load_saved_filters(self) -> Dict[str, List[Dict]]:
        """저장된 필터 로드"""
        return st.session_state.get('saved_filters', {})
    
    def _load_filter(self, conditions_data: List[Dict]):
        """필터 불러오기"""
        self.filter_system.clear_conditions()
        for condition_data in conditions_data:
            condition = FilterCondition.from_dict(condition_data)
            self.filter_system.add_condition(condition)
    
    def _delete_filter(self, name: str):
        """필터 삭제"""
        if 'saved_filters' in st.session_state and name in st.session_state.saved_filters:
            del st.session_state.saved_filters[name]