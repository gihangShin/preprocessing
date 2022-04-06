import json
import math
import os
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np


# HandlingDataset 연산만 생각
class HandlingDataset:

    def __init__(self, app, dsDAO, jhDAO):
        self.app = app
        self.dsDAO = dsDAO
        self.jhDAO = jhDAO

    #########################################################################
    #########################################################################
    # 0. 공통 소스
    #########################################################################
    # 0-1. Job_History 저장
    def insert_job_history_into_database(self, ds):
        target_id = 'user' + str(random.randint(100, 500))

        job_history = {
            'file_name': ds.file_id,
            'job_id': ds.job_id,
            'version': ds.version,
            'job_request_user_id': target_id,
            'content': ds.job_params_to_json()
        }

        self.jhDAO.insert_job_history(job_history=job_history)

    #########################################################################
    # 0-2. dataset 샘플링
    def sampling_dataset(self, ds):
        self.app.logger.info('sampling parameter 없음 Default value SEQ/ROW/FRT/50')
        sampling_method = 'SEQ'
        ord_value = 50
        ord_row = 'ROW'
        ord_set = 'FRT'
        # else:
        ########################################
        # session 말고 DB에서 불러올 예정 수정 필요 #
        ########################################
        # pass
        # sampling_method = session['sampling_method']
        # ord_value = int(session['ord_value'])
        # ord_row = session['ord_row']
        # if 'ord_set' in session:
        #     ord_set = session['ord_set']
        # else:
        #     ord_set = 'FRT'
        # msg = sampling_method + '/'
        # msg += ord_row + '/'
        # msg += ord_set + '/'
        # msg += str(ord_value)
        # self.app.logger.info('sampling parameter ' + msg)

        if sampling_method == 'RND':
            # Data Frame 셔플 수행
            ds.dataset = ds.dataset.sample(frac=1).reset_index(drop=True)
        sampled_df = pd.DataFrame()
        if ord_row == 'ROW':
            if ord_set == 'FRT':
                sampled_df = ds.dataset.iloc[:ord_value]
            elif ord_set == 'BCK':
                sampled_df = ds.dataset.iloc[-ord_value:, :]
        elif ord_row == 'PER':
            df_len = len(ds.dataset)
            df_per = int(df_len * ord_value / 100)
            if ord_set == 'FRT':
                sampled_df = ds.dataset.iloc[:df_per, :]
            elif ord_set == 'BCK':
                sampled_df = ds.dataset.iloc[-df_per:, :]

        ds.dataset = sampled_df
        return ds

    # 0-3. redirect_preprocess
    def redirect_preprocess(self, ds):
        job_id = ds.job_id
        if job_id == 'delete_column':
            ds = self.delete_column(ds)
        elif job_id == 'missing_value':
            ds = self.missing_value(ds)
        elif job_id == 'set_col_prop':
            ds = self.set_col_prop(ds)
        elif job_id == 'set_col_prop_to_datetime':
            ds = self.set_col_prop_to_datetime(ds)
        elif job_id == 'split_datetime':
            ds = self.split_datetime(ds)
        elif job_id == 'dt_to_str_format':
            ds = self.dt_to_str_format(ds)
        elif job_id == 'diff_datetime':
            ds = self.diff_datetime(ds)
        elif job_id == 'change_column_order':
            ds = self.change_column_order(ds)
        elif job_id == 'case_sensitive':
            ds = self.case_sensitive(ds)
        elif job_id == 'replace_by_input_value':
            ds = self.replace_by_input_value(ds)
        elif job_id == 'remove_space_front_and_rear':
            ds = self.remove_space_front_and_rear(ds)
        elif job_id == 'drop_duplicate_row':
            ds = self.drop_duplicate_row(ds)
        elif job_id == 'calculating_column':
            ds = self.calculating_column(ds)
        else:
            print('ERRORERRORERRORERRORERROR')
        return ds

    ##########################################################################
    ##########################################################################
    # 1. 전처리 동작
    ##########################################################################
    # 1-1. 열 삭제
    def delete_column(self, ds):
        self.app.logger.info('delete_column / ' + str(ds.job_params))
        ds.dataset = ds.dataset.drop(columns=ds.job_params['columns'], axis=1)
        ds.data_types = ds.get_types()
        ds.sync_dataset_with_dtypes()
        return ds

    ##########################################################################
    # 1-2. 결측치 처리 (열 연산)
    def missing_value(self, ds):
        self.app.logger.info('missing_value / ' + str(ds.job_params))

        return self.handling_missing_value(ds)

    def handling_missing_value(self, ds):
        missing_value = ds.job_params['method']
        if missing_value == 'remove':  # ok
            ds = self.remove_missing_value(ds)
        elif missing_value == 'mean':  # ok
            ds = self.fill_missing_value_mean(ds)
        elif missing_value == 'median':  # ok
            ds = self.fill_missing_value_median(ds)
        elif missing_value in {'ffill', 'bfill'}:  # ok
            ds = self.fill_missing_value(ds)
        elif missing_value == 'first_row':  # 미구현
            ds = self.fill_missing_value_first_row()
        elif missing_value == 'input':  # ok
            ds = self.fill_missing_value_specified_value(ds)
        return ds

    def remove_missing_value(self, ds):
        if 'colomns' in ds.job_params:
            ds.dataset = ds.dataset.dropna(subset=[ds.job_params['columns']])
        else:
            ds.dataset = ds.dataset.dropna(axis=1)
        return ds

    def fill_missing_value(self, ds):
        if 'colomns' in ds.job_params:
            ds.dataset[[ds.job_params['columns']]] = ds.dataset[[ds.job_params['columns']]].fillna(
                method=ds.job_params['method'], axis=1)
        else:
            ds.dataset = ds.dataset.fillna(method=ds.job_params['method'], axis=1)
        return ds

    def fill_missing_value_specified_value(self, ds):
        if 'colomns' in ds.job_params:
            ds.dataset[[ds.job_params['columns']]] = ds.dataset[[ds.job_params['columns']]].fillna(
                value=ds.job_params['input_data'])
        else:
            ds.dataset = ds.dataset.fillna(value=ds.job_params['input_data'])
        return ds

    def fill_missing_value_median(self, ds):
        ds.dataset[ds.job_params['columns']] = ds.dataset[ds.job_params['columns']].fillna(
            ds.dataset[ds.job_params['columns']].median())
        return ds

    def fill_missing_value_mean(self, ds):
        ds.dataset[ds.job_params['columns']] = ds.dataset[ds.job_params['columns']].fillna(
            ds.dataset[ds.job_params['columns']].mean())
        return ds

    ##########################################################################
    # 1-3. 컬럼 속성 변경
    # 에러 처리 제외, 기능만 구현하게 코딩함.
    def set_col_prop(self, ds):
        ds.dataset[ds.job_params['column']] = ds.dataset[ds.job_params['column']].astype(ds.job_params['type'])
        ds.data_types = ds.get_types()
        return ds

    ##########################################################################
    # 1-4. 선택 열 date time 으로 변환 후 추가
    def set_col_prop_to_datetime(self, ds):
        dt_format = ''
        for ch in ds.job_params['dt_format']:
            if ch == '?':
                ch = '%'
            dt_format += ch
        if 'new_column_name' in ds.job_params:
            ds.dataset[ds.job_params['new_column_name']] = pd.to_datetime(ds.dataset[ds.job_params['selected_column']],
                                                                          format=dt_format)
        else:
            ds.dataset[ds.job_params['selected_column']] = pd.to_datetime(ds.dataset[ds.job_params['selected_column']],
                                                                          format=dt_format)
        dt_format2 = ''
        for ch in dt_format:
            if ch == '%':
                ch = '?'
            dt_format2 += ch
        ds.job_params['dt_format'] = dt_format2
        ds.data_types = ds.get_types()
        return ds

    ##########################################################################
    # 1-5. 날짜 형 컬럼 분할 ex 날짜 -> 년, 월, 일, 시 ..
    def split_datetime(self, ds):

        for unit in ds.job_params['unit_list']:
            temp = str(ds.job_params['column']) + '_' + unit
            ds.dataset[temp] = self.split_variable_to_unit(ds, unit=unit)
        ds.data_types = ds.get_get_types()
        return ds

    def split_variable_to_unit(self, ds, unit):
        if unit == 'year':
            return ds.dataset[ds.job_params['column']].dt.year
        elif unit == 'month':
            return ds.dataset[ds.job_params['column']].dt.month
        elif unit == 'month_name':
            return ds.dataset[ds.job_params['column']].dt.month_name()
        elif unit == 'day':
            return ds.dataset[ds.job_params['column']].dt.day
        elif unit == 'hour':
            return ds.dataset[ds.job_params['column']].dt.hour
        elif unit == 'dayofweek':
            return ds.dataset[ds.job_params['column']].dt.dayofweek
        elif unit == 'day_name':
            return ds.dataset[ds.job_params['column']].dt.day_name()
        else:
            print("EEEEERRRRRRRRRROOOORRRRR")
            return ds

    ##########################################################################
    # 1-6. 날짜 처리(문자열로)
    def dt_to_str_format(self, ds):
        dt_format = ''
        for ch in ds.job_params['dt_format']:
            if ch == '?':
                ch = '%'
            dt_format += ch

        if 'new_column_name' in ds.job_params:
            ds.dataset[ds.job_params['new_column_name']] = ds.dataset[ds.job_params['column']].dt.strftime(dt_format)
        else:
            ds.dataset[ds.job_params['column']] = ds.dataset[ds.job_params['column']].dt.strftime(dt_format)

        dt_format2 = ''
        for ch in dt_format:
            if ch == '%':
                ch = '?'
            dt_format2 += ch
        ds.job_params['dt_format'] = dt_format2

        ds.data_types = ds.get_get_types()
        return ds

    ##########################################################################
    # 1-7. 날짜 처리(기준 일로 부터 날짜 차이)
    def diff_datetime(self, ds):

        year = ds.job_params['year']
        month = ds.job_params['month']
        day = ds.job_params['day']

        if 'hour' in ds.job_params:
            hour = ds.job_params['hour']
            dt_diff = ds.dataset[ds.job_params['column']] - datetime(year, month, day, hour)
        else:
            dt_diff = ds.dataset[ds.job_params['column']] - datetime(year, month, day)

        new_column_name = "diff" + str(year) + '-' + str(month) + '-' + str(day) + 'with' + str(ds.job_params['column'])
        if ds.job_params['unit'] == 'day':
            ds.dataset[new_column_name] = dt_diff.dt.days
        if ds.job_params['unit'] == 'minute':
            ds.dataset[new_column_name] = dt_diff.dt.total_seconds() / 60
        if ds.job_params['unit'] == 'hour':
            ds.dataset[new_column_name] = dt_diff.dt.total_seconds() / 360

        ds.data_types = ds.get_get_types()
        return ds

    ##########################################################################
    # 1-8. 컬럼 순서 변경
    def change_column_order(self, ds):
        ds.dataset = ds.dataset.iloc[:, list(ds.job_params['col_order_list'])]
        ds.data_types = ds.get_get_types()

        return ds

    ##########################################################################
    # 1-9. 대소문자 변환
    def case_sensitive(self, ds):
        if ds.job_params['str_type'] == 'UPP':
            ds.dataset[ds.job_params['column']] = ds.dataset[ds.job_params['column']].str.upper()
        elif ds.job_params['str_type'] == 'LOW':
            ds.dataset[ds.job_params['column']] = ds.dataset[ds.job_params['column']].str.lower()
        elif ds.job_params['str_type'] == 'CAP':
            ds.dataset[ds.job_params['column']] = ds.dataset[ds.job_params['column']].str.capitalize()
        elif ds.job_params['str_type'] == 'TIT':
            ds.dataset[ds.job_params['column']] = ds.dataset[ds.job_params['column']].str.title()
        else:
            pass

        ds.data_types = ds.get_get_types()
        return ds

    ##########################################################################
    # 1-10. 치환 입력값으로 교체
    def replace_by_input_value(self, ds):
        if ds.job_params['method'] == 'default':
            ds.dataset[ds.job_params['column']].replace(to_replace=ds.job_params['to_replace'],
                                                        value=ds.job_params['value'], inplace=True)
        elif ds.job_params['method'] == 'regex':
            to_replace = "(.*)" + str(ds.job_params['to_replace']) + "(.*)"
            value = r"\1" + str(ds.job_params['value']) + r"\2"
            ds.dataset[ds.job_params['column']].replace(to_replace=to_replace, value=value, regex=True, inplace=True)

        ds.data_types = ds.get_get_types()
        return ds

    ##########################################################################
    # 1-11. 공백 제거 (앞 뒤만 해당, 문자 사이 X)
    def remove_space_front_and_rear(self, ds):
        ds.dataset[ds.job_params['column']] = ds.dataset[ds.job_params['column']].str.strip()
        ds.data_types = ds.get_get_types()
        return ds

    ##########################################################################
    # 1-12. 중복 행 제거
    def drop_duplicate_row(self, ds):
        if 'keep' in ds.job_params:
            ds.dataset.drop_duplicates(subset=ds.job_params['columns'], keep=ds.job_params['keep'], inplace=True)
        else:
            ds.dataset.drop_duplicates(subset=ds.job_params['columns'], keep='first', inplace=True)
        ds.data_types = ds.get_get_types()
        return ds

    ##########################################################################
    # 1-13. 중복 값 확인                                            (단순 조회)
    def show_duplicate_row(self, ds):
        return ds.dataset[ds.job_params['column']].value_counts()

    ##########################################################################
    # 1-14. 연산
    def calculating_column(self, ds):
        if ds.job_params['method'] == 'arithmetic':
            ds = self.calc_arithmetic(ds)

        elif ds.job_params['method'] == 'function':
            ds = self.calc_function(ds)
        ds.data_types = ds.get_get_types()
        return ds

    # 1-14-1. 연산 동작(함수 선택 시)
    def calc_function(self, ds):
        function = ds.job_params['calc_function']
        columns = ds.job_params['columns']
        # 여러 컬럼에서만 동작하는 함수 단일 컬럼 X
        # mean, max, min, median, std, var
        if function == 'mean':
            result = ds.dataset[[columns]].mean(axis=1)
        elif function == 'max':
            result = ds.dataset[[columns]].max(axis=1)
        elif function == 'min':
            result = ds.dataset[[columns]].min(axis=1)
        elif function == 'median':
            result = ds.dataset[[columns]].median(axis=1)
        elif function == 'std':
            result = ds.dataset[[columns]].std(axis=1)
        elif function == 'var':
            result = ds.dataset[[columns]].var(axis=1)

        # 단일 컬럼에서만 동작하는 함수
        # sin, cos, abs, log,

        elif function == 'sin':
            result = np.sin(ds.dataset[[columns]])
        elif function == 'cos':
            result = np.cos(ds.dataset[[columns]])
        elif function == 'abs':
            result = np.abs(ds.dataset[[columns]])
        elif function == 'log':
            result = np.log(ds.dataset[[columns]])

        column_name = function + '(' + columns + ')'
        ds.dataset[column_name] = result
        return ds

    # 1-14-2. 연산 동작(산술 연산 선택 시)
    def calc_arithmetic(self, ds):
        operator = ds.job_params['operator']
        column1 = ds.job_params['column1']
        operand1 = ds.dataset[column1]
        # 2번 피연산자
        # 1. column명
        # 2. 상수
        # 3. column의 집계함수 값
        if ds.job_params['value_type'] == 'column':
            column2 = ds.job_params['value']
            operand2 = ds.dataset[column2]
        elif ds.job_params['value_type'] == 'aggregate':
            operand2, column2 = self.calc_column_aggregate_function(ds)
        elif ds.job_params['value_type'] == 'constant':
            operand2 = float(ds.job_params['value'])
            column2 = operand2

        if operator == 'add':
            result = operand1 + operand2
            operator = '+'

        elif operator == 'min':
            result = operand1 - operand2
            operator = '-'

        elif operator == 'mul':
            result = operand1 * operand2
            operator = '*'

        elif operator == 'div':
            result = operand1 / operand2
            operator = '/'

        elif operator == 'remainder':
            result = operand1 % operand2
            operator = '%'

        column_name = column1 + operator + column2
        ds.dataset[column_name] = result
        return ds

    # 1-14-3. 두번째 피연산자 == 컬럼의 집계값 사용 시
    def calc_column_aggregate_function(self, ds):
        column2 = ds.job_params['value']
        function = ds.job_params['column_function']
        result = 0
        if function == 'max':
            result = ds.dataset[column2].max(axis=0)
        elif function == 'min':
            result = ds.dataset[column2].min(axis=0)
        elif function == 'mean':
            result = ds.dataset[column2].mean(axis=0)
        elif function == 'median':
            result = ds.dataset[column2].median(axis=0)
        elif function == 'std':  # 표준편차
            result = ds.dataset[column2].std(axis=0)
        elif function == 'var':  # 분산
            result = ds.dataset[column2].var(axis=0)

        column_name = function + '(' + column2 + ')'
        return result, column_name

    ###########################################################################
    ###########################################################################
    # 2. 추출 동작
    ###########################################################################

    # 원본 파일 동작 재 수행
    def redo_job_history(self, ds):
        i = 0
        for row in self.get_job_historys(ds):
            ds.job_id = row['job_id']
            ds.job_params = row['content']
            self.app.logger.info('redo action ' + str(i) + ". " + str(ds.job_id))
            i += 1
            ds = self.redirect_preprocess(ds=ds)
        return ds

    # 3-1. job_history load
    def get_job_historys(self, ds):
        return self.jhDAO.select_job_history_by_file_name_and_version(file_name=ds.file_id,
                                                                      version=ds.version)
