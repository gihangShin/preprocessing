import json
import math
import os
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from numpyencoder import NumpyEncoder
from flask import session


class Preprocessing:
    target_id_seq = 1;

    def __init__(self, app, dsDAO, jhDAO):
        self.app = app
        self.dsDAO = dsDAO
        self.jhDAO = jhDAO

    ########################################################################################

    # 1-1
    # dataset 신규 등록만!!
    # 1. file을 server/project01/origin_data/ 저장
    # server/project01/p_data/ <filename>_V<version>_D<dateime>.<extension> 형식 저장
    #############################################
    # 테스트용 코드 웹단에서 이루어질 예정 == 곧 삭제  #
    #############################################
    def upload_file_to_server(self, payload):
        project_name = payload['project_name']
        file_name_type = payload['file_name']

        project_dir = './server/' + project_name
        os.makedirs(project_dir + '/origin_data', exist_ok=True)
        os.makedirs(project_dir + '/p_data', exist_ok=True)

        file_name = file_name_type.split('.')[-2]
        extention = file_name_type.split('.')[-1]
        org_url = project_dir + '/origin_data/' + file_name_type
        extention = 'csv'
        df = pd.read_csv('./data/bsdpbsdp.csv')
        # if extention == 'json':
        #     df_json = payload['file']
        #     df = pd.read_json(df_json)
        # elif extention == 'csv':
        #     df_csv = payload['file']
        #     df = pd.read_csv(df_csv)
        df.to_json(org_url, force_ascii=False)

        version = 1.0
        new_url = project_dir + '/p_data/' + file_name + '_V' + str(round(version, 2)) + '.json'
        df.to_json(new_url, force_ascii=False)

        self.app.logger.info('org_url : ' + org_url)
        self.app.logger.info('new_url : ' + new_url)
        return "1234"
        # return self.sampling_dataset(df).to_json()

    # 1-2-1. 기존 파일 불러오기
    def load_existed_file(self, payload):
        project_name = payload['project_name']
        file_name = payload['file_name']
        version = payload['version']

        df = self.load_org_file(payload=payload)
        df_dtypes = self.get_dtype_of_dataframe(df=df)
        df = df.astype(str)

        payload_response = {
            'dataset': self.sampling_dataset(df).to_json(force_ascii=False),
            'dataset_dtypes': df_dtypes
        }

        return payload_response

    # 1-2-2. 파일 load 모듈화 Dataframe 객체 반환
    def load_org_file(self, payload=None):
        project_name = payload['project_name']
        file_name = payload['file_name']
        version = str(round(float(payload['version']), 2))

        project_dir = './server/' + project_name + '/p_data/'
        full_file_name = file_name + '_V' + version + '.json'

        url = project_dir + full_file_name
        self.app.logger.info('load_org_file_url : ' + url)
        df = pd.read_json(url)
        return df

    # 1-2-3. DataFrame columns dtype 불러오기 (-> dict)
    def get_dtype_of_dataframe(self, df):
        df = df.convert_dtypes()
        df_dtypes = df.dtypes.to_dict()
        for k, v in df_dtypes.items():
            df_dtypes[k] = str(v)
        return df_dtypes

    # 1-3. sampled_parameter 설정
    #############################################
    # 테스트용 코드 웹단에서 이루어질 예정 == 곧 삭제  #
    #############################################
    def set_sampling_parameter(self, payload):
        session['sampling_method'] = payload['sampling_method']
        session['ord_row'] = payload['ord_row']
        session['ord_value'] = payload['ord_value']
        df = self.load_org_file()
        if 'ord_set' in payload:  # default FRT
            session['ord_set'] = payload['ord_set']
        else:
            session['ord_set'] = 'FRT'

        return self.sampling_dataset(df).to_json()

    # 1-3-2. sampling_dataset(dataframe)
    # return dataFrame
    def sampling_dataset(self, df):
        if 'sampling_method' not in session:
            self.app.logger.info('sampling parameter 없음 Default value SEQ/ROW/FRT/50')
            sampling_method = 'SEQ'
            ord_value = 50
            ord_row = 'ROW'
            ord_set = 'FRT'
        else:
            ########################################
            # session 말고 DB에서 불러올 예정 수정 필요 #
            ########################################
            sampling_method = session['sampling_method']
            ord_value = int(session['ord_value'])
            ord_row = session['ord_row']
            if 'ord_set' in session:
                ord_set = session['ord_set']
            else:
                ord_set = 'FRT'
            msg = sampling_method + '/'
            msg += ord_row + '/'
            msg += ord_set + '/'
            msg += str(ord_value)
            self.app.logger.info('sampling parameter ' + msg)
        # ord_set 값 없을 시 default FRT

        if sampling_method == 'RND':
            # Data Frame 셔플 수행
            df = df.sample(frac=1).reset_index(drop=True)
        sampled_df = pd.DataFrame()
        if ord_row == 'ROW':
            if ord_set == 'FRT':
                sampled_df = df.iloc[:ord_value]
            elif ord_set == 'BCK':
                sampled_df = df.iloc[-ord_value:, :]
        elif ord_row == 'PER':
            df_len = len(df)
            df_per = int(df_len * ord_value / 100)
            if ord_set == 'FRT':
                sampled_df = df.iloc[:df_per, :]
            elif ord_set == 'BCK':
                sampled_df = df.iloc[-df_per:, :]
        return sampled_df

    ########################################################################################

    # 2. 전처리 동작
    # view <-> 상위 메서드 에서만 df.to_json() 예정
    # 상위 <-> 하위 메서드 에선 DataFrame 객체 송수신

    # 2-0-1. 동작 내역 DB 저장 함수
    def insert_job_history(self, payload, job_id):
        dict_payload = dict(payload)
        target_id = 'admin' + str(random.randint(100, 300))

        if 'file_name' in dict_payload:
            del dict_payload['file_name']
        if 'version' in dict_payload:
            del dict_payload['version']

        job_history = {
            'file_name': payload['file_name'],
            'job_id': job_id,
            'version': payload['version'],
            'job_request_user_id': target_id,
            'content': json.dumps(dict_payload, ensure_ascii=False)
        }

        self.jhDAO.insert_job_history(job_history=job_history)

    # 2-0-2. dataset 로드 후 dtype init
    def init_dtype_to_dataset(self, payload):
        pass

    # 2-1. 열 삭제
    def delete_column(self, payload, df=None):
        column_name = payload['column_name']
        # column_id = payload['column_id']
        df = self.get_df(payload=payload, df=df)

        self.app.logger.info('delete_column / ' + column_name)
        df = df.drop([column_name], axis=1)
        dataset_dtypes = self.get_dtype_of_dataframe(df)
        if df is None:
            del payload['dataset']
            self.insert_job_history(payload, job_id='delete_column')
            return_object = {
                'dataset': df.to_json(force_ascii=False),
                'dataset_dtypes': dataset_dtypes
            }
            return return_object
        else:
            return df, dataset_dtypes

    # 2-2. 결측치 처리
    # 열 연산만
    def missing_value(self, payload, df=None):
        missing_value = payload['m_value']
        columns = payload['columns']

        if df is None:
            dataset = self.get_df_from_payload(payload=payload)
        else:
            dataset = df

        self.app.logger.info('missing value / ' + missing_value + " / " + columns)
        if missing_value == 'remove':  # ok
            dataset = self.remove_missing_value(dataset=dataset, columns=columns)
        elif missing_value == 'mean':  # ok
            dataset = self.fill_missing_value_mean(dataset=dataset, columns=columns)
        elif missing_value == 'median':  # ok
            dataset = self.fill_missing_value_median(dataset=dataset, columns=columns)
        elif missing_value == 'ffill':  # ok
            dataset = self.fill_missing_value_front(dataset=dataset, columns=columns)
        elif missing_value == 'bfill':  # ok
            dataset = self.fill_missing_value_back(dataset=dataset, columns=columns)
        elif missing_value == 'first_row':  # 미구현
            dataset = self.fill_missing_value_first_row()
        elif missing_value == 'input':  # ok
            input_data = payload['input_data']
            dataset = self.fill_missing_value_specified_value(dataset=dataset, columns=columns,
                                                              input_data=input_data)

        if df is None:
            return_object = {
                'dataset': dataset.to_json(force_ascii=False),
                'dataset_dtypes': payload['dataset_dtypes']
            }
            del payload['dataset']
            del payload['dataset_dtypes']
            self.insert_job_history(payload=payload, job_id='missing_value')
            return return_object
        else:
            return dataset, payload['dataset_dtypes']

    # 2-2. 하위 메서드

    def remove_missing_value(self, dataset, columns=None):
        if columns is None:
            dataset = dataset.dropna(axis=1)
        else:
            dataset = dataset.dropna(subset=[columns])
        return dataset

    # 이전값 채우기
    def fill_missing_value_front(self, dataset, columns=None):
        if columns is None:
            dataset = dataset.fillna(method='ffill', axis=1)
        else:
            dataset[[columns]] = dataset[[columns]].ffill()
        return dataset

    # 다음값 채우기
    def fill_missing_value_back(self, dataset, columns=None):
        if columns is None:
            dataset = dataset.fillna(method='bfill', axis=1)
        else:
            dataset[[columns]] = dataset[[columns]].bfill()
        return dataset

    def fill_missing_value_specified_value(self, dataset, input_data, columns=None):
        if columns is None:
            dataset = dataset.fillna(input_data)
        else:
            dataset[[columns]] = dataset[[columns]].fillna(value=input_data)
        return dataset

    def fill_missing_value_median(self, dataset, columns):
        dataset[columns] = dataset[columns].fillna(dataset[columns].median())
        return dataset

    def fill_missing_value_mean(self, dataset, columns):
        dataset[columns] = dataset[columns].fillna(dataset[columns].mean())
        return dataset

    # 2-3. 연산

    # 2-3-1. sampled_dataset 에서 연산용 데이터셋 추출
    # calculating
    # 테스트 완료
    def get_calc_dataset(self, payload, df=None):
        # calc_df == 계산 용 dataset
        # 그냥 수치형 다 받아옴
        if df is None:
            dataset = self.get_df_from_payload(payload)
            calc_df = dataset.select_dtypes(include=['int64', 'float64'])
            dataset_dtypes = self.get_dtype_of_dataframe(calc_df)
            response_object = {
                'dataset': calc_df.to_json(force_ascii=False),
                'dataset_dtypes': dataset_dtypes
            }
            return response_object
        else:
            dataset = df
            calc_df = dataset.select_dtypes(include=['int64', 'float64'])
            return calc_df

    # 2-3-2. 연산
    # 테스트 완료
    def calculating_column(self, payload, df=None):
        # method == arithmetic -> column1, (column2 or scala or 집계 데이터), operator
        #           return 연산 완료+추가된 calc_dataset
        # method == function(aggregate, Statistical) -> function, (column or scala)
        #           return 연산 완료+추가된 calc_dataset
        method = payload['method']

        if method == 'arithmetic':
            calc_df = self.calc_arithmetic(payload=payload, df=df)

        elif method == 'function':
            calc_df = self.calc_function(payload=payload, df=df)

        dataset_dtypes = self.get_dtype_of_dataframe(calc_df)
        if df is None:
            result = {
                'calc_dataset': calc_df.to_json(force_ascii=False),
                'dataset_dtypes': dataset_dtypes
            }

            del payload['calc_dataset']
            del payload['dataset_dtypes']
            if 'calc_job_history' not in payload:
                c_j_history = list()
                c_j_history.append(payload)
            else:
                c_j_history = payload['calc_job_history']
                del payload['calc_job_history']
                c_j_history.append(payload)

            result['calc_job_history'] = c_j_history
            return json.dumps(result)
        else:
            # 추출 시 반복작업 용
            return calc_df, dataset_dtypes

    # 2-3-2-1. 연산 동작(함수 선택 시)
    def calc_function(self, payload, df=None):
        if df is None:
            calc_df = self.get_calc_dataset_from_payload(payload)
        else:
            calc_df = df
        function = payload['calc_function']
        columns = payload['columns']

        # 여러 컬럼에서만 동작하는 함수 단일 컬럼 X
        # mean, max, min, median, std, var
        if function == 'mean':
            result = calc_df[[columns]].mean(axis=1)
        elif function == 'max':
            result = calc_df[[columns]].max(axis=1)
        elif function == 'min':
            result = calc_df[[columns]].min(axis=1)
        elif function == 'median':
            result = calc_df[[columns]].median(axis=1)
        elif function == 'std':
            result = calc_df[[columns]].std(axis=1)
        elif function == 'var':
            result = calc_df[[columns]].var(axis=1)

        # 단일 컬럼에서만 동작하는 함수
        # sin, cos, abs, log,

        elif function == 'sin':
            result = np.sin(calc_df[[columns]])
        elif function == 'cos':
            result = np.cos(calc_df[[columns]])
        elif function == 'abs':
            result = np.abs(calc_df[[columns]])
        elif function == 'log':
            result = np.log(calc_df[[columns]])

        column_name = function + '(' + columns + ')'
        calc_df[column_name] = result
        return calc_df

    # 2-3-2-2. 연산 동작(산술 연산 선택 시)
    # 산술 연산 + - * / %
    def calc_arithmetic(self, payload, df=None):
        calc_df = self.get_df(payload=payload, df=df)
        operator = payload['operator']
        column1 = payload['column1']
        operand1 = calc_df[column1]
        # 2번 피연산자
        # 1. column명
        # 2. 상수
        # 3. column의 집계함수 값
        value_type = payload['value_type']
        if value_type == 'column':
            column2 = payload['value']
            operand2 = calc_df[column2]
        elif value_type == 'column_func':
            if df is None:
                operand2, column2 = self.calc_function_column(payload=payload)
            else:
                operand2, column2 = self.calc_function_column(payload=payload, df=calc_df)
        elif value_type == 'constant':
            operand2 = float(payload['value'])
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
        calc_df[column_name] = result
        return calc_df

    # 2-3-2-2-1. 두번째 피연산자 == 컬럼의 집계값 사용 시
    def calc_function_column(self, payload, df=None):
        column2 = payload['value']
        function = payload['column_function']
        if df is None:
            calc_df = self.get_calc_dataset_from_payload(payload)
        else:
            calc_df = df
        result = 0
        if function == 'max':
            result = df[column2].max(axis=0)
        elif function == 'min':
            result = df[column2].min(axis=0)
        elif function == 'mean':
            result = df[column2].mean(axis=0)
        elif function == 'median':
            result = df[column2].median(axis=0)
        elif function == 'std':  # 표준편차
            result = df[column2].std(axis=0)
        elif function == 'var':  # 분산
            result = df[column2].var(axis=0)

        column_name = function + '(' + column2 + ')'
        return result, column_name

    def get_calc_dataset_from_payload(self, payload):
        dataset = payload['calc_dataset']
        if type(dataset) == dict:
            return pd.DataFrame(dataset)
        else:
            return pd.read_json(dataset)

    # 2-3-3. calc_dataset 에서 추출할 컬럼 선택 후 기존 데이터셋(sampled_dataset)으로 결합
    def select_calc_column_to_combine(self, payload, origin_df=None, calc_df=None):
        selected_columns = payload['selected_columns']
        if origin_df is None:
            calc_dataset = self.get_calc_dataset_from_payload(payload)
            dataset = self.get_df_from_payload(payload, change_type=False)
        else:
            calc_dataset = calc_df
            dataset = origin_df

        dataset[selected_columns] = calc_dataset[selected_columns]
        dataset_dtypes = self.get_dtype_of_dataframe(dataset)

        if origin_df is None:
            # 추출 시 재수행 동작이 아니라면 DB저장
            job_history = payload['calc_job_history']
            calc_payload = {
                'file_name': payload['file_name'],
                'version': payload['version'],
                'selected_columns': selected_columns,
                'calc_job_history': job_history  # list[json] 형식
            }
            self.insert_job_history(payload=calc_payload, job_id='calc_columns')

            return_object = {
                'dataset': dataset.to_json(force_ascii=False),
                'dataset_dtypes': dataset_dtypes
            }
            return return_object
        else:
            return dataset, dataset_dtypes

    # 2-4. 컬럼 속성 변경
    # 에러 처리 제외, 기능만 구현하게 코딩함.
    def set_col_prop(self, payload, df=None):
        dataset = self.get_df(payload=payload, df=df)
        column_name = payload['column_name']
        types = payload['type']

        dataset[column_name] = dataset[column_name].astype(types)
        dataset_dtypes = self.get_dtype_of_dataframe(dataset)
        if df is None:
            response_json = {
                'dataset': dataset.to_json(force_ascii=False),
                'dataset_dtypes': dataset_dtypes
            }

            job_history_payload = {
                'file_name': payload['file_name'],
                'version': payload['version'],
                'column_name': column_name,
                'type': types  # list[json] 형식
            }

            self.insert_job_history(payload=job_history_payload, job_id='set_col_prop')
            return response_json
        else:
            return df, dataset_dtypes

    # 2-5-1. 선택 열 date time 으로 변환 후 추가
    def set_col_prop_to_datetime(self, payload, df=None):
        dataset = self.get_df(payload=payload, df=df)
        selected_column = payload['selected_column']
        dt_format = payload['format']

        if df is not None:
            dt_format2 = ""
            for ch in dt_format:
                if ch == '?':
                    ch = '%'
                dt_format2 += ch
            dt_format = dt_format2

        if 'column_name' in payload:
            column_name = payload['column_name']
            dataset[column_name] = pd.to_datetime(dataset[selected_column], format=dt_format)
        else:
            dataset[selected_column] = pd.to_datetime(dataset[selected_column], format=dt_format)
        dataset_dtypes = self.get_dtype_of_dataframe(df=dataset)

        if df is None:
            dataset = dataset.astype(str)

            response_json = {
                'dataset': dataset.to_json(force_ascii=False),
                'dataset_dtypes': dataset_dtypes
            }

            # sql excute 할 때 %는 키워드로 인식 -->  % -> ?
            # 더 나은 방법이 있으면 수정 예정
            dt_format2 = ""
            for ch in dt_format:
                if ch == '%':
                    ch = '?'
                dt_format2 += ch

            del payload['dataset']
            del payload['dataset_dtypes']

            payload['format'] = dt_format2

            self.insert_job_history(payload=payload, job_id="col_to_datetime")

            return response_json
        else:
            return dataset, dataset_dtypes

    def split_variable_datetime(self, payload, df=None):
        unit_list = payload['unit_list']
        column_name = payload['column_name']
        dataset = self.get_df(payload=payload, df=df)

        for unit in unit_list:
            temp = str(column_name) + '_' + unit
            dataset[temp] = self.split_variable_to_unit(dataset, column_name=column_name, unit=unit)

        dataset_dtypes = self.get_dtype_of_dataframe(dataset)

        if df is None:
            dataset = dataset.astype(str)

            response_json = {
                'dataset': dataset.to_json(force_ascii=False),
                'dataset_dtypes': dataset_dtypes
            }

            del payload['dataset']
            del payload['dataset_dtypes']

            self.insert_job_history(payload=payload, job_id="split_var_dt")

            return response_json
        else:
            return dataset, dataset_dtypes

    # dt.dayofweek
    # dt.day_name
    def split_variable_to_unit(self, dataset, column_name, unit):
        if unit == 'year':
            return dataset[column_name].dt.year
        elif unit == 'month':
            return dataset[column_name].dt.month
        elif unit == 'month_name':
            return dataset[column_name].dt.month_name()
        elif unit == 'day':
            return dataset[column_name].dt.day
        elif unit == 'hour':
            return dataset[column_name].dt.hour
        elif unit == 'dayofweek':
            return dataset[column_name].dt.dayofweek
        elif unit == 'day_name':
            return dataset[column_name].dt.day_name()
        else:
            print("EEEEERRRRRRRRRROOOORRRRR")
            return dataset

    # 2-5-3. 날짜 처리(문자열로)
    def dt_to_str_format(self, payload, df=None):
        dataset = self.get_df(payload=payload, df=df)
        column_name = payload['column_name']
        dt_format = payload['format']
        if df is not None:
            dt_format2 = ""
            for ch in dt_format:
                if ch == '?':
                    ch = '%'
                dt_format2 += ch
            dt_format = dt_format2

        if 'new_column_name' in payload:
            new_column_name = payload['new_column_name']
            dataset[new_column_name] = dataset[column_name].dt.strftime(dt_format)
        else:
            dataset[column_name] = dataset[column_name].dt.strftime(dt_format)

        dataset_dtypes = self.get_dtype_of_dataframe(dataset)
        if df is None:
            dataset = dataset.astype(str)
            response_json = {
                'dataset': dataset.to_json(force_ascii=False),
                'dataset_dtypes': dataset_dtypes
            }

            del payload['dataset']
            del payload['dataset_dtypes']

            dt_format2 = ""
            for ch in dt_format:
                if ch == '%':
                    ch = '?'
                dt_format2 += ch
            payload['format'] = dt_format2

            self.insert_job_history(payload=payload, job_id="dt_to_str_format")

            return response_json
        else:
            return dataset, dataset_dtypes

    # 2-5-4. 날짜 처리(기준 일로 부터 날짜 차이)
    def diff_datetime(self, payload, df=None):
        dataset = self.get_df(payload=payload, df=df)
        unit = payload['unit']
        column_name = payload['column_name']

        year = payload['year']
        month = payload['month']
        day = payload['day']

        if 'hour' in payload:
            hour = payload['hour']
            dt_diff = dataset[column_name] - datetime(year, month, day, hour)
        else:
            dt_diff = dataset[column_name] - datetime(year, month, day)

        new_column_name = "diff" + str(year) + '-' + str(month) + '-' + str(day) + '-' + str(unit)
        if unit == 'day':
            dataset[new_column_name] = dt_diff.dt.days
        if unit == 'minute':
            dataset[new_column_name] = dt_diff.dt.total_seconds() / 60
        if unit == 'hour':
            dataset[new_column_name] = dt_diff.dt.total_seconds() / 360

        dataset_dtypes = self.get_dtype_of_dataframe(dataset)

        if df is None:
            dataset = dataset.astype(str)
            response_json = {
                'dataset': dataset.to_json(force_ascii=False),
                'dataset_dtypes': dataset_dtypes
            }
            del payload['dataset']
            del payload['dataset_dtypes']

            self.insert_job_history(payload=payload, job_id="diff_datetime")

            return response_json
        else:
            return dataset, dataset_dtypes

    # 2-6 컬럼 순서 변경
    def column_order_change(self, payload, df=None):
        dataset = self.get_df(payload=payload, df=df)
        col_order_list = list(payload['col_order_list'])

        dataset = dataset.iloc[:, col_order_list]

        dataset_dtypes = self.get_dtype_of_dataframe(dataset)

        if df is None:
            dataset = dataset.astype(str)
            response_json = {
                'dataset': dataset.to_json(force_ascii=False),
                'dataset_dtypes': dataset_dtypes
            }

            del payload['dataset']
            del payload['dataset_dtypes']

            self.insert_job_history(payload=payload, job_id="column_order_change")

            return response_json
        else:
            return dataset, dataset_dtypes

    # 2-7 대소문자 변환
    def col_prop_string_change(self, payload, df=None):
        dataset = self.get_df(payload=payload, df=df)
        column_name = payload['column_name']
        str_type = payload['str_type']

        if str_type == 'UPP':
            dataset[column_name] = dataset[column_name].str.upper()
        elif str_type == 'LOW':
            dataset[column_name] = dataset[column_name].str.lower()
        elif str_type == 'CAP':
            dataset[column_name] = dataset[column_name].str.capitalize()
        elif str_type == 'TIT':
            dataset[column_name] = dataset[column_name].str.title()
        else:
            pass

        dataset_dtypes = self.get_dtype_of_dataframe(dataset)

        if df is None:
            dataset = dataset.astype(str)
            response_json = {
                'dataset': dataset.to_json(force_ascii=False),
                'dataset_dtypes': dataset_dtypes
            }

            del payload['dataset']
            del payload['dataset_dtypes']

            self.insert_job_history(payload=payload, job_id="change_column_order")

            return response_json
        else:
            return dataset, dataset_dtypes

    # 2-8 치환 - 입력값으로 교체
    def col_prop_string_search_replace(self, payload, df=None):
        dataset = self.get_df(payload=payload, df=df)

        column_name = payload['column_name']
        method = payload['method']
        to_replace = payload['to_replace']
        value = payload['value']

        if method == 'default':
            dataset[column_name].replace(to_replace=to_replace, value=value, inplace=True)
        elif method == 'regex':
            to_replace = "(.*)"+str(to_replace)+"(.*)"
            value = r"\1"+str(value)+r"\2"
            dataset[column_name].replace(to_replace=to_replace, value=value, regex=True, inplace=True)

        dataset_dtypes = self.get_dtype_of_dataframe(dataset)

        if df is None:
            dataset = dataset.astype(str)
            response_json = {
                'dataset': dataset.to_json(force_ascii=False),
                'dataset_dtypes': dataset_dtypes
            }

            del payload['dataset']
            del payload['dataset_dtypes']

            self.insert_job_history(payload=payload, job_id="replace_col_value")

            return response_json
        else:
            return dataset, dataset_dtypes

    # 2-9 공백제거
    def remove_space(self, payload, df=None):
        dataset = self.get_df(payload=payload, df=df)
        column_name = payload['column_name']

        dataset[column_name] = dataset[column_name].str.strip()

        dataset_dtypes = self.get_dtype_of_dataframe(dataset)

        if df is None:
            dataset = dataset.astype(str)
            response_json = {
                'dataset': dataset.to_json(force_ascii=False),
                'dataset_dtypes': dataset_dtypes
            }

            del payload['dataset']
            del payload['dataset_dtypes']

            self.insert_job_history(payload=payload, job_id="replace_space")

            return response_json
        else:
            return dataset, dataset_dtypes

    # 2-10-1 중복 값 확인
    def row_control_show_duplicate_row(self, payload, df=None):
        dataset = self.get_df(payload=payload, df=df)
        columns = payload['columns']

        duplicate_values = dataset[columns].value_counts()
        dataset_dtypes = self.get_dtype_of_dataframe(dataset)

        if df is None:
            dataset = dataset.astype(str)
            response_json = {
                'dataset': dataset.to_json(force_ascii=False),
                'dataset_dtypes': dataset_dtypes,
                'duplicate_values': duplicate_values.to_json(force_ascii=False)
            }

            return response_json
        else:
            return dataset, dataset_dtypes

    def row_control_drop_duplicate_row(self, payload, df=None):
        dataset = self.get_df(payload=payload, df=df)
        columns = payload['columns']

        dataset = dataset.drop_duplicates(subset=columns)
        dataset_dtypes = self.get_dtype_of_dataframe(dataset)

        if df is None:
            dataset = dataset.astype(str)
            response_json = {
                'dataset': dataset.to_json(force_ascii=False),
                'dataset_dtypes': dataset_dtypes
            }

            del payload['dataset']
            del payload['dataset_dtypes']

            self.insert_job_history(payload=payload, job_id="replace_space")

            return response_json
        else:
            return dataset, dataset_dtypes
    # 3. 데이터 추출(저장)
    # parameter file_name, version
    def export_project(self, payload, method='minor'):
        patch = 0.01

        file_name = payload['file_name']
        project_name = payload['project_name']

        org_version = float(payload['version'])
        version = 0.00
        if method == 'minor':
            version = org_version + patch
        elif method == 'major':
            version = math.floor(org_version) + 1

        version = format(version, '.2f')

        ######################################
        # session 말고 DB에 저장 or request 로 받아올 예정 수정 필요 #
        ######################################
        self.app.logger.info('export_project version: ' + version)
        # url = "./server/" + project_name + '/p_data/' + file_name + '_V' + version + '_D' + date + '.json'
        url = "./server/" + project_name + '/p_data/' + file_name + '_V' + version + '.json'

        df = self.redo_job_history(payload=payload)
        df.to_json(url, force_ascii=False)
        print(df.head())
        print("#############")
        print(df.columns)
        df = df.astype(str)
        return df.to_json(force_ascii=False)

    # 3-1. job_history load
    def get_job_historys_by_file_name_and_version(self, payload):
        file_name = payload['file_name']
        version = payload['version']
        result_set = self.jhDAO.select_job_history_by_file_name_and_version(file_name=file_name,
                                                                            version=version)
        return result_set

    # 3-2. 추출 전 동작 재수행
    def redo_job_history(self, payload):
        result_set = self.get_job_historys_by_file_name_and_version(payload=payload)
        df = self.load_org_file(payload=payload)
        dataset_dtypes = self.get_dtype_of_dataframe(df=df)
        i = 1
        for row in result_set:
            job_id = row['job_id']
            content = row['content']
            self.app.logger.info('redo action ' + str(i) + ". " + str(job_id))
            i += 1
            content['dataset_dtypes'] = dataset_dtypes
            df, dataset_dtypes = self.redo_jobs(job_id=job_id, content=content, df=df)
        return df

    def redo_jobs(self, job_id, content, df):
        if job_id == 'missing_value':
            return self.missing_value(content, df=df)
        elif job_id == 'delete_column':
            return self.delete_column(content, df=df)
        elif job_id == 'calc_columns':
            return self.calc_columns_redo(content, df=df)
        elif job_id == 'set_col_prop':
            return self.set_col_prop(content, df=df)
        elif job_id == 'col_to_datetime':
            return self.set_col_prop_to_datetime(content, df=df)
        elif job_id == 'split_var_dt':
            return self.split_variable_datetime(content, df=df)
        elif job_id == 'dt_to_str_format':
            return self.dt_to_str_format(content, df=df)
        elif job_id == 'diff_datetime':
            return self.diff_datetime(content, df=df)
        else:
            print('EEEEEEEEEEEEEEEEEEEEEEERRRRRRRRRRRRRRRRRRROOOOOOOOOOR')

    def calc_columns_redo(self, content, df):
        calc_df = self.get_calc_dataset(content, df=df)
        dataset_dtypes = self.get_dtype_of_dataframe(calc_df)
        for payload in content['calc_job_history']:
            payload['dataset_dtypes'] = dataset_dtypes
            calc_df, dataset_dtypes = self.calculating_column(payload, df=calc_df)
        return self.select_calc_column_to_combine(content, origin_df=df, calc_df=calc_df)

    ########################################################################################
    ########################################################################################
    ########################################################################################
    ########################################################################################
    ########################################################################################

    # 확인용
    def show_dataset_all(self):
        dataset = self.dsDAO.select_dataset()

    def insert_test(self, payload):
        self.dsDAO.insert_test(payload)

    # 세션 정보 확인용
    def print_session_keys(self):
        for k in session.keys():
            print(k)

    # def redo_job_history(self, payload):
    #     result_set = self.get_dataset_jobs_in_session(payload=payload)
    #     version = payload['version']
    #     # session에 버전에 맞는 데이터셋 저장
    #     df_json = self.get_dataset_from_server(version)
    #
    #     for row in result_set:
    #         job_id = row['job_id']
    #         content = row['content']
    #         self.app.logger.info('redo action ' + str(job_id) + ' / ' + str(content))
    #
    #         content.update(df_json)
    #         df_json = self.redo_jobs(job_id=job_id, content=content)
    #     return self.sampling_dataset()

    # httpie 사용 시 session 유지 X 초기화됨
    # http -v --admin [method] url
    # 원래 파일 이름, DB에서 가져와야함
    # 테스트용~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_df_from_directory(self, url='./server/project01/origin_data/sampledtrain.json', method='minor'):
        project_name, file_name, version, date, extension = self.split_url(url)

        if extension == 'json':
            df = pd.read_json(url)
        if extension == 'csv':
            df = pd.read_csv(url)

        session['current_df'] = df.to_dict('list')
        session['current_filename'] = file_name
        session['current_version'] = version
        session['project_name'] = project_name
        session['extension'] = extension
        # minor, major upgrade 설정

    def get_dataset_from_server(self, version):
        url = './server/'
        url += session['project_name']
        url += '/p_data/'
        url += session['current_filename']
        url += '_V' + version + '.json'
        print(url)
        df = pd.read_json(url)

        self.app.logger.info('load_df_from_server / url = ' + url)
        session['current_version'] = version
        return df.to_json()

    # url 양식 /directory/<filename>_V<version>_D<dateime>.<extension>
    # input test : /directory/sampledtrain_V1.00_20220323.csv
    def split_url(self, url):
        full_url = url.split('/')[-1]
        project_name = url.split('/')[-3]
        file_name = '.'.join(full_url.split('.')[:-1])
        extension = full_url.split('.')[-1]

        if '_D' in file_name and '_V' in file_name:
            split_date = file_name.split('_D')
            split_version = split_date[0].split('_V')

            file_name = split_version[0]
            f_date = split_date[-1]
            version = float(split_version[1])
        else:
            # 신규 등록하는 데이터 셋일 때
            version = 1.00
            f_date = datetime.today().strftime('%Y%m%d')

        return project_name, file_name, version, f_date, extension

    def get_dataset_jobs_in_session(self, payload):
        file_name = session['file_name']
        version = payload['version']
        seq = payload['seq']
        result_set = self.dsDAO.select_dataset_jobs(file_name=file_name,
                                                    version=version,
                                                    seq=seq)
        for r in result_set:
            print(r)
        return result_set

    def init_dataset_table(self):
        self.dsDAO.init_dataset()

    def get_df(self, payload, df=None):
        if df is None:
            if 'calc_dataset' in payload:
                return self.get_calc_dataset_from_payload(payload=payload)
            else:
                return self.get_df_from_payload(payload)
        else:
            return df

    # DataFrame return
    def get_df_from_payload(self, payload, change_type=True):
        dataset = payload['dataset']
        if type(dataset) == dict:
            df = pd.DataFrame(dataset)
        else:
            df = pd.read_json(dataset)

        if 'dataset_dtypes' in payload and change_type is True:
            dataset_dtypes = dict(payload['dataset_dtypes'])
            print(dataset_dtypes)
            df = df.astype(dataset_dtypes)

        return df

    # method major 버전 증가 ex 1.05 -> 2.00
    # method minor 버전 증가 ex 2.04 -> 2.05
    # 임의 설정 파일 명 ./server/<projectname>/p_date/<filename>_V<version>.(csv, json)

    # 예외처리는 일단 나중으로 미루자

    def insert_dataset(self, payload, job_id):
        # 작업 내용 dataset Table insert
        # 사용한 함수,
        jcontent = json.dumps(payload)
        # id 정보, name 정보 = app 이나 database에서 가져오기 예상됨
        target_id = 'admin' + str(random.randint(100, 300))

        dataset = {
            'target_id': target_id,
            'version': session['current_version'],
            'name': session['current_filename'],
            'job_id': job_id,
            'content': jcontent
        }

        self.dsDAO.insert_dataset(dataset=dataset)

    # 결측 수식적용

    # 결측 모델 적용

    # 음수값 처리
    # 일단 단일 열 처리
    # column list or str
    # 음수 값 -> 양수로 method = 'positive'
    # 음수 값 -> 0     method = 'tozero'
    # 행 제거 ->       method = 'drop'
    def preprocessing_negative_value(self, columns, method='positive'):
        df = session['current_df']
        if method == 'drop':
            idx = df[df[columns] < 0].index()
            df = df.drop(idx)
        else:
            s = pd.DataFrame(df[columns])
            if method == 'positive':
                s[s < 0] = s[s < 0] * -1
            if method == 'tozero':
                s[s < 0] = 0
            # if method == 'delete':
            df[columns] = s
            df.to_csv('./preprocessing/data/sampledtrain_test.csv')

    # 소수점 처리

    # 소문자 변환

    # 첫 문자 대문자

    # 대문자 변환

    # 경과시간 계산

    # 날짜 형식 변환

    # 날짜 부분 추출

    # 연산

    # 수식 비교??

    # 내맘대로 해야지

    # 팝업창에서 데이터셋 검색 시 호출 -> 조회
    # /profile/{projectId}/data
    # pathparameter : projectid
    # bodyparameter : currentDatasetId, datasetName

    # 테이블 작업 - 삭제 - 비어있는 모든 행
    # 모든 컬럼의 데이터가 비어 있는 행을 삭제 처리한다.
