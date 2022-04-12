import json
import math
import os
import random
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
        self.app.logger.info('sampling parameter X -> Default SEQ/ROW/FRT/50')
        sampling_method = 'SEQ'
        ord_value = 5000
        ord_row = 'ROW'
        ord_set = 'FRT'

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
        self.app.logger.info('%s [%s]' % (ds.job_id, str(ds.job_params)))
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
        elif job_id == 'drop_row':
            ds = self.drop_row(ds)
        elif job_id == 'rename_col':
            ds = self.rename_col(ds)
        elif job_id == 'split_col':
            ds = self.split_col(ds)
        elif job_id == 'missing_data_model':
            ds = self.missing_data_model(ds)
        elif job_id == 'unit_conversion':
            ds = self.unit_conversion(ds)
        else:
            print('ERRORERRORERRORERRORERROR')
        return ds

    ##########################################################################
    ##########################################################################
    # 1. 전처리 동작
    ##########################################################################
    # 1-1. 열 삭제
    def delete_column(self, ds):
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
        ds.data_types = ds.get_types()
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

        ds.data_types = ds.get_types()
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

        ds.data_types = ds.get_types()
        return ds

    ##########################################################################
    # 1-8. 컬럼 순서 변경
    def change_column_order(self, ds):
        ds.dataset = ds.dataset.iloc[:, list(ds.job_params['col_order_list'])]
        ds.data_types = ds.get_types()

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

        ds.data_types = ds.get_types()
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

        ds.data_types = ds.get_types()
        return ds

    ##########################################################################
    # 1-11. 공백 제거 (앞 뒤만 해당, 문자 사이 X)
    def remove_space_front_and_rear(self, ds):
        ds.dataset[ds.job_params['column']] = ds.dataset[ds.job_params['column']].str.strip()
        ds.data_types = ds.get_types()
        return ds

    ##########################################################################
    # 1-12. 중복 행 제거
    def drop_duplicate_row(self, ds):
        if 'keep' in ds.job_params:
            ds.dataset.drop_duplicates(subset=ds.job_params['columns'], keep=ds.job_params['keep'], inplace=True)
        else:
            ds.dataset.drop_duplicates(subset=ds.job_params['columns'], keep='first', inplace=True)
        ds.data_types = ds.get_types()
        return ds

    ##########################################################################
    # 조회 1. 중복 값 확인                                            (단순 조회)
    def show_duplicate_row(self, ds):
        return ds.dataset[ds.job_params['column']].value_counts()

    ##########################################################################
    # 1-13. 연산
    def calculating_column(self, ds):
        if ds.job_params['method'] == 'arithmetic':
            ds = self.calc_arithmetic(ds)

        elif ds.job_params['method'] == 'function':
            ds = self.calc_function(ds)
        ds.data_types = ds.get_types()
        return ds

    # 1-13-1. 연산 동작(함수 선택 시)
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

    # 1-13-2. 연산 동작(산술 연산 선택 시)
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

    # 1-13-3. 두번째 피연산자 == 컬럼의 집계값 사용 시
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

    ##########################################################################
    # 14. 열 삭제
    def drop_row(self, ds):
        drop_type = ds.job_params['type']
        if drop_type == 'INPT':
            # 지정 값 일치 삭제 INPT
            index = ds.dataset[ds.dataset[ds.job_params['column']] == ds.job_params['input']].index
        elif drop_type == 'INVL':
            # 유효하지 않은 데이터 삭제 INVL
            # 일단 결측 삭제
            index = ds.dataset[ds.dataset[ds.job_params['column']].isna() == True].index
        elif drop_type == 'NEGA':
            # 음수 값 로우 삭제 NEGA
            if ds.data_types[ds.job_params['column']] not in ('int', 'float', 'int64', 'float64'):
                self.app.logger.info('column [%s] is not (Int, float) type' % ds.job_params['column'])
                return ds
            index = ds.dataset[ds.dataset[ds.job_params['column']] < 0].index
        else:
            pass
        ds.dataset.drop(index, inplace=True, axis=0)
        ds.data_types = ds.get_types()
        return ds

    ##########################################################################
    # 15. 컬럼 이름 변경
    def rename_col(self, ds):
        ds.dataset.rename(columns={ds.job_params['column_name']: ds.job_params['new_column_name']}, inplace=True)
        ds.data_types = ds.get_types()
        return ds

    ##########################################################################
    # 16. 컬럼 분할 ( 구분자, 컬럼 길이로 분할, 역분할(뒤에서 부터))
    def split_col(self, ds):
        if ds.job_params['type'] == 'SEP':
            col_name = '%s_%s_by_%s' % (ds.job_params['type'], ds.job_params['column'], ds.job_params['input'])
            ds.dataset[col_name] = ds.dataset[ds.job_params['column']].str.split(ds.job_params['input']).str[
                ds.job_params['position'] - 1]

        elif ds.job_params['type'] == 'LEN':
            if ds.job_params['reverse'] is True:
                input_value = abs(int(ds.job_params['input'])) * -1
                col_name = '%s_%s_by_%s' % (ds.job_params['type'], ds.job_params['column'], str(input_value))
                ds.dataset[col_name] = ds.dataset[ds.job_params['column']].str[input_value:]
            else:
                input_value = abs(int(ds.job_params['input']))
                col_name = '%s_%s_by_%s' % (ds.job_params['type'], ds.job_params['column'], str(input_value))
                ds.dataset[col_name] = ds.dataset[ds.job_params['column']].str[:input_value]
        ds.data_types = ds.get_types()
        return ds

    ##########################################################################
    # 17. 결측치 처리 머신 러닝 모델 활용
    def missing_data_model(self, ds):
        # 분류, 회귀 구분
        # method

        if ds.job_params['method'] == 'regression':
            return self.regression_model(ds)
        elif ds.job_params['method'] == 'classification':
            return self.classification_model(ds)

    def regression_model(self, ds):
        X_target_is_not_0, X_target_is_0 = self.split_null_or_not(ds)

        models = list()
        models.append(LinearRegression())
        models.append(Lasso())

        model = self.test_model(X_target_is_not_0, models, ds.job_params['target_column'])
        ds.dataset[ds.job_params['target_column']] = self.execute_model(ds, X_target_is_not_0, X_target_is_0, model)
        ds.data_types = ds.get_types()
        return ds

    def classification_model(self, ds):
        X_target_is_not_0, X_target_is_0 = self.split_null_or_not(ds)

        models = list()
        models.append(LogisticRegression(solver='saga', max_iter=2000))
        models.append(RandomForestClassifier(max_depth=10))

        model = self.test_model(X_target_is_not_0, models, ds.job_params['target_column'])
        ds.dataset[ds.job_params['target_column']] = self.execute_model(ds, X_target_is_not_0, X_target_is_0, model)
        ds.data_types = ds.get_types()
        return ds

    def split_null_or_not(self, ds):
        target_column = ds.job_params['target_column']
        feature_list = list(ds.job_params['feature_list'])

        df = pd.DataFrame()
        # feature_scaling
        for feature in feature_list:
            if ds.data_types[feature] in ('categories', 'string', 'object'):
                # 그냥 다 더미화
                scaled_data = pd.get_dummies(ds.dataset[feature])
                scaled_df = scaled_data
            if ds.data_types[feature] in ('int64', 'float64', 'int', 'float'):
                # 그냥 다 정규화 해버리자
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(ds.dataset[[feature]])
                scaled_df = pd.DataFrame(scaled_data, columns=[feature])
            if ds.data_types[feature] in ('datetime64[ns]'):
                # 그냥 다 정규화 해버리자
                scaled_df = pd.DataFrame()
                scaled_df['year'] = ds.dataset[feature].dt.year
                scaled_df['month'] = ds.dataset[feature].dt.month
                scaled_df['day'] = ds.dataset[feature].dt.day

            df = pd.concat([df, scaled_df], axis=1)
        df = pd.concat([df, ds.dataset[target_column]], axis=1)
        print(df.columns)
        df[target_column].fillna(0, inplace=True)
        X_target_is_0 = df[df[target_column] == 0]
        X_target_is_not_0 = df[df[target_column] != 0]

        print(X_target_is_0.head())
        print(X_target_is_not_0.head())

        return X_target_is_not_0, X_target_is_0

    def test_model(self, data_is_not_0, models, target_column):
        target = data_is_not_0[target_column]
        data_is_not_0.drop(columns=[target_column], inplace=True)
        x_train, x_test, y_train, y_test = train_test_split(data_is_not_0,
                                                            target,
                                                            test_size=0.2, random_state=5)

        RMSE = list()
        i = 0
        for model in models:
            model.fit(x_train, y_train)
            RMSE.append(mean_squared_error(y_test, model.predict(x_test), squared=False))
            print('RMSE_model_%d : %f' % (i + 1, RMSE[i]))
            i += 1

        tmp = min(RMSE)
        index = RMSE.index(tmp)

        print("%s / %f" % (models[index], tmp))

        return models[index]

    def execute_model(self, ds, X_target_is_not_0, X_target_is_0, model):
        y_train = ds.dataset[[ds.job_params['target_column']]]
        y_train.columns = [ds.job_params['target_column']]

        X_target_is_0.drop(columns=[ds.job_params['target_column']], inplace=True)
        target_predict = model.predict(X_target_is_0)
        X_target_is_0[ds.job_params['target_column']] = target_predict
        X_target_is_not_0[ds.job_params['target_column']] = y_train

        data = pd.concat([X_target_is_not_0, X_target_is_0])
        data[ds.job_params['target_column']] = data[ds.job_params['target_column']].round(2)
        data.sort_index(inplace=True)

        return data[ds.job_params['target_column']]

    ##########################################################################
    # 조회 2. 수식 비교 조회 ex) 몸무게 > 70 인 row
    def show_conditioned_row(self, ds):
        column = ds.job_params['column']
        operator = ds.job_params['operator']
        value = ds.job_params['value']

        if operator == '==':
            # 다른 타입도 가능
            return ds.dataset[ds.dataset[column] == value]
        elif operator == '!=':
            return ds.dataset[ds.dataset[column] != value]
        elif operator == '<=':
            return ds.dataset[ds.dataset[column] <= value]
        elif operator == '<':
            return ds.dataset[ds.dataset[column] < value]
        elif operator == '>=':
            return ds.dataset[ds.dataset[column] >= value]
        elif operator == '>':
            return ds.dataset[ds.dataset[column] > value]
        elif operator == 'between':
            value2 = ds.job_params['value2']
            return ds.dataset[(value <= ds.dataset[column]) & (value2 >= ds.dataset[column])]
        elif operator == 'else':
            value2 = ds.job_params['value2']
            return ds.dataset[(value > ds.dataset[column]) | (value2 < ds.dataset[column])]
        else:
            return 'ERROR'

    ##########################################################################
    # 18. 단위 변환 ex) kg -> g
    def unit_conversion(self, ds):

        method = ds.job_params['method']
        current_unit = ds.job_params['current_unit']
        conversion_unit = ds.job_params['conversion_unit']

        column = ds.job_params['column']
        if current_unit in column:
            column_name = column.split('(')[0]
        else:
            column_name = column

        new_column_name = "%s(%s)" % (column_name, conversion_unit)

        if method == 'temperature':
            if current_unit == 'Celsius':
                # 섭씨 -> 화씨
                # F = C(1.8) + 32
                ds.dataset[new_column_name] = (ds.dataset[column] * 1.8) + 32
            elif current_unit == 'Fahrenheit':
                # 화씨 -> 섭씨
                # C = (F-32) / 1.8
                ds.dataset[new_column_name] = (ds.dataset[column] - 32) / 1.8
        else:
            unit_data = self.get_unit(method)
            ds.dataset[new_column_name] = ds.dataset[column] / unit_data[current_unit] * unit_data[conversion_unit]
        ds.dataset[new_column_name] = ds.dataset[new_column_name].round(2)
        ds.data_types = ds.get_types()
        return ds

    def get_unit(self, method):
        if method == 'length':
            # 기준 m
            return {
                "cm": 100,
                "mm": 1000,
                "m": 1,
                "km": 0.001,
                "in": 39.370079,
                "ft": 3.28084,
                "yd": 1.093613,
                "mile": 0.000621
            }
        elif method == 'weight':
            # 기준 kg
            return {
                "mg": 1000000,
                "g": 1000,
                "kg": 1,
                "t": 0.001,
                "kt": 1e-6,
                "gr": 15432.3584,
                "oz": 35.273962,
                "lb": 2.204623
            }
        elif method == 'area':
            # 기준 m^2
            return {
                "m^2": 1,
                "a": 0.01,
                "ha": 0.0001,
                "km^2": 1e-6,
                "ft^2": 10.76391,
                "yd^2": 15432.3584,
                "ac": 0.000247105,
                "평": 0.3025
            }
        elif method == 'volume':
            # 기준 m
            return {
                "l": 1,
                "cc": 1000,
                "ml": 1000,
                "dl": 10,
                "cm^3": 1000,
                "m^3": 0.001,
                "in^3": 61.023744,
                "ft^3": 0.035314667,
                "yd^3": 0.001307951,
                "gal": 0.264172052,
                "bbl": 0.0062932662
            }
        elif method == 'speed':
            # 기준 m/s
            return {
                "m/s": 1,
                "m/h": 3600,
                "km/s": 0.001,
                "km/h": 3.6,
                "in/s": 39.370079,
                "in/h": 141732.283,
                "ft/s": 3.28084,
                "ft/h": 11811.0236,
                "mi/s": 0.000621,
                "mi/h": 2.236936,
                "kn": 1.943844,
                "mach": 0.002941
            }

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
            i += 1
            self.app.logger.info('redo action ' + str(i) + ". ")
            ds = self.redirect_preprocess(ds=ds)
        return ds

    # 3-1. job_history load
    def get_job_historys(self, ds):
        return self.jhDAO.select_job_history_by_file_name_and_version(file_name=ds.file_id,
                                                                      version=ds.version)
