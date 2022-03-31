import json
import math
import os
import random
from datetime import datetime

import pandas as pd
import numpy as np
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

        if extention == 'json':
            df_json = payload['file']
            df = pd.read_json(df_json)
        elif extention == 'csv':
            df_csv = payload['file']
            df = pd.read_csv(df_csv)
        df.to_json(org_url, force_ascii=False)

        version = 1.0
        new_url = project_dir + '/p_data/' + file_name + '_V' + str(round(version, 2)) + '.json'
        df.to_json(new_url, force_ascii=False)

        session['project_name'] = project_name
        session['file_name'] = file_name
        session['version'] = version

        self.app.logger.info('org_url : ' + org_url)
        self.app.logger.info('new_url : ' + new_url)

        return self.sampling_dataset(df).to_json()

    # 1-2
    # 기존 파일 불러오기
    def load_existed_file(self, payload):
        project_name = payload['project_name']
        file_name = payload['file_name']
        version = payload['version']

        session['project_name'] = project_name
        session['file_name'] = file_name
        session['version'] = version

        df = self.load_org_file(payload=payload)

        msg = project_name + '/' + file_name + '/V' + version
        self.app.logger.info('load_existed_file : ' + msg)

        return self.sampling_dataset(df).to_json()

    # 1-2-2. 파일 load 모듈화 Dataframe 객체 반환
    def load_org_file(self, payload=None):
        print('###############################')
        print(payload)
        if payload is None:
            project_name = session['project_name']
            file_name = session['file_name']
            version = session['version']
        else:
            project_name = payload['project_name']
            file_name = payload['file_name']
            version = payload['version']

        project_dir = './server/' + project_name
        url = project_dir + '/p_data/' + file_name + '_V' + version + '.json'
        self.app.logger.info('load_org_file_url : ' + url)
        df = pd.read_json(url)
        return df

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

    # 2-0. 동작 내역 DB 저장 함수
    def insert_job_history(self, payload, job_id):
        content_json = json.dumps(payload)
        target_id = 'admin' + str(random.randint(100, 300))

        job_history = {
            'file_name': session['file_name'],
            'job_id': job_id,
            'version': session['version'],
            'job_request_user_id': 'user123',
            'content': content_json
        }

        self.jhDAO.insert_job_history(job_history=job_history)

    # 2-1. 열 삭제

    # 열 삭제
    def delete_column(self, payload, redo=False):
        column_name = payload['column']
        dataset_json = payload['dataset']
        del payload['dataset']

        dataset = pd.read_json(dataset_json)

        self.app.logger.info('delete_column / ' + column_name)
        dataset = dataset.drop([column_name], axis=1)
        if redo is False:
            self.insert_job_history(payload, job_id='delete_column')
        return dataset.to_json(force_ascii=False)

    # 2-2. 결측치 처리
    # 열 연산만
    def missing_value(self, payload, redo=False):
        missing_value = payload['m_value']
        columns = payload['columns']
        dataset_json = payload['dataset']
        del payload['dataset']

        dataset = pd.read_json(dataset_json)
        self.app.logger.info('missing value / ' + str(payload))
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
        if redo is False:
            self.insert_job_history(payload=payload, job_id='missing_value')

        return dataset.to_json(force_ascii=False)

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
    def get_calc_dataset(self, payload, redo=False):
        df = self.get_df_from_payload(payload)
        # calc_df == 계산 용 dataset
        # 그냥 수치형 다 받아옴
        calc_df = df.select_dtypes(include=['int64', 'float64'])

        if redo is False:
            return calc_df.to_json
        else:
            return calc_df

    # 2-3-2. 연산
    # 테스트 완료
    def calculating_column(self, payload, redo=False):
        # method == arithmetic -> column1, (column2 or scala or 집계 데이터), operator
        #           return 연산 완료+추가된 calc_dataset
        # method == function(aggregate, Statistical) -> function, (column or scala)
        #           return 연산 완료+추가된 calc_dataset
        method = payload['method']

        if method == 'arithmetic':
            calc_df = self.calc_arithmetic(payload=payload)

        elif method == 'function':
            calc_df = self.calc_function(payload=payload)



        if redo is False:
            result = {
                'calc_dataset': calc_df.to_json(force_ascii=False)
            }
            # DB 저장은 마지막 수행
            del payload['dataset']
            del payload['calc_dataset']

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
            return calc_df.to_json(force_ascii=False)

    # 2-3-2-1. 연산 동작(함수 선택 시)
    def calc_function(self, payload):
        calc_df = pd.read_json(payload['calc_dataset'])
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
    def calc_arithmetic(self, payload):
        calc_df = pd.read_json(payload['calc_dataset'])
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
            operand2, column2 = self.calc_function_column(payload=payload)
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
    def calc_function_column(self, payload):
        column2 = payload['value']
        function = payload['column_function']
        df = self.get_df_from_payload(payload=payload)
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

    # 2-3-3. calc_dataset 에서 추출할 컬럼 선택 후 기존 데이터셋(sampled_dataset)으로 결합
    def select_calc_column_to_combine(self, payload, redo=False):
        selected_columns = payload['selected_columns']
        calc_dataset = pd.read_json(payload['calc_dataset'])
        dataset = pd.read_json(payload['dataset'])

        if redo is False:
            # 추출 시 재수행 동작이 아니라면 DB저장
            job_history = payload['calc_job_history']
            calc_payload = {
                'selected_columns': selected_columns,
                'calc_job_history': job_history  # list[json] 형식
            }
            self.insert_job_history(payload=calc_payload, job_id='calc_columns')

        result_dataset = calc_dataset[[selected_columns]]
        dataset = pd.concat([dataset, result_dataset], axis=1)

        return dataset.to_json(force_ascii=False)

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
        session['version'] = version

        self.app.logger.info('export_project version: ' + version)
        # url = "./server/" + project_name + '/p_data/' + file_name + '_V' + version + '_D' + date + '.json'
        url = "./server/" + project_name + '/p_data/' + file_name + '_V' + version + '.json'

        df = self.redo_job_history(payload=payload)

        df.to_json(url, force_ascii=False)

    # 3-1. job_history load
    def get_job_historys_by_file_name_and_version(self, payload):
        file_name = payload['file_name']
        version = payload['version']
        result_set = self.jhDAO.select_job_history_by_file_name_and_version(file_name=file_name,
                                                                            version=version)
        for r in result_set:
            print(r)
        return result_set

    # 3-2. 추출 전 동작 재수행
    def redo_job_history(self, payload):
        result_set = self.get_job_historys_by_file_name_and_version(payload=payload)

        df = self.load_org_file(payload=payload)
        df_json = df.to_json(force_ascii=False)
        for row in result_set:
            job_id = row['job_id']
            content = row['content']
            self.app.logger.info('redo action ' + str(job_id) + ' / ' + str(content))
            content['dataset'] = df_json
            df_json = self.redo_jobs(job_id=job_id, content=content)
            df_1 = pd.read_json(df_json)
        df = pd.read_json(df_json)
        return df

    def redo_jobs(self, job_id, content):
        if job_id == 'missing_value':
            return self.missing_value(content, redo=True)
        elif job_id == 'delete_column':
            return self.delete_column(content, redo=True)
        elif job_id == 'calc_columns':
            return self.calc_columns_redo(content)

    def calc_columns_redo(self, content):
        calc_dataset = self.get_calc_dataset(content)
        for payload in content['calc_job_history']:
            payload['calc_dataset'] = calc_dataset.to_json(force_ascii=False)
            calc_dataset = self.calculating_column(payload, redo=True)

        return self.select_calc_column_to_combine(content, redo=True)

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

    # DataFrame return
    def get_df_from_payload(self, payload):
        df_json = payload['dataset']
        df = pd.read_json(df_json)
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
