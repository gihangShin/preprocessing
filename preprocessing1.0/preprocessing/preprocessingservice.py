import json
import math
import os
import random
from datetime import datetime

import pandas as pd
from flask import session


class Preprocessing:
    target_id_seq = 1;

    def __init__(self, app, database):
        self.app = app
        self.db = database

    # 확인용
    def show_dataset_all(self):
        dataset = self.db.select_dataset()

    def insert_test(self, payload):
        self.db.insert_test(payload)

    # 세션 정보 확인용
    def print_session_keys(self):
        for k in session.keys():
            print(k)

    def move_job_history(self, payload):
        result_set = self.get_dataset_jobs_in_session(payload=payload)
        version = payload['version']
        seq = payload['seq']
        # session에 버전에 맞는 데이터셋 저장
        self.load_df_from_server(version)

        for row in result_set:
            job_id = row['job_id']
            content = row['content']
            row = 0
            self.app.logger.info('redo action ' + str(job_id) + ' / ' + str(content))
            self.redo_jobs(job_id=job_id, content=content)

    def redo_jobs(self, job_id, content):
        if job_id == 'missing_value':
            self.missing_value(content)
        elif job_id == 'delete_column':
            self.delete_column(content)

    # dataset 신규 등록만!!
    # 1. file을 server/project01/origin_data/ 저장
    # 2. file을 불러와서
    # server/project01/p_data/ <filename>_V<version>_D<dateime>.<extension> 형식 저장
    def upload_dataset(self, df, file_name, project_name='project01'):
        # 1. file을 server/project01/origin_data/ 저장
        project_dir = './server/' + project_name
        os.makedirs(project_dir + '/origin_data', exist_ok=True)
        os.makedirs(project_dir + '/p_data', exist_ok=True)
        file_name = file_name.split('/')[-1]
        org_url = project_dir + '/origin_data/' + file_name

        df.to_json(org_url)

        # 2. file을 불러와서
        # server/project01/p_data/ <filename>_V<version>_D<dateime>.<extension> 형식 저장
        project_name, file_name, version, date, extension = self.split_url(org_url)
        if extension == 'json':
            df = pd.read_json(org_url)
        elif extension == 'csv':
            df = pd.read_csv(org_url)
        new_url = project_dir + '/p_data/' + file_name + '_V' + str(round(version, 2)) + '_D' + date + '.' + extension
        df.to_json(new_url, force_ascii=False)

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
        self.print_session_keys()

    def load_df_from_server(self, version):
        url = './server/'
        url += session['project_name']
        url += '/p_data/'
        url += session['current_filename']
        url += '_V' + version + '.json'
        print(url)
        df = pd.read_json(url)

        self.app.logger.info('load_df_from_server / url = ' + url)
        session['current_df'] = df.to_dict('list')
        session['current_version'] = version

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
        file_name = session['current_filename']
        version = payload['version']
        seq = payload['seq']
        result_set = self.db.select_dataset_jobs(file_name=file_name,
                                                 version=version,
                                                 seq=seq)
        for r in result_set:
            print(r)
        return result_set

    def show_df_from_session(self):
        df = self.get_df_from_session()
        self.app.logger.info('session_dataframe\n' + df.head())

    def get_df_from_session(self):
        dict_obj = session['current_df']
        df = pd.DataFrame(dict_obj)
        return df

    def save_df_in_session(self, df):
        session['current_df'] = df.to_dict('list')

    # method major 버전 증가 ex 1.05 -> 2.00
    # method minor 버전 증가 ex 2.04 -> 2.05
    # 임의 설정 파일 명 ./server/<projectname>/p_date/<filename>_V<version>.(csv, json)
    def save_df_in_server(self, df=None, method='minor'):
        patch = 0.01

        df = self.get_df_from_session()
        file_name = session['current_filename']
        org_version = float(session['current_version'])
        # project_name = session['project_name']
        project_name = 'project01'
        date = datetime.today().strftime('%Y%m%d')
        version = 0.00
        if method == 'minor':
            version = org_version + patch
        elif method == 'major':
            version = math.floor(org_version) + 1

        version = format(version, '.2f')
        session['current_version'] = version
        self.app.logger.info('save_df_in_server version: ' + version)
        # url = "./server/" + project_name + '/p_data/' + file_name + '_V' + version + '_D' + date + '.json'
        url = "./server/" + project_name + '/p_data/' + file_name + '_V' + version + '.json'

        # url = './data/server_data/' + file_name + '_V' + version + '.json'
        df.to_json(url, force_ascii=False)

    # 예외처리는 일단 나중으로 미루자

    # 열 삭제
    def delete_column(self, payload):
        column_name = payload['column']
        self.app.logger.info('delete_column / ' + column_name)
        df = self.get_df_from_session()
        df = df.drop([column_name], axis=1)
        self.save_df_in_session(df)
        self.insert_dataset(payload, job_id='delete_column')

    # columns -> 파라미터
    # 열 연산만
    def missing_value(self, payload):
        missing_value = payload['m_value']
        columns = payload['columns']
        self.app.logger.info('missing value / ' + str(payload))
        if missing_value == 'remove':  # ok
            df = self.remove_missing_value(columns=columns)
        elif missing_value == 'mean':  # ok
            df = self.fill_missing_value_mean(columns=columns)
        elif missing_value == 'median':  # ok
            df = self.fill_missing_value_median(columns=columns)
        elif missing_value == 'ffill':  # ok
            df = self.fill_missing_value_front(columns=columns)
        elif missing_value == 'bfill':  # ok
            df = self.fill_missing_value_back(columns=columns)
        elif missing_value == 'first_row':  # 미구현
            df = self.fill_missing_value_first_row()
        elif missing_value == 'input':  # ok
            input_data = payload['input_data']
            df = self.fill_missing_value_specified_value(columns=columns, input_data=input_data)

        self.save_df_in_session(df)  # session에 df만 저장
        self.insert_dataset(payload=payload, job_id='missing_value')

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

        self.db.insert_dataset(dataset=dataset)

    def remove_missing_value(self, columns=None):
        df = self.get_df_from_session()
        if columns is None:
            df = df.dropna(axis=1)
        else:
            df = df.dropna(subset=[columns])
        return df

    # 이전값 채우기
    def fill_missing_value_front(self, columns=None):
        df = self.get_df_from_session()
        if columns is None:
            df = df.fillna(method='ffill', axis=1)
        else:
            df[[columns]] = df[[columns]].ffill()
        return df

    # 다음값 채우기
    def fill_missing_value_back(self, columns=None):
        df = self.get_df_from_session()
        if columns is None:
            df = df.fillna(method='bfill', axis=1)
        else:
            df[[columns]] = df[[columns]].bfill()
        return df

    def fill_missing_value_specified_value(self, input_data, columns=None):
        df = self.get_df_from_session()
        if columns is None:
            df = df.fillna(input_data)
        else:
            df[[columns]] = df[[columns]].fillna(value=input_data)
        return df

    def fill_missing_value_median(self, columns):
        df = self.get_df_from_session()
        df[columns] = df[columns].fillna(df[columns].median())
        return df

    def fill_missing_value_mean(self, columns):
        df = self.get_df_from_session()
        df[columns] = df[columns].fillna(df[columns].mean())
        return df

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
