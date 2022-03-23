import pandas as pd
import math
from flask import session
from datetime import datetime
import os
from werkzeug.utils import secure_filename
from flask_session import Session


class Preprocessing:
    app = None
    df = None

    def __init__(self, app):
        self.app = app

    # 세션 정보 확인용
    def print_session_keys(self):
        for k in session.keys():
            print(k)

    # dataset 신규 등록만!!
    # 받아올 수 있는 값 project name, file
    # 1. file을 server/project01/origin_data/ 저장
    # 2. file을 불러와서
    # server/project01/p_data/ <filename>_V<version>_D<dateime>.<extension> 형식 저장
    def upload_dataset(self, df, file_name, project_name='project01'):
        # 1. file을 server/project01/origin_data/ 저장
        project_dir = './server/' + project_name
        # 서버 폴더 내 프로젝트명으로 된 디렉토리 생성
        os.makedirs(project_dir + '/origin_data', exist_ok=True)
        os.makedirs(project_dir + '/p_data', exist_ok=True)
        # 이후 실제 file 객체 불러올 때
        # file_name = secure_filename(file.filename)
        file_name = file_name.split('/')[-1]
        org_url = project_dir + '/origin_data/' + file_name

        # file.save(org_url)
        df.to_json(org_url)

        # 2. file을 불러와서
        # server/project01/p_data/ <filename>_V<version>_D<dateime>.<extension> 형식 저장

        file_name, version, date, extension = self.split_url(org_url)
        if extension == 'json':
            df = pd.read_json(org_url)
        elif extension == 'csv':
            df = pd.read_csv(org_url)
        new_url = project_dir + '/p_data/' + file_name + '_V' + str(round(version,2)) + '_D' + date + '.' + extension
        df.to_json(new_url)

    # httpie 사용 시 session 유지 X 초기화됨
    # http -v --admin [method] url
    # 원래 파일 이름, DB에서 가져와야함
    # 원본파일은 버전이 없음 -> 버전정보 나중에 구현
    def load_df_from_directory(self, url='./data/origin_data/sampledtrain.csv', method='minor'):
        file_name, version, date, extension = self.split_url(url)
        # file_name, extension = self.split_url(url)
        if extension == 'json':
            df = pd.read_json(url)
        if extension == 'csv':
            df = pd.read_csv(url)
        self.save_df_in_session(df)
        session['current_df'] = df.to_dict('list')
        session['current_filename'] = file_name
        session['current_version'] = version
        session['extension'] = extension
        # 파일을 읽어보고 version == 1.00 -> 첫 데이터
        if version == 1.00:
            self.save_df_in_directory(method='first')
        else:
            # minor, major upgrade 설정
            self.save_df_in_directory(method=method)
        self.print_session_keys()

    # url 양식 /directory/<filename>_V<version>_D<dateime>.<extension>
    # input test : /directory/sampledtrain_V1.00_20220323.csv
    def split_url(self, url):
        full_url = url.split('/')[-1]
        file_name = '.'.join(full_url.split('.')[:-1])
        extension = full_url.split('.')[-1]

        # 처음 들어오는 데이터셋인지 확인(filename 양식으로 확인)
        # 이후 db에서 값을 가져와서 확인도 가능
        if '_D' in file_name and '_V' in file_name:
            # 기존 파일 일 때
            # 날짜정보 _D 로 구분 추출
            split_date = file_name.split('_D')
            f_date = split_date[-1]
            split_array = split_date[0].split('_V')
            file_name = split_array[0]
            version = float(split_array[1])
        else:
            # 신규 등록하는 데이터 셋일 때
            version = 1.00
            f_date = datetime.today().strftime('%Y%m%d')
        print('filename')
        print(file_name)
        print('version')
        print('%.2f' % version)
        print('datetime')
        print(f_date)
        print('extension')
        print(extension)
        return file_name, version, f_date, extension
        # return file_name_version, extension

    # type = (originfile, preprocessed)
    # 업로드는 나중에
    # def upload_dataset_to_server_directory(self, url):

    def show_df_from_session(self):
        df = self.get_df_from_session()
        print(df.head())

    def get_df_from_session(self):
        dict_obj = session['current_df']
        df = pd.DataFrame(dict_obj)
        return df

    def save_df_in_session(self, df):
        session['current_df'] = df.to_dict('list')

    # method major 버전 증가 ex 1.05 -> 2.00
    # method minor 버전 증가 ex 2.04 -> 2.05
    # 임의 설정 파일 명 ./data/server_data/<filename>_V<version>.(csv, json)
    def save_df_in_directory(self, method='minor'):
        patch = 0.01
        df = self.get_df_from_session()
        file_name = session['current_filename']
        if method == 'first':
            version = 1.00
        else:
            org_version = float(session['current_version'])
            version = 0.00
            if method == 'minor':
                version = org_version + patch
            elif method == 'major':
                version = math.floor(org_version) + 1

        version = format(version, '.2f')

        url = './data/server_data/' + file_name + '_V' + version + '.json'
        df.to_json(url, force_ascii=False)

    # def __init__(self, app, url)
    # url db에서 정보 불러오기
    # def __init__(self, app):
    # df = pd.read_csv('./preprocessing/data/sampledtrain.csv', sep=',')
    # app.dataFrame = df
    # self.app = app
    # self.df = df

    # 예외처리는 일단 나중으로 미루자

    # 데이터 처리
    # 결측치 삭제 행, 열
    # axis = 0 -> 행 삭제
    # axis = 1 -> 열 삭제

    # columns -> 파라미터
    # 열 연산만
    def missing_value(self, missing_value, columns=None, input_data=None):
        # missing_value
        print('missing_value before')
        self.show_df_from_session()
        if missing_value == 'remove':  # ok
            df = self.remove_missing_value(columns=columns)
        elif missing_value == 'mean':  # ok
            df = self.fill_missing_value_mean(columns=columns)
        elif missing_value == 'median':  # ok
            df = self.fill_missing_value_median(columns=columns)
        elif missing_value == 'ffill':  # ok
            df = self.fill_missing_value_front(col=columns)
        elif missing_value == 'bfill':  # ok
            df = self.fill_missing_value_back(columns=columns)
        elif missing_value == 'first_row':
            df = self.fill_missing_value_first_row()
        elif missing_value == 'input':  # ok
            df = self.fill_missing_value_specified_value(columns=columns, input_data=input_data)

        print('missing_value after')
        self.save_df_in_session(df)  # session에 df 저장
        self.show_df_from_session()

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
    # GET
    # 데이터셋 검색
    # pathparameter : projectid
    # bodyparameter : currentDatasetId, datasetName
    ## 데이터셋을 db에서 호출??
    ## 서버 내 가공파일 호출???
    # json 타입으로 리턴
    # request.json을 그냥 받아오는게 깔끔할지도
    def showDataset(self, projectId, currentDatasetId, datasetName):
        url = './preprocessing/' + projectId + '/data/' + datasetName
        df = pd.read_csv(url, sep=',')
        return df

    # 테이블 작업 - 삭제 - 비어있는 모든 행
    # 모든 컬럼의 데이터가 비어 있는 행을 삭제 처리한다.
