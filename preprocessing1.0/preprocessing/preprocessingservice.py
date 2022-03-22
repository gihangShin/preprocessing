import pandas as pd
import math
from flask import session


class PreprocessingService:
    app = None
    df = None

    # 원래 파일 이름, DB에서 가져와야함
    def load_df_from_directory(self, url):
        file_name, version, extension = self.split_url(url)

        if extension == 'json':
            df = pd.read_json(url)
        if extension == 'csv':
            df = pd.read_csv(url)

        session['current_df'] = df
        session['current_filename'] = file_name
        session['current_version'] = version

    def split_url(self, url):
        full_name = url.split('/')[-1]
        file_name_version = full_name.split('.')[0]
        extension = full_name.split('.')[1]

        file_name = file_name_version.split('_')[0]
        version = file_name_version.split('_')[1]
        return (file_name, version, extension)

    # type = (originfile, preprocessed)
    # 업로드는 나중에
    # def upload_dataset_to_server_directory(self, url):



    def get_df_in_session(self):
        return session['current_df']

    def save_df_in_session(self, df):
        session['current_df'] = df

    # method major 버전 증가 ex 1.05 -> 2.00
    # method minor 버전 증가 ex 2.04 -> 2.05
    # 임의 설정 파일 명 ./directory/pdata/<filename>_<version>.(csv, json)
    def save_df_in_directory(self, method='minor'):
        patch = 0.01
        df = session['current_df']
        file_name = session['current_filename']
        org_version = float(session['current_version'])
        version = 0.0
        if method == 'minor':
            version = org_version + patch
        else:
            version = math.floor(org_version) + 1
        version = format(version, '.2f')

        url = './preprocessing/pdata/' + file_name + '_' + version + '.json'
        df.to_json(url)

    # def __init__(self, app, url)
    # url db에서 정보 불러오기
    # def __init__(self, app):
    # df = pd.read_csv('./preprocessing/data/sampled_train.csv', sep=',')
    # app.dataFrame = df
    # self.app = app
    # self.df = df

    # csv 읽기
    def getbankcsv(self):
        df = pd.read_csv('./preprocessing/data/bank.csv')
        return df

    def getsampletraincsv(self):
        return self.df

    # 예외처리는 일단 나중으로 미루자

    # 데이터 처리
    # 결측치 삭제 행, 열
    # axis = 0 -> 행 삭제
    # axis = 1 -> 열 삭제
    def delete_missing_value(self, axis=0):
        self.df = self.df.dropna(axis=axis)
        self.df.to_csv('./preprocessing/data/sampled_traindropna.csv')
        self.app.dataFrame = self.df

    # 이전값 채우기
    def fill_missing_value_pre(self, columns=None):
        print(self.df.isna().sum())
        print(self.df.head())

        if columns is None:
            # 지정한 column이 없을 시 전체 지정 값 채우기
            self.df = self.df.fillna(method='ffill')
        else:
            # 지정한 column이 있을 시 해당 열만 지정 값 채우기
            # columns dtype == 리스트 or 문자열
            self.df = self.df.fillna(method='ffill', columns=columns)
        print(self.df.isna().sum())
        print(self.df.head())
        self.df.to_csv('./preprocessing/data/sampled_trainffill.csv')

    # 다음값 채우기
    def fill_missing_value_back(self, columns=None):
        print(self.df.head())

        if columns is None:
            # 지정한 column이 없을 시 전체 지정 값 채우기
            self.df = self.df.fillna(method='bfill')
        else:
            # 지정한 column이 있을 시 해당 열만 지정 값 채우기
            # columns dtype == 리스트 or 문자열
            self.df = self.df.fillna(method='bfill', columns=columns)
        print(self.df.head())
        self.df.to_csv('./preprocessing/data/sampled_trainbfill.csv')

    # 지정값 채우기 specified value
    def fill_missing_value_specified_value(self, specified_value, columns=None):
        print(self.df.head())

        if columns is None:
            # 지정한 column이 없을 시 전체 지정 값 채우기
            self.df = self.df.fillna(specified_value)
        else:
            # 지정한 column이 있을 시 해당 열만 지정 값 채우기
            # columns dtype == 리스트 or 문자열
            self.df = self.df.fillna(specified_value, columns=columns)

        print(self.df.head())
        self.df.to_csv('./preprocessing/data/sampled_trainbfill.csv')

    # 표준값 채우기
    # 한번에 한 열씩 동작 가정

    def fill_missing_value_std(self, column, method='mean'):
        # 평균값 mean
        if method == 'mean':
            self.df[column] = self.df[column].fillna(self.df[column].mean())
        # 중앙값 median
        if method == 'median':
            self.df[column] = self.df[column].fillna(self.df[column].median())

    # 결측 수식적용

    # 결측 모델 적용

    # 음수값 처리
    # 일단 단일 열 처리
    # column list or str
    # 음수 값 -> 양수로 method = 'positive'
    # 음수 값 -> 0     method = 'tozero'
    # 행 제거 ->       method = 'drop'
    def preprocessing_negative_value(self, columns, method='positive'):
        if method == 'drop':
            idx = self.df[self.df[columns] < 0].index()
            self.df = self.df.drop(idx)
        else:
            s = pd.DataFrame(self.df[columns])
            if method == 'positive':
                s[s < 0] = s[s < 0] * -1
            if method == 'tozero':
                s[s < 0] = 0
            # if method == 'delete':
            self.df[columns] = s
            self.df.to_csv('./preprocessing/data/sampled_train_test.csv')

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
