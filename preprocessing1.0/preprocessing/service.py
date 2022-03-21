import pandas as pd

class preprocessingService:

    # csv 읽기
    def getbankcsv(self):
        df = pd.read_csv('./preprocessing/data/bank.csv')
        return df

    def getsampletraincsv(self):
        df = pd.read_csv('./preprocessing/data/sampled_train.csv', sep=',')
        return df

    # 데이터 처리
    # 결측치 삭제 행, 열
    # axis = 0 -> 행 삭제
    # axis = 1 -> 열 삭제
    def delete_missing_value(self, axis):
        df = pd.read_csv('./preprocessing/data/bank.csv', sep=',')
        print(df.head())
        df = df.dropna(axis=axis)
        print(df.head())
        df.to_csv('./preprocessing/data/bank2.csv')



    # 이전값 채우기
    def fill_missing_value_pre(self):
        df = pd.read_csv('./preprocessing/data/bank.csv', sep=',')
        print(df.describe())
        df = df.fillna(method='ffill')
        print(df.describe())
        df.to_csv('./preprocessing/data/bank2.csv')

    # 다음값 채우기
    # 지정값 채우기
    # 표준값 채우기
    # 결측 수식적용
    # 결측 모델 적용









































    # 내맘대로 해야지


    # 팝업창에서 데이터셋 검색 시 호출 -> 조회
    #/profile/{projectId}/data
    #GET
    #데이터셋 검색
    # pathparameter : projectid
    # bodyparameter : currentDatasetId, datasetName
    ## 데이터셋을 db에서 호출??
    ## 서버 내 가공파일 호출???
    # json 타입으로 리턴
    # request.json을 그냥 받아오는게 깔끔할지도
    def showDataset(self, projectId, currentDatasetId, datasetName):
        url = './preprocessing/'+projectId+'/data/'+datasetName
        df = pd.read_csv(url, sep=',')
        return df

    # 테이블 작업 - 삭제 - 비어있는 모든 행
    # 모든 컬럼의 데이터가 비어 있는 행을 삭제 처리한다.



