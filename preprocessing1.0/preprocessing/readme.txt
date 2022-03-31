비지니스 로직 구현(실질적인 전처리 로직 구현)





# 1. 파일 접근

    # 1-1. 데이터 업로드(처음 가정) / version == 1.00
    # 업로드(org, preprocess[version 정보 추가])
    # return sampled_dataset
TEST    http --session=gihang -v POST localhost:5000/project project_name=project03 file_name=train_sampled.json file=@preprocessing1.0/data/train_sampled.json
    # 1-2. project_name, file_name, version parameter를 통해 load(기존 등록된 데이터 활용)
    # parameter project_name, file_name, version -> target_file 추출 후 sampled
    # return sampled_dataset

    # file_name, version 정보 session 내 보관

TEST    http --session=gihang -v POST localhost:5000/project/load project_name=project03 file_name=train_sampled version=1.0

    # 1-3. sampled_parameter 설정
    # defualt -> SEQ/ROW/FRT/50
    # 사용자 설정도 가능
    # 샘플링 정보는 파일별로 나누는게 나을거같음
    # 일단 session 내 유지

TEST    http --session=gihang -v POST localhost:5000/set_sampling_parameter sampling_method=RND ord_row=ROW ord_value=20

# 2. 전처리 동작
    # parameter sampled_dataset, job_id, content
    # DB(job_history) 동작 내용 등록
    #       - file_name, version, job_id, content
    # return sampled_dataset

    연산 동작
    테스트
    1.
    # http --session=gihang -v POST localhost:5000/calculate/get_calc_dataset dataset=@preprocessing1.0/server/project03/origin_data/train_sampled.json
    2.
    http --session=gihang -v POST localhost:5000/calculate/calculating dataset=@preprocessing1.0/server/project03/origin_data/train_sampled.json calc_dataset=@preprocessing1.0/server/project03/origin_data/train_calc.json method=function calc_function=sin columns=jibun


# 3. 데이터 추출(저장)
    # parameter file_name, version
    # DB(job_history) file_name, version 조회 (seq asc)
    # 조회된 행 의 동작 반복 수행
    # <file_name>_V<version+0.01>.json 추출
    # 추출된 파일 정보 DB(Dataset) 추가


 http --session=gihang -v POST localhost:5000/project/export file_name=train_sampled project_name=project03 version=1.0










-datetime
url 양식 /directory/<filename>_V<version>_D<dateime>.<extension>


list
     파라미터 payload(json 타입 통일 하는게 나아보임)
    - 작업중
        결측치 ( DB 테이블 연동 X )
        missing_value(self, missing_value, columns=None, input_data=None):
            결측치 삭제
                remove_missing_value(self, columns=None)
            이전값 채우기
                fill_missing_value_front(self, columns=None):
            다음값 채우기
                fill_missing_value_back(self, columns=None):
            지정값 채우기
                fill_missing_value_specified_value(self, input_data, columns=None):
            표준값 채우기
                중앙값
                    fill_missing_value_median(self, columns):
                평균값
                    fill_missing_value_mean(self, columns):

        컬럼 속성 편집

        컬럼삭제                    대충 해놈

    - 미완료
        경과시간계산
        날짜형식변환
        날짜부분추출
        연산
        수식 비교

        소문자로 변환
        첫문자 대문자로 변환
        대문자로 변환

        컬럼삭제                    대충 해놈
        지정값 일치 삭제
        유효하지 않은 데이터 삭제
        음수값 삭제

        결측치 삭제
        이전값 채우기
        다음값 채우기
        지정값 채우기
        표준값 채우기
        결측 수식적용
        결측 모델 적용

        IQR규칙적용 추출
        밀도기반 추출
        군집기반 추출
        입력값 적용

        음수값 처리
        소수점 처리

        사용자 입력값 편집

        작업스텝 조회
        작업규칙 저장


testcode
httpie
    - 데이터셋 불러오기
    http --session=gihang -v POST localhost:5000/

    - 데이터셋 서버에 저장
    http --session=gihang -v POST localhost:5000/save_df

    - 작업 이력 불러오기
    http --session=gihang -v POST localhost:5000/move_job_history version=1.01 seq=2

    - 전처리 동작 수행
    http --session=gihang -v POST localhost:5000/preprocessing/delete_column column=apt
    http --session=gihang -v POST localhost:5000/preprocessing/missingvalue m_value=remove columns=jibun