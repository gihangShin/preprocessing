from flask import request, jsonify, current_app, Response, render_template, g, session, Blueprint, redirect
import pandas as pd
import logging
import os


def create_endpoints(app, service):
    # 필요하면 encoding

    preprocessing_service = service

    # blueprint
    # preprocessing
    bp_preprocessing = Blueprint('preprocessing', __name__, url_prefix='/preprocessing')

    # # profiling
    # bp_profiling = Blueprint('profiling', __name__, url_prefix='/profiling')
    # app.register_blueprint(bp_profiling)

    ###################################################################
    ###################################################################
    ###################################################################
    ###################################################################
    ###################################################################

    # 1. 파일 접근

    # 1-1. 데이터 업로드(처음 가정) / version == 1.00
    # 업로드(org, preprocess[version 정보 추가])
    # return sampled_dataset

    #############################################
    # 테스트용 코드 웹단에서 이루어질 예정 == 곧 삭제  #
    #############################################
    # /project/<project_id>/file
    @app.route('/project', methods=['POST'])
    def file_upload():
        payload = request.get_json(force=True)
        return preprocessing_service.upload_file_to_server(payload=payload)

    # 1-2. project_name, file_name, version parameter를 통해 load(기존 등록된 데이터 활용)
    # parameter project_name, file_name, version -> target_file 추출 후 sampled
    # return sampled_dataset

    @app.route('/project/load', methods=['POST'])
    def load_existed_file():
        payload = request.get_json(force=True)
        return preprocessing_service.load_existed_file(payload=payload)

    # project_name, file_name, version 정보 session 내 보관

    # 1-3-1. sampled_parameter 설정
    # defualt -> SEQ/ROW/FRT/50
    # 사용자 설정도 가능
    # 샘플링 정보는 파일별로 나누는게 나을거같음
    # 일단 session 내 유지

    #############################################
    # 테스트용 코드 웹단에서 이루어질 예정 == 곧 삭제  #
    #############################################
    @app.route('/set_sampling_parameter', methods=['POST', 'GET'])
    def set_sampling():
        payload = request.get_json(force=True)
        return preprocessing_service.set_sampling_parameter(payload=payload)

    ###################################################################

    # 2. 전처리 동작
    # parameter sampled_dataset, job_id, content
    # DB(job_history) 동작 내용 등록
    #       - file_name, version, job_id, content
    # return sampled_dataset

    # 2-1. 열 삭제
    # /profile/{projectId}/data/{datasetId}/col-prop/columns/delete
    # @bp_preprocessing.route('/profile/<project_id>/data/<
    #
    #
    # dataset_id>/col-prop/columns/delete', methods=['POST', 'GET'])
    @bp_preprocessing.route('/delete_column', methods=['POST', 'GET'])
    def delete_column(project_id, dataset_id):
        payload = request.get_json(force=True)
        payload['project_id'] = project_id
        payload['dataset_id'] = dataset_id
        return preprocessing_service.delete_column(payload=payload)

    # 2-2. 결측치 처리
    @bp_preprocessing.route('/missingvalue', methods=['POST', 'GET'])
    def missing_value():
        payload = request.get_json(force=True)
        return preprocessing_service.missing_value(payload=payload)

    # 2-3. 연산
    # 미완성
    # 연산 calculating 1안 개선
    # 기존 dataset default parameter
    # -> 유지 할 곳이 없어서 이렇게 함

    # 2-3-1. sampled_dataset 에서 연산용 데이터셋 추출

    # 1, 2. UI에서 연산에 활용할 기존 컬럼을 선택 후 request
    # 파라미터 dataset(sampled), calc_columns(여러개일 시 리스트타입)  XXXXX
    # dataset의 테이블 컬럼 중 수치형 항목만 선택
    # return sampled_dataset, calc_dataset(calc_columns)
    @app.route('/calculate/get_calc_dataset', methods=['POST'])
    def get_calc_dataset():
        payload = request.get_json(force=True)
        return preprocessing_service.get_calc_dataset(payload=payload)

    # 2-3-2. 연산
    # 파라미터 calc_dataset, method
    # method == arithmetic -> column1, (column2 or scala or 집계 데이터), operator
    #           return 연산 완료+추가된 calc_dataset
    # method == aggregate, Statistical -> function, (column or scala)
    #           return 연산 완료+추가된 calc_dataset
    # 추가된 column명 -> ex) colA + colB, std(colA)
    @app.route('/calculate/calculating', methods=['POST', 'GET'])
    def calculating():
        payload = request.get_json(force=True)
        return preprocessing_service.calculating_column(payload=payload)

    # 2-3-3. calc_dataset 에서 추출할 컬럼 선택 후 기존 데이터셋(sampled_dataset)으로 결합
    # 선택한 컬럼 추가하기
    # 선택한 컬럼 기존 dataset(sampled)에 열 추가
    # parameter calc_dataset, columns
    # calc_dataset에서 column 추출 후 sampled_dataset 에 열 추가
    # DB(job_history) 추가
    # ->    job_id : calculate
    #       expression : 추출한 column_name ex) "colA + std(colB * colC)"
    # 이후 redo
    # return sampled_dataset(열 추가됨)
    @app.route('/calculate/select_result_columns', methods=['POST'])
    def export_calc_result():
        payload = request.get_json(force=True)
        return preprocessing_service.select_calc_column_to_combine(payload=payload)

    # 2-4. 컬럼 속성 변경
    @app.route('/set_col_prop', methods=['POST'])
    def set_col_prop():
        payload = request.get_json(force=True)
        return preprocessing_service.set_col_prop(payload=payload)

    # 2-5. 날짜 처리
    # 2-5-1. 선택 열 [date time] 으로 변환 후 추가
    @app.roure('/set_col_prop_to_datetime', methods=['POST'])
    def set_col_prop_to_datetime():
        payload = request.get_json(force=True)
        return preprocessing_service.set_col_prop_to_datetime()

    ###################################################################

    # 3. 데이터 추출(저장)
    # parameter file_name, version
    # DB(job_history) file_name, version 조회 (seq asc)
    # 조회된 행 의 동작 반복 수행
    # <file_name>_V<version+0.01>.json 추출
    # 추출된 파일 정보 DB(Dataset) 추가
    # @app.route('/project/{projectId}/export/{datasetId}', methods=['POST'])
    @app.route('/project/export', methods=['POST', 'GET'])
    def export_project():
        payload = request.get_json(force=True)
        return preprocessing_service.export_project(payload=payload)

    #########################################################################
    #########################################################################
    #########################################################################
    #########################################################################
    #########################################################################
    # 테스트용
    @app.route('/', methods=['POST', 'GET'])
    def index():
        print('endpoint 확인')
        print(app.url_map)
        session.clear()
        return '테스트 /'

    @app.route('/get_dataset_all', methods=['POST', 'GET'])
    def get_dataset_all():
        preprocessing_service.get_dataset_jobs_in_session()
        return '1234'

    @app.route('/redo_job_history', methods=['POST', 'GET'])
    def redo_job_history():
        payload = request.get_json(force=True)
        preprocessing_service.redo_job_history(payload=payload)
        # return 'org 데이터 가공 처리 후 추출(저장)'
        return '1234'

    # 불러오기
    # 파라미터 session init
    @app.route('/project/load', methods=['POST'])
    def project_load():

        return '1234'

    @app.route('/load_df_from_directory', methods=['GET'])
    def load_df_from_dictionary():
        payload = request.get_json(force=True)
        if 'url' in payload:
            url = './data/server_data/' + payload['url']
            preprocessing_service.load_df_from_directory(url=url)
        else:
            preprocessing_service.load_df_from_directory()
        return '1234'

    @app.route('/preprocessing_negative_value', methods=['GET'])
    def preprocessing_negative_value():
        column = 'transaction_real_price'
        preprocessing_service.preprocessing_negative_value(columns=column)
        return '1234'

    # blueprint 등록
    # 밑에서 설정해야 동작 왜?

    app.register_blueprint(bp_preprocessing)
