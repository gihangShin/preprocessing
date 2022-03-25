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

    @app.route('/show_current_df', methods=['POST', 'GET'])
    def show_current_df():
        preprocessing_service.show_df_from_session()
        return '1234'

    @app.route('/', methods=['POST', 'GET'])
    def index():
        print('endpoint 확인')
        print(app.url_map)
        # payload = request.get_json(force=True)
        preprocessing_service.load_df_from_directory()
        preprocessing_service.get_df_from_session()
        preprocessing_service.show_dataset_all()
        # preprocessing_service.insert_test(payload=payload)
        return '1234'

    @app.route('/save_df', methods=['POST', 'GET'])
    def save_df():
        preprocessing_service.save_df_in_server()
        return '1234'

    @app.route('/get_dataset_all', methods=['POST', 'GET'])
    def get_dataset_all():
        preprocessing_service.get_dataset_jobs_in_session()
        return '1234'

    @app.route('/move_job_history', methods=['POST', 'GET'])
    def get_dataset():
        payload = request.get_json(force=True)
        # payload [ version, seq ]
        preprocessing_service.move_job_history(payload=payload)
        return '1234'

    @bp_preprocessing.route('/missingvalue', methods=['POST', 'GET'])
    def missing_value():
        payload = request.get_json(force=True)
        preprocessing_service.missing_value(payload=payload)
        return '1234'

    @bp_preprocessing.route('/delete_column', methods=['POST', 'GET'])
    def delete_column():
        payload = request.get_json(force=True)
        preprocessing_service.delete_column(payload=payload)
        return '1234'

    # 관리자 권한
    # 프로젝트+데이터셋(로컬) 등록
    # 데이터셋을 데이터 서버로 업로드
    @app.route('/project', methods=['POST'])
    def project_upload():
        payload = request.get_json(force=True)
        project_name = payload['projectname']
        file_name = payload['filename']

        df = pd.read_json(payload['dataset'])
        print(df.head())

        preprocessing_service.upload_dataset(df, file_name, project_name=project_name)
        return '1234'

    # 테스트용으로 불러오기도 만들어야함
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
