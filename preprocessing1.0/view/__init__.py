from flask import request, jsonify, current_app, Response, render_template, g, session, Blueprint, redirect
import pandas as pd
import logging
import os


def create_endpoints(app, service):
    # 필요하면 encoder

    preprocessing_service = service

    # blueprint 도메인 나누기
    # preprocessing
    bp_preprocessing = Blueprint('preprocessing', __name__, url_prefix='/preprocessing')

    # # profiling
    # bp_profiling = Blueprint('profiling', __name__, url_prefix='/profiling')
    # app.register_blueprint(bp_profiling)

    # 결측치 처리
    # /preprecessing/missingvalue
    # 파라미터 m_value
    # 일단 column 기준
    # Response Parameter -> DB에 저장할 값들
    # jobId
    # jobs
    # targetColumns
    # dataTarget

    @app.route('/', methods=['POST', 'GET'])
    def index():
        print('111111111111111111111111111')
        print(app.url_map)
        preprocessing_service.load_df_from_directory()
        preprocessing_service.save_df_in_directory()
        return render_template("index.html")

    @bp_preprocessing.route('/missingvalue', methods=['POST', 'GET'])
    def missing_value():
        payload = request.get_json(force=True)
        print(payload)
        m_value = payload['m_value']
        if 'columns' in payload:
            print('==1==')
            columns = payload['columns']  # 단일 str객체 ? 배열 객체?
            print('colunm')
            print(columns)
            if 'input_data' in payload:
                print('==2==')
                input_data = payload['input_data']
                preprocessing_service.missing_value(missing_value=m_value, columns=columns, input_data=input_data)
                return '성공'
            else:
                print('==3==')
                preprocessing_service.missing_value(missing_value=m_value, columns=columns)
                return '성공'
        elif 'columns' not in request.args:
            print('==14==')
            preprocessing_service.missing_value(missing_value=m_value)
        return '성공'

    # 관리자 권한
    # 프로젝트+데이터셋(로컬) 등록
    # 데이터셋을 데이터 서버로 업로드
    # 받는 파라미터
    #   projectName string
    #   filename    file    (?) 일단 알아서
    @app.route('/project', methods=['POST'])
    def project_upload():
        payload = request.get_json(force=True)
        project_name = payload['projectname']
        file_name = payload['filename']
        print(request.files)
        if 'dataset' not in payload:
            print('No file part')
            return '1234'

        df = pd.read_json(payload['dataset'])
        print(df.head())

        # upload_file.save(os.path.join(ROOT_PATH, upload_file.filename))
        # print(upload_file.filename)
        # save_originfile_in_server
        preprocessing_service.upload_dataset(df, file_name, project_name=project_name)
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

    ##############################################################################
    ## 세션 동작 확인용
    @app.route('/set_session', methods=['GET'])
    def set_session():
        session['test'] = 'test1234'
        text = session['test']
        return text

    @app.route('/test_session', methods=['GET'])
    def test_session():
        if 'test' not in session.keys():
            return 'test is None'
        else:
            text = session['test']
            return text

    # blueprint 등록
    app.register_blueprint(bp_preprocessing)
