from flask import request, jsonify, current_app, Response, g, session, Blueprint
import pandas as pd



def create_endpoints(app, service):
    # 필요하면 encoder

    preprocessing_service = service

    # blueprint 도메인 나누기
    # preprocessing
    bp_preprocessing = Blueprint('preprocessing', __name__, url_prefix='/preprocessing')
    app.register_blueprint(bp_preprocessing)
    # profiling
    bp_profiling = Blueprint('profiling',__name__,url_prefix='/profiling')
    app.register_blueprint(bp_profiling)

    # 결측치 처리
    # /preprecessing/missingvalue
    # 파라미터 missingvalue
    # 일단 column 기준
    @bp_preprocessing.route('/missingvalue', methods=['GET'])
    def missing_value():
        payload = request.get_json(force=True)
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

    @app.route('/load_df_from_directory', methods=['GET'])
    def load_df_from_dictionary():
        preprocessing_service.load_df_from_directory()
        return '1234'


    @app.route('/preprocessing_negative_value', methods=['GET'])
    def preprocessing_negative_value():
        column = 'transaction_real_price'
        preprocessing_service.preprocessing_negative_value(columns=column)
        return '1234'















    ## 세션 동작 확인
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
