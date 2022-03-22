from flask import request, jsonify, current_app, Response, g, session
import pandas as pd


def create_endpoints(app, service):
    # 필요하면 encoder

    PreprocessingService = service



    @app.route('/getbankcsv', methods=['GET'])
    def getbankcsv():
        df = PreprocessingService.getbankcsv()
        print(df.head())
        return df.to_html()

    @app.route('/getsampletraincsv', methods=['GET'])
    def getsampletraincsv():
        df = PreprocessingService.getsampletraincsv()
        print(df.head())
        return df.to_html()

    @app.route('/delete_missing_value', methods=['GET'])
    def delete_missing_value():
        if 'axis' in request.args:
            payload = request.json
            axis = int(payload['axis'])
        else:
            axis = 0
        PreprocessingService.delete_missing_value(axis)
        return '1234'

    @app.route('/fill_missing_value_pre', methods=['GET'])
    def fill_missing_value_pre():
        PreprocessingService.fill_missing_value_pre()
        return '1234'

    @app.route('/fill_missing_value_back', methods=['GET'])
    def fill_missing_value_back():
        PreprocessingService.fill_missing_value_back()
        return '1234'

    @app.route('/fill_missing_value_std', methods=['GET'])
    def fill_missing_value_std():
        payload = request.json
        method = payload['method']
        # 수치형 컬럼인지 확인해야함
        column = payload['column']
        print(column)
        PreprocessingService.fill_missing_value_std(column, method=method)
        return '1234'

    @app.route('/preprocessing_negative_value', methods=['GET'])
    def preprocessing_negative_value():
        column = 'transaction_real_price'
        PreprocessingService.preprocessing_negative_value(columns=column)
        return '1234'


    ## 세션 동작 확인
    @app.route('/set_session', methods=['GET'])
    def set_session():
        session['test'] = 'test1234'
        text = session['test']
        return text

    @app.route('/test_session', methods=['GET'])
    def test_session():
        if 'test' in session.keys():
            return 'test is None'
        else :
            text = session['test']
            return text

