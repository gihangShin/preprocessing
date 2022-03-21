from flask import request, jsonify, current_app, Response, g
import pandas as pd


def create_endpoints(app, service):
    # 필요하면 encoder

    preprocessing_service = service

    # preprocessing_service.__init__(app)

    #
    @app.route('/getbankcsv', methods=['GET'])
    def getbankcsv():
        df = preprocessing_service.getbankcsv()
        print(df.head())
        return df.to_html()

    @app.route('/getsampletraincsv', methods=['GET'])
    def getsampletraincsv():
        df = preprocessing_service.getsampletraincsv()
        print(df.head())
        return df.to_html()

    @app.route('/delete_missing_value', methods=['GET'])
    def delete_missing_value():
        payload = request.json
        axis = int(payload['axis'])
        preprocessing_service.delete_missing_value(axis)
        return '1234'

    @app.route('/fill_missing_value_pre', methods=['GET'])
    def fill_missing_value_pre():
        preprocessing_service.fill_missing_value_pre()
        return '1234'

    @app.route('/fill_missing_value_back', methods=['GET'])
    def fill_missing_value_back():
        preprocessing_service.fill_missing_value_back()
        return '1234'

    @app.route('/fill_missing_value_std', methods=['GET'])
    def fill_missing_value_std():
        payload = request.json
        method = payload['method']
        # 수치형 컬럼인지 확인해야함
        column = payload['column']
        print(column)
        preprocessing_service.fill_missing_value_std(column, method=method)
        return '1234'

    @app.route('/preprocessing_negative_value', methods=['GET'])
    def preprocessing_negative_value():
        column = 'transaction_real_price'
        preprocessing_service.preprocessing_negative_value(columns=column)
        return '1234'

