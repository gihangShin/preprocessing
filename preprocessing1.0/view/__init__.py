from flask import request, jsonify, current_app, Response, g
import pandas as pd

def create_endpoints(app, service):
    # 필요하면 encoder

    preprocessing_service = service

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
        preprocessing_service.delete_missing_value()
        return '1234'

    @app.route('/fill_missing_value_pre', methods=['GET'])
    def fill_missing_value_pre():
        preprocessing_service.fill_missing_value_pre()
        return '1234'