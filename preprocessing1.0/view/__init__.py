from flask import request, jsonify, current_app, Response, render_template, g, session, Blueprint, redirect
import pandas as pd
import logging
import os


def create_endpoints(app, service):
    ps = service

    # blueprint
    # preprocessing
    bp_preprocessing = Blueprint('preprocessing', __name__, url_prefix='/preprocessing')

    ###################################################################
    ###################################################################
    ###################################################################

    # (처음) 불러오기
    @app.route('/project/load', methods=['POST'])
    def load():
        payload = request.get_json(force=True)
        return ps.load(payload=payload)

    # 데이터셋 추출
    @app.route('/project/export', methods=['POST', 'GET'])
    def export_project():
        payload = request.get_json(force=True)
        return ps.export(payload=payload)

    ###############################################################
    # 전처리 동작
    @bp_preprocessing.route('/delete_column', methods=['POST', 'GET'])
    def delete_column():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, method='delete_column')

    # 2-2. 결측치 처리
    @bp_preprocessing.route('/missing_value', methods=['POST', 'GET'])
    def missing_value():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, method='missing_value')

    # 2-4. 컬럼 속성 변경
    @bp_preprocessing.route('/set_col_prop', methods=['POST'])
    def set_col_prop():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, method='set_col_prop')

    # 2-5. 날짜 처리
    # 2-5-1. 선택 열 [date time] 으로 변환 후 추가
    @bp_preprocessing.route('/set_col_prop_to_datetime', methods=['POST'])
    def set_col_prop_to_datetime():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, method='set_col_prop_to_datetime')

    # 2-5-2. 날짜 처리(분할 하기)
    @bp_preprocessing.route('/datetime/spite_variable', methods=['POST'])
    def split_variable_datetime():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, method='split_datetime')

    # 2-5-3. 날짜 처리(문자열로)
    @bp_preprocessing.route('/datetime/dt_to_str_format', methods=['POST'])
    def dt_to_str_format():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, method='dt_to_str_format')

    # 2-5-4. 날짜 처리(기준 일로 부터 날짜 차이)
    @bp_preprocessing.route('/datetime/diff_datetime', methods=['POST'])
    def diff_datetime():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, method='diff_datetime')

    # 2-6 컬럼 순서 변경
    @bp_preprocessing.route('/change_column_order', methods=['POST'])
    def columns_order_change():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, method='change_column_order')

    # 2-7 대, 소문자 변환
    @bp_preprocessing.route('/col_prop/case_sensitive', methods=['POST'])
    def col_prop_string_change():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, method='case_sensitive')

    # 2-8 치환 - 입력값으로 교체
    @bp_preprocessing.route('/col_prop/string/replace_by_input_value', methods=['POST'])
    def col_prop_string_search_replace():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, method='replace_by_input_value')

    # 2-9 공백 제거
    @bp_preprocessing.route('/col_prop/string/remove_space_front_and_rear', methods=['POST'])
    def remove_space():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, method='remove_space_front_and_rear')

    # 2-10-1 중복 값 확인
    @bp_preprocessing.route('/row_control/show_duplicate_row', methods=['POST'])
    def row_control_show_duplicate_row():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, method='show_duplicate_row')

    # 2-10-2 중복 값 처리
    @bp_preprocessing.route('/row_control/drop_duplicate_row', methods=['POST'])
    def row_control_duplicate_row():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, method='drop_duplicate_row')

    # 연산 처리
    @bp_preprocessing.route('/calculateing', methods=['POST', 'GET'])
    def calculating():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, method='calculating_column')
