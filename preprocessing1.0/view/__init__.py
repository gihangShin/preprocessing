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
    # 1
    @bp_preprocessing.route('/delete_column', methods=['POST', 'GET'])
    def delete_column():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, job_id='delete_column')

    # 2. 결측치 처리
    @bp_preprocessing.route('/missing_value', methods=['POST', 'GET'])
    def missing_value():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, job_id='missing_value')

    # 3. 컬럼 속성 변경
    @bp_preprocessing.route('/set_col_prop', methods=['POST'])
    def set_col_prop():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, job_id='set_col_prop')

    # 4. 날짜 처리
    # 선택 열 [date time] 으로 변환 후 추가(혹은 변경)
    @bp_preprocessing.route('/set_col_prop_to_datetime', methods=['POST'])
    def set_col_prop_to_datetime():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, job_id='set_col_prop_to_datetime')

    # 5. 날짜 처리(분할 하기)
    @bp_preprocessing.route('/datetime/spite_variable', methods=['POST'])
    def split_variable_datetime():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, job_id='split_datetime')

    # 6. 날짜 처리(문자열로)
    @bp_preprocessing.route('/datetime/dt_to_str_format', methods=['POST'])
    def dt_to_str_format():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, job_id='dt_to_str_format')

    # 7. 날짜 처리(기준 일로 부터 날짜 차이)
    @bp_preprocessing.route('/datetime/diff_datetime', methods=['POST'])
    def diff_datetime():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, job_id='diff_datetime')

    # 8. 컬럼 순서 변경
    @bp_preprocessing.route('/change_column_order', methods=['POST'])
    def columns_order_change():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, job_id='change_column_order')

    # 9. 대, 소문자 변환
    @bp_preprocessing.route('/col_prop/case_sensitive', methods=['POST'])
    def col_prop_string_change():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, job_id='case_sensitive')

    # 10. 치환 - 입력값으로 교체
    @bp_preprocessing.route('/col_prop/string/replace_by_input_value', methods=['POST'])
    def col_prop_string_search_replace():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, job_id='replace_by_input_value')

    # 11. 공백 제거
    @bp_preprocessing.route('/col_prop/string/remove_space_front_and_rear', methods=['POST'])
    def remove_space():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, job_id='remove_space_front_and_rear')

    # 조회 1. 중복 값 확인
    @bp_preprocessing.route('/row_control/show_duplicate_row', methods=['POST'])
    def row_control_show_duplicate_row():
        payload = request.get_json(force=True)
        return ps.show(payload=payload, job_id='show_duplicate_row')

    # 12. 중복 값 처리
    @bp_preprocessing.route('/row_control/drop_duplicate_row', methods=['POST'])
    def row_control_duplicate_row():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, job_id='drop_duplicate_row')

    # 13. 연산 처리
    @bp_preprocessing.route('/calculateing', methods=['POST', 'GET'])
    def calculating():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, job_id='calculating_column')

    # 14. 열 삭제 (지정값 일치, 유효하지 않은 데이터, 음수(수치형만 해당))
    @bp_preprocessing.route('/row_control/drop_row', methods=['POST', 'GET'])
    def drop_row():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, job_id='drop_row')

    # 15. 컬럼 이름 변경
    @bp_preprocessing.route('/rename_col', methods=['POST', 'GET'])
    def rename_col():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, job_id='rename_col')

    # 16. 컬럼 분할 ( 구분자, 컬럼 길이로 분할, 역분할(뒤에서 부터))
    @bp_preprocessing.route('/col_prop/split_col', methods=['POST', 'GET'])
    def split_col():
        payload = request.get_json(force=True)
        return ps.preprocessing(payload=payload, job_id='split_col')

    app.register_blueprint(bp_preprocessing)
