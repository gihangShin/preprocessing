세션유지 httpie 활용법
    $http -v --admin [method] url parameter

엔드포인트(mapping)

blueprint 도메인
    preprocessing
        bp_preprocessing = Blueprint('preprocessing', __name__, url_prefix='/preprocessing')
    profiling
        bp_profiling = Blueprint('profiling', __name__, url_prefix='/profiling')
    app 내 등록
        app.register_blueprint(blueprint 객체)
            -> @bp_preprocessing == url_prefix='/preprocessing'
            -> @bp_profiling == url_prefix='/profiling'

    결측치 처리 (행은 처리 x)
        @bp_preprocessing.route('/missingvalue', methods=['GET'])
        parameter
            m_value
                remove, ffill, bfill ....
            columns 선택 값
                # 현재는 단일 열 처리, 다중 열 고려 해야 할듯
                <column 명>
            input_data 선택 값 (지정값 채우기 시)
                # dtype 예외처리 필요
                <data>

        return 처리 X
