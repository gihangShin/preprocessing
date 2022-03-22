from flask import Flask, session
from flask_cors import CORS

from preprocessing import preprocessingservice
from view import create_endpoints

__name__ = "__main__"

# session -> 권한 정보 저장 -> 데이터 접근 권한 ( 서버단에서 처리??)
#         -> 현재 처리중인 DataFrame 정보 저장 ( 매번 데이터를 저장할 순 없음.)
#         -> 사용자 정보?? -> 서버단??
#         -> api는 요청받은 처리만?? (이렇게 하면 보안문제)
#
def create_app():
    app = Flask(__name__)

    app.secret_key='1dsaidzicoqj1515'
    CORS(app)
    if __name__ == "__main__":
        app.run(debug=True)
    # preprocessing 전처리 (service)
    pre_service = service.PreprocessingService(app)

    # 엔드포인트 생성
    create_endpoints(app, pre_service)

    return app
