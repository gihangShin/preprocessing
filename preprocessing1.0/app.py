from flask import Flask
from flask_cors import CORS

from preprocessing import service
from view import create_endpoints

__name__ = "__main__"
def create_app():
    app = Flask(__name__)

    CORS(app)
    if __name__ == "__main__":
        app.run(debug=True)
    # preprocessing 전처리 (service)
    pre_service = service.preprocessingService()

    # 엔드포인트 생성
    create_endpoints(app, pre_service)

    return app
