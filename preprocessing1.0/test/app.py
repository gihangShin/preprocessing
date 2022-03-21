from flask import Flask

app = Flask('test')


@app.route('/ping', methods=['GET'])
def ping():
    return 'pong'
