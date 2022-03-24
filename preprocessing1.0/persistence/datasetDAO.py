import json

import psycopg2
from sqlalchemy import text


class DatasetDao:

    def __init__(self, db):
        self.db = db

    def insert_test(self, payload):
        test_content = {
            "name": "testname",
            "age": 14
        }

        test_content = json.dumps(test_content)

        payload['content'] = '1234'

        print(payload)
        self.cursor.execute("insert into dataset values(%s)" % payload['name'])

    def insert_dataset(self, dataset):
        print('insert_data')
        print(dataset)

        sql = "INSERT INTO dataset( target_id, version, name, content) VALUES("
        sql += "'" + dataset['target_id'] + "', "
        sql += str(dataset['version']) + ", "
        sql += "'" + dataset['name'] + "', "
        sql += "'" + dataset['content'] + "') "
        print(sql)

        self.db.execute(sql)
