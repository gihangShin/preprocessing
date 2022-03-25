import json


class DatasetDao:

    def __init__(self, db, app):
        self.db = db
        self.app = app
        self.logger = app.logger

    def insert_test(self, payload):
        test_content = {
            "name": "testname",
            "age": 14
        }

        payload['content'] = '1234'

        print(payload)
        self.cursor.execute("insert into dataset values(%s)" % payload['name'])

    def insert_dataset(self, dataset):
        self.logger.info('insert data dataset')

        sql = "INSERT INTO dataset( target_id, version, job_id, name, content) VALUES("
        sql += "'" + dataset['target_id'] + "', "
        sql += str(dataset['version']) + ", "
        sql += "'" + dataset['job_id'] + "', "
        sql += "'" + dataset['name'] + "', "
        sql += "'" + dataset['content'] + "') "

        self.db.execute(sql)

    # dataset 조회 전체
    # result_set return 타입 list[dict]
    def select_dataset(self, file_name=None):
        if file_name is None:
            self.logger.info('select dataset')
            sql = "SELECT * FROM dataset"
        else:
            self.logger.info('select dataset where name = filename')
            sql = "SELECT * FROM dataset where name = '" + file_name + "' order by seq DSC"

        result_set = list()
        result = self.db.execute(sql)
        for row in result:
            result_set.append(dict(row))

        return result_set

    def select_dataset_jobs(self, file_name, version, seq):
        sql = "SELECT * FROM dataset"
        sql += " where name='" + file_name + "' AND version=" + version + " AND seq <= " + seq
        sql += " order by seq ASC"
        result_set = list()
        result = self.db.execute(sql)
        for row in result:
            result_set.append(dict(row))

        return result_set
