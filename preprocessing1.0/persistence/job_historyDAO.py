import json


class JobHistoryDao:

    def __init__(self, db, app):
        self.db = db
        self.app = app
        self.logger = app.logger

    def insert_job_history(self, job_history):
        self.logger.info('insert data dataset')

        sql = "INSERT INTO preparation_job_history( file_id, job_id, version, job_request_user_id, content) VALUES("
        sql += "'" + job_history['file_name'] + "', "
        sql += "'" + job_history['job_id'] + "', "
        sql += str(job_history['version']) + ", "
        sql += "'" + job_history['job_request_user_id'] + "', "
        sql += "'" + job_history['content'] + "') "

        self.logger.info('execute sql -> \n' + sql)
        self.db.execute(sql)

    def init_dataset(self):
        sql = 'DELETE FROM dataset'
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

    # return type list[dict]
    def select_job_history_by_file_name_and_version(self, file_name, version):
        sql = "SELECT * FROM preparation_job_history"
        sql += " where file_id='" + file_name + "' AND version=" + str(round(version,2))
        sql += " order by seq ASC"
        self.logger.info('execute sql -> \n' + sql)
        result = self.db.execute(sql)
        result_set = list()
        for row in result:
            result_set.append(dict(row))

        return result_set
