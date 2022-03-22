import psycopg2

class Database:
    def __init__(self):
        self.db = psycopg2.connect(host='localhost',
                              dbname='postgres',
                              user='postgres',
                              password='1234',
                              port=5432)

        self.cursor = db.cursor()

    def __del__(self):
        self.db.close()
        self.cursor.close()

    def execute(self, query, args={}):
        self.cursor.execute(query, args)
        row = self.cursor.fetchall()
        return row

    def commit(self):
        self.cursor.commit()