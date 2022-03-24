import psycopg2


class Database:
    def __init__(self):
        self.db = psycopg2.connect(host='localhost',
                                   dbname='gihangdb',
                                   user='gihang',
                                   password='1234',
                                   port=5432)

        self.cursor = self.db.cursor()


    def __del__(self):
        self.db.close()
        self.cursor.close()

    def get_db(self):
        return self.db
    # def execute(self, query, args={}):
    #     print(args)
    #     self.cursor.execute(query, args)
    #
    # def commit(self):
    #     self.cursor.commit()
