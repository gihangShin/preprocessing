import psycopg2

try:
    conn = psycopg2.connect(host='localhost',
                              dbname='gihangdb',
                              user='gihang',
                              password='1234',
                              port=5432)
except:
    print('not connected')