db = {
    'user': 'gihang',
    'password': '1234',
    'host': 'localhost',
    'port': '5432',
    'database': 'gihangdb'
}
# postgresql://[user[:password]@][netloc][:port][/dbname][?param1=value1&...]
DB_URL = f"postgresql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}"