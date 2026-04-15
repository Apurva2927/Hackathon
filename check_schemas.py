import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
conn_string = os.getenv("DATABASE_URL")
try:
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()
    cur.execute("SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema')")
    tables = cur.fetchall()
    print("Tables found:")
    for t in tables:
        print(t)
    conn.close()
except Exception as e:
    print(e)
