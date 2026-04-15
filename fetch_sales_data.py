import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def fetch_sales_data():
    conn_string = os.getenv("DATABASE_URL")
    try:
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        
        tables_to_check = ['leads', 'objections', 'followups']
        
        for table in tables_to_check:
            print(f"\n--- Table: {table} ---")
            try:
                cursor.execute(f'SELECT * FROM "{table}" LIMIT 5')
                rows = cursor.fetchall()
                if rows:
                    colnames = [desc[0] for desc in cursor.description]
                    print(f"Columns: {colnames}")
                    for row in rows:
                        print(row)
                else:
                    print("Table is empty (0 rows).")
            except psycopg2.errors.UndefinedTable:
                conn.rollback() # rollback current transaction block
                print(f"Table '{table}' does not exist in the database.")
            except Exception as e:
                conn.rollback()
                print(f"Error fetching from {table}: {e}")
                
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    fetch_sales_data()
