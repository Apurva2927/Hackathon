import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def fetch_data():
    conn_string = os.getenv("DATABASE_URL")
    try:
        print("Connecting to the database...")
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        
        # Determine what tables are available
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = cursor.fetchall()
        
        print("Tables in 'public' schema:")
        for table in tables:
            print(f"- {table[0]}")
            
        # Fetching top 5 rows from the first table found
        if tables:
            first_table = tables[0][0]
            print(f"\nFetching data from table '{first_table}':")
            cursor.execute(f'SELECT * FROM "{first_table}" LIMIT 5')
            rows = cursor.fetchall()
            
            # Fetch column names
            colnames = [desc[0] for desc in cursor.description]
            print(f"Columns: {colnames}")
            for row in rows:
                print(row)
        else:
            print("No tables found in the 'public' schema.")
            
        cursor.close()
        conn.close()
        print("Connection closed.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_data()
