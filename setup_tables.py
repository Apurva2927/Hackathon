import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

def setup_and_fetch():
    conn_string = os.getenv("DATABASE_URL")
    try:
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # 1. Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255),
                company VARCHAR(255),
                description TEXT,
                score INTEGER,
                score_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS objections (
                id SERIAL PRIMARY KEY,
                text TEXT,
                response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS followups (
                id SERIAL PRIMARY KEY,
                prospect VARCHAR(255),
                last_interaction TIMESTAMP,
                days_since INTEGER,
                email VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 2. Insert dummy data if empty
        cursor.execute("SELECT COUNT(*) FROM leads")
        if cursor.fetchone()['count'] == 0:
            print("Populating initial data...")
            cursor.execute("""
                INSERT INTO leads (name, company, description, score, score_reason) 
                VALUES ('John Doe', 'Acme Corp', 'Looking for AI automation tools', 8, 'Strong budget and urgent need.');
                
                INSERT INTO objections (text, response) 
                VALUES ('Too expensive', 'We have flexible pricing plans.');
                
                INSERT INTO followups (prospect, last_interaction, days_since, email)
                VALUES ('John Doe', CURRENT_TIMESTAMP, 2, 'john@acme.com');
            """)
        
        conn.commit()
        
        # 3. Fetch data
        tables = ['leads', 'objections', 'followups']
        for table in tables:
            print(f"\n--- Data from '{table}' ---")
            cursor.execute(f"SELECT * FROM {table};")
            rows = cursor.fetchall()
            if not rows:
                print("No data found.")
            else:
                for row in rows:
                    print(dict(row))
                    
        cursor.close()
        conn.close()
        print("\nSetup and fetch completed.")
    except Exception as e:
        print(f"Error: {e}")
        if 'conn' in locals() and not conn.closed:
            conn.rollback()

if __name__ == "__main__":
    setup_and_fetch()
