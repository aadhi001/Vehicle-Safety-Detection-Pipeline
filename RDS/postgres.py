import psycopg2

db_host = 'your_host_name'
db_name = 'your_db_name'
db_user = 'your_user'
db_pass = 'your_pass'

connection = psycopg2.connect(host = db_host, database = db_name, user = db_user, password = db_pass)
print("Connected to the database")

cursor = connection.cursor()
# Define and execute the CREATE TABLE command
create_table_query = """
CREATE TABLE vehicle_safety_data (
    id SERIAL PRIMARY KEY,
    image_key VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    location VARCHAR(50),
    speed FLOAT,
    classification VARCHAR(10) CHECK (classification IN ('critical', 'safe')),
    pedestrians_detected INT DEFAULT 0
);
"""
cursor.execute(create_table_query)

# Commit the transaction
connection.commit()
print("Table 'vehicle_safety_data' created successfully")

# Verify the table (optional)
cursor.execute("SELECT version()")
db_version = cursor.fetchone()
print("Database version:", db_version)
cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'vehicle_safety_data'
    """)
table_exists = cursor.fetchone()
print("Does the table exist, ", table_exists)
cursor.close() 