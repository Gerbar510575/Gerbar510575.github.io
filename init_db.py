import database
import os

if __name__ == "__main__":
    if not os.path.exists(database.DB_FILE):
        print("Creating new database file...")
        open(database.DB_FILE, 'a').close()
    
    database.init_db()
    print("Database initialized successfully.")