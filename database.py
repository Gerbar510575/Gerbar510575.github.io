import sqlite3
import datetime

DB_FILE = 'ZSNR.db'

def connect_db():
    """建立資料庫連線"""
    conn = sqlite3.connect(DB_FILE)
    return conn

def init_db():
    """初始化資料庫並建立資料表"""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parking_spots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parking_name TEXT NOT NULL,
            available_spaces INTEGER,
            timestamp DATETIME NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_parking_data(data):
    """將 API 取得的資料儲存到資料庫"""
    conn = connect_db()
    cursor = conn.cursor()
    current_time = datetime.datetime.now().isoformat()
    
    records_saved = 0
    for item in data:
        try:
            parking_name = item.get('ParkingSegmentName', {}).get('Zh_tw', '')
            available_spaces = item.get('AvailableSpaces')
            
            if parking_name and available_spaces is not None:
                cursor.execute("""
                    INSERT INTO parking_spots (parking_name, available_spaces, timestamp)
                    VALUES (?, ?, ?)
                """, (parking_name, available_spaces, current_time))
                records_saved += 1
        except Exception as e:
            print(f"儲存資料時發生錯誤：{e}")

    conn.commit()
    conn.close()
    print(f"成功儲存 {records_saved} 筆資料到資料庫。")

def get_latest_data():
    """
    從資料庫取得最新的停車位資料。
    由於 TDX 資料會重複，此函式會只回傳每條路段最新的那一筆。
    """
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT parking_name, available_spaces
        FROM parking_spots
        WHERE (parking_name, timestamp) IN (
            SELECT parking_name, MAX(timestamp)
            FROM parking_spots
            GROUP BY parking_name
        )
    """)
    rows = cursor.fetchall()
    conn.close()
    
    latest_data = []
    for row in rows:
        latest_data.append({"name": row[0], "remain": row[1]})
        
    return latest_data