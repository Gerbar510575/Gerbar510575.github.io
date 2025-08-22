from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import requests
import database # 導入我們剛剛建立的資料庫模組
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# 設定您的 TDX 應用程式 ID 和金鑰
app_id = os.environ.get('TDX_APP_ID')
app_key = os.environ.get('TDX_APP_KEY')

auth_url = "https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token"
availability_data_url = "https://tdx.transportdata.tw/api/basic/v1/Parking/OnStreet/ParkingSegmentAvailability/City/Taipei?%24top=33&%24format=JSON"

def get_auth_header():
    # 您的 TDX 認證邏輯...
    content_type = 'application/x-www-form-urlencoded'
    grant_type = 'client_credentials'
    return {
        'content-type': content_type,
        'grant_type': grant_type,
        'client_id': app_id,
        'client_secret': app_key
    }

def get_data_header(auth_response):
    # 您的 TDX 資料頭部邏輯...
    auth_json = auth_response.json()
    access_token = auth_json.get('access_token')
    return {
        'authorization': 'Bearer ' + access_token,
        'Accept-Encoding': 'gzip'
    }

# 新增一個根路由來服務 index.html
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route("/parking_spots", methods=["GET"])
def get_parking_spots():
    try:
        # 1. 取得 TDX 認證令牌
        auth_params = get_auth_header()
        auth_response = requests.post(auth_url, data=auth_params)
        auth_response.raise_for_status()
        data_headers = get_data_header(auth_response)
        
        print("TDX 認證成功！")

        # 2. 取得停車位可用性資料
        availability_response = requests.get(availability_data_url, headers=data_headers)
        availability_response.raise_for_status()

        print("TDX 資料請求成功！")
        
        availability_data = availability_response.json().get('CurbParkingSegmentAvailabilities', [])
        print(f"從 TDX 取得的資料數量: {len(availability_data)}")
        
        # 3. 將資料儲存到資料庫 (呼叫 database 模組中的函式)
        database.save_parking_data(availability_data)
        
        # 4. 從資料庫取得最新資料並回傳給前端
        # 這確保前端顯示的資料來自於資料庫，而不是直接來自 API
        formatted_data = database.get_latest_data()
        
        return jsonify(formatted_data)

    except requests.exceptions.HTTPError as err:
        print(f"TDX API 請求錯誤: {err}")
        return jsonify({"error": f"TDX API 請求錯誤: {err}"}), 500
    except Exception as err:
        print(f"發生其他錯誤: {err}")
        return jsonify({"error": f"發生其他錯誤: {err}"}), 500

if __name__ == "__main__":
    database.init_db()  # 在應用程式啟動時初始化資料庫
    app.run(debug=True, host='0.0.0.0', port=5000)