from flask import Flask, jsonify
from flask_cors import CORS
import requests
import json

app = Flask(__name__)
CORS(app)

# 設定您的 TDX 應用程式 ID 和金鑰
# 注意：這些是硬編碼的值，建議在生產環境中使用環境變數
app_id = 'gerbarg8-8ed9452a-9756-4e2d'
app_key = '5f56ce95-d8ab-44d9-848a-c6d62104d09c'

auth_url = "https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token"

# 初始化一個全域變數來追蹤每次 API 請求的 $skip 值
# 每次呼叫 /parking_spots 時，這個值會增加 10
current_skip_offset = 0

def get_auth_header(app_id, app_key):
    content_type = 'application/x-www-form-urlencoded'
    grant_type = 'client_credentials'
    return {
        'content-type': content_type,
        'grant_type': grant_type,
        'client_id': app_id,
        'client_secret': app_key
    }

def get_data_header(auth_response):
    auth_json = auth_response.json()
    access_token = auth_json.get('access_token')
    return {
        'authorization': 'Bearer ' + access_token,
        'Accept-Encoding': 'gzip'
    }

@app.route("/parking_spots", methods=["GET"])
def get_parking_spots():
    # 聲明我們要修改全域變數 current_skip_offset
    global current_skip_offset

    try:
        # 1. 取得認證令牌
        auth_params = get_auth_header(app_id, app_key)
        auth_response = requests.post(auth_url, data=auth_params)
        auth_response.raise_for_status()
        data_headers = get_data_header(auth_response)
        
        print("TDX 認證成功！")

        # 根據當前的 current_skip_offset 值來動態構建 API URL
        # 我們始終設定 $top=10，並根據 current_skip_offset 調整 $skip
        dynamic_availability_data_url = f"https://tdx.transportdata.tw/api/basic/v1/Parking/OnStreet/ParkingSegmentAvailability/City/Taipei?%24top=10&%24skip={current_skip_offset}&%24format=JSON"
        
        print(f"正在請求 TDX API，URL: {dynamic_availability_data_url}")

        # 2. 取得停車位可用性資料
        availability_response = requests.get(dynamic_availability_data_url, headers=data_headers)
        availability_response.raise_for_status()

        # 檢查資料請求是否成功
        print("TDX 資料請求成功！")
        
        availability_data = availability_response.json().get('CurbParkingSegmentAvailabilities', [])
        # 檢查回傳的資料內容
        print(f"從 TDX 取得的資料數量: {len(availability_data)}")
        
        formatted_data = []
        for item in availability_data:
            formatted_data.append({
                "name": item.get('ParkingSegmentName', {}).get('Zh_tw', ''),
                "remain": item.get('AvailableSpaces')
            })
        
        # 重要的步驟：將 $skip 值增加 10，以便下次請求時跳過更多資料
        current_skip_offset += 10
        print(f"下次 API 請求的 $skip 值將為: {current_skip_offset}")

        return jsonify(formatted_data)

    except requests.exceptions.HTTPError as err:
        print(f"TDX API 請求錯誤: {err}")
        return jsonify({"error": f"TDX API 請求錯誤: {err}"}), 500
    except Exception as err:
        print(f"發生其他錯誤: {err}")
        return jsonify({"error": f"發生其他錯誤: {err}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)