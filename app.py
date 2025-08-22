from flask import Flask, jsonify
from flask_cors import CORS
import requests
import json

app = Flask(__name__)
CORS(app)

# 設定您的 TDX 應用程式 ID 和金鑰
app_id = 'gerbarg8-8ed9452a-9756-4e2d'
app_key = '5f56ce95-d8ab-44d9-848a-c6d62104d09c'

auth_url = "https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token"
availability_data_url = "https://tdx.transportdata.tw/api/basic/v1/Parking/OnStreet/ParkingSegmentAvailability/City/Taipei?%24top=10&%24skip=10&%24format=JSON"

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
    try:
        # 1. 取得認證令牌
        auth_params = get_auth_header(app_id, app_key)
        auth_response = requests.post(auth_url, data=auth_params)
        auth_response.raise_for_status()
        data_headers = get_data_header(auth_response)
        
        # 檢查認證是否成功
        print("TDX 認證成功！")

        # 2. 取得停車位可用性資料
        availability_response = requests.get(availability_data_url, headers=data_headers)
        availability_response.raise_for_status()

        # 檢查資料請求是否成功
        print("TDX 資料請求成功！")
        
        #print(availability_response.json().get('CurbParkingSegmentAvailabilities'))

        #availability_data = availability_response.json().get('ParkingSegmentAvailabilities', [])
        availability_data = availability_response.json().get('CurbParkingSegmentAvailabilities', [])
        # 檢查回傳的資料內容
        print(f"從 TDX 取得的資料數量: {len(availability_data)}")
        
        formatted_data = []
        for item in availability_data:
            formatted_data.append({
                "name": item.get('ParkingSegmentName', {}).get('Zh_tw', ''),
                #"remain": item.get('SpaceAvailability')
                "remain": item.get('AvailableSpaces')
            })
        
        return jsonify(formatted_data)

    except requests.exceptions.HTTPError as err:
        print(f"TDX API 請求錯誤: {err}")
        return jsonify({"error": f"TDX API 請求錯誤: {err}"}), 500
    except Exception as err:
        print(f"發生其他錯誤: {err}")
        return jsonify({"error": f"發生其他錯誤: {err}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)