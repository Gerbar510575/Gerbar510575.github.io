# 使用官方的 Python 3.9 映像檔作為基礎
FROM python:3.9-slim

# 設定工作目錄
WORKDIR /app

# 將本地的 requirements.txt 檔案複製到容器內
COPY requirements.txt .

# 更新 pip
RUN pip install --upgrade pip

# 安裝專案所需的套件
RUN pip install --no-cache-dir -r requirements.txt

# 將專案中的所有檔案複製到容器的工作目錄
COPY . .

# 設定環境變數，以確保 Flask 程式能正確啟動
ENV FLASK_APP=app.py

# 暴露 Flask 預設的 5000 Port
EXPOSE 5000

# 定義容器啟動時執行的指令
#CMD ["python3", "-m", "flask", "run", "--host", "0.0.0.0"]
# 執行資料庫初始化腳本，然後再啟動 Flask 應用程式
CMD ["/bin/sh", "-c", "python3 init_db.py && python3 -m flask run --host 0.0.0.0"]