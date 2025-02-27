from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
from datetime import datetime
import json

# Đọc dữ liệu tọa độ các huyện từ file JSON
with open('mekong_districts.json', 'r', encoding='utf-8') as f:
    district_data = json.load(f)

# Khởi tạo Flask app
app = Flask(__name__)

# Đường dẫn tệp CSV để lưu dữ liệu dự đoán
csv_file_path = 'future_predictions.csv'

# Kiểm tra nếu tệp CSV đã tồn tại và lấy dữ liệu AQI từ tệp
def get_existing_aqi(district):
    if os.path.exists(csv_file_path):
        predictions_df = pd.read_csv(csv_file_path)
        # Lọc dữ liệu dự đoán cho huyện cụ thể
        filtered = predictions_df[predictions_df['District'] == district]
        if not filtered.empty:
            # Trả về danh sách các giá trị AQI dự đoán kèm thời gian
            return filtered[['Timestamp', 'AQI Prediction']].to_dict('records')  # Trả về cả timestamp và AQI
    return None

# Định nghĩa route cho giao diện
@app.route('/')
def index():
    return render_template('index.html')

# API lấy thông tin huyện
@app.route('/get_districts', methods=['GET'])
def get_districts():
    return jsonify(district_data)

# API lấy AQI từ CSV
@app.route('/get_aqi', methods=['POST'])
def get_aqi():
    data = request.get_json()
    print('Received data:', data)  # Log để kiểm tra dữ liệu nhận từ client
    district = data.get('district')

    if not district:
        return jsonify({'error': 'Thiếu thông tin huyện'}), 400

    # Kiểm tra xem dữ liệu dự đoán đã tồn tại trong tệp CSV hay chưa
    existing_aqi = get_existing_aqi(district)
    if existing_aqi is not None:
        return jsonify({
            'district': district,
            'aqi_list': existing_aqi,
            'source': 'cached'
        })
    else:
        return jsonify({'error': 'Không có dữ liệu AQI cho huyện này'}), 404

if __name__ == '__main__':
    app.run(debug=True)
