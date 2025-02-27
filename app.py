from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
from datetime import datetime
import json
import subprocess
import threading

# Khởi tạo Flask app
app = Flask(__name__)

# Đọc dữ liệu tọa độ các huyện từ file JSON
with open('mekong_districts_updated.json', 'r', encoding='utf-8') as f:
    district_data = json.load(f)

# Biến toàn cục để lưu trạng thái quá trình huấn luyện mô hình
training_in_progress = False

# Đường dẫn tệp CSV để lưu dữ liệu dự đoán
csv_file_path = '7_hours_future_predictions_standardized.csv'
# Đường dẫn tệp Excel dữ liệu lịch sử chất lượng không khí
xlsx_file_path = 'all_districts_air_quality_data.xlsx'

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

# Kiểm tra nếu dữ liệu có cần cập nhật hay không (cách ngày hiện tại tối đa là 1 ngày)
def is_data_up_to_date():
    if os.path.exists(xlsx_file_path):
        data = pd.read_excel(xlsx_file_path)
        # Lấy ngày cuối cùng trong dữ liệu
        last_date = pd.to_datetime(data['dt'].max()).date()
        # Lấy ngày hiện tại
        current_date = datetime.now().date()
        # Kiểm tra nếu last_date cách current_date tối đa là 1 ngày
        return (current_date - last_date).days > 1
    return False

# Hàm cập nhật tiến trình và huấn luyện mô hình
def update_data_and_train_model():
    global training_in_progress
    training_in_progress = True  # Đặt trạng thái là đang huấn luyện
    print("Training in progress:", training_in_progress)

    try:
        # Bắt đầu tiến trình mô phỏng lấy dữ liệu
        print("Bắt đầu lấy dữ liệu mới...")
        subprocess.run(["python3", "data.py"], check=True)
        print("Đã hoàn tất lấy dữ liệu.")

        # Cập nhật tiến trình
        print("Bắt đầu huấn luyện mô hình...")
        subprocess.run(["python3", "model.py"], check=True)
        print("Hoàn tất huấn luyện mô hình.")

    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi cập nhật dữ liệu hoặc huấn luyện mô hình: {e}")
    finally:
        training_in_progress = False  # Đặt trạng thái là đã hoàn tất
        print("Training completed:", training_in_progress)

# API kiểm tra trạng thái của quá trình huấn luyện
@app.route('/check_training_status', methods=['GET'])
def check_training_status():
    global training_in_progress
    return jsonify({'training_in_progress': training_in_progress})

# Định nghĩa route cho giao diện
@app.route('/')
def index():
    return render_template('index1.html')

# API lấy thông tin huyện
@app.route('/get_districts', methods=['GET'])
def get_districts():
    return jsonify(district_data)

@app.route('/check_csv', methods=['GET'])
def check_csv():
    if os.path.exists(csv_file_path):
        predictions_df = pd.read_csv(csv_file_path)
        return predictions_df.head().to_json()  # Hiển thị 5 dòng đầu tiên của file CSV
    else:
        return jsonify({'error': 'Không tìm thấy tệp dữ liệu dự đoán'}), 500

# API lấy danh sách ngày và giờ có sẵn trong tệp CSV dựa vào quận
@app.route('/get_available_dates_and_times', methods=['POST'])
def get_available_dates_and_times():
    data = request.get_json()
    district = data.get('district')

    if not district:
        return jsonify({'error': 'Thiếu thông tin huyện'}), 400

    if os.path.exists(csv_file_path):
        predictions_df = pd.read_csv(csv_file_path)

        # Chuyển đổi cột 'Timestamp' thành datetime để xử lý
        predictions_df['Timestamp'] = pd.to_datetime(predictions_df['Timestamp'])

        # Lọc dữ liệu cho huyện đã chọn
        filtered = predictions_df[predictions_df['District'] == district].copy()

        if filtered.empty:
            return jsonify({'error': 'Không có dữ liệu cho huyện này'}), 404

        # Tách giá trị ngày và giờ độc nhất bằng .loc[]
        filtered.loc[:, 'Date'] = filtered['Timestamp'].dt.date
        filtered.loc[:, 'Time'] = filtered['Timestamp'].dt.strftime('%H:%M')

        available_dates = filtered['Date'].unique().tolist()
        times_by_date = {str(date): filtered[filtered['Date'] == date]['Time'].unique().tolist() for date in available_dates}

        return jsonify({
            'district': district,
            'available_dates': times_by_date
        })
    else:
        return jsonify({'error': 'Không tìm thấy tệp dữ liệu dự đoán'}), 500



# API lấy AQI từ CSV
@app.route('/get_aqi', methods=['POST'])
def get_aqi():
    # Nhận dữ liệu từ frontend
    data = request.get_json()
    district = data.get('district')
    date = data.get('date')
    hour = data.get('hour')

    # In ra các giá trị nhận được từ frontend để kiểm tra
    print(f"Received district: {district}, date: {date}, hour: {hour}")

    if not district or not date or not hour:
        print("Thiếu thông tin huyện, ngày hoặc giờ")  # In thông báo lỗi
        return jsonify({'error': 'Thiếu thông tin huyện, ngày hoặc giờ'}), 400

    # Kiểm tra nếu tệp CSV tồn tại
    if os.path.exists(csv_file_path):
        predictions_df = pd.read_csv(csv_file_path)
        predictions_df['Timestamp'] = pd.to_datetime(predictions_df['Timestamp'])

        # In ra dữ liệu từ CSV để kiểm tra
        print("First 5 rows of predictions_df:")
        print(predictions_df.head())

        # Lọc dữ liệu dựa vào huyện, ngày và giờ
        filtered = predictions_df[
            (predictions_df['District'] == district) & 
            (predictions_df['Timestamp'].dt.date == pd.to_datetime(date).date()) & 
            (predictions_df['Timestamp'].dt.strftime('%H:%M') == hour)
        ]

        # In ra kết quả sau khi lọc để kiểm tra
        print(f"Filtered data for district: {district}, date: {date}, hour: {hour}")
        print(filtered)

        # Kiểm tra nếu dữ liệu lọc không rỗng
        if not filtered.empty:
            aqi_value = filtered['AQI Prediction'].values[0]
            aqi_value = int(aqi_value)
            print(f"AQI Prediction found: {aqi_value}")  # In ra giá trị AQI
            return jsonify({
                'district': district,
                'date': date,
                'hour': hour,
                'aqi_prediction': aqi_value,
                'source': 'cached'
            })
        else:
            print("Không có dữ liệu AQI cho thời gian đã chọn")
            return jsonify({'error': 'Không có dữ liệu AQI cho thời gian đã chọn'}), 404
    else:
        print("Không tìm thấy tệp dữ liệu dự đoán")
        return jsonify({'error': 'Không tìm thấy tệp dữ liệu dự đoán'}), 500


# API để bắt đầu cập nhật dữ liệu và huấn luyện mô hình trong một luồng riêng
@app.route('/update_and_train', methods=['POST'])
def update_and_train():
    global training_in_progress
    if training_in_progress:
        return jsonify({'status': 'already_training'}), 200

    # Khởi động luồng riêng để không chặn request
    threading.Thread(target=update_data_and_train_model).start()
    return jsonify({'status': 'started'}), 200



@app.route('/get_aqi_24h', methods=['POST'])
def get_aqi_24h():
    data = request.json
    district = data.get('district')
    date = data.get('date')
    
    # Fetch AQI data for the 24-hour period
    aqi_hourly = get_aqi_data_for_24_hours(district, date)  # Hàm này sẽ lấy dữ liệu từ database hoặc API
    
    return jsonify({"aqi_hourly": aqi_hourly})

def get_aqi_data_for_24_hours(district, date):
    # Placeholder: Thay với code truy vấn dữ liệu thực tế cho district và date
    return [{"time": f"{hour}:00", "aqi": 50 + hour} for hour in range(24)]


@app.route('/get_daily_aqi', methods=['POST'])
def get_daily_aqi():
    data = request.get_json()
    district = data.get('district')
    date = data.get('date')

    if not district or not date:
        return jsonify({'error': 'Thiếu thông tin huyện hoặc ngày'}), 400

    # Đọc dữ liệu từ tệp CSV
    if os.path.exists(csv_file_path):
        predictions_df = pd.read_csv(csv_file_path)
        predictions_df['Timestamp'] = pd.to_datetime(predictions_df['Timestamp'])

        # Lọc dữ liệu cho huyện và ngày được chọn
        filtered = predictions_df[
            (predictions_df['District'] == district) &
            (predictions_df['Timestamp'].dt.date == pd.to_datetime(date).date())
        ]

        # Kiểm tra nếu dữ liệu lọc không rỗng
        if not filtered.empty:
            hourly_aqi = {row['Timestamp'].strftime('%H:%M'): int(row['AQI Prediction']) for _, row in filtered.iterrows()}
            return jsonify({'hourly_aqi': hourly_aqi})
        else:
            return jsonify({'error': 'Không có dữ liệu AQI cho ngày đã chọn'}), 404
    else:
        return jsonify({'error': 'Không tìm thấy tệp dữ liệu dự đoán'}), 500


if __name__ == '__main__':
    app.run(debug=True)
