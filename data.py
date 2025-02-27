import json
import pandas as pd
from airpyllution.airpyllution import get_pollution_history
from datetime import datetime, timedelta

# Đọc dữ liệu từ file JSON
with open('mekong_districts_updated.json', 'r', encoding='utf-8') as file:
    locations = json.load(file)

# Đặt khóa API của bạn vào đây
api_key = "7351241f3499ddb1bed51ef73196abc9" 

end_date = datetime.now()
start_date = end_date - timedelta(days=210)

# Chuyển đổi thời gian sang định dạng UNIX timestamp
end_timestamp = int(end_date.timestamp())
start_timestamp = int(start_date.timestamp())

# Hàm để lấy dữ liệu chất lượng không khí
def get_historical_air_pollution_data(lat, lon, api_key, start_timestamp, end_timestamp):
    try:
        data = get_pollution_history(start_timestamp, end_timestamp, lat, lon, api_key)
        return pd.DataFrame(data)  # Đảm bảo dữ liệu được trả về dưới dạng DataFrame
    except Exception as e:
        print(f"Error fetching data for coordinates ({lat}, {lon}): {e}")
        return pd.DataFrame()  # Trả về DataFrame rỗng nếu có lỗi

# Hàm tính AQI theo tiêu chuẩn US EPA
def calc_aqi_us(concentration, pollutant):
    if pollutant == 'PM2.5':
        c_low = [0, 12.1, 35.5, 55.5, 150.5, 250.5, 350.5]
        c_high = [12, 35.4, 55.4, 150.4, 250.4, 350.4, 500.4]
        i_low = [0, 51, 101, 151, 201, 301, 401]
        i_high = [50, 100, 150, 200, 300, 400, 500]
    elif pollutant == 'PM10':
        c_low = [0, 55, 155, 255, 355, 425, 505]
        c_high = [54, 154, 254, 354, 424, 504, 604]
        i_low = [0, 51, 101, 151, 201, 301, 401]
        i_high = [50, 100, 150, 200, 300, 400, 500]
    else:
        return 'Invalid pollutant type'
    c = concentration
    for i in range(len(c_low)):
        if c_low[i] <= int(c) <= c_high[i]:
            aqi = ((i_high[i] - i_low[i]) / (c_high[i] - c_low[i])) * (c - c_low[i]) + i_low[i]
            return round(aqi, 1)
    if c > c_high[-1]:
        aqi = ((i_high[-1] - i_low[-1]) / (c_high[-1] - c_low[-1])) * (c - c_low[-1]) + i_low[-1]
        return round(aqi, 1)
    else:
        return 'Input concentration is below AQI scale'

# DataFrame chính để lưu trữ tất cả dữ liệu
all_data = pd.DataFrame()

# Duyệt qua tất cả các tỉnh và huyện để lấy dữ liệu chất lượng không khí
for province, districts in locations.items():
    for district, coords in districts.items():
        # Lấy dữ liệu chất lượng không khí
        historical_data = get_historical_air_pollution_data(coords["lat"], coords["lon"], api_key, start_timestamp, end_timestamp)

        if not historical_data.empty:
            # Thêm thông tin tỉnh, huyện, và tọa độ địa lý vào DataFrame
            historical_data['Province'] = province
            historical_data['District'] = district
            
            # Tính AQI cho PM2.5 và PM10
            historical_data['AQI_PM2.5'] = historical_data['pm2_5'].apply(lambda x: calc_aqi_us(x, "PM2.5"))
            historical_data['AQI_PM10'] = historical_data['pm10'].apply(lambda x: calc_aqi_us(x, "PM10"))

            # Chọn giá trị AQI cao nhất làm AQI tổng hợp
            historical_data['AQI'] = historical_data[['AQI_PM2.5', 'AQI_PM10']].select_dtypes(include='number').max(axis=1).apply(lambda x: round(x, 0))


            # In 5 hàng đầu tiên của mỗi huyện
            print(f"First 5 rows of data for {district}, {province}:\n", historical_data.head(5))

            # Append to the main DataFrame
            all_data = pd.concat([all_data, historical_data], ignore_index=True)

# Lưu dữ liệu vào file Excel
all_data.to_excel("all_districts_air_quality_data.xlsx", index=False)
