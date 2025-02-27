import json
from geopy.geocoders import Nominatim
import time

# Khởi tạo geolocator với user agent cụ thể
geolocator = Nominatim(user_agent="geoapiExercises")

# Đọc dữ liệu từ file JSON
with open('mekong_districts_updated.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Hàm để lấy địa chỉ chi tiết từ tọa độ
def get_location_details(lat, lon):
    try:
        location = geolocator.reverse(f"{lat}, {lon}")
        return location.address if location else "Không tìm thấy địa chỉ"
    except Exception as e:
        return f"Error: {str(e)}"

# Duyệt qua các tỉnh và các địa điểm, in ra địa chỉ chi tiết
for province, locations in data.items():
    print(f"Tỉnh: {province}")
    for location_name, coords in locations.items():
        lat = coords["lat"]
        lon = coords["lon"]
        address = get_location_details(lat, lon)
        print(f"  Địa điểm: {location_name}, Toạ độ: ({lat}, {lon})")
        print(f"  Địa chỉ chi tiết: {address}\n")
        time.sleep(1)  # Thêm thời gian chờ 1 giây giữa các yêu cầu để tránh bị chặn
