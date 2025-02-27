from geopy.geocoders import Nominatim
from unidecode import unidecode
import json
import math

# Tạo đối tượng geolocator từ OpenStreetMap
geolocator = Nominatim(user_agent="geo_checker")

# Hàm để lấy tọa độ thực tế từ OpenStreetMap với cả tên có dấu và không dấu
def get_real_coordinates(district, province):
    # Thử tìm với tên có dấu trước
    location = geolocator.geocode(f"{district}, {province}, Vietnam")
    
    if location is None:
        # Nếu không tìm thấy, thử với tên không dấu
        district_ascii = unidecode(district)
        province_ascii = unidecode(province)
        location = geolocator.geocode(f"{district_ascii}, {province_ascii}, Vietnam")
    
    if location and math.isfinite(location.latitude) and math.isfinite(location.longitude):
        return location.latitude, location.longitude
    return None

# Đọc và kiểm tra file JSON chứa tọa độ
with open('mekong_districts_update.json', 'r+', encoding='utf-8') as f:
    districts_data = json.load(f)
    updated_districts_data = {}

    # Kiểm tra và tự động cập nhật tọa độ không chính xác
    for province, districts in districts_data.items():
        updated_districts = {}
        for district, coordinates in districts.items():
            district_lat = coordinates['lat']
            district_lon = coordinates['lon']
            real_coordinates = get_real_coordinates(district, province)

            if real_coordinates:
                real_lat, real_lon = real_coordinates
                if district_lat != real_lat or district_lon != real_lon:
                    print(f"Tọa độ hiện tại của {district}, {province} không chính xác.")
                    print(f"Tọa độ thực tế: (lat: {real_lat}, lon: {real_lon})")
                    # Tự động cập nhật tọa độ nếu chúng là số hữu hạn
                    if math.isfinite(real_lat) and math.isfinite(real_lon):
                        updated_districts[district] = {'lat': real_lat, 'lon': real_lon}
                        print(f"Tọa độ của {district}, {province} đã được tự động cập nhật.\n")
                else:
                    updated_districts[district] = coordinates  # Giữ tọa độ cũ nếu chính xác
                    print(f"Tọa độ của {district}, {province} đã chính xác.\n")
            else:
                # Giữ nguyên tọa độ cũ nếu không thể lấy tọa độ mới
                updated_districts[district] = coordinates
                print(f"Không thể lấy tọa độ thực tế của {district}, {province}, giữ nguyên tọa độ cũ.\n")
        
        # Chỉ thêm các huyện hợp lệ vào danh sách của tỉnh
        if updated_districts:
            updated_districts_data[province] = updated_districts

    # Ghi lại dữ liệu sau khi chỉnh sửa vào file JSON
    f.seek(0)
    f.truncate()
    json.dump(updated_districts_data, f, ensure_ascii=False, indent=4)
