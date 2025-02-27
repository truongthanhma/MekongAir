import pandas as pd
import json
from datetime import datetime, timedelta
from geopy.distance import distance
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import SAGEConv, GINConv
from torch.nn import BatchNorm1d
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Đọc dữ liệu từ Excel và JSON (tọa độ huyện)
data = pd.read_excel('all_districts_air_quality_data.xlsx')
data['dt'] = pd.to_datetime(data['dt'])  # Chuyển đổi thành dạng ngày tháng

with open('mekong_districts.json', 'r', encoding='utf-8') as f:
    location_data = json.load(f)

# Lấy tọa độ của các huyện
def get_lat_lon(row, location_data):
    province, district = row['Province'], row['District']
    if province in location_data and district in location_data[province]:
        return location_data[province][district]['lat'], location_data[province][district]['lon']
    return None, None

data['latitude'], data['longitude'] = zip(*data.apply(lambda row: get_lat_lon(row, location_data), axis=1))

# Tạo danh sách các huyện và map mỗi huyện với một chỉ số duy nhất
districts = list(data['District'].unique())
district_to_idx = {district: idx for idx, district in enumerate(districts)}

# Xây dựng cạnh chỉ dựa trên khoảng cách địa lý
def build_edges_from_distance(data, distance_threshold):
    edges = []
    districts_unique = data[['Province', 'District', 'latitude', 'longitude']].drop_duplicates()

    for i, row1 in districts_unique.iterrows():
        for j, row2 in districts_unique.iterrows():
            if i != j:
                dist = distance((row1['latitude'], row1['longitude']), (row2['latitude'], row2['longitude'])).km
                if dist < distance_threshold:
                    edges.append((district_to_idx[row1['District']], district_to_idx[row2['District']]))
    return edges

# Xây dựng cạnh
edges = build_edges_from_distance(data, distance_threshold=50)

# Thêm cột giờ trong ngày và ngày trong tuần dưới dạng sin-cos
data['hour_sin'] = np.sin(2 * np.pi * data['dt'].dt.hour / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['dt'].dt.hour / 24)
data['day_sin'] = np.sin(2 * np.pi * data['dt'].dt.dayofweek / 7)
data['day_cos'] = np.cos(2 * np.pi * data['dt'].dt.dayofweek / 7)

# Chuẩn bị StandardScaler cho các đặc trưng không khí và AQI
feature_scaler = StandardScaler()
aqi_scaler = StandardScaler()

# Huấn luyện các scaler cho đặc trưng không khí và AQI
feature_scaler.fit(data[['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']])
aqi_scaler.fit(data[['AQI']])

# Chuẩn bị dữ liệu cho GNN theo từng giờ
def prepare_graph_data_hourly(data, date, hour, feature_scaler, aqi_scaler, edges):
    hour_data = data[(data['dt'].dt.date == pd.to_datetime(date).date()) & (data['dt'].dt.hour == hour)]
    
    if hour_data.empty:
        return None

    # Lấy đặc trưng không khí và thời gian dưới dạng sin-cos
    features = feature_scaler.transform(hour_data[['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']])
    time_features = hour_data[['hour_sin', 'hour_cos', 'day_sin', 'day_cos']].values
    node_features = torch.tensor(np.concatenate([features, time_features], axis=1), dtype=torch.float)

    # Chuẩn hóa chỉ số AQI
    aqi_scaled = aqi_scaler.transform(hour_data[['AQI']])
    labels = torch.tensor(aqi_scaled, dtype=torch.float).squeeze()

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = add_self_loops(edge_index)[0]

    return Data(x=node_features, edge_index=edge_index, y=labels)

# Tạo dữ liệu cho từng giờ của mỗi ngày
dates = data['dt'].dt.date.unique()
hours = range(24)
graph_data_hourly = []

for date in dates:
    for hour in hours:
        graph_data = prepare_graph_data_hourly(data, date, hour, feature_scaler, aqi_scaler, edges)
        if graph_data:
            graph_data_hourly.append(graph_data)

# Xây dựng mô hình GNN-GIN-SAGE-LSTM
class GNN_LSTM_Model(nn.Module):
    def __init__(self, num_node_features, hidden_size, lstm_hidden_size, num_layers, use_gin=True):
        super(GNN_LSTM_Model, self).__init__()
        self.use_gin = use_gin
        self.batch_norm = BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        if use_gin:
            self.gin = GINConv(
                nn.Sequential(
                    nn.Linear(num_node_features, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
            )
        else:
            self.sage = SAGEConv(num_node_features, hidden_size)
        self.lstm = nn.LSTM(hidden_size, lstm_hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, graph_data, hidden_state=None):
        if self.use_gin:
            x = self.gin(graph_data.x, graph_data.edge_index)
        else:
            x = self.sage(graph_data.x, graph_data.edge_index)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = x.unsqueeze(0)  # Thêm chiều thời gian cho LSTM
        out, hidden_state = self.lstm(x, hidden_state)
        out = self.fc(out).squeeze(2)
        return out, hidden_state

# Huấn luyện mô hình
def train_model(train_loader, model, num_epochs=100, lr=0.0001, patience=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    best_loss, no_improvement = float('inf'), 0

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            hidden_state = None
            output, hidden_state = model(data, hidden_state)
            output = output.squeeze()
            loss = criterion(output, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss}')
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss, no_improvement = avg_loss, 0
        else:
            no_improvement += 1
        if no_improvement >= patience:
            print(f'Early stopping tại epoch {epoch + 1}')
            break

# Đánh giá mô hình
def evaluate_model(graph_data, model, aqi_scaler):
    model.eval()
    true_values = []
    predicted_values = []
    hidden_state = None

    with torch.no_grad():
        for data in graph_data:
            output, hidden_state = model(data, hidden_state)
            predictions_scaled = output.squeeze().cpu().numpy()
            predictions_original = aqi_scaler.inverse_transform(predictions_scaled.reshape(-1, 1))

            true_values.extend(aqi_scaler.inverse_transform(data.y.cpu().numpy().reshape(-1, 1)))
            predicted_values.extend(predictions_original)

    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)

    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)

    return mae, mse, rmse

# Hàm để chuẩn bị dữ liệu cho giờ tiếp theo dựa trên dự đoán hiện tại
def update_data_for_next_hour(current_data, predicted_aqi):
    new_data = current_data.clone()

    # Cập nhật nhãn AQI cho giờ tiếp theo bằng giá trị dự đoán của giờ trước
    new_data.y = torch.tensor(predicted_aqi, dtype=torch.float)

    # Giải mã các giá trị hour_sin và hour_cos để tăng giờ thêm 1
    hour = np.arctan2(new_data.x[:, -4].numpy(), new_data.x[:, -3].numpy()) * (24 / (2 * np.pi))
    hour = (hour + 1) % 24
    new_data.x[:, -4] = torch.tensor(np.sin(2 * np.pi * hour / 24), dtype=torch.float)
    new_data.x[:, -3] = torch.tensor(np.cos(2 * np.pi * hour / 24), dtype=torch.float)

    # Giải mã các giá trị day_sin và day_cos để tăng ngày thêm 1 nếu cần
    day_of_week = np.arctan2(new_data.x[:, -2].numpy(), new_data.x[:, -1].numpy()) * (7 / (2 * np.pi))
    day_of_week = (day_of_week + (hour == 0)) % 7  # Tăng ngày nếu giờ mới bắt đầu ngày mới
    new_data.x[:, -2] = torch.tensor(np.sin(2 * np.pi * day_of_week / 7), dtype=torch.float)
    new_data.x[:, -1] = torch.tensor(np.cos(2 * np.pi * day_of_week / 7), dtype=torch.float)

    return new_data

# Hàm dự đoán AQI cho tương lai
def predict_aqi_for_future(model, start_data, aqi_scaler, district_names, start_time, future_hours=1):
    model.eval()
    predictions = []
    hidden_state = None
    current_data = start_data
    current_time = start_time

    with torch.no_grad():
        for hour in range(future_hours):
            output, hidden_state = model(current_data, hidden_state)
            predictions_scaled = output.squeeze().cpu().numpy()
            predictions_original = aqi_scaler.inverse_transform(predictions_scaled.reshape(-1, 1))

            # Lưu dự đoán cho giờ hiện tại với ngày giờ tương ứng
            hour_predictions = {
                "timestamp": current_time,
                "predictions": {district_names[i]: round(predictions_original[i][0], 2) for i in range(len(district_names))}
            }
            predictions.append(hour_predictions)

            # Tăng thời gian thêm một giờ
            current_time += timedelta(hours=1)

            # Cập nhật dữ liệu cho giờ tiếp theo
            current_data = update_data_for_next_hour(current_data, predictions_original)

    return predictions

# Khởi tạo và huấn luyện mô hình
model = GNN_LSTM_Model(num_node_features=12, hidden_size=64, lstm_hidden_size=64, num_layers=2, use_gin=True)
train_loader = DataLoader(graph_data_hourly, batch_size=32, shuffle=True)

# Huấn luyện mô hình
train_model(train_loader, model, num_epochs=100, lr=0.0001, patience=10)

# Đánh giá mô hình trên toàn bộ dữ liệu
mae, mse, rmse = evaluate_model(graph_data_hourly, model, aqi_scaler)
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Lưu các giá trị vào file txt
output_file = 'evaluation_metrics.txt'  # Tên file bạn muốn lưu
with open(output_file, 'w') as file:
    file.write(f"MAE: {mae:.4f}\n")
    file.write(f"MSE: {mse:.4f}\n")
    file.write(f"RMSE: {rmse:.4f}\n")

print(f"Đã lưu các chỉ số đánh giá vào file {output_file}")

# Dự báo chỉ số AQI cho các giờ trong tương lai
district_names = list(data['District'].unique())
start_data = graph_data_hourly[-1]
start_time = pd.to_datetime(data['dt'].max())

# Số giờ muốn dự báo
future_hours = 7 * 24
future_predictions = predict_aqi_for_future(model, start_data, aqi_scaler, district_names, start_time, future_hours)

# Hiển thị kết quả dự báo cho từng giờ
for prediction in future_predictions:
    print("--------------------------------------------------------------------------------------")
    print(f"Dự đoán AQI cho {prediction['timestamp']}:")
    for district, aqi in prediction['predictions'].items():
        print(f"Huyện: {district}, Dự đoán AQI: {aqi:.2f}")

# Lưu dự báo vào tệp CSV
predictions_list = []
for prediction in future_predictions:
    for district, aqi in prediction['predictions'].items():
        predictions_list.append({
            "Timestamp": prediction['timestamp'],
            "District": district,
            "AQI Prediction": int(aqi)
        })

predictions_df = pd.DataFrame(predictions_list)
predictions_df.to_csv('7_hours_future_predictions_standardized.csv', index=False)
print("Kết quả dự đoán đã được lưu vào tệp CSV.")

# Lưu toàn bộ mô hình
torch.save(model.state_dict(), 'gnn_lstm_model_standardized.pth')
print("Mô hình đã được lưu thành công với StandardScaler.")