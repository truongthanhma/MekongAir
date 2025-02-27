import pandas as pd
import json
from datetime import datetime, timedelta
from geopy.distance import distance
from sklearn.preprocessing import MinMaxScaler
import torch
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Đọc dữ liệu từ Excel và JSON (tọa độ huyện)
data = pd.read_excel('all_districts_air_quality_data.xlsx')
data['dt'] = pd.to_datetime(data['dt'])

with open('mekong_districts.json', 'r', encoding='utf-8') as f:
    location_data = json.load(f)

# Lấy tọa độ của các huyện
def get_lat_lon(row, location_data):
    province, district = row['Province'], row['District']
    if province in location_data and district in location_data[province]:
        return location_data[province][district]['lat'], location_data[province][district]['lon']
    return None, None

data['latitude'], data['longitude'] = zip(*data.apply(lambda row: get_lat_lon(row, location_data), axis=1))

# Xây dựng cạnh dựa trên khoảng cách địa lý
def build_edges_from_distance(data, distance_threshold):
    edges = []
    districts = data[['Province', 'District', 'latitude', 'longitude']].drop_duplicates()
    for i, row1 in districts.iterrows():
        for j, row2 in districts.iterrows():
            if i != j:
                dist = distance((row1['latitude'], row1['longitude']), (row2['latitude'], row2['longitude'])).km
                if dist < distance_threshold:
                    edges.append((district_to_idx[row1['District']], district_to_idx[row2['District']]))
    return edges

# Map mỗi huyện với một chỉ số duy nhất
districts = list(data['District'].unique())
district_to_idx = {district: idx for idx, district in enumerate(districts)}

# Xây dựng cạnh
edges = build_edges_from_distance(data, distance_threshold=50)

# Chuẩn bị dữ liệu cho GAT-LSTM theo từng giờ
def prepare_graph_data_hourly(data, date, hour, feature_scaler, aqi_scaler, edges):
    hour_data = data[(data['dt'].dt.date == pd.to_datetime(date).date()) & (data['dt'].dt.hour == hour)]
    
    if hour_data.empty:
        return None

    features = feature_scaler.transform(hour_data[['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']])
    time_features = hour_data[['hour_of_day', 'day_of_week']].values
    node_features = torch.tensor(np.concatenate([features, time_features], axis=1), dtype=torch.float)

    aqi_scaled = aqi_scaler.transform(hour_data[['AQI']])
    labels = torch.tensor(aqi_scaled, dtype=torch.float).squeeze()

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = add_self_loops(edge_index)[0]

    return Data(x=node_features, edge_index=edge_index, y=labels)

# Thêm cột giờ trong ngày và ngày trong tuần
data['hour_of_day'] = data['dt'].dt.hour
data['day_of_week'] = data['dt'].dt.dayofweek

# Chuẩn bị các scaler
feature_scaler = MinMaxScaler()
aqi_scaler = MinMaxScaler()
time_scaler = MinMaxScaler()

# Huấn luyện các scaler
data[['hour_of_day', 'day_of_week']] = time_scaler.fit_transform(data[['hour_of_day', 'day_of_week']])
feature_scaler.fit(data[['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']])
aqi_scaler.fit(data[['AQI']])

# Tạo dữ liệu cho từng giờ của mỗi ngày
dates = data['dt'].dt.date.unique()
hours = range(24)
graph_data_hourly = []

for date in dates:
    for hour in hours:
        graph_data = prepare_graph_data_hourly(data, date, hour, feature_scaler, aqi_scaler, edges)
        if graph_data:
            graph_data_hourly.append(graph_data)

# Xây dựng mô hình GAT-LSTM
class GAT_LSTM_Model(nn.Module):
    def __init__(self, num_node_features, hidden_size, lstm_hidden_size, num_heads, num_layers):
        super(GAT_LSTM_Model, self).__init__()
        self.gat = GATConv(num_node_features, hidden_size, heads=num_heads)
        self.lstm = nn.LSTM(hidden_size * num_heads, lstm_hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, graph_data, hidden_state=None):
        x, edge_index = graph_data.x, graph_data.edge_index
        x = self.gat(x, edge_index)
        x = torch.relu(x)
        x = x.unsqueeze(0)  # Thêm chiều batch cho LSTM
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
            print(f'Early stopping at epoch {epoch + 1}')
            break

# Dự đoán AQI cho giờ trong tương lai
def predict_aqi_for_future(model, start_data, aqi_scaler, district_names, start_time, future_hours=1):
    model.eval()
    predictions = []
    hidden_state = None
    current_data = start_data
    current_time = start_time

    for hour in range(future_hours):
        output, hidden_state = model(current_data, hidden_state)
        predictions_scaled = output.squeeze().detach().cpu().numpy()
        predictions_original = aqi_scaler.inverse_transform(predictions_scaled.reshape(-1, 1))

        hour_predictions = {
            "timestamp": current_time,
            "predictions": {district_names[i]: round(predictions_original[i][0]) for i in range(len(district_names))}
        }
        predictions.append(hour_predictions)

        current_time += timedelta(hours=1)
        current_data = update_data_for_next_hour(current_data, predictions_original)

    return predictions

# Cập nhật dữ liệu cho giờ tiếp theo dựa trên dự đoán hiện tại
def update_data_for_next_hour(current_data, predicted_aqi):
    new_data = current_data.clone()
    new_data.y = torch.tensor(predicted_aqi, dtype=torch.float)
    new_data.x[:, -2] += 1
    new_data.x[:, -2] = new_data.x[:, -2] % 24
    new_data.x[:, -1] = (new_data.x[:, -1] + (new_data.x[:, -2] == 0).float()) % 7
    return new_data

# Khởi tạo và huấn luyện mô hình
model = GAT_LSTM_Model(num_node_features=10, hidden_size=64, lstm_hidden_size=64, num_heads=4, num_layers=2)
train_loader = DataLoader(graph_data_hourly, batch_size=32, shuffle=True)

train_model(train_loader, model, num_epochs=100, lr=0.0001, patience=10)

# Hàm đánh giá mô hình
def evaluate_model(graph_data, model, aqi_scaler):
    model.eval()
    true_values = []
    predicted_values = []
    hidden_state = None

    for data in graph_data:
        output, hidden_state = model(data, hidden_state)
        predictions_scaled = output.squeeze().detach().cpu().numpy()
        predictions_original = aqi_scaler.inverse_transform(predictions_scaled.reshape(-1, 1))

        true_values.extend(aqi_scaler.inverse_transform(data.y.cpu().numpy().reshape(-1, 1)))
        predicted_values.extend(predictions_original)

    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)

    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)

    return mae, mse, rmse

# Đánh giá mô hình sau khi huấn luyện
mae, mse, rmse = evaluate_model(graph_data_hourly, model, aqi_scaler)
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Dự đoán AQI cho tương lai
district_names = list(data['District'].unique())
start_data = graph_data_hourly[-1]
start_time = pd.to_datetime(data['dt'].max())

future_hours = 7
future_predictions = predict_aqi_for_future(model, start_data, aqi_scaler, district_names, start_time, future_hours)

# In kết quả dự đoán
for prediction in future_predictions:
    print("--------------------------------------------------------------------------------------")
    print(f"Dự đoán AQI cho {prediction['timestamp']}:")
    for district, aqi in prediction['predictions'].items():
        print(f"Huyện: {district}, Dự đoán AQI: {aqi:.2f}")

# Lưu mô hình
torch.save(model.state_dict(), 'gat_lstm_model.pth')
print("Mô hình GAT-LSTM đã được lưu thành công.")
