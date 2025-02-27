import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu
models = ['GAT-LSTM', 'GIN-LSTM', 'GraphSAGE-LSTM']
mae = [2.6432, 9.2647, 4.6025]
mse = [15.2023, 201.2255, 45.7276]
rmse = [3.9132, 14.1854, 6.7622]
# Vẽ biểu đồ
x = np.arange(len(models))  # Vị trí của các mô hình trên trục x

# Tạo biểu đồ với nhiều thanh (bar) cho mỗi mô hình
width = 0.2  # Độ rộng của từng thanh
fig, ax = plt.subplots(figsize=(10, 6))

bar1 = ax.bar(x - width, mae, width, label='MAE', color='skyblue')
bar2 = ax.bar(x, mse, width, label='MSE', color='salmon')
bar3 = ax.bar(x + width, rmse, width, label='RMSE', color='lightgreen')

# Thêm thông tin cho biểu đồ
ax.set_xlabel('Mô hình')
ax.set_ylabel('Giá trị')
ax.set_title('So sánh MAE, MSE và RMSE của các mô hình - Dữ liệu 7 tháng')  # Tiêu đề đã được thay đổi
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Hiển thị giá trị lên từng cột
def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Đẩy giá trị lên trên cột
                    textcoords="offset points",
                    ha='center', va='bottom')

# Thêm giá trị cho từng cột
add_values(bar1)
add_values(bar2)
add_values(bar3)

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
