import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '7_days_future_predictions.csv'
data = pd.read_csv(file_path)

# Convert 'Timestamp' to datetime format for easier filtering
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Filter for the first 24 hours (0 AM to 11 PM) of each district
districts = data['District'].unique()

for district in districts:
    # Extract data for the district
    district_data = data[data['District'] == district].sort_values('Timestamp')
    
    # Find the first day with a full 24-hour set of data
    valid_start_date = None
    for date in district_data['Timestamp'].dt.normalize().unique():
        # Filter data for this date
        day_data = district_data[district_data['Timestamp'].dt.normalize() == date]
        
        # Check if the day has 24 entries (hours)
        if len(day_data) == 24:
            valid_start_date = date
            break
    
    # If no valid day found, skip the district
    if valid_start_date is None:
        print(f"No valid 24-hour data found for {district}. Skipping...")
        continue

    # Filter data for the valid day
    district_data_24h = district_data[district_data['Timestamp'].dt.normalize() == valid_start_date]
    
    # Plot AQI prediction for the first valid 24 hours
    plt.figure(figsize=(10, 6))
    plt.plot(district_data_24h['Timestamp'], district_data_24h['AQI Prediction'], marker='o', linestyle='-')
    
    # Add title and labels
    plt.title(f'24-Hour AQI Prediction for {district}')
    plt.xlabel('Timestamp')
    plt.ylabel('AQI Prediction')
    
    # Set x-axis major ticks to show every hour
    plt.xticks(district_data_24h['Timestamp'], district_data_24h['Timestamp'].dt.strftime('%H:%M'), rotation=45)
    
    # Ensure proper layout
    plt.tight_layout()
    
    # Display the plot
    plt.show()
