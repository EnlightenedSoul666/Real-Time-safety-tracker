import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import serial
from datetime import datetime, timedelta
import pytz  # Ensure this is installed: pip install pytz

def get_current_ist_time():
    # Get the current UTC time
    utc_time = datetime.utcnow()
    
    # Convert to Indian Standard Time (IST)
    ist_time = utc_time + timedelta(hours=5, minutes=30)
    return ist_time.strftime('%Y-%m-%d %H:%M:%S')
# Load the dataset
dataset = pd.read_csv('chennai_crime_dataset.csv')

# Step 1: Convert one-hot encoded labels into a single 'Safety_Level' column
def get_safety_level(row):
    if row['Red'] == 1:
        return 'Red'
    elif row['Yellow'] == 1:
        return 'Yellow'
    elif row['Green'] == 1:
        return 'Green'

dataset['Safety_Level'] = dataset.apply(get_safety_level, axis=1)

# Step 2: Prepare the features (latitude and longitude) and labels (safety level)
X = dataset[['Latitude', 'Longitude']]
y = dataset['Safety_Level']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Step 5: Save the trained model to a file
joblib.dump(knn_model, 'crime_model.pkl')
print("Model trained and saved as 'crime_model.pkl'")

# Load the trained KNN model
model_path = 'crime_model.pkl'
model = joblib.load(model_path)

def classify_location(lat, lon):
    # Convert the input coordinates to a 2D list
    coordinates = [[float(lat), float(lon)]]
    # Predict the safety level using the KNN model
    prediction = model.predict(coordinates)
    return prediction[0]

# Main loop to read data and classify location
try:
    arduino = serial.Serial(port='COM5', baudrate=9600, timeout=1)
    
    while True:
        if arduino.in_waiting > 0:
            try:
            # Read and decode data from Arduino
                data = arduino.readline().decode('utf-8', errors='ignore').strip()
            
            # Check if the data contains both latitude and longitude
                if "Latitude:" in data and "Longitude:" in data:
                    parts = data.split()  # Split by spaces
                    lat = float(parts[1].replace(',', ''))  # Get latitude value
                    lon = float(parts[3].replace(',', ''))  # Get longitude value

                    print(f"Latitude: {lat}, Longitude: {lon}")
                    ist_time = get_current_ist_time()
                    print(f"Timestamp (IST): {ist_time}")
                    print(f"Latitude: {lat}, Longitude: {lon}")

                # Predict safety level
                    safety_level = classify_location(lat, lon)
                    print(f"Predicted Safety Level: {safety_level}")
                else:
                    print(f"Unexpected data format: {data}")
            except (IndexError, ValueError) as e:
                print(f"Data error: {e}, raw data: {data}")

except KeyboardInterrupt:
    print("\nExiting...")
    
finally:
    arduino.close()
