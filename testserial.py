import serial

# Open the serial port
arduino = serial.Serial(port='COM5', baudrate=9600, timeout=1)

lat = 0.0  # Initialize latitude
lon = 0.0  # Initialize longitude

try:
    while True:
        if arduino.in_waiting > 0:
            # Read and decode data from Arduino
            data = arduino.readline().decode('utf-8', errors='ignore').strip()
            
            # Ensure the data format is correct before processing
            if "Latitude=" in data and "Longitude=" in data:
                parts = data.split()
                try:
                    lat = float(parts[1])  # Update latitude
                    lon = float(parts[3])  # Update longitude
                    print(lat,lon)
                except (IndexError, ValueError) as e:
                    print(f"Data error: {e}, raw data: {data}")
except KeyboardInterrupt:
    print("\nExiting...")
finally:
    arduino.close()

