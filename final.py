import cv2
import numpy as np
import requests

# Function to calculate distance
def calculate_distance(known_width, focal_length, observed_width):
    return (known_width * focal_length) / observed_width

# Function to get weather information
def get_weather(api_key, city):
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(base_url)
    weather_data = response.json()

    if weather_data["cod"] == 200:
        weather_description = weather_data["weather"][0]["description"]
        temperature = weather_data["main"]["temp"]
        humidity = weather_data["main"]["humidity"]
        wind_speed = weather_data["wind"]["speed"]

        return weather_description, temperature, humidity, wind_speed
    else:
        return "Error", None, None, None

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print("YOLO LOADED")

# Replace 'YOUR_OPENWEATHERMAP_API_KEY' with your actual OpenWeatherMap API key
api_key = '5b12b44c37eb32abd61621d14bd240e1'

# Replace 'CITY_NAME' with the city you want to get the weather for (e.g., 'Mumbai', 'Delhi', 'Bangalore')
city_name = 'Karkala'

# Example values (for demonstration, replace with actual values)
known_width_of_object = 1.0  # Width of the object in meters
focal_length = 1000  # Focal length of the camera in pixels (hypothetical value)

address_var = input("Enter IP Address: ")
vid = cv2.VideoCapture("http://" + address_var + "/video")

while True:
    # Capture the video frame by frame
    ret, frame = vid.read()
    if not ret:
        break

    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
                # Calculate observed width of the object in pixels
                observed_width = w  # Assuming you're using the width of the bounding box
                
                # Calculate distance
                distance = calculate_distance(known_width_of_object, focal_length, observed_width)
                
                # Display distance on the frame
                cv2.putText(frame, f"Distance: {distance:.2f} meters", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1 / 2, color, 2)

            # Get weather information for the specified city
            weather_description, temperature, humidity, wind_speed = get_weather(api_key, city_name)
            
            # Display weather information on the frame
            if weather_description != "Error":
                weather_info = f"Weather in {city_name}: {weather_description}, {temperature}°C, Humidity: {humidity}%, Wind Speed: {wind_speed} m/s"
                cv2.putText(frame, weather_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Image", frame)
    
    # Check for key press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
