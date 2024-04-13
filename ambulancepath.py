from ultralytics import YOLO
import cv2
import numpy as np

# Get user input for the route number
x = input("Enter the route in which the ambulance needs to travel (1, 2, 3 ,4): ")
reach_time_ambulance=int(input("enter the time to reach the signal by the ambulance:"))

# Initialize the model
model = YOLO("yolov8n.pt")

# Define image paths based on user input
if x == "1":
    image_paths = ["inference/images/signal11.jpg"]
elif x == "2":
    image_paths = ["inference/images/signal22.jpg"]
elif x == "3":
    image_paths = ["inference/images/signal33.jpg"]
else:
    image_paths = ["inference/images/signal44.jpg"]

# Define a function to process images
def process_image(image_path, model):
    # Predict on an image
    detection_output = model.predict(source=image_path, conf=0.25)

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Counters for the number of cars, buses, and total vehicles
    car_count = 0
    bus_count = 0
    total_vehicle_count = 0

    # Define a list of vehicle classes to detect (excluding persons)
    vehicle_classes = ["car", "bus", "truck", "motorcycle", "bicycle"]

    # Iterate through the predictions and draw bounding boxes on the image
    for detection in detection_output:
        for box in detection.boxes:
            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            class_id = int(box.cls[0])

            # Convert the coordinates from float to int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get the class name of the detected object
            class_name = detection.names[class_id]

            # Check if the detection is a vehicle (excluding persons)
            if class_name in vehicle_classes:
                # Draw the bounding box on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness of 2

                # Optionally, add text labels (class and confidence) on the image
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Increment total vehicle count
                total_vehicle_count += 1

                # Check if the detection is a car or bus
                if class_name == "car":
                    car_count += 1
                elif class_name == "bus":
                    bus_count += 1

    # Display the image using OpenCV
    cv2.imshow(f"Detections - {image_path}", image)

    # Print the count of cars, buses, and total vehicles
    print(f"Number of cars: {car_count}")
    print(f"Number of buses: {bus_count}")
    print(f"Total number of vehicles: {total_vehicle_count}")

    # Determine timing based on car and bus counts
    if total_vehicle_count>10:
       print("green time is:%d",((150+total_vehicle_count)/reach_time_ambulance)*100)
    else:
       print("on the red signal there is no vehicle")
        
    # Return car count, bus count, and total vehicle count
    return car_count, bus_count, total_vehicle_count

# Initialize cycle count
count = 0

# Iterate through each image path
for path in image_paths:
    # Process the image and get car, bus, and total vehicle counts
    car_count, bus_count, total_vehicle_count = process_image(path, model)

    # Alternate pedestrian time based on cycle count
    count += 1
    if count % 2 == 0:
        print("This time is allotted for the pedestrian = 10 sec")

    # Wait for a key press and then close the display window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
