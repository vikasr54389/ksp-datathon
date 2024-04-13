from ultralytics import YOLO
import cv2
import numpy as np

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Define a list of image paths
image_paths = ["inference/images/signal11.jpg",
               "inference/images/signal22.jpg",
               "inference/images/signal33.jpg",
               "inference/images/signal44.jpg"]

# Define the region of interest (ROI) as a rectangle on the image
# The coordinates are (x1, y1) as top-left corner and (x2, y2) as bottom-right corner
roi_x1, roi_y1 = 100, 300  # Top-left corner of the ROI
roi_x2, roi_y2 = 900, 900  # Bottom-right corner of the ROI

# Define a list of allowed class names (vehicles to detect)
allowed_classes = ["car", "bus", "tuk-tuk", "motorcycle", "bicycle"]

# Iterate through each image path
count = 0
for image_path in image_paths:
    # Predict on an image
    detection_output = model.predict(source=image_path, conf=0.10)

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Draw the ROI on the image (blue rectangle with thickness of 2)
    cv2.rectangle(image, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)  # Blue color (BGR format)

    # Counters for the number of cars, buses, and total vehicles within the ROI
    car_count = 0
    bus_count = 0
    total_vehicles = 0  # New counter for total vehicles

    # Iterate through the predictions and draw bounding boxes on the image
    for detection in detection_output:
        # Each detection contains bounding boxes, class IDs, confidence scores, etc.
        for box in detection.boxes:
            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID

            # Convert the coordinates from float to int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Check if the bounding box is fully or partially inside the ROI
            if not (x2 < roi_x1 or x1 > roi_x2 or y2 < roi_y1 or y1 > roi_y2):
                # If it is, check if the class name is in the allowed classes list
                class_name = detection.names[class_id]
                if class_name in allowed_classes:
                    total_vehicles += 1  # Increment total vehicle count

                    # Increment specific class counters based on class name
                    if class_name == "car":
                        car_count += 1
                    elif class_name == "bus":
                        bus_count += 1
                    # Draw the bounding box on the image if within the ROI
            if x2 >= roi_x1 and x1 <= roi_x2 and y2 >= roi_y1 and y1 <= roi_y2:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness of 2
                # Optionally, add text labels (class and confidence) on the image
                label = f"{detection.names[class_id]}: {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image using OpenCV
    cv2.imshow(f"Detections with ROI - {image_path}", image)

    # Print the count of cars, buses, and total vehicles within the ROI
    print(f"Number of cars in the ROI: {car_count}")
    print(f"Number of buses in the ROI: {bus_count}")
    print(f"Total number of vehicles in the ROI: {total_vehicles}")

    # Determine the time settings based on the number of vehicles
    if total_vehicles > 10:
        print("green time allotted:%d",((total_vehicles+(car_count*1.5)+(bus_count*2))/400)*200)
    else:
        print("No Vehicle Found in the Region of Interest, turn on red signal")

    count += 1
    if count % 2 == 0:
        print("This time is allotted for the pedestrian = 10 sec")

    # Wait for a key press and then close the display window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
