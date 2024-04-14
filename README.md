'''
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
'''

##Import Statements:

It imports the YOLO model from the ultralytics library and the OpenCV (cv2) and NumPy (np) libraries.
Model Loading:

A pretrained YOLOv8n model is loaded from a file ("yolov8n.pt").
Image Paths:

A list of image paths (image_paths) is defined for the images that need to be processed.
Region of Interest:

The code defines a region of interest (ROI) as a rectangle on the image using top-left and bottom-right coordinates (roi_x1, roi_y1, roi_x2, roi_y2).
Allowed Classes:

A list of allowed class names (allowed_classes) is defined. These are the types of vehicles the code aims to detect, such as cars, buses, motorcycles, and bicycles.
Main Loop:

The code iterates through each image path in the list:
Prediction:
For each image, it uses the YOLO model to predict objects in the image with a confidence threshold of 0.10.
Image Loading:
It loads the image using OpenCV.
Draw ROI:
It draws the defined ROI as a blue rectangle on the image.
Counters:
It initializes counters for the number of cars, buses, and total vehicles within the ROI.
Processing Predictions:
It iterates through the predictions, drawing bounding boxes and labels for detected objects in the image and counting the allowed classes within the ROI.
Display and Output:
The processed image is displayed using OpenCV.
The count of cars, buses, and total vehicles within the ROI is printed.
The code decides whether to allot green time or turn on a red signal based on the total number of vehicles within the ROI.
Every second image in the list also triggers a 10-second pedestrian crossing time.
Cleanup:
After each image is processed, the code waits for a key press and then closes the display window.

'''from ultralytics import YOLO
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
'''

##User Input:

The code starts by prompting the user for a route number (x) and the time it will take for the ambulance to reach the signal (reach_time_ambulance).
Model Initialization:user input will be taken an signal from google maps in the future and implemented in web server

A pretrained YOLOv8n model is loaded from a file (yolov8n.pt).
Image Paths:

Based on the route number provided by the user, a specific image path is selected. Each route corresponds to a different image path.
Function Definition:

The code defines a function process_image that takes an image path and the YOLO model as input parameters.
Inside the function:
Prediction:
The function uses the YOLO model to predict objects in the image with a confidence threshold of 0.25.
Image Loading:
It loads the image using OpenCV.
Counters:
It initializes counters for the number of cars, buses, and total vehicles in the image.
Processing Predictions:
The function iterates through the predictions, drawing bounding boxes and labels for detected objects in the image.
It counts the number of vehicles (cars, buses, and total vehicles) detected.
Display and Output:
The function displays the processed image using OpenCV and prints the count of cars, buses, and total vehicles.
Timing Calculation:
The function calculates and prints the green time allotted based on the total vehicle count and the time it takes for the ambulance to reach the signal.
The function returns the counts of cars, buses, and total vehicles.
Main Loop:

The code iterates through each image path in the list:
For each path, it processes the image using the function process_image and gets the counts of cars, buses, and total vehicles.
It alternates pedestrian time every second iteration by printing a 10-second pedestrian crossing time.
After processing each image, the code waits for a key press and then closes the display window.

'''import sys
from heapq import heappop, heappush
from colorama import Fore
import winsound

def dijkstra(graph, source, destinations):
    """
    Dijkstra's shortest path algorithm.

    Args:
      graph: A dictionary of dictionaries, where each key is a node and the value is a dictionary
        of neighbors and their distances.
      source: The starting node.
      destinations: A set of nodes to find the nearest one.

    Returns:
      A tuple of the shortest distance and the nearest destination node.
    """
    inf = sys.maxsize
    node_data = {}
    for node in graph:
        node_data[node] = {'cost': inf, 'predecessors': set()}
    node_data[source]['cost'] = 0
    min_heap = [(0, source)]
    while min_heap:
        cost, temp = heappop(min_heap)
        if temp in destinations:
            break
        for neighbor, distance in graph[temp].items():
            new_cost = cost + distance
            if new_cost < node_data[neighbor]['cost']:
                node_data[neighbor]['cost'] = new_cost
                node_data[neighbor]['predecessors'] = {temp}
                heappush(min_heap, (new_cost, neighbor))
    min_distance = min(node_data[destination]['cost'] for destination in destinations)
    nearest_destination = next(destination for destination in destinations if node_data[destination]['cost'] == min_distance)
    return min_distance, nearest_destination

# Example usage
graph = {
  'am': {'d': 130 },
  'a': {'b': 180, 'c': 150, 'h': 120 },
  'b': {'d': 130, 'c': 170, 'a': 180 }, 
  'c': {'b': 170, 'd': 170, 'e': 90, 'f': 140, 'a': 150 },
  'd': {'am': 130, 'e': 150, 'c': 170, 'b': 170 },
  'e': {'d': 150, 'c': 90, 'f': 120 },
  'f': {'c': 140, 'e': 120, 'h': 140 },
  'h': {'f': 140, 'a': 120 },
  'x': {'e': 50,'d': 100 }
}

user_defined_point = 'x' # Replace 'x' with the user-defined point
nearest_nodes = {'e'}  # Replace with the nodes you want to find the nearest one

shortest_distance, nearest_node = dijkstra(graph, user_defined_point, nearest_nodes)

if shortest_distance == sys.maxsize:
    print("No path found from", user_defined_point, "to any of the destinations")


#def time_scheduling(user_defined_point,shortest_distance,nearest_node):

if shortest_distance<=750:
    f=2000
    d=5000
    winsound.Beep (f,d)
    print(shortest_distance)
else:
    print("no nearest ambulance")
'''
##Import Statements:

sys: Provides access to system-specific parameters and functions. The code uses sys.maxsize as a representation of infinity.
heapq: Provides a heap queue (priority queue) implementation. The code uses it to maintain a priority queue of nodes based on the shortest distance found so far.
colorama: A library to provide color formatting in terminal outputs (unused in the code).
winsound: A library for playing sound on Windows.
Function dijkstra:

The function implements Dijkstra's shortest path algorithm. It takes a graph, a source node, and a set of destination nodes as inputs.
The function initializes a dictionary node_data to store the cost (shortest distance) and predecessors for each node.
The source node's cost is initialized to 0, and the function uses a min heap to prioritize nodes based on the cost.
The function iterates through the min heap, updating the cost and predecessors of neighboring nodes if a shorter path is found.
When the function reaches a destination node, it breaks out of the loop.
The function returns the shortest distance to a destination node and the nearest destination node.
Example Usage:

The code includes an example graph represented as a dictionary, where each key is a node and its value is another dictionary of neighbors and distances.
The user-defined point (user_defined_point) is set to 'x', and the destination nodes (nearest_nodes) are set to a set containing 'e'.
The function dijkstra is called with the graph, the user-defined point, and the destination nodes.
If no path is found from the user-defined point to any of the destinations, the code prints an appropriate message.
Time Scheduling and Sound:

The code includes logic to play a sound and print the shortest distance if it is less than or equal to 750.
The code uses the winsound library to play a beep sound with a frequency of 2000 Hz and a duration of 5000 ms.
The function successfully implements Dijkstra's algorithm and calculates the shortest path from a source node to a set of destination nodes. The code uses the shortest distance to decide whether to play a sound and print the distance.

'''
import sys
from heapq import heappop, heappush
from colorama import Fore
import winsound

def dijkstra_1(graph, source, destination):
  """
  Dijkstra's shortest path algorithm.

  Args:
    graph: A dictionary of dictionaries, where each key is a node and the value is a dictionary
      of neighbors and their distances.
    source: The starting node.
    destination: The ending node.

  Returns:
    A tuple of the shortest distance and the shortest path.
  """
  inf = sys.maxsize
  node_data = {}
  for node in graph:
    node_data[node] = {'cost': inf, 'predecessors': set()}
  node_data[source]['cost'] = 0
  min_heap = [(0, source)]
  while min_heap:
    cost, temp = heappop(min_heap)
    if temp == destination:
      break
    for neighbor, distance in graph[temp].items():
      new_cost = cost + distance
      if new_cost < node_data[neighbor]['cost']:
        node_data[neighbor]['cost'] = new_cost
        node_data[neighbor]['predecessors'] = {temp}
        heappush(min_heap, (new_cost, neighbor))
  if node_data[destination]['cost'] == inf:
    return None, None
  path = []
  current = destination
  while current != source:
    path.append(current)
    current = next(iter(node_data[current]['predecessors']))
  path.append(source)
  path.reverse()
  return node_data[destination]['cost'], path

# Example usage
graph = {
  'am': {'d': 130 },
  'a': {'b': 180, 'c': 150, 'h': 120 },
  'b': {'d': 130, 'c': 170, 'a': 180 }, 
  'c': {'b': 170, 'd': 170, 'e': 90, 'f': 140, 'a': 150 },
  'd': {'am': 130, 'e': 150, 'c': 170, 'b': 170 },
  'e': {'d': 150, 'c': 90, 'f': 120 },
  'f': {'c': 140, 'e': 120, 'h': 140 },
  'h': {'f': 140, 'a': 120 }
}

source = 'am'
destination = 'h'

shortest_distance, shortest_path = dijkstra_1(graph, source, destination)

if shortest_distance is None:
  print("No path found from", source, "to", destination)
else:
  print("Shortest distance from", source, "to", destination, "is", shortest_distance)
  print("Shortest path:", "->".join(shortest_path))

def dijkstra_2(graph, source, destination, avoid_nodes):
    """
    Dijkstra's shortest path algorithm with the option to avoid specific nodes.

    Args:
      graph: A dictionary of dictionaries, where each key is a node and the value is a dictionary
        of neighbors and their distances.
      source: The starting node.
      destination: The ending node.
      avoid_nodes: A set of nodes to avoid in the path.

    Returns:
      A tuple of the shortest distance and the shortest path.
    """
    inf = sys.maxsize
    node_data = {}
    for node in graph:
        node_data[node] = {'cost': inf, 'predecessors': set()}
    node_data[source]['cost'] = 0
    min_heap = [(0, source)]
    while min_heap:
        cost, temp = heappop(min_heap)
        if temp == destination:
            break
        for neighbor, distance in graph[temp].items():
            new_cost = cost + distance
            if new_cost < node_data[neighbor]['cost'] and neighbor not in avoid_nodes:
                node_data[neighbor]['cost'] = new_cost
                node_data[neighbor]['predecessors'] = {temp}
                heappush(min_heap, (new_cost, neighbor))
    if node_data[destination]['cost'] == inf:
        return None, None
    path = []
    current = destination
    while current != source:
        path.append(current)
        current = next(iter(node_data[current]['predecessors']))
    path.append(source)
    path.reverse()
    return node_data[destination]['cost'], path

# Example usage for two vehicles
graph = {
  'am': {'d': 130 },
  'a': {'b': 180, 'c': 150, 'h': 120 },
  'b': {'d': 130, 'c': 170, 'a': 180 }, 
  'c': {'b': 170, 'd': 170, 'e': 90, 'f': 140, 'a': 150 },
  'd': {'am': 130, 'e': 150, 'c': 170, 'b': 170 },
  'e': {'d': 150, 'c': 90, 'f': 120 },
  'f': {'c': 140, 'e': 120, 'h': 140 },
  'h': {'f': 140, 'a': 120 },
  'x': {'e': 50, 'd': 100 }
}

source_vehicle1 = 'a'
source_vehicle2 = 'b'
destination = 'h'
avoid_nodes_vehicle1 = {'b'}
avoid_nodes_vehicle2 = {'a'}

shortest_distance_vehicle1, shortest_path_vehicle1 = dijkstra_2(graph, source_vehicle1, destination, avoid_nodes_vehicle1)
shortest_distance_vehicle2, shortest_path_vehicle2 = dijkstra_2(graph, source_vehicle2, destination, avoid_nodes_vehicle2)

if shortest_distance_vehicle1 is None or shortest_distance_vehicle2 is None:
    print("No valid paths found for both vehicles.")
else:
    print("Shortest distance for Vehicle 1:", shortest_distance_vehicle1)
    print("Shortest path for Vehicle 1:", "->".join(shortest_path_vehicle1))
    print("Shortest distance for Vehicle 2:", shortest_distance_vehicle2)
    print("Shortest path for Vehicle 2:", "->".join(shortest_path_vehicle2))
'''
##Function dijkstra_1:

This function implements Dijkstra's algorithm to find the shortest path and distance between a source node and a destination node in a graph.
The function uses a priority queue (min_heap) to process nodes based on the cost (shortest distance) from the source node.
If the destination node is reached, the function terminates the search.
Once the shortest path is found, the function constructs the path from the destination to the source using the predecessors' data.
The function returns the shortest distance and the shortest path.
Function dijkstra_2:

This function is similar to dijkstra_1, but with an additional option to avoid specific nodes during the pathfinding process.
The function accepts an additional argument avoid_nodes, which is a set of nodes that should be avoided in the path.
The function follows the same approach as dijkstra_1 but includes a check to avoid processing neighbors that are in the avoid_nodes set.
The function returns the shortest distance and the shortest path that avoids the specified nodes.
Example Usage:

In the example usage for dijkstra_1, the function finds the shortest distance and path from the source node 'am' to the destination node 'h'.
In the example usage for dijkstra_2, the function finds the shortest paths and distances for two vehicles, with each vehicle avoiding specific nodes in the graph.
The code checks if valid paths were found for both vehicles, and if so, it prints the shortest distance and path for each vehicle.
