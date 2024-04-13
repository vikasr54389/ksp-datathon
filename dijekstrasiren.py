import sys
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

