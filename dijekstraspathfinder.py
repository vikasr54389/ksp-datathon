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
