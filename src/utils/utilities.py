import numpy as np
import sys
import copy

# Calculate the distance between two data points
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))
# print(euclidean_distance(np.array([1, 2, 3]), np.array([4, 5, 6])))

# A class that represents the nodes
class Node:

    # Constructor to initialize the node
    def __init__(self, nodeInfo):
        self.patientID = nodeInfo[0]
        self.data = nodeInfo[1:]

    # Method to evaluate the equality of two nodes
    def __eq__(self, other):
        return self.patientID == other.patientID

    # Method to return the hash value of the node
    def __hash__(self):
        return hash(self.patientID)

    # Method to return the string representation of the node
    def __repr__(self):
        return f"Patient {self.patientID}"
    
# A class that represents the graph
class Graph:

    # Constructor to initialize the graph
    def __init__(self, nodes):
        self.nodes = [Node(nodeInfo) for nodeInfo in nodes]
        self.adjacency_matrix = None
        self.adjacency_list = {}
        for node in self.nodes:
            self.adjacency_list[node] = []
        self.visited = set()
        self.queue = []

    # Method to compute the adjacency matrix of the graph
    def compute_adjacency_matrix(self):
        self.adjacency_matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                self.adjacency_matrix[i][j] = euclidean_distance(self.nodes[i].data, self.nodes[j].data)

    # Method to add a node to the graph
    def add_node(self, node):
        self.nodes.append(node)
        self.adjacency_list[node] = []

    # Method to add an edge between two nodes
    def add_edge(self, node1, node2):
        self.adjacency_list[node1].append(node2)
        self.adjacency_list[node2].append(node1)

    # Method to remove an edge between two nodes
    def remove_edge(self, node1, node2):
        self.adjacency_list[node1].remove(node2)
        self.adjacency_list[node2].remove(node1)

    # Method to return the edges of the graph
    def get_edges(self):
        edges = []
        for node in self.adjacency_list:
            for neighbour in self.adjacency_list[node]:
                if {node, neighbour} not in edges:
                    edges.append({node, neighbour})
        return edges

    def get_sorted_edges(self, reversed=False):
        return sorted(self.get_edges(), key=lambda edge: self.adjacency_matrix[self.nodes.index(list(edge)[0])][self.nodes.index(list(edge)[1])], reverse=reversed)

    # Method to return the neighbours of a node
    def get_neighbours(self, node):
        return self.adjacency_list[node]

    # Method to return the MST of the graph using Prim's algorithm
    def get_mst(self):
        self.visited = set()
        mst = Graph([])
        mst.add_node(self.nodes[0])
        self.visited.add(self.nodes[0])
        while len(mst.nodes) != len(self.nodes):
            min_distance = sys.maxsize
            min_node = None
            parent_node = None
            for node in self.visited:
                for neighbour in self.get_neighbours(node):
                    if neighbour not in self.visited:
                        distance = self.adjacency_matrix[self.nodes.index(node)][self.nodes.index(neighbour)]
                        if distance < min_distance:
                            min_distance = distance
                            min_node = neighbour
                            parent_node = node
            mst.add_node(min_node)
            mst.add_edge(min_node, parent_node)
            self.visited.add(min_node)
        mst.compute_adjacency_matrix()
        return mst

    # Method to remove MST edges from Graph
    def remove_mst_edges(self, mst):
        for edge in mst.get_edges():
            self.remove_edge(list(edge)[0], list(edge)[1])
    
    # Method to remove k inconcisitent edges in mst
    def remove_k_inconsistent_edges(self, mst, k, type):
        current_mst = copy.deepcopy(mst)
        inconsistent_edges = []
        sorted_edges = current_mst.get_sorted_edges(reversed=True)
        # Inconsistent edges are the k edges with the highest weight
        if type == "LONGEST_EDGE":
            inconsistent_edges = sorted_edges[:k]
        # Inconsistent edges are the edges which upon removal reduces standard deviation of the graph
        elif type == "MAX_STD_DEV":
            max_std_dev = current_mst.get_std_dev()
            for edge in sorted_edges:
                current_mst.remove_edge(list(edge)[0], list(edge)[1])
                std_dev = current_mst.get_std_dev()
                if std_dev < max_std_dev:
                    max_std_dev = std_dev
                    inconsistent_edges.append(edge)    
        # Remove the inconsistent edges
        for edge in inconsistent_edges:
            mst.remove_edge(list(edge)[0], list(edge)[1])

    # Method to return the standard deviation of the graph
    def get_std_dev(self):
        distances = []
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if i != j:
                    distances.append(self.adjacency_matrix[i][j])
        return np.std(distances)
    