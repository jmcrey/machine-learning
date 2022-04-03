from queue import LifoQueue

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Set, Union
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import os
import csv
import time
import heapq

from geopy.distance import distance
from typing_extensions import Self

# `scipy` is *only* used to store the large and sparse adjacency matrix. 
# None of the scientific mechanisms are used with it!
from scipy.sparse import lil_matrix
from scipy.sparse._csr import csr_matrix
from scipy.sparse._coo import coo_matrix
import numpy as np
import pandas as pd


class DuplicateNodeError(Exception):
    pass


@dataclass(eq=False)
class Node:
    node_id: int
    weight: float
    heuristic: Optional[float] = None
    parent_node: Optional[Self] = None

    def __str__(self) -> str:
        return f'Node({self.node_id})'

    def __eq__(self, __o: object) -> bool:
        """ Two Nodes are equal if their IDs match. Parent Node IDs and Weights are ignored. """
        if not isinstance(__o, Node):
            raise TypeError(f"Cannot compare 'Node' object to {__o.__class__.__name__}")
        return self.node_id == __o.node_id
    
    def __ne__(self, __o: object) -> bool:
        """ Two Nodes are not equal if their IDs do not match. Parent Node IDs and Weights are ignored. """
        if not isinstance(__o, Node):
            raise TypeError(f"Cannot compare 'Node' object to {__o.__class__.__name__}")
        return self.node_id != __o.node_id

    def __add__(self, __o: object) -> Self:
        """ Add the number to the weight and return a new Node object. """
        if not isinstance(__o, float) and not isinstance(__o, int):
            raise TypeError(f"Cannot add 'Node' with {__o.__class__.__name__} object. Must be 'float' or 'int'")
        weight = self.weight + __o
        if weight < 0:
            raise ValueError("A 'Node' object's weight cannot be negative")
        return Node(self.node_id, weight, self.heuristic, self.parent_node)

    def __sub__(self, __o: object) -> Self:
        """ Subtract the number to the weight and return a new Node object. """
        if not isinstance(__o, float) and not isinstance(__o, int):
            raise TypeError(f"Cannot add 'Node' with {__o.__class__.__name__} object")
        weight = self.weight - __o
        if weight < 0:
            raise ValueError("A 'Node' object's weight cannot be negative")
        return Node(self.node_id, weight, self.heuristic, self.parent_node)

    def __lt__(self, __o: object) -> bool:
        """ Compare the weight of 'self' and the given Node. Return True if the weight of 'self' is less than the given Node. """
        if not isinstance(__o, Node):
            raise TypeError(f"Cannot compare 'Node' object to {__o.__class__.__name__}")
        x = self.weight + self.heuristic if self.heuristic else self.weight
        y = __o.weight + __o.heuristic if __o.heuristic else __o.weight
        return x < y

    def __le__(self, __o: object) -> bool:
        """ Compare the weight of 'self' and the given Node. Return True if the weight of 'self' is less than or equal to the 
            given Node. """
        if not isinstance(__o, Node):
            raise TypeError(f"Cannot compare 'Node' object to {__o.__class__.__name__}")
        x = self.weight + self.heuristic if self.heuristic else self.weight
        y = __o.weight + __o.heuristic if __o.heuristic else __o.weight
        return x <= y
    
    def __gt__(self, __o: object) -> bool:
        """ Compare the weight of 'self' and the given Node. Return True if the weight of 'self' is greater than the given Node. """
        if not isinstance(__o, Node):
            raise TypeError(f"Cannot compare 'Node' object to {__o.__class__.__name__}")
        x = self.weight + self.heuristic if self.heuristic else self.weight
        y = __o.weight + __o.heuristic if __o.heuristic else __o.weight
        return x > y
    
    def __ge__(self, __o: object) -> bool:
        """ Compare the weight of 'self' and the given Node. Return True if the weight of 'self' is greater than or equal
            to the given Node. """
        if not isinstance(__o, Node):
            raise TypeError(f"Cannot compare 'Node' object to {__o.__class__.__name__}")
        x = self.weight + self.heuristic if self.heuristic else self.weight
        y = __o.weight + __o.heuristic if __o.heuristic else __o.weight
        return x >= y


class AdjacencyMatrix:

    REQUIRED_COLS = {'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'trip_distance'}

    def __init__(self) -> None:
        self._coordinate_map: Dict[Tuple[float, float], int] = dict()
        self._inverse_map: Dict[int, Tuple[float, float]] = dict()
        self._matrix: Optional[csr_matrix] = None

    def exists(self, node: Union[int, Node]) -> bool:
        """ Checks whether the Node exists in the graph """
        if isinstance(node, Node):
            id_ = node.node_id
        else:
            id_ = node

        return bool(self._inverse_map.get(id_))
    
    def get_location(self, node: Union[int, Node]) -> Optional[Tuple[float, float]]:
        """ Returns the original weight for a given Node """
        if isinstance(node, Node):
            id_ = node.node_id
        else:
            id_ = node

        return self._inverse_map.get(id_)

    def get_node(self, node_id: int, weight: float, heuristic: Callable = None, parent_node: Node = None, **kwargs) -> Node:
        """ Builds a node from the given arguments.
        
            Parameters:
                node_id (int): The ID of the node
                weight (float): The weight of the node
                heuristic (Callable): A function to compute the heuristic of the Node. Must accept a `node_id` argument.
                                      Any other args can be passed using the kwargs argument.
                parent_node (Node): A parent Node
                **kwargs: Any arguments needed by the heuristic function
            
            Returns:
                Node(node_id, weight, heuristic, parent_node)
        """
        h = None
        if heuristic:
            h = heuristic(node_id=node_id, **kwargs)
            if not isinstance(h, float) and not h is None:
                raise ValueError(f"'heuristic' must return a 'float' or 'None' value, not '{h.__class__.__name__}'")

        return Node(
            node_id=node_id,
            weight=weight,
            heuristic=h,
            parent_node=parent_node
        )

    def get_children(self, node: Node, heuristic: Callable = None, **kwargs) -> List[Node]:
        """ Returns all the weights and Node IDs of the non-zero children of the given Node ID.
        
            Parameters:
                nodeid (int): The node that you are interested in
            
            Returns:
                List[Tuple[int, float]]: A list of (Weight, Node ID)
            
            Example:
                _matrix = [
                    [0, 1, 0],
                    [0, 3, 2]
                ]
                >>> adjacency.get(Node(1, 0, None, None))
                [(3, 1), (2, 2)]
        """
        if self._matrix is None:
            raise ValueError("Must call 'build()' before this function")

        _, node_ids = self._matrix[node.node_id].nonzero()
        children: List[Node] = [
            self.get_node(node_id, self._matrix[node.node_id, node_id], heuristic=heuristic, parent_node=node, **kwargs) 
            for node_id in node_ids
        ]
        return children

    def build(self, df: pd.DataFrame) -> None:
        """ Builds the undirected graph in the form of an adjaceny matrix 
        
                1) Rounds the pickup and dropoff latitude and longitude to the nearest 4th decimal place
                2) Assumes that the graph is undirected (i.e. a trip from A to B means that a trip from B to A is possible)
                3) Pairs the pickup latitude and longitude as the Axis 0, and drop off latitude and longitude as the Axis 1
                4) Any non-zero entry in the matrix is a possible trip
                5) The value is the trip distance from A to B (or from B to A)
        """
        self._validate(df)
        coordinate_cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
        df.loc[:, coordinate_cols] = df[coordinate_cols].round(4)
        df = df[list(self.REQUIRED_COLS)]
        self._build_map(df)
        self._build_matrix(df)

    def _validate(self, df: pd.DataFrame) -> None:
        """ Validates that the CSV has all the needed columns """
        if self.REQUIRED_COLS - set(df.columns):
            raise ValueError(f"Missing required columns: {', '.join(self.REQUIRED_COLS - set(df.columns))}")

    def _build_matrix(self, df: pd.DataFrame) -> None:
        """ Builds the adjancency matrix, given starting x and y """
        if not self._coordinate_map:
            raise ValueError("Must call '_build_map()' before this function")

        n = len(self._coordinate_map.keys())
        # n x n adjacency matrix where any 0 means that the route is not possible
        matrix = lil_matrix((n, n), dtype=np.float64)
        for _, row in df.iterrows():
            x = (row['pickup_latitude'], row['pickup_longitude'])
            y = (row['dropoff_latitude'], row['dropoff_longitude'])
            w = row['trip_distance']

            # Get the Node ID from the map
            start_id = self._coordinate_map[x]
            dest_id = self._coordinate_map[y]

            # Check for trip collision
            # If there is a trip collision, pick the minimum weight as the trip weight
            if matrix[start_id, dest_id] != 0:
                w = min([w, matrix[start_id, dest_id], matrix[dest_id, start_id]])

            # Assign (x, y) and (y, x) to be the weight for the single trip
            # Both are assigned because we assume that a trip is possible from both x -> y and y -> x
            matrix[start_id, dest_id] = w
            matrix[dest_id, start_id] = w

        self._matrix = matrix.tocsr()

    def _build_map(self, df: pd.DataFrame) -> Dict[Tuple[float, float], int]:
        """ Creates a coordinate map given a list of start and end coordinates.
        
            Parameters:
                df (pd.DataFrame): The DataFrame containing the pickup latitude and longitude, and the dropoff latitude and longitude

            Sets:
                _coordinate_map: dict = {
                    (lat, long): 0,
                    ...
                }
                _inverse_map: dict = {
                    0: (lat, long),
                    ...
                }
        """
        # Get the unique coordinates
        unique_coords = set()
        unique_coords.update(zip(df['pickup_latitude'].tolist(), df['pickup_longitude'].tolist()))
        unique_coords.update(zip(df['dropoff_latitude'].tolist(), df['dropoff_longitude'].tolist()))

        # Sort the values for reproducability
        sorted_coords = sorted(unique_coords)
        self._coordinate_map = dict(zip(sorted_coords, range(len(sorted_coords))))
        self._inverse_map = {v: k for k, v in self._coordinate_map.items()}

    def save(self, path: str, node_file: str = 'nodes.csv', edge_file: str = 'edges.csv') -> None:
        """ Writes the nodes and the edges to a CSV file """
        if not self._coordinate_map or self._matrix is None:
            raise ValueError("Must call 'build()' before calling this function")

        if not os.path.isdir(path):
            raise ValueError("'path' must be a directory")

        node_path = os.path.join(path, node_file)
        edge_path = os.path.join(path, edge_file)
        self._save_nodes(node_path)
        self._save_edges(edge_path, self._matrix.tocoo())

    def _save_nodes(self, path: str) -> None:
        """ Writes the nodes to a CSV file """
        with open(path, 'w+', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['nodeid', 'lat', 'long'])
            writer.writeheader()
            for k, v in self._inverse_map.items():
                writer.writerow(dict(nodeid=k, lat=v[0], long=v[1]))

    def _save_edges(self, path: str, matrix: coo_matrix) -> None:
        """ Writes all the edges to a CSV file """
        with open(path, 'w+', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['nodeid1', 'nodeid2'])
            writer.writeheader()
            for i, j, v in zip(matrix.row, matrix.col, matrix.data):
                if v:
                    writer.writerow(dict(nodeid1=i, nodeid2=j))


class PriorityQueue:

    def __init__(self) -> None:
        self.queue = []
        heapq.heapify(self.queue)

    def push(self, node: Node) -> None:
        """ Pushes a new Node onto the priority queue """
        if node in self.queue:
            raise ValueError("'PriorityQueue' cannot have duplicate entries")
        heapq.heappush(self.queue, node)

    def pop(self) -> Node:
        """ Pops the next item in the queue and returns the Node """
        return heapq.heappop(self.queue)
    
    def get(self, index: int) -> Node:
        """ Returns the node at the given index. Raises IndexError if the index is out of range. """
        return self.queue[index]
    
    def empty(self) -> bool:
        """ Returns whether the queue is empty or not """
        return not self.queue

    def find(self, node: Node) -> Tuple[int, Optional[Node]]:
        """ Finds the node in the queue and returns its index and the node. Otherwise, will return -1 and None. """
        try:
            idx = self.queue.index(node)
            return idx, self.get(idx)
        except ValueError:
            return -1, None

    def replace(self, index: int, node: Node) -> None:
        """ Replaces the node at the index with the new Node by removing the old node and pushing the new one into
            the Priority Queue. """
        del self.queue[index]
        heapq.heappush(self.queue, node)


class UniformCostSearch:

    def __init__(self, graph: AdjacencyMatrix) -> None:
        self.graph = graph
        self.explored: Set[int] = set()
        self.queue = PriorityQueue()

    def backtrack(self, current_node: Node) -> LifoQueue:
        """ Backtracks the path from current node to the first parent node """
        stack = LifoQueue()
        while current_node.parent_node is not None:
            current_node -= current_node.parent_node.weight
            stack.put_nowait(current_node)
            current_node = current_node.parent_node
        stack.put_nowait(current_node)
        return stack

    def search(self, start_node_id: int, goal_node_id: int) -> Optional[LifoQueue]:
        """ Searches the graph for the best path from the given start node to the given destination node """
        if start_node_id == goal_node_id:
            return self.backtrack(self.graph.get_node(start_node_id, 0))

        if not self.graph.exists(start_node_id):
            raise ValueError(f"The Start Node {start_node_id} does not exist")
        
        if not self.graph.exists(goal_node_id):
            raise ValueError(f"The Goal Node {goal_node_id} does not exist")
        
        return self._search_graph(start_node_id, goal_node_id)

    def _search_graph(self, start_node_id: int, goal_node_id: int) -> Optional[LifoQueue]:
        """ Implements the UCS search function using the parent's weight as the heuristic """
        node = self.graph.get_node(start_node_id, 0)
        
        self.queue.push(node)  # Push the start node into the queue

        while node.node_id != goal_node_id and not self.queue.empty():
            node = self.queue.pop()
            self.explore(node)

        if node.node_id == goal_node_id:
            return self.backtrack(node)
        else:
            return None

    def explore(self, node: Node, **kwargs) -> None:
        """ Explore the given node by getting all its children and adding them to the priority queue """
        children = self.graph.get_children(node, self.heuristic, **kwargs)
        if children:
            self.add(children)
        self.explored.add(node.node_id)

    def add(self, children: List[Node]) -> None:
        """ Adds the children to the queue, or updates the weight if it is already in the queue """
        for child in children:

            # Ignore any child that has already been searched
            if child.node_id in self.explored:
                continue

            if child.parent_node:
                child += child.parent_node.weight  # Keep a running total of the weight
            
            self.push(child)

    def push(self, child: Node) -> None:
        """ Pushes the child node to the frontier """
        idx, current = self.queue.find(child)
        if current and current <= child:
            return  # Do nothing, the child is in the queue with less or equal weight
        elif current and current > child:
            self.queue.replace(idx, child)  # Delete the current item in the queue and replace with the new node
        else:
            self.queue.push(child)

    def heuristic(self, node_id: int, **kwargs) -> float:
        """ Defines the heuristic for the model """
        pass


class AStar(UniformCostSearch):

    """ For this A* Algorithm, I use the formula: Cost(Next) = Total_Cost(Start, Next) + H(Next).

        This formula is consistent with Wikipedia's description, and seems to be an equally valid approach to
        A*. It is also more consistent with the Uniform Cost Search.

        Wikipedia: https://en.wikipedia.org/wiki/A*_search_algorithm
    """

    def __init__(self, graph: AdjacencyMatrix) -> None:
        super().__init__(graph)

    def _search_graph(self, start_node_id: int, goal_node_id: int) -> Optional[LifoQueue]:
        """ Implements the UCS search function using the parent's weight as the heuristic """
        goal_location = self.graph.get_location(goal_node_id)  # Get the goal node's location from the graph
        node = self.graph.get_node(start_node_id, 0)

        self.queue.push(node)  # Push the parent node into the queue

        while node.node_id != goal_node_id and not self.queue.empty():
            node = self.queue.pop()
            self.explore(node, goal_location=goal_location)

        if node.node_id == goal_node_id:
            return self.backtrack(node)
        else:
            return None

    def heuristic(self, node_id: int, **kwargs) -> float:
        """ Defines the heuristic for the search algorithm. In this case, it is the absolute distance from the
            node to the goal state. The distance is computed using geopy.

            Note that this can be change to use the algorithm discussed in class: Cost(Next) = Cost(Next) + H(Next) - H(Parent).
            If desired, the **kwargs must include the parent's heuristic. You can see the comments below the return statment
            for an example.
        """
        goal_location = kwargs['goal_location']
        node_location = self.graph.get_location(node_id)
        # Get the absolute distance between the two locations
        node_distance = np.absolute(distance(node_location, goal_location).miles)
        return node_distance
        # H(Next) - H(Node)
        # parent_heuristic = kwargs.get('parent_heuristic', 0.0)
        # return node_distance - parent_heuristic



def _parse_args() -> Namespace:
    """ Parses the command line arguments """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter ,description=(
        'Creates an adjacency matrix based on Taxi data, and calculates and reports the shortest '
        'path to a given destination using the given algorithm.'
        )
    )
    parser.add_argument('-p', '--path', required=False, dest='path', default='nyc_taxi_data.csv',
                        help='The path to the Taxi data')
    parser.add_argument('-s', '--start', dest='start', type=int, required=False, default=0, help='The ID of the start node')
    parser.add_argument('-d', '--dest', dest='destination', type=int, required=False, default=1, help='The ID of the destination node')
    parser.add_argument('-a', '--alg', dest='algorithm', required=False, default='UCS', choices=['UCS', 'AStar'],
                        help='The algorithm to use to calculate the path. Either Uniform Cost Search (UCS) or A* (a*)')
    parser.add_argument('--save', dest='save_state', required=False, default=False, action='store_true',
                        help='Write the graph to disk. Must specify \'--output\' if set')
    parser.add_argument('-o', '--output', required=False, dest='output', help='The directory to output the graphs nodes and edges.')
    return parser.parse_args()


def _validate_args(args: Namespace) -> None:
    """ Validates the given arguments """
    if not os.path.exists(args.path):
        raise ValueError(f"Path to data must exist. {args.path} path does not exist!")

    if not os.path.isfile(args.path) or not args.path.endswith('csv'):
        raise ValueError("Path must be a CSV file. Path is not a file!")

    if args.save_state and not (args.output and os.path.exists(args.output)):
        raise ValueError(f"Output path must be a directory and exist. {args.output} does not exist!")
    
    if args.save_state and not os.path.isdir(args.output):
        raise ValueError(f"Output must be a directory. Output path is not a directory!")


def search(adjacency_matrix: AdjacencyMatrix, algorithm: str, start_node_id: int, goal_node_id: int) -> Optional[LifoQueue]:
    """ Runs the Search Algorithm and returns the results """
    if algorithm == 'UCS':
        alg = UniformCostSearch(adjacency_matrix)
    else:
        alg = AStar(adjacency_matrix)
    
    return alg.search(start_node_id, goal_node_id)


def report(time_taken: int, path: LifoQueue = None) -> None:
    """ Reports the path from the start node to the goal node, as well as the time taken to run the algorithm """
    if not path:
        print("Path: No path found!")
        print("Total Cost of Path: N/A")
    else:
        nodes: List[Node] = []
        while not path.empty():
            nodes.append(path.get_nowait())

        output = 'Path: '
        total_weight = 0.0
        for i, node in enumerate(nodes):
            if i == 0:
                output += f"{str(node)}"
            else:
                output += f' --{np.round(node.weight, 4)}--> {str(node)}'
                total_weight += node.weight
        print(output)
        print(f"Total Cost of Path: {np.round(total_weight, 4)}")
    print(f"Total Time Taken: {np.round(time_taken, 4)}s")


def main():
    args = _parse_args()
    _validate_args(args)

    build_start_time = time.time()
    trip_data = pd.read_csv(args.path)
    adjacency = AdjacencyMatrix()
    adjacency.build(trip_data)

    if args.save_state:
        adjacency.save(args.output)

    build_end_time = time.time() - build_start_time
    print(f"Graph Build Time: {build_end_time}s")

    search_start_time = time.time()
    path = search(adjacency, args.algorithm, args.start, args.destination)
    search_end_time = time.time() - search_start_time
    report(search_end_time, path)



if __name__ == '__main__':
    main()