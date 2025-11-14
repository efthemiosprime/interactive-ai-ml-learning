import React, { useState, useEffect } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const codeExamples = {
  astar: `# A* Pathfinding Algorithm - Complete Implementation
import heapq

def astar(maze, start, end):
    """
    A* Algorithm Formula: f(n) = g(n) + h(n)
    
    where:
    - g(n) = cost from start to node n
    - h(n) = heuristic estimate from n to goal
    - f(n) = estimated total cost
    """
    
    def heuristic(a, b):
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(node):
        """Get valid neighbors (Graph Theory: edges)"""
        row, col = node
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < len(maze) and 0 <= c < len(maze[0]):
                if maze[r][c] == 0:  # Not a wall
                    neighbors.append((r, c))
        return neighbors
    
    # Initialize: Priority queue with f-score
    open_set = [(0, 0, start, [start])]  # (f_score, g_score, node, path)
    visited = set()
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    
    while open_set:
        # Get node with lowest f-score
        current_f, current_g, current, path = heapq.heappop(open_set)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        # Goal reached!
        if current == end:
            return path
        
        # Explore neighbors
        for neighbor in get_neighbors(current):
            if neighbor in visited:
                continue
            
            # Calculate g-score (cost from start)
            tentative_g = current_g + 1
            
            # Calculate f-score (g + heuristic)
            tentative_f = tentative_g + heuristic(neighbor, end)
            
            # Update if better path found
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_f
                heapq.heappush(open_set, (tentative_f, tentative_g, neighbor, path + [neighbor]))
    
    return []  # No path found

# Example usage
maze = [
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
end = (4, 4)

path = astar(maze, start, end)
print(f"Path found: {path}")`,

  bfs: `# Breadth-First Search (BFS)
from collections import deque

def bfs(maze, start, end):
    """
    BFS explores all nodes at current depth before moving deeper.
    Guarantees shortest path in unweighted graphs.
    
    Graph Theory: Level-order traversal
    """
    
    def get_neighbors(node):
        row, col = node
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < len(maze) and 0 <= c < len(maze[0]):
                if maze[r][c] == 0:
                    neighbors.append((r, c))
        return neighbors
    
    # Queue: FIFO (First In, First Out)
    queue = deque([(start, [start])])
    visited = set([start])
    
    while queue:
        current, path = queue.popleft()
        
        if current == end:
            return path
        
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return []

# Time Complexity: O(V + E) where V = vertices, E = edges
# Space Complexity: O(V)`,

  dfs: `# Depth-First Search (DFS)
def dfs(maze, start, end):
    """
    DFS explores as far as possible before backtracking.
    Uses stack (LIFO) instead of queue.
    
    Graph Theory: Depth-first traversal
    """
    
    def get_neighbors(node):
        row, col = node
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < len(maze) and 0 <= c < len(maze[0]):
                if maze[r][c] == 0:
                    neighbors.append((r, c))
        return neighbors
    
    # Stack: LIFO (Last In, First Out)
    stack = [(start, [start])]
    visited = set([start])
    
    while stack:
        current, path = stack.pop()
        
        if current == end:
            return path
        
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))
    
    return []

# Time Complexity: O(V + E)
# Space Complexity: O(V)`,

  dijkstra: `# Dijkstra's Algorithm
import heapq

def dijkstra(maze, start, end):
    """
    Dijkstra's finds shortest path in weighted graphs.
    Similar to A* but without heuristic.
    
    Formula: dist[v] = min(dist[v], dist[u] + weight(u, v))
    """
    
    def get_neighbors(node):
        row, col = node
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < len(maze) and 0 <= c < len(maze[0]):
                if maze[r][c] == 0:
                    neighbors.append((r, c))
        return neighbors
    
    # Initialize distances
    distances = {start: 0}
    previous = {}
    unvisited = set()
    
    # Add all nodes to unvisited
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 0:
                node = (i, j)
                if node != start:
                    distances[node] = float('inf')
                unvisited.add(node)
    
    while unvisited:
        # Find unvisited node with minimum distance
        current = min(unvisited, key=lambda n: distances.get(n, float('inf')))
        
        if distances.get(current, float('inf')) == float('inf'):
            break
        
        unvisited.remove(current)
        
        if current == end:
            # Reconstruct path
            path = []
            node = end
            while node:
                path.append(node)
                node = previous.get(node)
            return path[::-1]
        
        # Update distances to neighbors
        for neighbor in get_neighbors(current):
            if neighbor in unvisited:
                alt = distances[current] + 1  # Weight = 1 for unweighted
                if alt < distances.get(neighbor, float('inf')):
                    distances[neighbor] = alt
                    previous[neighbor] = current
    
    return []

# Time Complexity: O((V + E) log V) with priority queue
# Space Complexity: O(V)`
};

export default function MazeSolver() {
  const [gridSize, setGridSize] = useState(8);
  const [algorithm, setAlgorithm] = useState('astar');
  const [maze, setMaze] = useState([]);
  const [path, setPath] = useState([]);
  const [visited, setVisited] = useState([]);
  const [isAnimating, setIsAnimating] = useState(false);
  const [start, setStart] = useState([0, 0]);
  const [end, setEnd] = useState([7, 7]);
  const [selectedStep, setSelectedStep] = useState('overview');

  // Initialize maze
  useEffect(() => {
    if (gridSize > 0) {
      initializeMaze();
    }
  }, [gridSize]);

  const initializeMaze = () => {
    if (gridSize <= 0) return;
    const newMaze = Array(gridSize).fill(null).map(() => Array(gridSize).fill(0));
    // Add some walls randomly (0 = empty, 1 = wall)
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        if (Math.random() < 0.25 && (i !== 0 || j !== 0) && (i !== gridSize - 1 || j !== gridSize - 1)) {
          newMaze[i][j] = 1;
        }
      }
    }
    setMaze(newMaze);
    setPath([]);
    setVisited([]);
    setStart([0, 0]);
    setEnd([gridSize - 1, gridSize - 1]);
  };

  // Heuristic function (Manhattan distance)
  const heuristic = (a, b) => {
    return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
  };

  // Get neighbors
  const getNeighbors = (node) => {
    const [row, col] = node;
    const neighbors = [];
    const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // Up, Down, Left, Right

    for (const [dr, dc] of directions) {
      const newRow = row + dr;
      const newCol = col + dc;
      if (
        newRow >= 0 && newRow < gridSize &&
        newCol >= 0 && newCol < gridSize &&
        maze[newRow][newCol] === 0
      ) {
        neighbors.push([newRow, newCol]);
      }
    }
    return neighbors;
  };

  // A* Algorithm
  const astar = () => {
    const openSet = [start];
    const cameFrom = new Map();
    const gScore = new Map();
    const fScore = new Map();
    const visitedSet = new Set();
    const openSetSet = new Set([start.toString()]);

    gScore.set(start.toString(), 0);
    fScore.set(start.toString(), heuristic(start, end));

    const visitedOrder = [];

    while (openSet.length > 0) {
      openSet.sort((a, b) => {
        const fA = fScore.get(a.toString()) || Infinity;
        const fB = fScore.get(b.toString()) || Infinity;
        return fA - fB;
      });

      const current = openSet.shift();
      openSetSet.delete(current.toString());
      
      if (visitedSet.has(current.toString())) continue;
      
      visitedSet.add(current.toString());
      visitedOrder.push(current);

      if (current[0] === end[0] && current[1] === end[1]) {
        const path = [];
        let node = current;
        while (node) {
          path.unshift(node);
          node = cameFrom.get(node.toString());
        }
        return { path, visited: visitedOrder };
      }

      for (const neighbor of getNeighbors(current)) {
        if (visitedSet.has(neighbor.toString())) continue;

        const tentativeGScore = (gScore.get(current.toString()) || 0) + 1;

        if (!gScore.has(neighbor.toString()) || tentativeGScore < gScore.get(neighbor.toString())) {
          cameFrom.set(neighbor.toString(), current);
          gScore.set(neighbor.toString(), tentativeGScore);
          fScore.set(neighbor.toString(), tentativeGScore + heuristic(neighbor, end));

          if (!openSetSet.has(neighbor.toString())) {
            openSet.push(neighbor);
            openSetSet.add(neighbor.toString());
          }
        }
      }
    }

    return { path: [], visited: visitedOrder };
  };

  // BFS Algorithm
  const bfs = () => {
    const queue = [[start]];
    const visitedSet = new Set([start.toString()]);
    const visitedOrder = [];

    while (queue.length > 0) {
      const path = queue.shift();
      const current = path[path.length - 1];
      visitedOrder.push(current);

      if (current[0] === end[0] && current[1] === end[1]) {
        return { path, visited: visitedOrder };
      }

      for (const neighbor of getNeighbors(current)) {
        if (!visitedSet.has(neighbor.toString())) {
          visitedSet.add(neighbor.toString());
          queue.push([...path, neighbor]);
        }
      }
    }

    return { path: [], visited: visitedOrder };
  };

  // DFS Algorithm
  const dfs = () => {
    const stack = [[start]];
    const visitedSet = new Set([start.toString()]);
    const visitedOrder = [];

    while (stack.length > 0) {
      const path = stack.pop();
      const current = path[path.length - 1];
      visitedOrder.push(current);

      if (current[0] === end[0] && current[1] === end[1]) {
        return { path, visited: visitedOrder };
      }

      for (const neighbor of getNeighbors(current)) {
        if (!visitedSet.has(neighbor.toString())) {
          visitedSet.add(neighbor.toString());
          stack.push([...path, neighbor]);
        }
      }
    }

    return { path: [], visited: visitedOrder };
  };

  // Dijkstra Algorithm
  const dijkstra = () => {
    const distances = new Map();
    const previous = new Map();
    const unvisited = new Set();
    const visitedOrder = [];

    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        if (maze[i][j] === 0) {
          const node = [i, j];
          distances.set(node.toString(), Infinity);
          unvisited.add(node.toString());
        }
      }
    }

    distances.set(start.toString(), 0);

    while (unvisited.size > 0) {
      let current = null;
      let minDist = Infinity;

      for (const nodeStr of unvisited) {
        const dist = distances.get(nodeStr);
        if (dist < minDist) {
          minDist = dist;
          current = nodeStr;
        }
      }

      if (!current || minDist === Infinity) break;

      unvisited.delete(current);
      const [row, col] = current.split(',').map(Number);
      visitedOrder.push([row, col]);

      if (row === end[0] && col === end[1]) {
        const path = [];
        let node = end.toString();
        while (node) {
          path.unshift(node.split(',').map(Number));
          node = previous.get(node);
        }
        return { path, visited: visitedOrder };
      }

      for (const neighbor of getNeighbors([row, col])) {
        if (unvisited.has(neighbor.toString())) {
          const alt = distances.get(current) + 1;
          if (alt < distances.get(neighbor.toString())) {
            distances.set(neighbor.toString(), alt);
            previous.set(neighbor.toString(), current);
          }
        }
      }
    }

    return { path: [], visited: visitedOrder };
  };

  const solveMaze = async () => {
    setIsAnimating(true);
    setPath([]);
    setVisited([]);

    let result;
    switch (algorithm) {
      case 'astar':
        result = astar();
        break;
      case 'bfs':
        result = bfs();
        break;
      case 'dfs':
        result = dfs();
        break;
      case 'dijkstra':
        result = dijkstra();
        break;
      default:
        result = astar();
    }

    for (let i = 0; i < result.visited.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 20));
      setVisited(result.visited.slice(0, i + 1));
    }

    if (result.path.length > 0) {
      for (let i = 0; i < result.path.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 50));
        setPath(result.path.slice(0, i + 1));
      }
    }

    setIsAnimating(false);
  };

  const getCellColor = (row, col) => {
    if (!maze || !maze[row] || maze[row][col] === undefined) return 'bg-white';
    if (start && start.length === 2 && row === start[0] && col === start[1]) return 'bg-green-500';
    if (end && end.length === 2 && row === end[0] && col === end[1]) return 'bg-red-500';
    if (maze[row][col] === 1) return 'bg-gray-800';
    if (path && path.some(p => p && p[0] === row && p[1] === col)) return 'bg-yellow-400';
    if (visited && visited.some(v => v && v[0] === row && v[1] === col)) return 'bg-blue-300';
    return 'bg-white';
  };

  const tutorialSteps = {
    overview: {
      title: 'Overview: Pathfinding Algorithms',
      content: `# Complete Pathfinding Tutorial

## What You'll Learn

This comprehensive tutorial covers:
1. **Graph Theory** - How mazes are represented as graphs
2. **A* Algorithm** - The optimal pathfinding algorithm
3. **Heuristics** - How to estimate distances efficiently
4. **Pathfinding** - Multiple algorithms and their trade-offs

## Why Learn Pathfinding?

Pathfinding algorithms are fundamental to:
- Game AI (NPC navigation)
- GPS navigation systems
- Robotics (robot path planning)
- Network routing
- Logistics and delivery optimization
- Any problem requiring optimal path finding

## Tutorial Structure

This tutorial is divided into 5 comprehensive steps:
1. **Graph Theory Foundations** - Understanding graphs, nodes, edges
2. **A* Algorithm Deep Dive** - Complete explanation with formulas
3. **Heuristics Explained** - Manhattan, Euclidean, and admissible heuristics
4. **Pathfinding Algorithms** - A*, BFS, DFS, Dijkstra comparison
5. **Complete Implementation** - Full code with detailed explanations`
    },
    graphTheory: {
      title: 'Step 1: Graph Theory Foundations',
      content: `# Graph Theory: Representing Mazes as Graphs

## What is a Graph?

A **graph** G = (V, E) consists of:
- **V (Vertices/Nodes)**: Set of points/cells in the maze
- **E (Edges)**: Connections between adjacent cells

## Maze as Graph

In our maze solver:
- **Each cell** = A vertex/node
- **Adjacent cells** = Edges (connections)
- **Walls** = No edge (blocked connection)

### Example:

\`\`\`python
# Maze representation
maze = [
    [0, 0, 1, 0],  # 0 = empty cell (node), 1 = wall (no edge)
    [0, 1, 1, 0],
    [0, 0, 0, 0]
]

# Graph representation:
# Nodes: (0,0), (0,1), (0,3), (1,0), (1,3), (2,0), (2,1), (2,2), (2,3)
# Edges: Connections between adjacent empty cells
#        (0,0) ↔ (0,1)  (edge exists)
#        (0,0) ↔ (1,0)  (edge exists)
#        (0,1) ↔ (0,0)  (no edge - wall at (0,2))
\`\`\`

## Key Graph Concepts

### 1. Adjacency
Two nodes are **adjacent** if connected by an edge.

**In maze**: Cells are adjacent if they share a side (up, down, left, right).

### 2. Path
A **path** is a sequence of nodes where consecutive nodes are adjacent.

**Example**: [(0,0), (0,1), (1,1), (2,1)] is a path.

### 3. Shortest Path
The path with minimum number of edges (or minimum total weight).

### 4. Graph Types

**Undirected Graph**: Edges have no direction (maze connections are bidirectional)
**Weighted Graph**: Edges have weights (in our maze, all edges have weight = 1)
**Unweighted Graph**: All edges have same weight

## Mathematical Representation

### Adjacency Matrix
\`\`\`python
# For a 4x4 maze, adjacency matrix A where:
# A[i][j] = 1 if nodes i and j are adjacent
# A[i][j] = 0 otherwise

adjacency_matrix = [
    [0, 1, 0, 1, 0, ...],  # Node 0 connected to nodes 1, 3
    [1, 0, 0, 0, 1, ...],  # Node 1 connected to nodes 0, 4
    # ...
]
\`\`\`

### Adjacency List (More Efficient)
\`\`\`python
# For each node, list its neighbors
adjacency_list = {
    (0, 0): [(0, 1), (1, 0)],      # Node (0,0) neighbors
    (0, 1): [(0, 0), (0, 2)],     # Node (0,1) neighbors
    # ...
}
\`\`\`

## Why Graph Theory Matters

1. **Abstraction**: Reduces complex maze to simple graph structure
2. **Algorithms**: Graph algorithms solve pathfinding problems
3. **Efficiency**: Graph representations enable efficient search
4. **Generalization**: Same concepts apply to any pathfinding problem

## Code: Building Graph from Maze

\`\`\`python
def build_graph(maze):
    """
    Convert maze to graph representation
    
    Returns: adjacency_list dictionary
    """
    graph = {}
    rows, cols = len(maze), len(maze[0])
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    
    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 0:  # Empty cell
                node = (i, j)
                neighbors = []
                
                # Check all 4 directions
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    
                    # Check bounds and if cell is empty
                    if (0 <= ni < rows and 0 <= nj < cols and 
                        maze[ni][nj] == 0):
                        neighbors.append((ni, nj))
                
                graph[node] = neighbors
    
    return graph

# Example usage
maze = [
    [0, 0, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0]
]

graph = build_graph(maze)
print(graph)
# Output: {
#     (0, 0): [(0, 1), (1, 0)],
#     (0, 1): [(0, 0), (0, 3)],
#     (0, 3): [(0, 1), (1, 3)],
#     ...
# }
\`\`\`

## Graph Properties in Pathfinding

- **Connected Graph**: All nodes reachable from start
- **Acyclic**: No cycles (for tree search)
- **Weighted**: Each edge has cost (usually 1 in grid)
- **Sparse**: Few edges compared to possible edges (efficient storage)`
    },
    astarAlgorithm: {
      title: 'Step 2: A* Algorithm Deep Dive',
      content: `# A* Algorithm: Complete Explanation

## What is A*?

**A*** (A-Star) is an informed search algorithm that finds the shortest path 
from start to goal using a heuristic function to guide the search.

## The A* Formula

### Core Formula:
\`\`\`
f(n) = g(n) + h(n)
\`\`\`

Where:
- **f(n)** = Total estimated cost from start to goal through node n
- **g(n)** = Actual cost from start to node n (known)
- **h(n)** = Heuristic estimate from node n to goal (estimated)

### Detailed Example with Sample Numbers:

Let's trace through a concrete example:

**Setup:**
- Start position: (0, 0)
- Current node n: (2, 3)
- Goal position: (5, 7)
- Each step costs: 1

**Step 1: Calculate g(n) - Actual cost from start**
- Path from start (0,0) to current (2,3):
  - (0,0) → (1,0) → (2,0) → (2,1) → (2,2) → (2,3)
  - Number of steps: 5
  - **g(n) = 5**

**Step 2: Calculate h(n) - Heuristic estimate to goal**
- Using Manhattan Distance: h(n) = |x₁ - x₂| + |y₁ - y₂|
- h(n) = |2 - 5| + |3 - 7|
- h(n) = |−3| + |−4|
- h(n) = 3 + 4
- **h(n) = 7**

**Step 3: Calculate f(n) - Total estimated cost**
- f(n) = g(n) + h(n)
- f(n) = 5 + 7
- **f(n) = 12**

**Interpretation:**
- We've traveled 5 steps from start (g = 5)
- We estimate 7 more steps to goal (h = 7)
- Total estimated cost: 12 steps

### Complete Example: Multiple Nodes

Let's see how A* evaluates multiple nodes:

**Maze Setup:**
\`\`\`
Start (0,0)     (0,1)     (0,2)
  (1,0)     (1,1)     (1,2)
  (2,0)     (2,1)  Goal (2,2)
\`\`\`

**Node Evaluations:**

**Node A: (0,1)**
- g(A) = 1 (1 step from start)
- h(A) = |0-2| + |1-2| = 2 + 1 = 3
- f(A) = 1 + 3 = **4**

**Node B: (1,0)**
- g(B) = 1 (1 step from start)
- h(B) = |1-2| + |0-2| = 1 + 2 = 3
- f(B) = 1 + 3 = **4**

**Node C: (1,1)**
- g(C) = 2 (2 steps from start, e.g., via (0,1))
- h(C) = |1-2| + |1-2| = 1 + 1 = 2
- f(C) = 2 + 2 = **4**

**Node D: (2,1)**
- g(D) = 3 (3 steps from start)
- h(D) = |2-2| + |1-2| = 0 + 1 = 1
- f(D) = 3 + 1 = **4**

**A* Decision:**
- All nodes have f = 4, so A* can choose any
- Typically explores in order: (1,1) or (2,1) as they're closer to goal
- Next step: (2,2) - Goal reached!

## How A* Works

### Step-by-Step Process:

1. **Initialize**:
   - Start node: g(start) = 0, f(start) = h(start)
   - Open set: [start]
   - Closed set: []

2. **Main Loop**:
   - Select node with **lowest f-score** from open set
   - Move it to closed set
   - If it's the goal, reconstruct path and return

3. **Explore Neighbors**:
   - For each neighbor:
     - Calculate tentative g-score: g(neighbor) = g(current) + cost(current, neighbor)
     - If better path found, update g and f scores
     - Add to open set if not already there

4. **Repeat** until goal found or open set empty

## Detailed Algorithm Explanation

### Data Structures:

\`\`\`python
# Priority queue (min-heap) for open set
open_set = []  # Nodes to explore, sorted by f-score

# Maps for tracking costs
g_score = {}   # Actual cost from start to each node
f_score = {}   # Estimated total cost (g + h)

# For path reconstruction
came_from = {}  # Parent of each node in optimal path

# Track visited nodes
closed_set = set()  # Nodes already explored
\`\`\`

### Complete A* Implementation:

\`\`\`python
import heapq

def astar(maze, start, end, heuristic):
    """
    A* Pathfinding Algorithm
    
    Parameters:
    - maze: 2D grid (0 = empty, 1 = wall)
    - start: (row, col) starting position
    - end: (row, col) goal position
    - heuristic: function h(n) estimating distance to goal
    
    Returns: path (list of nodes) or None
    """
    
    def get_neighbors(node):
        """Get valid adjacent cells"""
        row, col = node
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if (0 <= r < len(maze) and 0 <= c < len(maze[0]) and 
                maze[r][c] == 0):
                neighbors.append((r, c))
        return neighbors
    
    # Initialize
    open_set = [(0, start)]  # (f_score, node)
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    came_from = {}
    closed_set = set()
    
    while open_set:
        # Get node with lowest f-score
        current_f, current = heapq.heappop(open_set)
        
        # Skip if already processed
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        # Goal reached!
        if current == end:
            # Reconstruct path
            path = []
            node = end
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            return path[::-1]  # Reverse to get start->end
        
        # Explore neighbors
        for neighbor in get_neighbors(current):
            if neighbor in closed_set:
                continue
            
            # Calculate tentative g-score
            # Cost = 1 for grid (can be different for weighted graphs)
            tentative_g = g_score[current] + 1
            
            # If better path found
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                
                # Add to open set
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found
\`\`\`

## Why A* is Optimal

A* is **optimal** (finds shortest path) if:
1. **Heuristic is admissible**: h(n) ≤ actual distance to goal
2. **Heuristic is consistent**: h(n) ≤ cost(n, n') + h(n') for all neighbors n'

### Admissibility Proof:
If h(n) never overestimates, A* will never skip the optimal path.

## Time & Space Complexity

- **Time**: O(b^d) where b = branching factor, d = depth
- **Space**: O(b^d) for storing open/closed sets
- **Best case**: O(d) if heuristic is perfect
- **Worst case**: O(b^d) if heuristic provides no guidance

## A* vs Other Algorithms

| Algorithm | Optimal? | Uses Heuristic? | Time Complexity |
|-----------|----------|-----------------|-----------------|
| A*        | Yes      | Yes             | O(b^d)          |
| Dijkstra  | Yes      | No              | O((V+E)log V)   |
| BFS       | Yes*     | No              | O(V+E)          |
| DFS       | No       | No              | O(V+E)          |

*BFS optimal only for unweighted graphs`
    },
    heuristics: {
      title: 'Step 3: Heuristics Explained',
      content: `# Heuristics: Estimating Distances

## What is a Heuristic?

A **heuristic** h(n) is a function that estimates the cost from node n to the goal.
It's a "rule of thumb" that guides the search toward the goal.

## Why Use Heuristics?

1. **Speed**: Guides search directly toward goal (fewer nodes explored)
2. **Efficiency**: Reduces search space significantly
3. **Optimality**: With admissible heuristics, A* finds optimal path

## Common Heuristics for Grid Pathfinding

### 1. Manhattan Distance (L1 Norm)

**Formula**: h(n) = |x₁ - x₂| + |y₁ - y₂|

**Use case**: Grids where movement is only up/down/left/right (4-directional)

**Properties**:
- ✅ Admissible (never overestimates)
- ✅ Consistent
- ✅ Fast to compute: O(1)

**Example**:
\`\`\`python
def manhattan_distance(a, b):
    """Manhattan distance heuristic"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Example: (0, 0) to (3, 4)
# h = |0-3| + |0-4| = 3 + 4 = 7
\`\`\`

### Detailed Calculation Examples:

**Example 1: Basic Calculation**
- Point A: (2, 5)
- Point B: (7, 9)
- **Calculation:**
  - |x₁ - x₂| = |2 - 7| = |−5| = **5**
  - |y₁ - y₂| = |5 - 9| = |−4| = **4**
  - h(n) = 5 + 4 = **9**

**Example 2: Negative Coordinates**
- Point A: (−3, 2)
- Point B: (1, −4)
- **Calculation:**
  - |x₁ - x₂| = |−3 - 1| = |−4| = **4**
  - |y₁ - y₂| = |2 - (−4)| = |2 + 4| = |6| = **6**
  - h(n) = 4 + 6 = **10**

**Example 3: Same Row (Horizontal Movement)**
- Point A: (3, 0)
- Point B: (3, 8)
- **Calculation:**
  - |x₁ - x₂| = |3 - 3| = |0| = **0**
  - |y₁ - y₂| = |0 - 8| = |−8| = **8**
  - h(n) = 0 + 8 = **8**
  - *Interpretation: Only vertical movement needed*

**Example 4: Same Column (Vertical Movement)**
- Point A: (0, 5)
- Point B: (6, 5)
- **Calculation:**
  - |x₁ - x₂| = |0 - 6| = |−6| = **6**
  - |y₁ - y₂| = |5 - 5| = |0| = **0**
  - h(n) = 6 + 0 = **6**
  - *Interpretation: Only horizontal movement needed*

**Example 5: Pathfinding Context**
- Current node: (4, 2)
- Goal node: (8, 6)
- **Step-by-step:**
  1. Calculate x difference: |4 - 8| = 4
  2. Calculate y difference: |2 - 6| = 4
  3. Sum: 4 + 4 = **8**
  4. *Meaning: At least 8 steps needed (4 right + 4 down)*

**Visual Representation:**
\`\`\`
Start (0,0) → Goal (3,4)
   |←←←←←←←←←|
   |         |
   |         ↓
   |         Goal
   |
   Start
   
   Manhattan: 3 right + 4 down = 7 steps
\`\`\`

**Why Manhattan Distance Works:**
- In a grid with 4-directional movement, you can only move:
  - Up/Down (changes y)
  - Left/Right (changes x)
- Minimum steps = |Δx| + |Δy|
- This matches Manhattan distance exactly!

### 2. Euclidean Distance (L2 Norm)

**Formula**: h(n) = √((x₁ - x₂)² + (y₁ - y₂)²)

**Use case**: Continuous spaces or 8-directional movement

**Properties**:
- ✅ Admissible
- ✅ Consistent
- ⚠️ Slightly slower: O(1) but involves square root

**Example**:
\`\`\`python
import math

def euclidean_distance(a, b):
    """Euclidean distance heuristic"""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx*dx + dy*dy)

# Example: (0, 0) to (3, 4)
# h = √(3² + 4²) = √(9 + 16) = √25 = 5
\`\`\`

### 3. Chebyshev Distance (L∞ Norm)

**Formula**: h(n) = max(|x₁ - x₂|, |y₁ - y₂|)

**Use case**: 8-directional movement (diagonal allowed)

**Properties**:
- ✅ Admissible for 8-directional
- ✅ Consistent

**Example**:
\`\`\`python
def chebyshev_distance(a, b):
    """Chebyshev distance heuristic"""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

# Example: (0, 0) to (3, 4)
# h = max(3, 4) = 4
\`\`\`

## Heuristic Properties

### 1. Admissibility

**Definition**: h(n) ≤ actual cost from n to goal

**Why important**: Ensures A* finds optimal path

**Example**:
- Manhattan distance is admissible for 4-directional movement
- If actual path is 7 steps, h(n) ≤ 7

### 2. Consistency (Monotonicity)

**Definition**: h(n) ≤ cost(n, n') + h(n') for all neighbors n'

**Why important**: Ensures A* doesn't need to re-explore nodes

**Example**:
- If h(current) = 5 and moving to neighbor costs 1
- Then h(neighbor) ≥ 4 (consistent)

### 3. Dominance

**Definition**: h₁ dominates h₂ if h₁(n) ≥ h₂(n) for all n

**Why important**: Better heuristics explore fewer nodes

**Example**:
- Euclidean distance dominates Manhattan distance
- But both are admissible

## Choosing the Right Heuristic

| Movement Type | Best Heuristic | Reason |
|---------------|----------------|--------|
| 4-directional (grid) | Manhattan | Matches movement exactly |
| 8-directional | Chebyshev | Accounts for diagonals |
| Continuous space | Euclidean | Most accurate |
| Weighted edges | Custom | Based on edge weights |

## Heuristic Comparison Example

\`\`\`python
start = (0, 0)
goal = (3, 4)

# Manhattan: |0-3| + |0-4| = 7
# Euclidean: √(3² + 4²) = 5
# Chebyshev: max(3, 4) = 4

# For 4-directional movement:
# - Manhattan is most accurate (actual path = 7)
# - Euclidean underestimates (5 < 7)
# - Chebyshev underestimates more (4 < 7)

# All are admissible, but Manhattan is best!
\`\`\`

## Code: Implementing Heuristics

\`\`\`python
class Heuristics:
    """Collection of heuristic functions"""
    
    @staticmethod
    def manhattan(a, b):
        """Manhattan distance for 4-directional movement"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    @staticmethod
    def euclidean(a, b):
        """Euclidean distance for continuous/8-directional"""
        import math
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.sqrt(dx*dx + dy*dy)
    
    @staticmethod
    def chebyshev(a, b):
        """Chebyshev distance for 8-directional movement"""
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    
    @staticmethod
    def zero(a, b):
        """No heuristic (becomes Dijkstra)"""
        return 0

# Usage in A*
heuristic = Heuristics.manhattan
path = astar(maze, start, end, heuristic)
\`\`\`

## Heuristic Quality Impact

**Better heuristics** (closer to actual distance):
- ✅ Explore fewer nodes
- ✅ Faster execution
- ✅ Still find optimal path

**Worse heuristics** (further from actual):
- ⚠️ Explore more nodes
- ⚠️ Slower execution
- ✅ Still find optimal path (if admissible)

**Perfect heuristic** h*(n) = actual distance:
- ✅ Explores only nodes on optimal path
- ✅ Fastest possible
- ✅ Still optimal`
    },
    pathfinding: {
      title: 'Step 4: Pathfinding Algorithms Comparison',
      content: `# Pathfinding Algorithms: Complete Comparison

## Overview of Algorithms

We'll compare 4 major pathfinding algorithms:
1. **A*** - Informed search with heuristics
2. **BFS** - Breadth-first exploration
3. **DFS** - Depth-first exploration
4. **Dijkstra** - Weighted graph shortest path

## Algorithm Comparison Table

| Algorithm | Optimal? | Uses Heuristic? | Best For | Time Complexity |
|-----------|----------|-----------------|----------|-----------------|
| **A***    | ✅ Yes   | ✅ Yes          | Grid pathfinding | O(b^d) |
| **BFS**   | ✅ Yes*  | ❌ No           | Unweighted graphs | O(V+E) |
| **DFS**   | ❌ No    | ❌ No           | Path existence | O(V+E) |
| **Dijkstra** | ✅ Yes | ❌ No           | Weighted graphs | O((V+E)log V) |

*BFS optimal only for unweighted graphs

## 1. A* Algorithm

### How It Works:
- Uses **f(n) = g(n) + h(n)** to guide search
- Explores nodes with lowest f-score first
- Combines actual cost (g) with heuristic estimate (h)

### Advantages:
- ✅ Finds optimal path
- ✅ Efficient (guided by heuristic)
- ✅ Works for weighted and unweighted graphs

### Disadvantages:
- ⚠️ Requires good heuristic
- ⚠️ Memory intensive (stores all explored nodes)

### Code:
\`\`\`python
def astar(maze, start, end):
    open_set = [(0, start)]  # (f_score, node)
    g_score = {start: 0}
    came_from = {}
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        if current == end:
            return reconstruct_path(came_from, end)
        
        for neighbor in get_neighbors(current):
            tentative_g = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score, neighbor))
    
    return None
\`\`\`

## 2. Breadth-First Search (BFS)

### How It Works:
- Explores **level by level** (all nodes at depth d before depth d+1)
- Uses **queue** (FIFO: First In, First Out)
- Guarantees shortest path in **unweighted** graphs

### Advantages:
- ✅ Finds shortest path (unweighted graphs)
- ✅ Simple to implement
- ✅ No heuristic needed

### Disadvantages:
- ❌ Not optimal for weighted graphs
- ⚠️ Explores many unnecessary nodes
- ⚠️ Memory intensive

### Code:
\`\`\`python
from collections import deque

def bfs(maze, start, end):
    queue = deque([(start, [start])])
    visited = set([start])
    
    while queue:
        current, path = queue.popleft()
        
        if current == end:
            return path
        
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None
\`\`\`

### BFS Visualization:
\`\`\`
Level 0: [Start]
Level 1: [All neighbors of Start]
Level 2: [All neighbors of Level 1 nodes]
Level 3: [All neighbors of Level 2 nodes]
...
\`\`\`

## 3. Depth-First Search (DFS)

### How It Works:
- Explores **as deep as possible** before backtracking
- Uses **stack** (LIFO: Last In, First Out)
- May not find shortest path

### Advantages:
- ✅ Simple to implement
- ✅ Low memory (only stores current path)
- ✅ Good for path existence problems

### Disadvantages:
- ❌ **Not optimal** (may find long path)
- ⚠️ Can get stuck in deep branches
- ⚠️ May explore entire graph

### Code:
\`\`\`python
def dfs(maze, start, end):
    stack = [(start, [start])]
    visited = set([start])
    
    while stack:
        current, path = stack.pop()
        
        if current == end:
            return path
        
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))
    
    return None
\`\`\`

### DFS Visualization:
\`\`\`
Start → Neighbor1 → Neighbor1.1 → Neighbor1.1.1 → ...
         (backtrack if dead end)
         → Neighbor2 → Neighbor2.1 → ...
\`\`\`

## 4. Dijkstra's Algorithm

### How It Works:
- Finds shortest path in **weighted** graphs
- Similar to A* but **without heuristic**
- Always explores node with minimum distance first

### The Dijkstra Distance Update Formula:

**Core Formula:**
\`\`\`
dist[v] = min(dist[v], dist[u] + weight(u, v))
\`\`\`

Where:
- **dist[v]** = Current shortest distance to node v
- **dist[u]** = Shortest distance to node u (already known)
- **weight(u, v)** = Cost to travel from u to v
- **min(...)** = Keep the smaller value (shorter path)

### Detailed Example with Sample Numbers:

**Graph Setup:**
- Start node: A
- Nodes: A, B, C, D, E
- Edges with weights:
  - A → B: weight = 4
  - A → C: weight = 2
  - B → D: weight = 5
  - C → B: weight = 1
  - C → D: weight = 8
  - C → E: weight = 10
  - D → E: weight = 2

**Initial State:**
- dist[A] = 0 (start node)
- dist[B] = ∞ (unknown)
- dist[C] = ∞ (unknown)
- dist[D] = ∞ (unknown)
- dist[E] = ∞ (unknown)

**Iteration 1: Process A**
- Current node: A (dist = 0)
- Neighbors: B, C

**Update B:**
- dist[B] = min(dist[B], dist[A] + weight(A, B))
- dist[B] = min(∞, 0 + 4)
- dist[B] = min(∞, 4)
- **dist[B] = 4** ✅

**Update C:**
- dist[C] = min(dist[C], dist[A] + weight(A, C))
- dist[C] = min(∞, 0 + 2)
- dist[C] = min(∞, 2)
- **dist[C] = 2** ✅

**Current distances:**
- dist[A] = 0
- dist[B] = 4
- dist[C] = 2
- dist[D] = ∞
- dist[E] = ∞

**Iteration 2: Process C** (smallest unvisited: C with dist = 2)
- Current node: C (dist = 2)
- Neighbors: B, D, E

**Update B:**
- dist[B] = min(dist[B], dist[C] + weight(C, B))
- dist[B] = min(4, 2 + 1)
- dist[B] = min(4, 3)
- **dist[B] = 3** ✅ (Found shorter path!)

**Update D:**
- dist[D] = min(dist[D], dist[C] + weight(C, D))
- dist[D] = min(∞, 2 + 8)
- dist[D] = min(∞, 10)
- **dist[D] = 10** ✅

**Update E:**
- dist[E] = min(dist[E], dist[C] + weight(C, E))
- dist[E] = min(∞, 2 + 10)
- dist[E] = min(∞, 12)
- **dist[E] = 12** ✅

**Current distances:**
- dist[A] = 0
- dist[B] = 3 (updated from 4!)
- dist[C] = 2
- dist[D] = 10
- dist[E] = 12

**Iteration 3: Process B** (smallest unvisited: B with dist = 3)
- Current node: B (dist = 3)
- Neighbors: D

**Update D:**
- dist[D] = min(dist[D], dist[B] + weight(B, D))
- dist[D] = min(10, 3 + 5)
- dist[D] = min(10, 8)
- **dist[D] = 8** ✅ (Found shorter path!)

**Current distances:**
- dist[A] = 0
- dist[B] = 3
- dist[C] = 2
- dist[D] = 8 (updated from 10!)
- dist[E] = 12

**Iteration 4: Process D** (smallest unvisited: D with dist = 8)
- Current node: D (dist = 8)
- Neighbors: E

**Update E:**
- dist[E] = min(dist[E], dist[D] + weight(D, E))
- dist[E] = min(12, 8 + 2)
- dist[E] = min(12, 10)
- **dist[E] = 10** ✅ (Found shorter path!)

**Final distances:**
- dist[A] = 0
- dist[B] = 3
- dist[C] = 2
- dist[D] = 8
- dist[E] = 10

**Key Insight:**
- The formula **always keeps the minimum** distance
- When a shorter path is found, it **updates** the distance
- This ensures we find the **optimal** shortest path

### Step-by-Step Calculation Example:

**Simple Grid Example:**

**Setup:**
- Grid: 3x3
- Start: (0,0)
- Goal: (2,2)
- All edges have weight = 1

**Initial:**
- dist[(0,0)] = 0
- All others = ∞

**After processing (0,0):**
- dist[(0,1)] = min(∞, 0 + 1) = **1**
- dist[(1,0)] = min(∞, 0 + 1) = **1**

**After processing (0,1):**
- dist[(0,2)] = min(∞, 1 + 1) = **2**
- dist[(1,1)] = min(∞, 1 + 1) = **2**

**After processing (1,0):**
- dist[(1,1)] = min(2, 1 + 1) = **2** (no change)
- dist[(2,0)] = min(∞, 1 + 1) = **2**

**After processing (1,1):**
- dist[(1,2)] = min(∞, 2 + 1) = **3**
- dist[(2,1)] = min(∞, 2 + 1) = **3**

**After processing (2,1):**
- dist[(2,2)] = min(∞, 3 + 1) = **4** ✅ Goal reached!

**Final path length: 4 steps**

### Advantages:
- ✅ Finds optimal path (weighted graphs)
- ✅ No heuristic needed
- ✅ Works for any graph structure

### Disadvantages:
- ⚠️ Slower than A* (explores more nodes)
- ⚠️ Memory intensive

### Code:
\`\`\`python
import heapq

def dijkstra(maze, start, end):
    distances = {start: 0}
    previous = {}
    unvisited = set(all_nodes)
    
    while unvisited:
        # Get unvisited node with minimum distance
        current = min(unvisited, key=lambda n: distances.get(n, float('inf')))
        unvisited.remove(current)
        
        if current == end:
            return reconstruct_path(previous, end)
        
        for neighbor in get_neighbors(current):
            if neighbor in unvisited:
                alt = distances[current] + weight(current, neighbor)
                if alt < distances.get(neighbor, float('inf')):
                    distances[neighbor] = alt
                    previous[neighbor] = current
    
    return None
\`\`\`

## When to Use Which Algorithm?

### Use **A*** when:
- ✅ Grid-based pathfinding
- ✅ Need optimal path
- ✅ Have good heuristic
- ✅ **Best choice for most pathfinding problems**

### Use **BFS** when:
- ✅ Unweighted graph
- ✅ Need shortest path
- ✅ Simple implementation needed
- ✅ No heuristic available

### Use **DFS** when:
- ✅ Just need to find ANY path
- ✅ Memory is limited
- ✅ Path existence problem
- ❌ Don't need shortest path

### Use **Dijkstra** when:
- ✅ Weighted graph
- ✅ No good heuristic available
- ✅ Need optimal path
- ⚠️ A* usually better if heuristic exists

## Performance Comparison

### Example: 20x20 Grid Maze

| Algorithm | Nodes Explored | Time | Path Length |
|-----------|----------------|------|-------------|
| A*        | ~150           | Fast | Optimal (28) |
| BFS       | ~400           | Medium | Optimal (28) |
| DFS       | ~200           | Medium | Suboptimal (45) |
| Dijkstra  | ~350           | Slow | Optimal (28) |

**A* is the clear winner** for grid pathfinding!

## Code: Complete Comparison

\`\`\`python
def compare_algorithms(maze, start, end):
    """Compare all pathfinding algorithms"""
    
    results = {}
    
    # A*
    start_time = time.time()
    path_astar = astar(maze, start, end, manhattan_distance)
    time_astar = time.time() - start_time
    results['A*'] = {
        'path': path_astar,
        'length': len(path_astar) if path_astar else None,
        'time': time_astar
    }
    
    # BFS
    start_time = time.time()
    path_bfs = bfs(maze, start, end)
    time_bfs = time.time() - start_time
    results['BFS'] = {
        'path': path_bfs,
        'length': len(path_bfs) if path_bfs else None,
        'time': time_bfs
    }
    
    # DFS
    start_time = time.time()
    path_dfs = dfs(maze, start, end)
    time_dfs = time.time() - start_time
    results['DFS'] = {
        'path': path_dfs,
        'length': len(path_dfs) if path_dfs else None,
        'time': time_dfs
    }
    
    # Dijkstra
    start_time = time.time()
    path_dijkstra = dijkstra(maze, start, end)
    time_dijkstra = time.time() - start_time
    results['Dijkstra'] = {
        'path': path_dijkstra,
        'length': len(path_dijkstra) if path_dijkstra else None,
        'time': time_dijkstra
    }
    
    return results
\`\`\``
    },
    implementation: {
      title: 'Step 5: Complete Implementation',
      content: `# Complete Pathfinding Implementation

## Full Maze Solver with All Algorithms

This is a complete, production-ready implementation with:
- Graph representation
- All 4 algorithms (A*, BFS, DFS, Dijkstra)
- Path reconstruction
- Error handling
- Visualization support

## Complete Code

\`\`\`python
import heapq
from collections import deque
from typing import List, Tuple, Optional, Set, Dict

class MazeSolver:
    """
    Complete Maze Solver with Multiple Pathfinding Algorithms
    
    Supports:
    - A* (optimal with heuristic)
    - BFS (optimal for unweighted)
    - DFS (any path)
    - Dijkstra (optimal for weighted)
    """
    
    def __init__(self, maze: List[List[int]]):
        """
        Initialize solver with maze
        
        Args:
            maze: 2D list where 0 = empty cell, 1 = wall
        """
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0]) if maze else 0
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    
    def get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighbors of a node
        
        Graph Theory: Returns edges from current node
        """
        row, col = node
        neighbors = []
        
        for dr, dc in self.directions:
            r, c = row + dr, col + dc
            
            # Check bounds and if cell is empty (not a wall)
            if (0 <= r < self.rows and 0 <= c < self.cols and 
                self.maze[r][c] == 0):
                neighbors.append((r, c))
        
        return neighbors
    
    def manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Manhattan distance heuristic
        
        Formula: h(n) = |x₁ - x₂| + |y₁ - y₂|
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct path from start to goal using parent pointers
        
        Graph Theory: Traverse parent chain to build path
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Reverse to get start->end
    
    def astar(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A* Algorithm Implementation
        
        Formula: f(n) = g(n) + h(n)
        - g(n) = actual cost from start to n
        - h(n) = heuristic estimate from n to goal
        - f(n) = total estimated cost
        
        Returns: path (list of nodes) or None if no path exists
        """
        # Initialize data structures
        open_set = [(0, start)]  # Priority queue: (f_score, node)
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        f_score: Dict[Tuple[int, int], float] = {start: self.manhattan_distance(start, end)}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_set: Set[Tuple[int, int]] = set()
        
        while open_set:
            # Get node with lowest f-score
            current_f, current = heapq.heappop(open_set)
            
            # Skip if already processed
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Goal reached!
            if current == end:
                return self.reconstruct_path(came_from, current)
            
            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g-score
                # Cost = 1 for grid (can be different for weighted graphs)
                tentative_g = g_score[current] + 1
                
                # If better path found, update
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.manhattan_distance(neighbor, end)
                    
                    # Add to open set
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def bfs(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Breadth-First Search
        
        Explores level by level, guarantees shortest path in unweighted graphs
        """
        queue = deque([(start, [start])])
        visited: Set[Tuple[int, int]] = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if current == end:
                return path
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def dfs(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Depth-First Search
        
        Explores as deep as possible, may not find shortest path
        """
        stack = [(start, [start])]
        visited: Set[Tuple[int, int]] = {start}
        
        while stack:
            current, path = stack.pop()
            
            if current == end:
                return path
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, path + [neighbor]))
        
        return None
    
    def dijkstra(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Dijkstra's Algorithm
        
        Finds shortest path in weighted graphs (no heuristic)
        Formula: dist[v] = min(dist[v], dist[u] + weight(u, v))
        """
        distances: Dict[Tuple[int, int], float] = {start: 0}
        previous: Dict[Tuple[int, int], Tuple[int, int]] = {}
        unvisited: Set[Tuple[int, int]] = set()
        
        # Initialize all nodes
        for i in range(self.rows):
            for j in range(self.cols):
                if self.maze[i][j] == 0:
                    node = (i, j)
                    if node != start:
                        distances[node] = float('inf')
                    unvisited.add(node)
        
        while unvisited:
            # Find unvisited node with minimum distance
            current = min(unvisited, key=lambda n: distances.get(n, float('inf')))
            
            if distances.get(current, float('inf')) == float('inf'):
                break
            
            unvisited.remove(current)
            
            if current == end:
                return self.reconstruct_path(previous, current)
            
            # Update distances to neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor in unvisited:
                    alt = distances[current] + 1  # Weight = 1
                    if alt < distances.get(neighbor, float('inf')):
                        distances[neighbor] = alt
                        previous[neighbor] = current
        
        return None
    
    def solve(self, start: Tuple[int, int], end: Tuple[int, int], 
              algorithm: str = 'astar') -> Optional[List[Tuple[int, int]]]:
        """
        Solve maze using specified algorithm
        
        Args:
            start: Starting position (row, col)
            end: Goal position (row, col)
            algorithm: 'astar', 'bfs', 'dfs', or 'dijkstra'
        
        Returns:
            Path as list of nodes or None if no path exists
        """
        if algorithm == 'astar':
            return self.astar(start, end)
        elif algorithm == 'bfs':
            return self.bfs(start, end)
        elif algorithm == 'dfs':
            return self.dfs(start, end)
        elif algorithm == 'dijkstra':
            return self.dijkstra(start, end)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

# Example Usage
if __name__ == "__main__":
    # Example maze
    maze = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    
    solver = MazeSolver(maze)
    start = (0, 0)
    end = (4, 4)
    
    # Try different algorithms
    for algo in ['astar', 'bfs', 'dfs', 'dijkstra']:
        path = solver.solve(start, end, algo)
        if path:
            print(f"{algo.upper()}: Path found with {len(path)} steps")
            print(f"  Path: {path}")
        else:
            print(f"{algo.upper()}: No path found")
\`\`\`

## Code Explanation

### 1. Graph Representation
- **Maze → Graph**: Each cell is a node, adjacent cells are edges
- **get_neighbors()**: Returns edges from a node (Graph Theory)

### 2. A* Algorithm Details
- **Priority Queue**: Stores nodes sorted by f-score
- **g_score**: Actual cost from start (known)
- **h_score**: Heuristic estimate to goal (estimated)
- **f_score**: Total estimated cost (g + h)
- **Path Reconstruction**: Backtrack using parent pointers

### 3. Algorithm Differences

**A***:
- Uses heuristic to guide search
- Explores nodes with lowest f-score
- Optimal and efficient

**BFS**:
- Explores level by level
- Uses queue (FIFO)
- Optimal for unweighted graphs

**DFS**:
- Explores deep first
- Uses stack (LIFO)
- Not optimal but memory efficient

**Dijkstra**:
- No heuristic
- Explores by minimum distance
- Optimal for weighted graphs

## Testing the Implementation

\`\`\`python
# Test with different mazes
test_mazes = [
    # Simple maze
    [[0, 0, 0],
     [0, 1, 0],
     [0, 0, 0]],
    
    # Complex maze
    [[0, 0, 1, 0, 0],
     [0, 1, 1, 0, 1],
     [0, 0, 0, 0, 0],
     [1, 1, 0, 1, 0],
     [0, 0, 0, 0, 0]],
    
    # No path maze
    [[0, 1, 0],
     [1, 1, 1],
     [0, 1, 0]]
]

for i, maze in enumerate(test_mazes):
    print(f"\\nTesting Maze {i+1}:")
    solver = MazeSolver(maze)
    start = (0, 0)
    end = (len(maze)-1, len(maze[0])-1)
    
    path = solver.solve(start, end, 'astar')
    if path:
        print(f"  Path found: {len(path)} steps")
    else:
        print("  No path exists")
\`\`\`

## Performance Optimization Tips

1. **Use A*** for grid pathfinding (best choice)
2. **Good heuristic** = faster search
3. **Early termination** when goal found
4. **Efficient data structures** (heap for priority queue)
5. **Avoid redundant exploration** (closed set)

## Real-World Applications

- **Game AI**: NPC pathfinding
- **GPS Navigation**: Route planning
- **Robotics**: Robot navigation
- **Network Routing**: Packet routing
- **Logistics**: Delivery optimization`
    }
  };

  const renderTutorialContent = () => {
    const step = tutorialSteps[selectedStep];
    if (!step) return null;

    // Parse content and render with proper formatting
    const parts = [];
    const lines = step.content.split('\n');
    let currentCodeBlock = null;
    let currentCodeLanguage = 'python';
    let currentParagraph = [];

    const flushParagraph = () => {
      if (currentParagraph.length > 0) {
        const text = currentParagraph.join(' ').trim();
        if (text) {
          parts.push(
            <p key={`p-${parts.length}`} className="mb-3 text-gray-700 leading-relaxed">
              {text.split(/(\*\*.*?\*\*)/g).map((segment, idx) => {
                if (segment.startsWith('**') && segment.endsWith('**')) {
                  return <strong key={idx} className="font-semibold text-gray-900">{segment.slice(2, -2)}</strong>;
                }
                return segment;
              })}
            </p>
          );
        }
        currentParagraph = [];
      }
    };

    const flushCodeBlock = () => {
      if (currentCodeBlock) {
        parts.push(
          <div key={`code-${parts.length}`} className="my-4">
            <SyntaxHighlighter language={currentCodeLanguage} style={vscDarkPlus} showLineNumbers>
              {currentCodeBlock.join('\n')}
            </SyntaxHighlighter>
          </div>
        );
        currentCodeBlock = null;
      }
    };

    lines.forEach((line, idx) => {
      // Code block start/end (handle both ``` and \`\`\`)
      const trimmedLine = line.trim();
      const isCodeBlockStart = trimmedLine.startsWith('```') || trimmedLine.startsWith('\\`\\`\\`');
      if (isCodeBlockStart) {
        flushParagraph();
        if (currentCodeBlock === null) {
          // Start code block
          const cleanLine = trimmedLine.replace(/\\`/g, '`');
          const langMatch = cleanLine.match(/```(\w+)?/);
          currentCodeLanguage = langMatch && langMatch[1] ? langMatch[1] : 'python';
          currentCodeBlock = [];
        } else {
          // End code block
          flushCodeBlock();
        }
        return;
      }

      // Inside code block
      if (currentCodeBlock !== null) {
        currentCodeBlock.push(line);
        return;
      }

      // Headers
      if (line.startsWith('# ')) {
        flushParagraph();
        parts.push(<h2 key={`h2-${idx}`} className="text-xl font-bold text-gray-900 mt-6 mb-3">{line.substring(2)}</h2>);
        return;
      }
      if (line.startsWith('## ')) {
        flushParagraph();
        parts.push(<h3 key={`h3-${idx}`} className="text-lg font-semibold text-gray-800 mt-4 mb-2">{line.substring(3)}</h3>);
        return;
      }
      if (line.startsWith('### ')) {
        flushParagraph();
        parts.push(<h4 key={`h4-${idx}`} className="text-md font-semibold text-gray-700 mt-3 mb-2">{line.substring(4)}</h4>);
        return;
      }

      // Lists
      if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
        flushParagraph();
        const listText = line.trim().substring(2);
        parts.push(
          <li key={`li-${idx}`} className="ml-6 mb-1 text-gray-700">
            {listText.split(/(\*\*.*?\*\*)/g).map((segment, segIdx) => {
              if (segment.startsWith('**') && segment.endsWith('**')) {
                return <strong key={segIdx} className="font-semibold text-gray-900">{segment.slice(2, -2)}</strong>;
              }
              return segment;
            })}
          </li>
        );
        return;
      }

      // Empty line
      if (line.trim() === '') {
        flushParagraph();
        return;
      }

      // Regular text
      currentParagraph.push(line);
    });

    flushParagraph();
    flushCodeBlock();

    return (
      <div className="bg-white rounded-lg p-6 shadow-md">
        <h3 className="text-2xl font-bold text-gray-900 mb-4">{step.title}</h3>
        <div className="prose max-w-none">
          <div className="text-gray-700 leading-relaxed bg-gray-50 p-6 rounded-lg border border-gray-200">
            {parts}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="bg-purple-50 rounded-lg p-4 border-2 border-purple-200 mb-4">
        <h2 className="text-2xl font-bold text-purple-900 mb-2">Maze Solver - Complete Pathfinding Tutorial</h2>
        <p className="text-purple-800">
          A comprehensive step-by-step guide covering Graph Theory, A* Algorithm, Heuristics, and Pathfinding algorithms. 
          Learn how to implement and understand pathfinding from scratch.
        </p>
      </div>

      {/* Tutorial Step Selector */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Select Tutorial Step
        </label>
        <select
          value={selectedStep}
          onChange={(e) => setSelectedStep(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
        >
          <option value="overview">Overview: Pathfinding Algorithms</option>
          <option value="graphTheory">Step 1: Graph Theory Foundations</option>
          <option value="astarAlgorithm">Step 2: A* Algorithm Deep Dive</option>
          <option value="heuristics">Step 3: Heuristics Explained</option>
          <option value="pathfinding">Step 4: Pathfinding Algorithms Comparison</option>
          <option value="implementation">Step 5: Complete Implementation</option>
        </select>
      </div>

      {/* Tutorial Content */}
      {renderTutorialContent()}

      {/* Interactive Maze Solver */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <h3 className="text-lg font-bold text-gray-900 mb-4">Interactive Maze Solver</h3>
        
        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Grid Size
            </label>
            <select
              value={gridSize}
              onChange={(e) => {
                setGridSize(Number(e.target.value));
                initializeMaze();
              }}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            >
              <option value={8}>8x8</option>
              <option value={10}>10x10</option>
              <option value={12}>12x12</option>
              <option value={15}>15x15</option>
              <option value={20}>20x20</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Algorithm
            </label>
            <select
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            >
              <option value="astar">A* (A-Star)</option>
              <option value="bfs">BFS (Breadth-First Search)</option>
              <option value="dfs">DFS (Depth-First Search)</option>
              <option value="dijkstra">Dijkstra</option>
            </select>
          </div>

          <div className="flex items-end">
            <button
              onClick={solveMaze}
              disabled={isAnimating}
              className="w-full px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-semibold"
            >
              {isAnimating ? 'Solving...' : 'Solve Maze'}
            </button>
          </div>
        </div>

        <div className="flex gap-2 mb-4">
          <button
            onClick={initializeMaze}
            className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 font-semibold"
          >
            Generate New Maze
          </button>
        </div>

        {/* Maze Visualization */}
        <div className="flex justify-center">
          {maze && maze.length > 0 ? (
            <div
              className="grid gap-1 border-2 border-gray-800 p-2"
              style={{
                gridTemplateColumns: `repeat(${gridSize}, minmax(0, 1fr))`,
                width: 'fit-content'
              }}
            >
              {Array.from({ length: gridSize }).map((_, row) =>
                Array.from({ length: gridSize }).map((_, col) => (
                  <div
                    key={`${row}-${col}`}
                    className={`w-8 h-8 border border-gray-400 ${getCellColor(row, col)} transition-colors duration-150`}
                    style={{ minWidth: '32px', minHeight: '32px' }}
                  />
                ))
              )}
            </div>
          ) : (
            <div className="text-gray-500">Loading maze...</div>
          )}
        </div>

        {/* Legend */}
        <div className="mt-4 flex flex-wrap gap-4 justify-center text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-500 border border-gray-400"></div>
            <span>Start</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-500 border border-gray-400"></div>
            <span>End</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-gray-800 border border-gray-400"></div>
            <span>Wall</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-blue-300 border border-gray-400"></div>
            <span>Visited</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-yellow-400 border border-gray-400"></div>
            <span>Path</span>
          </div>
        </div>
      </div>

      {/* Algorithm Code Examples */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <h3 className="text-lg font-bold text-gray-900 mb-4">Algorithm Implementation</h3>
        <div className="bg-gray-900 rounded-lg overflow-hidden">
          <SyntaxHighlighter
            language="python"
            style={vscDarkPlus}
            customStyle={{ margin: 0, borderRadius: '0.5rem' }}
            showLineNumbers
          >
            {codeExamples[algorithm] || codeExamples.astar}
          </SyntaxHighlighter>
        </div>
      </div>
    </div>
  );
}
