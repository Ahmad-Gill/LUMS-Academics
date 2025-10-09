# Importing modules and libraries
from maze_visual import maze, agent
import sys
import argparse
from collections import deque
import time
from heapq import heappush, heappop

# Import any other modules you want to use here



# DO NOT CHANGE THESE LINES OF CODE
# ----------------------------------
ROWS = 20 # Number of rows in the maze
COLS = 20 # Number of columns in the maze
m = maze(ROWS, COLS) # Initialize the maze

# Load the maze from the csv file. You may need to change this path depending on where you save the files.
m.LoadMaze(loadMaze='maze_config.csv', theme="dark")

# ----------------------------------

# Helper function 
directions = ['N','W','S','E']
# directions = ['E','S','W','N']
def makeMovemetn(col,dir):
    x,y=col
    if dir == "N" :
        return(x-1,y)
    if dir == "W" :
        return(x,y-1)
    if dir == "S" :
        return(x+1,y)
    if dir == "E" :
        return(x,y+1)

def DFS(m, start, end):
    maze_map = m.maze_map 
    '''
    This function should implement the Depth First Search algorithm.
    The inputs to this function are:
        maze: The maze object
        start: The start position of the agent as a tuple (x,y)
        goal: The goal position of the agent as a tuple (x,y)
    The function should return:
        a list containing all the positions visited by the search algorithm
        a list containing the positions in the final path from the start to the goal
    '''

    visited_positions = []
    path_to_goal = []


    # TODO: Implement Depth-First Search (DFS) algorithm here
    # NOTE: Think sbout what direction you should explore first in the event of multiple options. For the purpose of this assignment, the start 
    #       position is always bottom right and the goal is always top left. Also think about what gets popped first using either stack or recursion.

    stack=[(start,[start])]
    visisted=set([start])
    visisted_list=[start]
    # visisted=set([start])
    while stack:
        cell, path=stack.pop()
        if cell == end:
                return visisted_list, path
        for dir in directions[::-1]:
                if maze_map[cell][dir] == 1:
                    next_cell = makeMovemetn(cell, dir)
                    if next_cell not in visisted:
                        visisted.add(next_cell)
                        visisted_list.append(next_cell)
                        if next_cell in maze_map:
                            stack.append((next_cell, path + [next_cell]))
    return visisted_list, []




def BFS(m, start, end):

    '''
    This function should implement the Breadth First Search algorithm.
    The inputs to this function are:
        maze: The maze object
        start: The start position of the agent as a tuple (x,y)
        goal: The goal position of the agent as a tuple (x,y)
    The function should return:
        a list containing all the positions visited by the search algorithm
        a list containing the positions in the final path from the start to the goal
    '''

    visited_positions = []
    maze_map = m.maze_map 
    path_to_goal = []

    # TODO: Implement Breadth-First Search (BFS) algorithm here

    stack=deque([(start,[start])])
    visisted=set([start])
    visisted_list=[start]
    # visisted=set([start])
    while stack:
        cell, path=stack.popleft()
        if cell == end:
                return visisted_list, path
        for dir in directions:
                if maze_map[cell][dir] == 1:
                    next_cell = makeMovemetn(cell, dir)
                    if next_cell not in visisted:
                        visisted.add(next_cell)
                        visisted_list.append(next_cell)
                        if next_cell in maze_map:
                            stack.append((next_cell, path + [next_cell]))


    return visisted_list, []




def heuristic(position, goal):
    '''
    This function should implement Euclidean Distance as the heuristic function used in A* algorithm.
    The inputs to this function are:
        position: The current position of the agent as a tuple (x,y)
        goal: The goal position of the agent as a tuple (x,y)
    The function should return:
        the heuristic value of the given position
    '''
    x_p,y_p=position
    x_g,y_g=goal
    return (((x_p-x_g)**2)+(y_g-y_p)**2)**0.5

def AStar(m, start, goal):
    maze_map = m.maze_map
    '''
    This function should implement the A* algorithm.
    The inputs to this function are:
        maze: The maze object
        start: The start position of the agent as a tuple (x,y)
        goal: The goal position of the agent as a tuple (x,y)
    The function should return:
        a list containing all the positions visited by the search algorithm
        a list containing the positions in the final path from the start to the goal
    '''


    visited_positions = []                 
    parent = {}                    
    g_n_cost = {start: 0}       
    queue = []   
    maze_map = m.maze_map                    

    heappush(queue, (heuristic(start, goal), start)) 
    while queue:
        f_n_cost, current = heappop(queue) 
        visited_positions.append(current)          
        if current == goal:              
            path = []
            while current in parent:      
                path.append(current)   
                current = parent[current]
            path.insert(0, start)         
            return visited_positions, path[::-1]       
        for dir in directions:
            if maze_map[current][dir] == 1:
                next_cell = makeMovemetn(current, dir)
                new_cost = g_n_cost[current] + 1  
                if next_cell not in g_n_cost or new_cost < g_n_cost[next_cell]:
                    g_n_cost[next_cell] = new_cost
                    parent[next_cell] = current
                    f_n_cost = new_cost + heuristic(next_cell, goal)
                    heappush(queue, (f_n_cost, next_cell))


    # TODO: Implement A* Search algorithm here
    # NOTE: You can assume the cost of moving one step is 1 for this maze
    #       You can use the Euclidean distance as the heuristic function for this assignment


    return visited_positions, []





# DO NOT CHANGE THE LINES OF CODE BELOW
# -------------------------------------
# This part of the code calls the search algorithms implemented above and displays the results on the maze
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bfs", help="Run BFS", action="store_true")
    parser.add_argument("-d", "--dfs", help="Run DFS", action="store_true")
    parser.add_argument("-a", "--astar", help="Run A* Search", action="store_true")

    args = parser.parse_args()

    start = (20, 20)
    goal = (1,1)
    end=goal

    explored, path_to_goal = [], []
    algorithm_name = ""
    if args.bfs:
        start_time = time.time()
        algorithm_name = "Breadth-First Search (BFS)"
        print(f"Running {algorithm_name}...")
        start_time = time.time()
        explored, path_to_goal = BFS(m, start, goal)
        end_time = time.time()
    elif args.dfs:
        start_time = time.time()
        algorithm_name = "Depth-First Search (DFS)"
        print(f"Running {algorithm_name}...")
        start_time = time.time()
        explored, path_to_goal = DFS(m, start, goal)
        end_time = time.time()
    elif args.astar:
        start_time = time.time()
        algorithm_name = "A* Search"
        print(f"Running {algorithm_name}...")
        start_time = time.time()
        explored, path_to_goal = AStar(m, start, goal)
        end_time = time.time()
    else:
        print("No search algorithm specified. See help below.")
        parser.print_help()
        sys.exit()
    
    # --- Statistics Calculation and Printing ---
    if start_time > 0:
        execution_time = end_time - start_time
        path_length = len(path_to_goal)
        nodes_explored = len(explored)

        print("\n--- Search Algorithm Statistics ---")
        print(f"Algorithm: {algorithm_name}")
        print(f"Execution Time: {execution_time:.4f} seconds")
        print(f"Path Length: {path_length} steps")
        print(f"Nodes Explored: {nodes_explored} nodes")
        print("-------------------------------------\n")
    # -------------------------------------------

    # If a path was found, start visualization
    if path_to_goal:
        print("Starting visualization...")
        a = agent(m, ROWS, COLS, filled=True)
        b = agent(m, ROWS, COLS, color="red")

        m.tracePath({a: explored}, delay=20)
        m.tracePath({b: path_to_goal}, delay=50)

        m.run()
    else:
        print("No path to the goal was found!")


if __name__ == "__main__":
    main()