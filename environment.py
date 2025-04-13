import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import pickle

class GridWorldEnv(gym.Env):
    def __init__(self):
        """
        Initialises the grid-world environment.
        """
        super().__init__()
        self.grid_size = 13  # size of the grid (13x13)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2 * self.grid_size ** 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # four possible actions: north, south, east, west
        self.state = None  # initial state of the agent
        self.goal_state = None 
        self.step_count = 0  # Track steps
        self.max_steps = 100  # Reset after 100 steps

        string_input = True
        if string_input:
            # Convert the maze structure into nodes
            self.string_representation = [
                "A1-A2", "A3-A4", "A4-A5", "A5-A6", "A6-A7", "A2-B2", "A3-B3",
                "A5-B5", "A7-B7", "B4-B5", "B6-B7", "B1-C1", "B2-C2", "B3-C3",
                "B6-C6", "C1-C2", "C2-C3", "C3-C4", "C4-C5", "C5-C6", "C6-C7",
                "C2-D2", "C5-D5", "C7-D7", "D1-D2", "D3-D4", "D4-D5", "D6-D7",
                "D1-E1", "D2-E2", "D3-E3", "D4-E4", "D5-E5", "D6-E6", "E2-F2",
                "E3-F3", "E5-F5", "E6-F6", "E7-F7", "F1-F2", "F2-F3", "F4-F5",
                "F6-F7", "F2-G2", "F5-G5", "F6-G6", "G1-G2", "G2-G3", "G3-G4",
                "G4-G5", "G5-G6", "G6-G7"
            ]

            self.listofcoords, self.towers_index, self.bridge_index = maze_structure(self.string_representation, self.grid_size)

            # Compute obstacles
            self.free_spaces = self.towers_index | self.bridge_index
            self.obstacles = set(range(self.grid_size ** 2)) - self.free_spaces

            #store dictionary of converting index back to string
            self.string_dictionary = index_to_string(self.towers_index, self.bridge_index, self.grid_size)

        else:
            self.obstacles = set()
            for _ in range(7):
                self.obstacles.add(random.randint(0, self.grid_size ** 2 - 1))
            self.free_spaces = set(range(self.grid_size ** 2)) - self.obstacles
            
    def reset(self, start_state=None, goal_state=None, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_count = 0  # Reset step count 

        if start_state is not None:
            self.state = start_state
        else:
            while True:
                self.state = random.randint(0, self.grid_size ** 2 - 1)  # random starting position
                if self.state not in self.obstacles:
                    break
        
        if goal_state is not None:
            self.goal_state = goal_state
        else:
            while True:
                self.goal_state = random.randint(0, self.grid_size ** 2 - 1)  # random goal position
                if self.goal_state != self.state and self.goal_state not in self.obstacles and self.goal_state not in self.bridge_index:
                    break
        
        return self._get_observation(), {}
    
    def step(self, action):
        self.step_count += 1  # Increment step count
        row, col = divmod(self.state, self.grid_size)  # compute row and column from state index

        # move based on the action
        if action == 0 and row > 0:  # north
            row -= 1
        elif action == 1 and row < self.grid_size - 1:  # south
            row += 1
        elif action == 2 and col < self.grid_size - 1:  # east
            col += 1
        elif action == 3 and col > 0:  # west
            col -= 1

        new_state = row * self.grid_size + col  # compute new state index

        if new_state in self.obstacles:
            new_state = self.state  # stay in the same position if moving into an obstacle

        reward = 0 if new_state == self.goal_state else -1  # reward is 1 if goal is reached
        
        terminated = new_state == self.goal_state  # True if goal is reached
        truncated = self.step_count >= self.max_steps  # True if max steps exceeded
        
        self.state = new_state  # update agent's state
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _get_observation(self):
        """
        Construct the observation as a concatenation of one-hot encoded state and goal.
        """
        state_one_hot = one_hot_encode(self.state, self.grid_size)
        goal_one_hot = one_hot_encode(self.goal_state, self.grid_size)
        return np.concatenate((state_one_hot, goal_one_hot))


def one_hot_encode(state, grid_size):
    """
    Convert a state index into a one-hot encoded vector.

    Args:
        state (int): The index representing the state within the grid.
        grid_size (int): The size of one dimension of the grid (grid is assumed to be square).

    Returns:
        np.ndarray: A one-hot encoded vector of length grid_size^2 with a 1 at the index
        corresponding to the state, and 0s elsewhere.
    """
    vec = np.zeros(grid_size ** 2)
    vec[state] = 1
    return vec
    

def maze_structure(string_representation, grid_size):
    letter_dictionary = {chr(65 + i): i * 2 for i in range(7)}  # {'A': 0, 'B': 2, ..., 'G': 12} - skipping 1 for bridges

    def coordinates(pair): 
        #Convert string representation to (x, y) coordinates.
        letter = pair[0] 
        num = int(pair[1])
        x = letter_dictionary[letter]  # Column (x-axis)
        y = 2 * (num - 1)              # Row (y-axis)
        return (x, y)

    def coord_to_index(x, y):
        return x + y * grid_size 

    coords = []  # List of (tower1, tower2, bridge) coordinates
    towers_index = set()  # Unique indices for towers
    bridge_index = set()  # Unique indices for bridges

    for connection in string_representation:
        tower1, tower2 = connection.split('-')
        tower1_coord = coordinates(tower1)
        tower2_coord = coordinates(tower2)

        # Compute bridge coordinate (midpoint)
        bridge_x = (tower1_coord[0] + tower2_coord[0]) // 2
        bridge_y = (tower1_coord[1] + tower2_coord[1]) // 2
        bridge_coord = (bridge_x, bridge_y)

        # Store coordinates
        coords.append((tower1_coord, tower2_coord, bridge_coord))

        # Convert coordinates to indices and store in sets
        towers_index.add(coord_to_index(*tower1_coord))
        towers_index.add(coord_to_index(*tower2_coord))
        bridge_index.add(coord_to_index(*bridge_coord))

    return coords, towers_index, bridge_index


def index_to_string(towers_index, bridge_index, grid_size):
    letter_dictionary = {i * 2: chr(65 + i) for i in range(7)}  # {0: 'A', 2: 'B', ..., 12: 'G'}
    
    def index_to_coordinates(index):
        x = index % grid_size
        y = index // grid_size
        return (x, y)
    
    def coordinates_to_pair(x, y):
        letter = letter_dictionary[x]
        num = y // 2 + 1
        return f"{letter}{num}"
    
    states_dict = {} 
    for index in sorted(towers_index):
        x, y = index_to_coordinates(index)
        states_dict[index] = coordinates_to_pair(x, y)
    
    for index in sorted(bridge_index):
        x, y = index_to_coordinates(index)
        bridge_representation = f"{coordinates_to_pair(x - 1, y)}-{coordinates_to_pair(x + 1, y)}" if x % 2 == 1 else f"{coordinates_to_pair(x, y - 1)}-{coordinates_to_pair(x, y + 1)}"
        states_dict[index] = bridge_representation
    
    return states_dict

### DON'T NEED IF STRING_INPUT = TRUE
def save_environment_config(env, filename="env_config.pkl"):
    config = {
        'obstacles': env.obstacles,
        'free_spaces': env.free_spaces,
        'state': env.state,
        'goal_state': env.goal_state,
        'grid_size': env.grid_size
    }

    with open(filename, 'wb') as f:
        pickle.dump(config, f)
    print("Environment configuration saved.")


def load_environment_config(filename="env_config.pkl"):
    with open(filename, 'rb') as f:
        config = pickle.load(f)

    # Create a new environment object
    env = GridWorldEnv()

    # Restore the saved configuration
    env.obstacles = config['obstacles']
    env.free_spaces = config['free_spaces']
    env.state = config['state']
    env.goal_state = config['goal_state']
    env.grid_size = config['grid_size']

    print("Environment configuration loaded.")
    return env