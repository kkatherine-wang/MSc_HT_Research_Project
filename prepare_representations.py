import pandas as pd
import numpy as np
import torch
import networkx as nx

from environment import *
from visualise import *
from environment import *
from peters_representations import *


def convert_action_to_direction(action_index):
    if action_index == 0:
         action = "N"
    if action_index == 1:
         action = "S"
    if action_index == 2:
         action = "E"
    if action_index == 3:
         action = "W"
    return action

##get shortest path distances
def get_all_distances(simple_maze):
    extended_maze = get_extended_simple_maze(simple_maze)
    coord2label = get_maze_coord2label(extended_maze)
    all_path_lengths = dict(nx.all_pairs_dijkstra_path_length(extended_maze))
    # translate to alphaneumeritc
    return {coord2label[k]: {coord2label[_k]: _v for _k, _v in v.items()} for k, v in all_path_lengths.items()}


def extract_layer_output(policy, obs, layer_name):
    """Extracts the output of a specific layer for a given observation."""
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
   
    with torch.no_grad():
        latent = policy.mlp_extractor(obs_tensor)

        if layer_name == "policy_activations":
            return latent[0].cpu().numpy() 
        elif layer_name == "value_activations":
            return latent[1].cpu().numpy()
        elif layer_name == "value":
            return policy.value_net(latent[1]).cpu().numpy() #latent 1 is used for value
        elif layer_name == "action":
            logits = policy.action_net(latent[0])  # Get raw logits
            probs = torch.softmax(logits, dim=-1)  # Convert to probabilities
            return probs.cpu().numpy()
            #return policy.action_net(latent[0]).cpu().numpy() #latent 0 is used for action
        else:
            raise ValueError("Invalid layer name.")


def store_all_activations(env, model):
    data = []
    sorted_tower_spaces = sorted(env.towers_index)
    maze_graph = simple_maze(env.string_representation)
    all_distances = get_all_distances(maze_graph)

    for goal in sorted_tower_spaces:
        for current in env.free_spaces: 
            if current == goal:  
                continue  
            obs, _ = env.reset(start_state=current, goal_state=goal)
            policy_activations = extract_layer_output(model.policy, obs, "policy_activations")
            value_activations = extract_layer_output(model.policy, obs, "value_activations")
            value_output = extract_layer_output(model.policy, obs, "value")
            action_output = extract_layer_output(model.policy, obs, "action")
            
            unit_policy_activation = policy_activations.flatten().tolist()
            unit_value_activations = value_activations.flatten().tolist()
            action_probabilities = action_output.flatten().tolist()

            #row = {("current", ""): current, ("goal", ""): goal}

            # Convert state indices to string representations
            string_current = env.string_dictionary.get(current, str(current))
            string_goal = env.string_dictionary.get(goal, str(goal))

            #store current state and goal state
            row = {("current", "index"): current, ("current", "string"): string_current, ("goal", "index"): goal, ("goal", "string"): string_goal}

            #calculate distances 
            #   euclidean 
            row_current, col_current = divmod(current, env.grid_size)
            row_goal, col_goal = divmod(goal, env.grid_size)
            euclidean_distance = np.sqrt((row_current - row_goal) ** 2 + (col_current - col_goal) ** 2)

            #   shortest_path_to_goal
            shortest_path_to_goal = all_distances[string_current][string_goal]
            row.update({("distances", "euclidean_distance"): euclidean_distance, ("distances", "shortest_path"): shortest_path_to_goal})

 
            for index, activation_value in enumerate(unit_policy_activation):
                row[("policy_activations", f"unit {index}")] = activation_value

            for index, activation_value in enumerate(unit_value_activations):
                row[("value_activations", f"unit {index}")] = activation_value

            row.update({("value_output", ""): value_output.item()})
            
            for index, probability in enumerate(action_probabilities):
                action = convert_action_to_direction(index)
                row[("action_probabilities", f"{action}")] = probability

            data.append(row)

    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.to_csv("store_all_activations.csv", index=False)
    return df

def store_policy_timestep_activations(env, model, num_repeats):
    data = []
    current_episode = 0
    maze_graph = simple_maze(env.string_representation)
    all_distances = get_all_distances(maze_graph)

    for _ in range(num_repeats):
        random_tower_goals = random.sample(sorted(env.towers_index), len(env.towers_index))  # Shuffle each time
    
        for goal in random_tower_goals:
            obs, _ = env.reset(goal_state=goal)
            goal_state = goal
            step_count = 0 
            
            while step_count < 30:
                prev_state = env.state  # Store the previous state before action
                action, _ = model.predict(obs, deterministic=False)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Skip storing data if the agent hit an obstacle and didn't move
                if env.state == prev_state:
                    step_count += 1
                    continue

                # Convert state indices to string representations
                string_current = env.string_dictionary.get(env.state, str(env.state))
                string_goal = env.string_dictionary.get(goal_state, str(goal_state))

                #store current state and goal state
                row = {("episode",""): current_episode, ("timestep",""): step_count, ("current", ""): string_current, ("goal", ""): string_goal}

                letter_action = convert_action_to_direction(action)

                row.update({("action",""):letter_action})

                #calculate distances 
                #   euclidean 
                row_current, col_current = divmod(env.state, env.grid_size)
                row_goal, col_goal = divmod(goal_state, env.grid_size)
                euclidean_distance = np.sqrt((row_current - row_goal) ** 2 + (col_current - col_goal) ** 2)

                #   shortest_path_to_goal
                current_label = env.string_dictionary.get(env.state, str(env.state))
                goal_label = env.string_dictionary.get(goal_state, str(goal_state))
                
                shortest_path_to_goal = all_distances[current_label][goal_label]
                row.update({("distances", "euclidean_distance"): euclidean_distance, ("distances", "shortest_path"): shortest_path_to_goal})

                policy_activations = extract_layer_output(model.policy, obs, "policy_activations")
                value_output = extract_layer_output(model.policy, obs, "value")
                action_output = extract_layer_output(model.policy, obs, "action")
                
                unit_policy_activation = policy_activations.flatten().tolist()
                action_probabilities = action_output.flatten().tolist()
    
                for index, activation_value in enumerate(unit_policy_activation):
                    row[("activations", f"unit {index}")] = activation_value
                
                row.update({("value_output", ""): value_output.item()})
            
                for index, probability in enumerate(action_probabilities):
                    action = convert_action_to_direction(index)
                    row[("action_probabilities", f"{action}")] = probability

                data.append(row)
                step_count += 1

                if env.state == goal_state or done:
                    break

            current_episode += 1
    
    dataframe = pd.DataFrame(data)
    dataframe.columns = pd.MultiIndex.from_tuples(dataframe.columns)
    dataframe.to_csv("store_policy_timestep_activations.csv", index=False)
    return dataframe

def store_value_timestep_activations(env, model, num_repeats):
    data = []
    current_episode = 0
    maze_graph = simple_maze(env.string_representation)
    all_distances = get_all_distances(maze_graph)

    for _ in range(num_repeats):
        random_tower_goals = random.sample(sorted(env.towers_index), len(env.towers_index))  # Shuffle each time
    
        for goal in random_tower_goals:
            obs, _ = env.reset(goal_state=goal)
            goal_state = goal
            step_count = 0 
            
            while step_count < 30:
                prev_state = env.state  # Store the previous state before action
                action, _ = model.predict(obs, deterministic=False)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Skip storing data if the agent hit an obstacle and didn't move
                if env.state == prev_state:
                    step_count += 1
                    continue

                # Convert state indices to string representations
                string_current = env.string_dictionary.get(env.state, str(env.state))
                string_goal = env.string_dictionary.get(goal_state, str(goal_state))

                #store current state and goal state
                row = {("episode",""): current_episode, ("timestep",""): step_count, ("current", ""): string_current, ("goal", ""): string_goal}

                letter_action = convert_action_to_direction(action)

                row.update({("action",""):letter_action})

                #calculate distances 
                #   euclidean 
                row_current, col_current = divmod(env.state, env.grid_size)
                row_goal, col_goal = divmod(goal_state, env.grid_size)
                euclidean_distance = np.sqrt((row_current - row_goal) ** 2 + (col_current - col_goal) ** 2)

                #   shortest_path_to_goal
                shortest_path_to_goal = all_distances[string_current][string_goal]
                row.update({("distances", "euclidean_distance"): euclidean_distance, ("distances", "shortest_path"): shortest_path_to_goal})

                value_activations = extract_layer_output(model.policy, obs, "value_activations")
                value_output = extract_layer_output(model.policy, obs, "value")
                action_output = extract_layer_output(model.policy, obs, "action")
                
                unit_value_activations = value_activations.flatten().tolist()
                action_probabilities = action_output.flatten().tolist()

                for index, activation_value in enumerate(unit_value_activations):
                    row[("activations", f"unit {index}")] = activation_value
                
                row.update({("value_output", ""): value_output.item()})
            
                for index, probability in enumerate(action_probabilities):
                    action = convert_action_to_direction(index)
                    row[("action_probabilities", f"{action}")] = probability

                data.append(row)
                step_count += 1

                if env.state == goal_state or done:
                    break

            current_episode += 1
    
    dataframe = pd.DataFrame(data)
    dataframe.columns = pd.MultiIndex.from_tuples(dataframe.columns)
    dataframe.to_csv("store_value_timestep_activations.csv", index=False)
    return dataframe


#### FUNCTIONS FOR SHARED NETWORK

def extract_shared_PPO_layer_output(policy, obs, layer_name):
    """Extracts the output of a specific layer for a given observation."""
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Ensure batch dimension
   
    with torch.no_grad():
        shared_output, shared_head_output = policy.mlp_extractor(obs_tensor)[:2]  
        # Extract first two shared layers

        if layer_name == "shared":
            return shared_output.cpu().numpy()  # First shared layer output
        elif layer_name == "shared_head":
            return shared_head_output.cpu().numpy()  # Second shared layer output
        elif layer_name == "value":
            return policy.value_net(shared_head_output).cpu().numpy()  # Value network output
        elif layer_name == "action":
            #return policy.action_net(shared_head_output).cpu().numpy()  # Action network (policy) output
            logits = policy.action_net(shared_head_output)
            # Apply softmax to convert logits to probabilities
            action_probs = torch.softmax(logits, dim=-1)
            return action_probs.cpu().numpy()
        else:
            raise ValueError("Invalid layer name. Choose 'shared', 'shared_head', 'value', or 'action'.")


def store_shared_PPO_all_activations(env, model):
    data = []
    sorted_tower_spaces = sorted(env.towers_index)
    maze_graph = simple_maze(env.string_representation)
    all_distances = get_all_distances(maze_graph)
    
    for goal in sorted_tower_spaces:
        for current in env.free_spaces: 
            if current == goal:  
                continue  
            obs, _ = env.reset(start_state=current, goal_state=goal)
            shared_output = extract_shared_PPO_layer_output(model.policy, obs, "shared_head")
            value_output = extract_shared_PPO_layer_output(model.policy, obs, "value")
            action_output = extract_shared_PPO_layer_output(model.policy, obs, "action")
            
            unit_activation = shared_output.flatten().tolist()
            action_probabilities = action_output.flatten().tolist()

            # Convert state indices to string representations
            string_current = env.string_dictionary.get(current, str(current))
            string_goal = env.string_dictionary.get(goal, str(goal))

            #store current state and goal state
            row = {("current", "index"): current, ("current", "string"): string_current, ("goal", "index"): goal, ("goal", "string"): string_goal}

            #calculate distances 
            #   euclidean 
            row_current, col_current = divmod(current, env.grid_size)
            row_goal, col_goal = divmod(goal, env.grid_size)
            euclidean_distance = np.sqrt((row_current - row_goal) ** 2 + (col_current - col_goal) ** 2)

            #   shortest_path_to_goal
            shortest_path_to_goal = all_distances[string_current][string_goal]
            row.update({("distances", "euclidean_distance"): euclidean_distance, ("distances", "shortest_path"): shortest_path_to_goal})

 
            for index, activation_value in enumerate(unit_activation):
                row[("activations", f"unit {index}")] = activation_value

            row.update({("value_output", ""): value_output.item()})
            
            for index, probability in enumerate(action_probabilities):
                action = convert_action_to_direction(index)
                row[("action_probabilities", f"{action}")] = probability

            data.append(row)

    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.to_csv("store_shared_PPO_all_activations.csv", index=False)
    return df

def store_shared_PPO_timestep_activations(env, model, num_repeats):
    data = []
    current_episode = 0
    maze_graph = simple_maze(env.string_representation)
    all_distances = get_all_distances(maze_graph)

    for _ in range(num_repeats):
        random_tower_goals = random.sample(sorted(env.towers_index), len(env.towers_index))  # Shuffle each time
    
        for goal in random_tower_goals:
            obs, _ = env.reset(goal_state=goal)
            goal_state = goal
            step_count = 0 
            
            while step_count < 30:
                prev_state = env.state  # Store the previous state before action
                action, _ = model.predict(obs, deterministic=False)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Skip storing data if the agent hit an obstacle and didn't move
                if env.state == prev_state:
                    step_count += 1
                    continue

                # Convert state indices to string representations
                string_current = env.string_dictionary.get(env.state, str(env.state))
                string_goal = env.string_dictionary.get(goal_state, str(goal_state))

                #store current state and goal state
                row = {("episode",""): current_episode, ("timestep",""): step_count, ("current", ""): string_current, ("goal", ""): string_goal}

                letter_action = convert_action_to_direction(action)

                row.update({("action",""):letter_action})

                #calculate distances 
                #   euclidean 
                row_current, col_current = divmod(env.state, env.grid_size)
                row_goal, col_goal = divmod(goal_state, env.grid_size)
                euclidean_distance = np.sqrt((row_current - row_goal) ** 2 + (col_current - col_goal) ** 2)

                #   shortest_path_to_goal
    
                shortest_path_to_goal = all_distances[string_current][string_goal]
                row.update({("distances", "euclidean_distance"): euclidean_distance, ("distances", "shortest_path"): shortest_path_to_goal})

                #extract activations
                shared_output = extract_shared_PPO_layer_output(model.policy, obs, "shared_head")
                value_output = extract_shared_PPO_layer_output(model.policy, obs, "value")
                action_output = extract_shared_PPO_layer_output(model.policy, obs, "action")
                
                unit_activation = shared_output.flatten().tolist()
                action_probabilities = action_output.flatten().tolist()
                
                for index, activation_value in enumerate(unit_activation):
                    row[("activations", f"unit {index}")] = activation_value
                
                row.update({("value_output", ""): value_output.item()})

                for index, probability in enumerate(action_probabilities):
                    action = convert_action_to_direction(index)
                    row[("action_probabilities", f"{action}")] = probability

                data.append(row)
                step_count += 1

                if env.state == goal_state or done:
                    break

            current_episode += 1
    
    dataframe = pd.DataFrame(data)
    dataframe.columns = pd.MultiIndex.from_tuples(dataframe.columns)
    dataframe.to_csv("store_shared_PPO_timestep_activations.csv", index=False)
    return dataframe

