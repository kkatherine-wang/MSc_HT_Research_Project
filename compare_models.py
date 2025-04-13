from visualise import *
from shared_ppo_architecture import *
from environment import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
        def __init__(self, eval_env, model_name, n_eval_episodes=50, eval_freq=5000000):
            super().__init__()
            self.eval_env = eval_env
            self.model_name = model_name
            self.episode_rewards = []
            self.total_episode_reward = 0 
            self.episode_count = 0
            self.n_eval_episodes = n_eval_episodes
            self.eval_freq = eval_freq
        
        def _on_step(self) -> bool:
            if "rewards" in self.locals:
                # Increment the total reward for the current episode
                self.total_episode_reward += self.locals["rewards"].item()  # Convert numpy array to scalar
            
            done = self.locals["dones"]
            if done:
                self.episode_rewards.append(self.total_episode_reward)
                self.episode_count += 1

            #uncomment below if want to visualise route throughout points in training
            """if self.n_calls % self.eval_freq == 0:
                self.visualize_route()"""

            return True
        
        def visualize_route(self):
            obs, _ = self.eval_env.reset()
            goal_state = self.eval_env.goal_state
            grid_size = self.eval_env.grid_size

            route = [self.eval_env.state]
            step_count = 0

            while step_count < 30:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated

                route.append(self.eval_env.state)
                step_count += 1

                if self.eval_env.state == goal_state or done:
                    break

            grid = np.zeros((grid_size, grid_size))
            for idx, pos in enumerate(route):
                row, col = divmod(pos, grid_size)
                grid[row, col] = idx + 1

            for obs in self.eval_env.obstacles:
                row, col = divmod(obs, grid_size)
                grid[row, col] = -2

            goal_row, goal_col = divmod(goal_state, grid_size)

            plt.figure(figsize=(5, 5))
            plt.imshow(grid, cmap="viridis", origin="lower",vmin=-2)
            plt.colorbar(label="Step Order")
            
            # Label steps
            for i, (row, col) in enumerate([divmod(pos, grid_size) for pos in route]):
                plt.text(col, row, str(i), ha="center", va="center", color="white")

            plt.scatter(goal_col, goal_row, color="red", marker="*", s=100, label="Goal")

            plt.title(f"{self.model_name} - Agent's Route to Goal (Timestep {self.n_calls})")
            plt.xticks(range(grid_size))
            plt.yticks(range(grid_size))
            plt.savefig(f"{self.model_name}_route_{self.n_calls}.png", dpi=300)
            plt.show()

def train_and_evaluate(agent_class, env, model_name, hyperparams, timesteps):
    model = agent_class("MlpPolicy", env, **hyperparams, verbose=0)
    callback = CustomCallback(env, model_name)
    model.learn(total_timesteps=timesteps, progress_bar=True, callback=callback)
    return model, callback.episode_rewards, callback.episode_count

def compare(env, hyperparameter_sets, num_runs):
    trained_models = {}
    data = []

    # Run training for different agent types
    for name, (agent_class, params) in hyperparameter_sets.items():
        print(f"Training {name}...")
        
        for run in range(num_runs):
            model, episode_rewards, total_episode_count = train_and_evaluate(
                agent_class,
                env,
                name,
                params,
                timesteps=10000000
            )
            
            # Save the model
            model.save(f"{name}_run_{run}.zip")

            # Store results
            for episode, reward in enumerate(episode_rewards):
                if len(data) <= episode:
                    data.append({("episode", ""): episode})
                data[episode][(f"{name}", f"run_{run}")] = reward
            
            print(episode_rewards)

        trained_models[name] = model

    # After collecting all the data, create the DataFrame
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.to_csv("episode_reward_values.csv", index=False)
    return df, trained_models


##WEIGHT DECAY COMPARIONS
def training_weight_decay(env, policy_kwargs):   
    model = PPO("MlpPolicy", env, learning_rate=1e-3, gamma=0.99, n_steps=2048, verbose=0, policy_kwargs=policy_kwargs)
    callback = CustomCallback(env, model_name="PPO_with_Weight_Decay")
    model.learn(total_timesteps=20000000, progress_bar=True, callback=callback)
    return model, callback.episode_rewards, callback.episode_count


def compare_weight_decay(env, weight_decay_list, num_runs):
    trained_models = {}
    data = []

    # Run training for different agent types
    for weight_decay in weight_decay_list:
        policy_kwargs = dict(
            activation_fn=nn.ReLU,
            optimizer_kwargs=dict(weight_decay=weight_decay)  ##L2 
        )
        print(f"Training with weight decay: {weight_decay}")
        
        for run in range(num_runs):
            model, episode_rewards, total_episode_count = training_weight_decay(env, policy_kwargs)

            model.save(f"PPO_{weight_decay}_{run}.zip")
                
            # Store results
            for episode, reward in enumerate(episode_rewards):
                if len(data) <= episode:
                    data.append({("episode", ""): episode})
                data[episode][(f"PPO_{weight_decay}", f"run_{run}")] = reward

            trained_models[weight_decay] = model

    # After collecting all the data, create the DataFrame
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.to_csv("episode_reward_values_weight_decay.csv", index=False)
    return df, trained_models

def rolling_mean(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def compare_runs_for_weight_decay(dataframe, weight_decay, max_episodes, window_size):
    plt.figure(figsize=(10, 6))
    for run_num in range(5):
        run = dataframe[f"{weight_decay}", f"run_{run_num}"]
        run = run[:max_episodes]
        if len(run) >= window_size:
            smoothed_stable = rolling_mean(run, window_size)
            plt.plot(smoothed_stable, label=f"Run {run_num}")

    # Labels and title
    plt.xlabel("Training Episodes")
    plt.ylabel(f"Rolling Mean Reward ({window_size} Episodes)")
    plt.title(f"Comparing Learning in Different Runs of PPO Models with Weight-Decay 1e-7")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid()
    plt.tight_layout()
    plt.show()


## COMPARING SHARED PPO HYPERPARAMETERS
def train_and_evaluate_shared(agent_class, env, model_name, hyperparams, timesteps):
    """Train an agent with given hyperparameters and return the model,
    reward curve, and training time.
    """
    model = agent_class(CustomActorCriticPolicy, env, **hyperparams, verbose=0)
    callback = CustomCallback(env, model_name)
    model.learn(total_timesteps=timesteps, progress_bar=True, callback=callback)

    return model, callback.episode_rewards, callback.episode_count


def compare_shared(env, hyperparameter_sets, num_runs):
    trained_models = {}
    training_times = {}
    data = []

    # Run training for different agent types
    for name, (agent_class, params) in hyperparameter_sets.items():
        print(f"Training {name}...")
        
        for run in range(num_runs):
            model, episode_rewards, total_episode_count = train_and_evaluate_shared(
                agent_class,
                env,
                name,
                params,
                timesteps=25000000
            )
            
            # Save the model
            model.save(f"{name}_run_{run}.zip")

            # Store results
            for episode, reward in enumerate(episode_rewards):
                if len(data) <= episode:
                    data.append({("episode", ""): episode})
                data[episode][(f"{name}", f"run_{run}")] = reward
            
            print(episode_rewards)

        trained_models[name] = model

    # After collecting all the data, create the DataFrame
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.to_csv("episode_reward_values_shared.csv", index=False)
    return df, training_times, trained_models

def compare_shared_runs(dataframe, model_name, max_episodes, window_size):
    plt.figure(figsize=(10, 6))
    for run_num in range(5):
        run = dataframe[f"{model_name}", f"run_{run_num}"]
        run = run[:max_episodes]
        if len(run) >= window_size:
            smoothed_stable = rolling_mean(run, window_size)
            plt.plot(smoothed_stable, label=f"Run {run_num}")

    # Labels and title
    plt.xlabel("Training Episodes")
    plt.ylabel(f"Rolling Mean Reward ({window_size} Episodes)")
    plt.title(f"Comparing Learning in Different Runs of PPO Model (LR=1e-4, vf = 0.25)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid()
    plt.tight_layout()
    plt.show()


## VISUALISING COMPARISONS ACROSS MODELS
def plot_all_models(dataframe, models_list, plot_title, window=1000, max_episodes = 400000):
    plt.style.use("ggplot")
    plt.style.use("seaborn-v0_8-white")
    plt.style.use("seaborn-v0_8-paper")
    plt.figure(figsize=(10, 6))
    
    for model_name in models_list:
        # Select all columns corresponding to the model's runs
        model_columns = [col for col in dataframe.columns if col[0] == model_name]

        # Extract episode numbers
        episodes = dataframe[("episode", "Unnamed: 0_level_1")]

        # Filter for max_episodes
        mask = episodes <= max_episodes
        episodes = episodes[mask]
        
        # Compute mean and standard deviation across runs for each episode
        means = dataframe[model_columns].mean(axis=1)[mask]
        stds = dataframe[model_columns].std(axis=1)[mask]
        N = dataframe[model_columns].count(axis=1)[mask]
        standard_errors = stds / np.sqrt(N)

        # Apply rolling window
        rolling_mean = means.rolling(window=window, min_periods=1).mean()
        rolling_se = standard_errors.rolling(window=window, min_periods=1).mean()

        # Plot rolling mean
        plt.plot(episodes, rolling_mean, label=f"{model_name}")
        plt.fill_between(episodes, rolling_mean - rolling_se, rolling_mean + rolling_se, alpha=0.2)

    plt.xlabel("Episodes")
    plt.ylabel("Rolling Mean Reward (1000 episodes)")
    plt.title(f"{plot_title}")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()






