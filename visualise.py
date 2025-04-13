from environment import *
from prepare_representations import *
from peters_representations import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import peters_plotting as mp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import spearmanr
from tqdm import tqdm

plt.style.use("ggplot")
plt.style.use("seaborn-v0_8-white")
plt.style.use("seaborn-v0_8-paper")

def all_plots(env, dataframe, distance, interested_units, num_components):
    #checks
    distance_value_check(dataframe, distance)
    linear_regression_for_values(dataframe)

    #place-direction
    place_direction_heatmap(env, dataframe, interested_units)
    plot_component_place_direction_heatmap(env, dataframe, num_components)

    #distance-to-goal tuning
    distance_tuning_curves(dataframe, distance, interested_units)
    heatmap_for_distances(dataframe, distance)


## TRAIN ROUTE
def visualize_trained_route(env, model):
    obs, _ = env.reset()
    goal_state = env.goal_state
    grid_size = env.grid_size

    route = [env.state]
    step_count = 0

    while step_count < 30:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        route.append(env.state)
        step_count += 1

        if env.state == goal_state or done:
            break

    grid = np.zeros((grid_size, grid_size))
    for idx, pos in enumerate(route):
        row, col = divmod(pos, grid_size)
        grid[row, col] = idx + 1

    for obs in env.obstacles:
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

    plt.title("Agent's Route to Goal")
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    plt.show()

### CHECK VALUE
def distance_value_check(dataframe, distance):
    plt.figure(figsize=(8, 5))

    distance_value = (dataframe.groupby([("distances", f"{distance}")])
                    .value_output.mean())
    
    distance_value = distance_value[distance_value.index <= 24]
    plt.plot(distance_value.index, distance_value.values, linestyle='-')

    # Labels, title, and legend
    plt.xlabel("Distance")
    plt.ylabel("Value Output")
    plt.title(f"Value Check")
    plt.xticks(range(int(min(distance_value.index)), int(max(distance_value.index)) + 1))  # Ensure integer x-ticks
    plt.show()

### PLACE DIRECTION PLOT
def place_direction_heatmap(env, dataframe, interested_units):
    for unit in interested_units:
        fig, ax = plt.subplots()
        values = (dataframe.groupby([("current", "Unnamed: 2_level_1"), ("action", "Unnamed: 4_level_1")]).activations.mean().activations[f"unit {unit}"])
        
        maze_graph = simple_maze(env.string_representation)
        mp.plot_directed_heatmap(maze_graph, values, ax=ax)

        ax.set_aspect('equal')

        # Display the plot
        plt.title(f"Unit {unit}")
        plt.show()


### Component Heatmaps
NMF_KWARGS = {
    "init": "random",
    "random_state": 0,
    "solver": "mu",
    "beta_loss": "kullback-leibler",
    "max_iter": 1000,
}

def get_nmf_df(place_direction_df, n_components):
    model = NMF(n_components=n_components, **NMF_KWARGS)

    data_matrix = place_direction_df.to_numpy()
    data_matrix = data_matrix.T
    
    decomp_components = model.fit(data_matrix).components_

    nmf_df = pd.DataFrame(
        data=decomp_components.T, 
        index=place_direction_df.index, 
        columns=range(n_components)
    )
    
    return nmf_df

def plot_component_place_direction_heatmap(env, dataframe, num_components):
    # Create heatmap dictionary
    heatmap_dict = {}
    for unit in range(64):
        values = dataframe.groupby([("current", "Unnamed: 2_level_1"), ("action", "Unnamed: 4_level_1")]).activations.mean().activations[f"unit {unit}"]
        heatmap_dict[f"unit {unit}"] = values

    # Create Heatmap Dataframe
    heatmap_df = pd.DataFrame(heatmap_dict)

    # Perform NMF
    nmf_df = get_nmf_df(heatmap_df, n_components=num_components)
    nmf_df.to_csv("nmf_df.csv", index=True)
    nmf_df = pd.read_csv("nmf_df.csv")
    
    nmf_df.columns = [eval(col) if isinstance(col, str) and col.startswith('(') else col for col in nmf_df.columns]

    plot_compontents = range(num_components)

    for component in plot_compontents:
        values = nmf_df.groupby([('current', 'Unnamed: 2_level_1'), ('action', 'Unnamed: 4_level_1')])[f'{component}'].mean()
        values = values.reset_index()
        values.columns = ['maze_position', 'direction', 'value']
        values = values.set_index(['maze_position', 'direction'])['value']

        maze_graph = simple_maze(env.string_representation)

        fig, ax = plt.subplots()
        mp.plot_directed_heatmap(maze_graph, values, ax=ax, colormap="Reds")

        plt.title(f"Component {component}")
        plt.show()


#### DISTANCE TUNING CURVES
def distance_tuning_curves(dataframe, distance, interested_units):
    plt.figure(figsize=(8, 5))

    #colour palatte to ensure that different units are plotted in different colours
    colourmap = cm.get_cmap("tab20", len(interested_units))  # Get enough colors for the number of units
    color_indices = np.linspace(0, len(interested_units) - 1, len(interested_units)).astype(int)
    colours = [colourmap(i / len(interested_units)) for i in color_indices]  # Normalize indices to spread colors


    # Loop over each unit and plot its tuning curve
    for i, unit in enumerate(interested_units):
        distance_tuning = (dataframe.groupby([("distances", f"{distance}")])
                        .activations.mean().activations[f"unit {unit}"])

        distance_tuning = distance_tuning[distance_tuning.index <= 24]

        if distance_tuning is None or distance_tuning.isna().all() or (distance_tuning == 0).all():
            continue
        
        plt.plot(distance_tuning.index, distance_tuning.values, 
                 linestyle='-', label=f"Unit {unit}", color=colours[i])


    # Labels, title, and legend
    plt.xlabel("Distance")
    plt.ylabel("Mean Activation")
    plt.title(f"Individual Unit Activations for Shortest-Path-to-Goal Distance")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
               ncol=8, fontsize='small', frameon=False)
    plt.xticks(range(int(min(distance_tuning.index)), int(max(distance_tuning.index)) + 1))  # Ensure integer x-ticks
    plt.show()

#### DISTANCE HEATMAP TUNING CURVES
def heatmap_for_distances(dataframe, distance):
    
    # Group by the distance column and calculate mean of activations
    distance_tuning = (dataframe.groupby([("distances", f"{distance}")])
                        .activations.mean())
    
    distance_tuning = distance_tuning[distance_tuning.index <= 24] 

    # Filter out units that have zero activation across all distances
    non_zero_cols = distance_tuning.columns[(distance_tuning != 0).any()]
    distance_tuning = distance_tuning[non_zero_cols]

    #normalise
    for column in distance_tuning.columns: 
        distance_tuning[column] = distance_tuning[column] / distance_tuning[column].abs().max() 

    # Flip it so rows are units
    distance_tuning = distance_tuning.T

    # Remove units with no activations (all zeros or NaN)
    active_units = distance_tuning.loc[:, (~distance_tuning.isna()).any(axis=0)]
    active_units = active_units.loc[:, (active_units != 0).any(axis=0)]

    # Get the column (distance to goal) where each row (unit) is at its max value
    max_distance = active_units.idxmax(axis=1)

    # Now sort the dataframe by this series
    sorted_df = active_units.loc[max_distance.sort_values().index]

    # Create custom y-tick labels for units
    unit_labels = [int(label[1].split()[-1]) for label in sorted_df.index]

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_df, cmap="YlGnBu", xticklabels=True, yticklabels=unit_labels)
    plt.xlabel(f'Distance ({distance})')
    plt.ylabel('Unit')
    plt.title('Heatmap of Activations by Unit and Distance')
    plt.show()

#### LINEAR REGRESSION
def linear_regression_for_values(df):
    # Extract relevant information
    x = df["activations"].values
    y = df[("distances", "shortest_path")].values  

    # Train-test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Plot actual vs. predicted
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", color="lightcoral")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")  # Perfect prediction line
    plt.xlabel("Actual Distance")
    plt.ylabel("Predicted Distance")
    plt.title("Predicted vs. Actual Distance")
    plt.show()

def linear_regression_with_significance_full_vector(df, distance_metric="shortest_path"):
    # Extract combined activation vector (each row is a vector)
    x = np.stack(df["activations"].values)
    y = df[("distances", distance_metric)].values  

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # === scikit-learn model for performance ===
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Sklearn Evaluation ---")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # === statsmodels model for significance ===
    X_train_const = sm.add_constant(X_train)
    stats_model = sm.OLS(y_train, X_train_const).fit()

    print(f"\n--- Statsmodels OLS Summary (Training Data) ---")
    print(stats_model.summary())

    # Get p-values (excluding intercept if desired: [1:] slices off const)
    pvals = stats_model.pvalues[1:]  # exclude intercept
    reject_bonf, pvals_bonf, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')

    # Print summary of corrected results
    print(f"\n--- Bonferroni Correction Summary ---")
    print(f"Uncorrected significant activations (p < 0.05): {sum(pvals < 0.05)}")
    print(f"Bonferroni corrected significant activations (p < 0.05): {sum(reject_bonf)}")

    # Optional: attach Bonferroni-corrected p-values to the summary
    bonferroni_table = pd.DataFrame({
        "coef": stats_model.params[1:],
        "pval_uncorrected": pvals,
        "pval_bonferroni": pvals_bonf,
        "significant_bonferroni": reject_bonf
    })
    print("\nTop Bonferroni-significant activations:")
    print(bonferroni_table[bonferroni_table["significant_bonferroni"]].sort_values("pval_bonferroni").head())

    # Plot actual vs predicted
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", color="lightcoral")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
    plt.xlabel("Actual Distance")
    plt.ylabel("Predicted Distance")
    plt.title("Predicted vs. Actual Distance")
    plt.show()

    return stats_model



#### VECTOR DISTANCE VS SPATIAL DISTANCE
def get_all_distances(simple_maze):
    extended_maze = get_extended_simple_maze(simple_maze)
    coord2label = get_maze_coord2label(extended_maze)
    all_path_lengths = dict(nx.all_pairs_dijkstra_path_length(extended_maze))
    # translate to alphaneumeritc
    return {coord2label[k]: {coord2label[_k]: _v for _k, _v in v.items()} for k, v in all_path_lengths.items()}

def analyze_specific_distance_correlation(env, dataframe, interested_goal_string, activations_name, distance_metric):
    maze_graph = simple_maze(env.string_representation)
    all_distances = get_all_distances(maze_graph)
    goal_index = dataframe[dataframe[('goal', 'string')] == interested_goal_string][('goal', 'index')].values[0]
    list_of_free_spaces = list(env.free_spaces)  # Convert the set to a list

    distance_array = []
    activation_distance_array = []

    # Extract activations for all free spaces
    activations_dict = {}
    for current in list_of_free_spaces:
        if current == goal_index:  
            continue
        goal_data = dataframe[(dataframe["current", "index"] == current) & (dataframe["goal", "string"] == interested_goal_string)]
        activations_values = goal_data[f'{activations_name}'].values[0]
        activations_dict[current] = activations_values

    # Compare between all combinations of current states
    for current1 in list_of_free_spaces:
        if current1 == goal_index:  
            continue
        for current2 in list_of_free_spaces:
            if current2 == goal_index or current1 >= current2:  
                continue  # Skip redundant combinations (current1, current2) = (current2, current1)
            
            # Calculate Euclidean distance between the activation vectors of current1 and current2
            activation_diff = np.linalg.norm(activations_dict[current1] - activations_dict[current2])
            activation_distance_array.append(activation_diff)

            # Calculate distances
            if distance_metric == "shortest_path":
                string_current1 = env.string_dictionary.get(current1, str(current1))
                string_current2 = env.string_dictionary.get(current2, str(current2))
                shortest_path_to_goal = all_distances[string_current1][string_current2]
                distance_array.append(shortest_path_to_goal)
            
            else: 
                row_current1, col_current1 = divmod(current1, env.grid_size)
                row_current2, col_current2 = divmod(current2, env.grid_size)
                euclidean_distance = np.sqrt((row_current1 - row_current2) ** 2 + (col_current1 - col_current2) ** 2)
                distance_array.append(euclidean_distance)

    correlation, _ = spearmanr(activation_distance_array, distance_array)
    print(f"Spearman correlation (Activation vs Distance) for goal {interested_goal_string}: {correlation:.4f}")

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=activation_distance_array, y=distance_array, alpha=0.5)
    plt.xlabel("Distance")
    plt.ylabel("Activation Vector Distance")
    plt.title(f"Distance (Goal {interested_goal_string})")
    plt.show()

def analyze_distance_correlation(env, dataframe, activations_name, distance_metric, n_permutations=5000):
    maze_graph = simple_maze(env.string_representation)
    all_distances = get_all_distances(maze_graph)
    list_of_free_spaces = list(env.free_spaces)  # Convert the set to a list

    activation_units = [col for col in dataframe.columns if col[0] == activations_name]
    grouped = dataframe.groupby(("current", "index"))[activation_units].mean()
    activations_dict = {current: grouped.loc[current].values for current in grouped.index}
    
    spatial_distance_array = []
    activation_distance_array = []
    pairs = []  # Store state pairs to track dependencies

    # Compare between all combinations of current states
    for current1 in list_of_free_spaces:
        for current2 in list_of_free_spaces:
            if current1 >= current2:  
                continue  # Skip redundant combinations (current1, current2) = (current2, current1)
            
            # Calculate Euclidean distance between the activation vectors of current1 and current2
            activation_diff = np.linalg.norm(activations_dict[current1] - activations_dict[current2])
            activation_distance_array.append(activation_diff)

            # Calculate distances
            if distance_metric == "shortest_path":
                string_current1 = env.string_dictionary.get(current1, str(current1))
                string_current2 = env.string_dictionary.get(current2, str(current2))
                shortest_path_to_goal = all_distances[string_current1][string_current2]
                spatial_distance_array.append(shortest_path_to_goal)
            
            else: 
                row_current1, col_current1 = divmod(current1, env.grid_size)
                row_current2, col_current2 = divmod(current2, env.grid_size)
                euclidean_distance = np.sqrt((row_current1 - row_current2) ** 2 + (col_current1 - col_current2) ** 2)
                spatial_distance_array.append(euclidean_distance)
            
            pairs.append((current1, current2))  # Track which states are paired

    # Calculate both correlations + p-values
    obs_corr, _ = spearmanr(spatial_distance_array, activation_distance_array)

    # Permutation test (preserves dependency structure)
    perm_corrs = np.zeros(n_permutations)
    for i in tqdm(range(n_permutations), desc="Running permutations"):
        # Shuffle activation distances *within state blocks*
        # This ensures pairs sharing a state aren't fully independent
        perm_act_dists = np.array(activation_distance_array)
        
        # Shuffle activations for each state across its pairs
        for state in list_of_free_spaces:
            # Find all pairs involving this state
            state_pair_indices = [idx for idx, (c1, c2) in enumerate(pairs) if state in (c1, c2)]
            
            if len(state_pair_indices) > 1:
                # Shuffle only the activation distances for this state's pairs
                perm_act_dists[state_pair_indices] = np.random.permutation(
                    perm_act_dists[state_pair_indices]
                )
        
        # Compute correlation for this permutation
        perm_corrs[i], _ = spearmanr(spatial_distance_array, perm_act_dists)

    # Calculate p-value (two-tailed)
    p_value = (np.sum(np.abs(perm_corrs) >= np.abs(obs_corr)) / n_permutations)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.regplot(x=spatial_distance_array, y=activation_distance_array, scatter_kws={'alpha':0.5})

    plt.xlabel("Spatial Distance (shortest path between states)")
    plt.ylabel("Activation Vector Distance")
    plt.title(f"Spatial vs Value Activations Distance")    
    plt.show()

    print(f"Spearman’s ρ: {obs_corr:.4f}, perm. p = {p_value:.4f}")

