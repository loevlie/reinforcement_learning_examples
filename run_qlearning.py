# visualize_qlearning.py

import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np

from environment import env  # Import your existing env class

def plot_environment(environment, iteration):
    """
    Draws the environment as a directed graph, labeling edges with current Q-values,
    and highlights the agent's current state.
    """
    # Build a directed graph
    G = nx.DiGraph()

    # Add all states as nodes
    for s in environment.states:
        G.add_node(s)

    # Add edges for each (state -> action) with Q-value as the weight
    for s in environment.q_table:
        for a, q_val in environment.q_table[s].items():
            # 'none' is in your Q-table for s6, skip that
            if a != 'none':
                G.add_edge(s, a, weight=round(q_val, 2))

    # A fixed position layout for nicer visuals:
    pos = {
        "s1": (0, 1),
        "s2": (1, 1),
        "s3": (2, 1),
        "s4": (0, 0),
        "s5": (1, 0),
        "s6": (2, 0)
    }

    # Clear current figure and draw the graph
    plt.clf()

    # Highlight the current state in a different color
    node_colors = [
        "yellow" if node == environment.current_state else "lightblue"
        for node in G.nodes()
    ]

    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        node_color=node_colors,
        node_size=1500,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15
    )

    # Draw edge labels = Q-values
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.title(f"Iteration {iteration} - Current State: {environment.current_state}")
    plt.axis("off")
    plt.pause(0.5)  # brief pause to update the interactive display

def run_q_learning_visualized(num_iterations=20):
    """
    Runs a short Q-learning process with epsilon-greedy,
    plots the environment after each move.
    """
    environment = env()  # Use your environment
    epsilon = 0.1

    # Turn on interactive plotting
    plt.ion()
    plt.figure(figsize=(8, 5))

    for i in range(num_iterations):
        # Decide whether to exploit or explore
        if random.random() > (1 - epsilon):
            # Exploit: pick the action with the highest Q-value
            q_values = list(environment.list_qtable_values().values())
            actions = list(environment.list_qtable_values().keys())
            best_action = actions[np.argmax(q_values)]
            next_action = best_action
        else:
            # Explore: pick a random action from available ones
            next_action = random.choice(environment.list_actions())

        # Take the action and update Q-table
        environment.take_action(next_action)

        # Plot the current state of the environment
        plot_environment(environment, i)

    # Once done, keep the final plot open
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_q_learning_visualized(num_iterations=20)
