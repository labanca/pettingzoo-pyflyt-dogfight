import numpy as np
state = np.array(
    [
        [
            [0,0,0],
            [0,1,1],
            [1,0,1],
            [1,1,0],
        ],
        [
            [1,1,1],
            [1,2,2],
            [2,1,2],
            [2,2,1],
        ]
    ]
    )


print(state[::-1][-1] - state[-1])


import numpy as np

# Assuming `num_drones` is a list of (4, 3) arrays
# where each element corresponds to the state of an agent
num_drones = 2

# Calculate the ground frame linear positions for all agents
ground_frame_linear_positions = [state[3, :] for state in state]

# Calculate the separation between agents
def calculate_separation(positions):
    num_agents = len(positions)
    separation = np.zeros((num_agents, num_agents, 3))  # Initialize a separation matrix

    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            # Calculate the difference in positions between agent i and agent j
            separation[i, j] = positions[i] - positions[j]
            separation[j, i] = -separation[i, j]  # Symmetric: i to j is the negative of j to i

    return separation

separation = calculate_separation(ground_frame_linear_positions)
pass
# Now, `separation` is a 3D NumPy array where `separation[i, j]` represents the difference in ground frame linear position between agent i and agent j.