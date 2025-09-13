import random
import matplotlib.pyplot as plt
from matplotlib import animation

# --- Grid Setup ---
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
goal = (4, 4)
moves = ['Up', 'Down', 'Left', 'Right']

population_size = 50
path_length = 25
generations = 150
mutation_rate = 0.1
epsilon = 1e-6  # for zero-weight protection

# --- Fitness Function ---
def fitness(path):
    x, y = start
    steps = 0
    for move in path:
        if move == 'Up' and x > 0: x -= 1
        elif move == 'Down' and x < len(grid)-1: x += 1
        elif move == 'Left' and y > 0: y -= 1
        elif move == 'Right' and y < len(grid[0])-1: y += 1
        steps += 1
        if grid[x][y] == 1:  # obstacle penalty
            return epsilon
        if (x, y) == goal:  # reached goal
            return 100 - steps
    # distance penalty if goal not reached
    dist = abs(goal[0]-x) + abs(goal[1]-y)
    return max(1, 50 - steps - dist*10)

# --- Population Initialization ---
population = [[random.choice(moves) for _ in range(path_length)] for _ in range(population_size)]

# --- GA Loop ---
best_path = None
best_fit = -float('inf')

for gen in range(1, generations+1):
    fitnesses = [fitness(path) for path in population]
    max_fit = max(fitnesses)
    if max_fit > best_fit:
        best_fit = max_fit
        best_path = population[fitnesses.index(max_fit)]

    print(f"Generation {gen}: Best Fitness = {best_fit:.4f}")

    if best_fit >= 100:  # goal reached
        print(f"\nGoal reached! Stopping early at generation {gen}")
        break

    # --- Selection ---
    selected = random.choices(population, weights=[f + epsilon for f in fitnesses], k=population_size)

    # --- Crossover ---
    next_gen = []
    for i in range(0, population_size, 2):
        p1 = selected[i]
        p2 = selected[i+1 if i+1 < population_size else 0]
        point = random.randint(1, path_length-1)
        child1 = p1[:point] + p2[point:]
        child2 = p2[:point] + p1[point:]
        next_gen.extend([child1, child2])

    # --- Mutation ---
    for path in next_gen:
        for i in range(path_length):
            if random.random() < mutation_rate:
                path[i] = random.choice(moves)

    population = next_gen[:population_size]

# --- Path Positions for Animation ---
x, y = start
path_positions = [(x, y)]
for move in best_path:
    if move == 'Up' and x > 0: x -= 1
    elif move == 'Down' and x < len(grid)-1: x += 1
    elif move == 'Left' and y > 0: y -= 1
    elif move == 'Right' and y < len(grid[0])-1: y += 1
    path_positions.append((x, y))

# --- Animation ---
fig, ax = plt.subplots()
ax.imshow(grid, cmap='gray_r')

# Plot path
path_x = [y for _, y in path_positions]
path_y = [x for x, _ in path_positions]
ax.plot(path_x, path_y, color='blue', linewidth=2, label='Path')

# Robot dot
robot_dot, = ax.plot([], [], 'ro', markersize=10, label='Robot')

def update(frame):
    x, y = path_positions[frame]
    robot_dot.set_data([y], [x])
    return robot_dot,

ani = animation.FuncAnimation(fig, update, frames=len(path_positions), interval=500, blit=True, repeat=False)
ax.legend()
plt.show()
