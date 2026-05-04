import random
import numpy as np
import matplotlib.pyplot as plt
#import assign_PE_max_output_filter
# Change this line in your GA code:
import assign_PE_max_output_filter_cpp as assign_PE_max_output_filter
import pretrained_models_torchvision
from pretrained_models_torchvision import get_pruned_weights_dict
from tqdm import tqdm
import time
import os
import csv

start_time = time.time()
log_filename = "feature26_mux2.txt"
log_filename_csv = "feature26_mux2.csv"
plot_filename = "feature26_mux2.png"
final_result = "feature26_mux2_final_results.csv"
# Optional: Clear the log file at the start of a new run
header = ["Generation", "Best_Fitness", "Avg_Fitness", "Global_Best_Fitness", "Global_Best_Count"]
with open(log_filename_csv, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
with open(log_filename, "w") as f:
    f.write(f"GA Run Started: {time.ctime()}\n\n")

# --- Load pruned weights ---
weights_dict = get_pruned_weights_dict(pruning_amount=0.71)

# --- GA Hyperparameters ---
POP_SIZE = 2000
NUM_GENERATIONS = 5000
MUTATION_RATE = 0.04
TOURNAMENT_SIZE = 6
ELITISM_COUNT = 5

# --- Target Layer ---
layer_name = 'features.26.weight'
#layer_name = 'features.4.1.block.0.0.weight'
arg1, arg2, arg3, max_mux_trans = 33, 45, 256, 2  # constants for fitness function

# --- Extract weights and transpose to [kH, kW, C_in, C_out] ---
layer_weights = weights_dict[layer_name]
weights_np = np.transpose(layer_weights, (2, 3, 1, 0))
C_in = weights_np.shape[2]
C_out = weights_np.shape[3]
fitness, count = assign_PE_max_output_filter.assign_PE_max_output_filter(
    arg1, arg2, arg3, weights_np, max_mux_trans
)
print(f"Fitness: {fitness}, Count: {count}")
with open(log_filename, "a") as f:
    f.write(f"Fitness: {fitness:.6f}, Count: {count}\n")
# --- GA Functions ---
def initialize_population(pop_size, C_in, C_out):
    return [(list(np.random.permutation(C_in)), list(np.random.permutation(C_out)))
            for _ in range(pop_size)]

def evaluate_fitness(population, weights):
    scores = []
    i=0
    for input_order, output_order in population:
        reordered = weights[:, :, input_order, :][:, :, :, output_order]
        fitness, count = assign_PE_max_output_filter.assign_PE_max_output_filter(arg1, arg2, arg3, reordered, max_mux_trans)
        if i==0 :
            print(fitness)
        scores.append((fitness, count))
        i=i+1
    return scores

def tournament_selection(pop, scores, k=TOURNAMENT_SIZE):
    return max(
        random.sample(list(zip(pop, scores)), k),
        key=lambda x: x[1][0]   # fitness only
    )[0]


def crossover_order(o1, o2):
    size = len(o1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b] = o1[a:b]
    fill = [x for x in o2 if x not in child]
    j = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill[j]
            j += 1
    return child

def mutate_order(order, rate=MUTATION_RATE):
    order = order[:]
    if random.random() < rate:
        a, b = random.sample(range(len(order)), 2)
        order[a], order[b] = order[b], order[a]
    return order

def crossover(p1, p2):
    return (crossover_order(p1[0], p2[0]), crossover_order(p1[1], p2[1]))

def mutate(ind):
    return (mutate_order(ind[0]), mutate_order(ind[1]))

# --- Real-time plot setup ---
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
line_best, = ax.plot([], [], label='Best Fitness', color='green')
line_avg, = ax.plot([], [], label='Average Fitness', color='blue')
ax.set_xlim(0, NUM_GENERATIONS)
ax.set_ylim(0, 1)
ax.set_xlabel("Generation")
ax.set_ylabel("Fitness")
ax.set_title("GA Progress (Live)")
ax.grid(True)
ax.legend()

# --- GA Execution ---
population = initialize_population(POP_SIZE, C_in, C_out)
global_best_fitness = -float('inf')
global_best_count = None
global_best_individual = None
best_history = []
avg_history = []

for gen in tqdm(range(NUM_GENERATIONS)):
    fitness_scores = evaluate_fitness(population, weights_np)
    fitness_only = [f for f, c in fitness_scores]
    best_idx = np.argmax(fitness_only)
    best_fit, best_count = fitness_scores[best_idx]
    best_ind = population[best_idx]

    if best_fit > global_best_fitness:
        global_best_fitness = best_fit
        global_best_count = best_count
        global_best_individual = best_ind

    best_history.append(best_fit)
    avg_history.append(np.mean(fitness_only))
    print(
    f"Generation {gen+1:>3} | "
    f"Best Fitness: {best_fit:.4f} | "
    f"Best Count: {best_count} | "
    f"Global Best: {global_best_fitness:.4f} | "
    f"Global Best Count: {global_best_count}"
    )
    
    # Update plot
    line_best.set_data(range(1, gen + 2), best_history)
    line_avg.set_data(range(1, gen + 2), avg_history)
    current_max = max(best_history + avg_history)
    if current_max > ax.get_ylim()[1]:
        ax.set_ylim(0, current_max * 1.1)
    plt.pause(0.01)
    if (gen + 1) % 1 == 0: # Saves every 10 generations to stay fast
        plt.savefig(plot_filename)
        with open(log_filename, "a") as f:
            f.write(f"Generation {gen+1:>3} | Best: {best_fit:.4f} | Global Best: {global_best_fitness:.4f} | Global Best Count: {global_best_count}\n")
            f.flush() # This ensures it saves even if the system crashes
        with open(log_filename_csv, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                gen + 1, 
                f"{best_fit:.6f}", 
                f"{np.mean(fitness_only):.6f}", 
                f"{global_best_fitness:.6f}", 
                global_best_count
            ])
    # Elitism + breeding
    elite = [population[i] for i in np.argsort(fitness_only)[-ELITISM_COUNT:]]
    new_pop = elite[:]
    while len(new_pop) < POP_SIZE:
        p1 = tournament_selection(population, fitness_scores)
        p2 = tournament_selection(population, fitness_scores)
        child = mutate(crossover(p1, p2))
        new_pop.append(child)
    population = new_pop

# --- Final output ---


print("\n✅ Global Best Input Order:", global_best_individual[0])
print("✅ Global Best Output Order:", global_best_individual[1])
print("\n✅ Global Best Fitness:", global_best_fitness)
print("✅ Corresponding Global Best Count:", global_best_count)

# Optional: reordered weights
best_weights_reordered = weights_np[:, :, global_best_individual[0], :][:, :, :, global_best_individual[1]]
end_time = time.time()
with open(log_filename, "a") as f:
    f.write(f" Global Best Input Order: {global_best_individual[0]}\n")
    f.write(f" Global Best Output Order: {global_best_individual[1]}\n")
    f.write(f"Total Runtime: {end_time - start_time:.2f} seconds\n")
print("Start:", time.ctime(start_time))
print("End:", time.ctime(end_time))
print("Total runtime:", end_time - start_time, "seconds")
with open(final_result, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Parameter", "Value"])
    writer.writerow(["Best_Input_Order", global_best_individual[0]])
    writer.writerow(["Best_Output_Order", global_best_individual[1]])
    writer.writerow(["Final_Fitness", global_best_fitness])
    writer.writerow(["Final_Count", global_best_count])
# --- ADD THESE AT THE END ---
plt.savefig(plot_filename) # Final high-quality save
with open(log_filename, "a") as f:
    f.write(f"\n Finished at: {time.ctime()}\n")
with open(log_filename, "a") as f:
    f.write(f"Total Global Best Fitness: {global_best_fitness:.4f}\n")
    f.write(f"Corresponding Global Best Count: {global_best_count}\n")
plt.ioff()
plt.show()