import sys

# Define the India map as a dictionary of dictionaries
india_map = {
    'Mumbai': {'Pune': 150, 'Surat': 280, 'Indore': 585},
    'Pune': {'Mumbai': 150, 'Hyderabad': 560},
    'Surat': {'Mumbai': 280, 'Ahmedabad': 265},
    'Indore': {'Mumbai': 585, 'Bhopal': 195},
    'Hyderabad': {'Pune': 560, 'Bangalore': 570},
    'Ahmedabad': {'Surat': 265, 'Jaipur': 675},
    'Bhopal': {'Indore': 195, 'Nagpur': 350},
    'Bangalore': {'Hyderabad': 570, 'Chennai': 345},
    'Nagpur': {'Bhopal': 350, 'Raipur': 285},
    'Raipur': {'Nagpur': 285, 'Bhubaneswar': 505},
    'Bhubaneswar': {'Raipur': 505, 'Kolkata': 440},
    'Jaipur': {'Ahmedabad': 675, 'Delhi': 270},
    'Chennai': {'Bangalore': 345, 'Hyderabad': 630},
    'Kolkata': {'Bhubaneswar': 440, 'Patna': 580},
    'Patna': {'Kolkata': 580, 'Delhi': 1050},
    'Delhi': {'Jaipur': 270, 'Patna': 1050}
}

# Heuristic values (estimated distance to Delhi)
heuristics = {
    'Mumbai': 1400,
    'Pune': 1380,
    'Surat': 1200,
    'Indore': 800,
    'Hyderabad': 1500,
    'Ahmedabad': 900,
    'Bhopal': 700,
    'Bangalore': 1800,
    'Nagpur': 1200,
    'Raipur': 1100,
    'Bhubaneswar': 1500,
    'Jaipur': 260,
    'Chennai': 1900,
    'Kolkata': 1600,
    'Patna': 1000,
    'Delhi': 0
}

# Recursive Best-First Search implementation
def rbfs_search(current_city, goal_city, path, f_limit):
    if current_city == goal_city:
        return path

    successors = india_map.get(current_city, {})
    if not successors:
        return None

    sorted_successors = sorted(successors, key=lambda x: successors[x] + heuristics[x])

    for city in sorted_successors:
        new_path = path + [city]
        f_value = successors[city] + heuristics[city]

        if f_value > f_limit:
            continue  # Skip this path, not within current f_limit

        result = rbfs_search(city, goal_city, new_path, min(f_limit, f_value))
        if result is not None:
            return result

    return None

def recursive_best_first_search(start_city, goal_city):
    f_limit = sys.maxsize
    path = [start_city]
    while True:
        result = rbfs_search(start_city, goal_city, path, f_limit)
        if result is not None:
            return result
        f_limit = sys.maxsize  # Retry with a high limit if needed

# Driver code
start_city = 'Mumbai'
goal_city = 'Delhi'
path = recursive_best_first_search(start_city, goal_city)

if path is None:
    print("Path not found!")
else:
    print("Path:", " -> ".join(path))
    total_cost = sum(india_map[path[i]][path[i + 1]] for i in range(len(path) - 1))
    print("Total Cost:", total_cost)
