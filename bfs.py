import queue as Q

# Define the graph
dict_gn = {
    'Mumbai': {'Pune': 75, 'Ahmedabad': 118, 'Bangalore': 140},
    'Delhi': {'Jaipur': 85, 'Ahmedabad': 90, 'Bangalore': 101, 'Hyderabad': 211},
    'Bangalore': {'Hyderabad': 120, 'Chennai': 138, 'Mumbai': 146},
    'Hyderabad': {'Bangalore': 120, 'Delhi': 211},
    'Chennai': {'Bangalore': 138},
    'Pune': {'Mumbai': 75},
    'Ahmedabad': {'Mumbai': 118, 'Delhi': 90},
    'Jaipur': {'Delhi': 85}
}

# BFS function
def bfs(start, goal):
    cityq = Q.Queue()
    visited = set()
    result = []

    cityq.put(start)
    visited.add(start)

    while not cityq.empty():
        city = cityq.get()
        result.append(city)

        if city == goal:
            return result

        for next_city in dict_gn.get(city, {}):
            if next_city not in visited:
                cityq.put(next_city)
                visited.add(next_city)

    return result

# Main function
def main():
    start = 'Mumbai'
    goal = 'Delhi'
    result = bfs(start, goal)
    print(f"BFS Traversal from {start} to {goal} is:")
    print(" -> ".join(result))

# Run the main function
main()
