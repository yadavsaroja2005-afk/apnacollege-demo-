# Graph definition
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

# DFS function
def dfs(start, goal):
    stack = [start]
    visited = set()
    result = []

    while stack:
        city = stack.pop()
        if city not in visited:
            visited.add(city)
            result.append(city)

            if city == goal:
                break

            # Reverse to preserve order similar to input
            stack.extend(reversed(dict_gn.get(city, {})))

    return result

# Main function
def main():
    start, goal = 'Mumbai', 'Delhi'
    path = dfs(start, goal)
    print(f"DFS Traversal from {start} to {goal} is:")
    print(" -> ".join(path))

# Run the main function
main()
