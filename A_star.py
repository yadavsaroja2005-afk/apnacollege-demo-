import queue as Q

dict_hn = {
    'Chandigarh': 1600,
    'Lucknow': 900,
    'Patna': 580,
    'Ranchi': 400,
    'Kolkata': 0,
    'Varanasi': 650,
    'Bhubaneswar': 300,
    'Nagpur': 950,
    'Raipur': 600
}

dict_gn = {
    'Chandigarh': {'Lucknow': 740, 'Nagpur': 1200},
    'Lucknow': {'Patna': 500, 'Varanasi': 320, 'Chandigarh': 740},
    'Patna': {'Ranchi': 340, 'Lucknow': 500},
    'Ranchi': {'Kolkata': 400, 'Patna': 340, 'Raipur': 610},
    'Kolkata': {},
    'Varanasi': {'Patna': 220, 'Lucknow': 320},
    'Bhubaneswar': {'Kolkata': 440, 'Raipur': 550},
    'Nagpur': {'Raipur': 290, 'Chandigarh': 1200},
    'Raipur': {'Ranchi': 610, 'Bhubaneswar': 550, 'Nagpur': 290}
}

def a_star(start, goal):
    def heuristic(city):
        return dict_hn[city]

    def get_fn(citystr):
        cities = citystr.split(' , ')
        gn = sum(dict_gn[cities[i]][cities[i + 1]] for i in range(len(cities) - 1))
        hn = heuristic(cities[-1])
        return gn + hn

    cityq = Q.PriorityQueue()
    cityq.put((get_fn(start), start, start))  # (total_cost, path_str, current_city)

    while not cityq.empty():
        total_cost, path, current_city = cityq.get()

        if current_city == goal:
            return f"{path} : : {total_cost}"

        for neighbor in dict_gn.get(current_city, {}):
            new_path = f"{path} , {neighbor}"
            cityq.put((get_fn(new_path), new_path, neighbor))

    return "Path not found"

# Run the A* algorithm
start, goal = 'Chandigarh', 'Kolkata'
print("The A* path with the total is:")
print(a_star(start, goal))
