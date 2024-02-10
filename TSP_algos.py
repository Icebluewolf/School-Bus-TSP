import random
from utils import distance


def nearest_neighbor(nodes, start, end):
    nodes = nodes.copy()
    ordered_nodes = []
    end._connections = [x for x in end._connections if x[0] != start]
    start._connections = [x for x in start._connections if x[0] != end]
    start._connections.append((end, 0))
    for i in nodes:
        i.create_connections(nodes)
    point = start
    total = 0
    while len(nodes) > 1:
        while True:
            s = point.get_shortest_link(remove=True)
            if s[0] in nodes:
                break
        # plt.plot([point.get_coord()[0], s[0].get_coord()[0]], [point.get_coord()[1], s[0].get_coord()[1]], "ro-")
        nodes.remove(point)
        ordered_nodes.append(point)
        point = s[0]
        total += s[1]
    ordered_nodes.append(point)
    ordered_nodes.append(start)
    # plt.plot([point.get_coord()[0], start.get_coord()[0]], [point.get_coord()[1], start.get_coord()[1]], "ro-")
    # plt.plot([end.get_coord()[0], start.get_coord()[0]], [end.get_coord()[1], start.get_coord()[1]], "go-")
    # plt.title("Nearest Neighbor")
    # plt.show()
    return ordered_nodes, total


def cheapest_insertion(cities, start, end):
    cities = cities.copy()
    num_cities = len(cities)
    unvisited = set(range(num_cities))
    visited = []
    # start_city = cities.index(start)
    start_city = 0
    visited.append(start_city)
    unvisited.remove(start_city)

    while unvisited:
        min_cost = float('inf')
        best_city = None
        best_position = None

        for current_city in visited:
            for next_city in unvisited:
                for i in range(len(visited) + 1):
                    city1 = cities[current_city]
                    city2 = cities[next_city]
                    if i == 0:
                        city3 = cities[visited[0]]
                    elif i == len(visited):
                        city3 = cities[visited[-1]]
                    else:
                        city3 = cities[visited[i]]

                    if (city1 == start and city2 == end) or (city2 == start and city1 == end):
                        cost = 0 - distance(city1, city3)
                    else:
                        cost = distance(city1, city2) + distance(city2, city3) - distance(city1, city3)

                    if cost < min_cost:
                        min_cost = cost
                        best_city = next_city
                        best_position = i

        visited.insert(best_position, best_city)
        unvisited.remove(best_city)

    return [cities[city_index] for city_index in visited]


def cheapest_insertion_2(nodes, start, end):
    nodes = nodes.copy()
    path = [start, end]
    try:
        nodes.remove(start)
        nodes.remove(end)
    except ValueError:
        pass

    for link in enumerate(path[:-1]):
        shortest = (None, float("inf"))
        for node in nodes:
            pass


def shortest_random(nodes, start, end, iterations=10000):
    nodes.copy()
    try:
        nodes.remove(start)
        nodes.remove(end)
    except ValueError:
        pass
    shortest = (nodes, float("inf"))

    for i in range(iterations):
        random.shuffle(nodes)
        total = distance(start, nodes[0])
        for n, current in enumerate(nodes[:-1]):
            total += distance(current, nodes[n])
        total += distance(nodes[-1], end)

        if total < shortest[1]:
            shortest = (nodes.copy(), total)

    shortest[0].insert(0, start)
    shortest[0].append(end)

    # for n, t in enumerate(shortest[0][:-1]):
    #     plt.plot([t.long, shortest[0][n+1].long], [t.lat, shortest[0][n+1].lat], "bo-")
    # plt.plot([start.long, shortest[0][0].long], [start.lat, shortest[0][0].lat], "bo-")
    # plt.plot([end.long, shortest[0][-1].long], [end.lat, shortest[0][-1].lat], "bo-")
    #
    # plt.title("Random")
    # plt.show()
    return shortest[0]
