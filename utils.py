import math
import matplotlib as plt
from project_dataclasses import Location
import random

def distance(start, end):
    # Calculate the Euclidean distance between two points
    x1, y1 = start.get_coord()
    x2, y2 = end.get_coord()
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # XY Distance
    # return abs(y2-y1) + abs(x2-y2)


def interpolate(num, in_min, in_max, out_min, out_max):
    return out_min + (float(num - in_min) / float(in_max - in_min) * (out_max - out_min))


def plot_path(path: list[Location], show: bool = False, color: tuple = (255, 0, 0)):
    for point in range(len(path) - 1):
        plt.plot(
            [path[point].get_coord()[0], path[point + 1].get_coord()[0]],
            [path[point].get_coord()[1], path[point + 1].get_coord()[1]],
            "o-",
            color=color,
        )
    if show:
        plt.xlim(-1, HIGHEST_POINT + 1)
        plt.ylim(-1, HIGHEST_POINT + 1)
        plt.show()


def get_groups(nodes: list[Location], number: int):
    groups = {}
    old_groups = {}
    # Get Initial Centers
    for i in range(number):
        center = random.choice(nodes).__copy__()
        groups[center] = []

    while groups.values() != old_groups.values() and set(groups.keys()) != set(old_groups.keys()):
        # Assign Points To The Center Closest To Them
        for node in nodes:
            shortest = (None, float("inf"))
            for center in groups.keys():
                d = distance(center, node)
                if d == 0:
                    continue
                if d < shortest[1]:
                    shortest = (center, d)
            groups[shortest[0]].append(node)

        # Calculate The Average Center
        old_groups = groups
        groups = {}
        # nodes.extend(old_groups.keys())
        for c, n in old_groups.items():
            fake_stop = Location(
                long=(sum([ni.long for ni in n]) + c.long) / (len(n) + 1),
                lat=(sum([ni.lat for ni in n]) + c.lat) / (len(n) + 1),
            )
            groups[fake_stop] = []

    # Display
    for c, n in old_groups.items():
        for ni in n:
            plt.plot([c.long, ni.long], [c.lat, ni.lat], "bo-")
    plt.title("Groups")
    plt.xlim(-1, HIGHEST_POINT + 1)
    plt.ylim(-1, HIGHEST_POINT + 1)
    plt.show()
    return old_groups.values()


def find_overlapping_points(*lists):
    overlapping_points = []

    # Iterate over each list
    for i, lst in enumerate(lists):
        # Iterate over each pair of consecutive points in the list
        for j in range(len(lst) - 1):
            point1 = lst[j]
            point2 = lst[j + 1]

            # Check if the pair of points exists in any other list
            for k, other_lst in enumerate(lists):
                if not other_lst.index(point1) == len(other_lst) - 1:
                    if k != i and other_lst[other_lst.index(point1) + 1] == point2:
                        overlapping_points.append((point1, point2))
                        break

    # Display
    for n, t in enumerate(overlapping_points[:-1]):
        plt.xlim(-1, HIGHEST_POINT + 1)
        plt.ylim(-1, HIGHEST_POINT + 1)
        plt.plot([t[0].long, t[1].long], [t[0].lat, t[1].lat], "go-")
    plt.title("Overlapping")
    plt.xlim(-1, HIGHEST_POINT + 1)
    plt.ylim(-1, HIGHEST_POINT + 1)
    plt.show()

    return overlapping_points

