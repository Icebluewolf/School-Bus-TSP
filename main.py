# Make total distance low
# Average distance per student close
# Gather from everywhere to central location
# Look for convergence of points for transfer locations
# BONUS-Change the school to student to see effects.


# PROOF OF CONCEPT

# Given: Students Addresses And Schools, Number Of Busses/Drivers
import os
import json
import matplotlib.lines
import requests.exceptions

import read_csv
import matplotlib.pyplot as plt
from matplotlib import colors
from statistics import median
import numpy as np
from python_tsp.distances import osrm_distance_matrix
from python_tsp.heuristics import solve_tsp_local_search

from utils import interpolate, distance
import clusters

# Get Subplots
fig, (student_loc_ax, cluster_ax) = plt.subplots(figsize=(8, 14), nrows=2, sharex="all", sharey="all",
                                                 layout="tight", gridspec_kw={"hspace": 0})
misc_fig, (box_ax, histo_ax) = plt.subplots(figsize=(14, 8), ncols=2)

# Load The Data Into Dataclasses
schools = read_csv.get_schools("data/SchoolList.csv")
students = read_csv.get_students("data/StudentsNEW.csv")

# Generate A Range Of Colors And Assign Them To Schools
colors = [colors.hsv_to_rgb((interpolate(n * (360 / len(schools)), 0, 360, 0, 1), 0.8, 0.7)) for n in
          range(len(schools))]
colors = {schools[n].name: colors[n] for n in range(len(colors))}

# Set Up The Student List In An Easy To Manage Order
students_grouped = {}
for student in students:
    students_grouped.setdefault(student.school, []).append(student)

# Get All The Student Locations Extracted
bounding_box = [float('inf'), float('-inf'), float('inf'), float('-inf')]
visible_schools = {}
inlier_plots = []
# outlier_plots = {int: set[str, list]} # TODO: rnage -# 0 +#   set(school, points)
outlier_plots = []
box_distances = []
# TODO: Make this recursive
for school, school_students in students_grouped.items():
    longs = []
    lats = []
    distances = []
    for student in school_students:
        long = student.house.long
        lat = student.house.lat
        longs.append(long)
        lats.append(lat)

        d = distance(student.house, school)
        student.distance_to_school = d
        distances.append(d)

        bounding_box[0] = min(bounding_box[0], long)
        bounding_box[1] = max(bounding_box[1], long)
        bounding_box[2] = min(bounding_box[2], lat)
        bounding_box[3] = max(bounding_box[3], lat)

    distances = np.array(distances)
    q1 = np.quantile(distances, 0.25, interpolation="midpoint")
    q3 = np.quantile(distances, 0.75, interpolation="midpoint")

    inlier = ([], [])
    outlier = ([], [])
    for student in school_students:
        if student.distance_to_school <= q1:
            pass
        if q1 <= student.distance_to_school <= q3:
            inlier[0].append(student.house.long)
            inlier[1].append(student.house.lat)
        else:
            outlier[0].append(student.house.long)
            outlier[1].append(student.house.lat)

    # plot() Returns A List Of Lines We Only Need The First One
    visible_schools[school.name] = True
    inlier_plots.append(student_loc_ax.plot(inlier[0], inlier[1], ".", color=colors.get(school.name, (0, 0, 0)),
                                            label=school.name)[0])
    outlier_plots.append(student_loc_ax.plot(outlier[0], outlier[1], ".", color=colors.get(school.name, (0, 0, 0)),
                                             label="_" + school.name)[0])
    box_distances.append([school.name, distances])
school_plots = student_loc_ax.scatter([school.long for school in schools], [school.lat for school in schools],
                                      marker="*",
                                      c=[colors.get(school.name, (0, 0, 0)) for school in schools],
                                      label="Schools",
                                      zorder=2.001)
# Setup A Line That Will Represent The Schools Scatter
school_plots_line = matplotlib.lines.Line2D([0], [0], linewidth=0, marker="*", label="Schools")

# Set Up Legend
school_legend = student_loc_ax.legend(handles=inlier_plots, loc="center left", bbox_to_anchor=(1.04, 0))
school_legend_lines = school_legend.get_lines()
legend_elements = [matplotlib.lines.Line2D([0], [0], linewidth=0,  marker="o", label="Inlier"),
                   matplotlib.lines.Line2D([0], [0], linewidth=0, marker="o", label="Outlier"),
                   school_plots_line, ]
other_legend = student_loc_ax.legend(handles=legend_elements,
                                     loc="center right",
                                     bbox_to_anchor=(-0.08, 0.5), )
# Add The Original Legend As MatPlotLib Tries To Be Smart When You Have 2 Legends And Removes The Original
student_loc_ax.add_artist(school_legend)
graphs = {}
for n, scatter in enumerate(school_legend_lines):
    scatter.set_picker(True)
    scatter.set_pickradius(10)
    graphs[scatter] = {"inlier": inlier_plots[n], "outlier": outlier_plots[n]}

for line in other_legend.get_lines():
    print(f"Other: {line}")
    line.set_picker(True)
    line.set_pickradius(10)


def on_pick(event):
    print("picked")
    if event.artist in school_legend.get_lines():
        on_school_pick(event)
    elif event.artist in other_legend.get_lines():
        on_other_pick(event)
    elif event.artist in cluster_leg.get_lines():
        on_cluster_pick(event)


def on_school_pick(event):
    is_visible = event.artist.get_visible()
    visible_schools[event.artist.get_label()] = not is_visible

    other_lines = other_legend.get_lines()

    if other_lines[0].get_visible():
        graphs[event.artist]["inlier"].set_visible(not is_visible)
    if other_lines[1].get_visible():
        graphs[event.artist]["outlier"].set_visible(not is_visible)

    # for graph in graphs[event.artist]:
    #     graph.set_visible(not is_visible)

    event.artist.set_visible(not is_visible)
    event.canvas.draw()


def on_other_pick(event):
    print(event.artist.get_label())
    is_visible = event.artist.get_visible()

    if event.artist.get_label() == "Inlier":
        graphs = inlier_plots
    elif event.artist.get_label() == "Outlier":
        graphs = outlier_plots
    elif event.artist.get_label() == "Schools":
        school_plots.set_visible(not is_visible)
        event.artist.set_visible(not is_visible)
        event.canvas.draw()
        return

    for graph in graphs:
        if visible_schools[graph.get_label().strip("_")]:
            graph.set_visible(not is_visible)

    event.artist.set_visible(not is_visible)
    event.canvas.draw()


def on_cluster_pick(event):
    # TODO: MAKE THIS CUSTOM
    print(event.artist.get_label())
    is_visible = event.artist.get_visible()

    for graph in cluster_points[event.artist.get_label()]:
        # if visible_schools[graph.get_label().strip("_")]:
        #     graph.set_visible(not is_visible)
        for lines in graph[1]:
            lines.set_visible(not is_visible)

    event.artist.set_visible(not is_visible)
    event.canvas.draw()


# Create Saved Data
loaded_data = {}
try:
    with open("data/results/gage_test.json", "r") as file:
        loaded_data = json.load(file)
        print(loaded_data)
except FileNotFoundError:
    print("Data file not found. Starting with a new dataset.")
except json.JSONDecodeError:
    print("No Data Found. Starting New Dataset")

with open("data/results/gage_test.json", "w") as file:

    # Cluster Graph
    source_name = "Gage"
    # Somehow Coords Are Backwards
    sources = np.array([[student.house.lat, student.house.long] for student in students if student.school.name == source_name])
    cluster_points = clusters.get_clusters(sources, cluster_ax, 10, 60)

    cluster_leg = cluster_ax.legend(loc="center right",
                                    bbox_to_anchor=(-0.08, 0.5), )
    for line in cluster_leg.get_lines():
        print(f"Cluster: {line}")
        line.set_picker(True)
        line.set_pickradius(10)

    # TSP Solver
    overall_distances = {}
    for name, clusters in cluster_points.items():
        try:
            if loaded_data[source_name][name][1]:
                continue
        except KeyError:
            pass
        print(f"\n{name}: ", end="")
        cluster_distances = []
        print(clusters)
        for n, (cluster, line) in enumerate(clusters):
            print(f"\n\t{len(cluster)}", end="")
            saved = False
            while True:
                try:
                    distance_matrix = osrm_distance_matrix(cluster, osrm_server_address="http://WolfyHost:5000",
                                                           osrm_batch_size=100)
                    order, total_distance = solve_tsp_local_search(distance_matrix, verbose=False)
                    cluster_distances.append(total_distance)
                    loaded_data.setdefault(source_name, {}).setdefault(name, []).append([order, total_distance])
                    break
                except requests.exceptions.ConnectionError:
                    if saved is False:
                        # Rewrite The File. Go to beginning, overwrite, delete leftover characters.
                        print(" saving", end="")
                        file.seek(0)
                        json.dump(loaded_data, file, indent=2)
                        file.truncate()
                        saved = True
                    else:
                        print(".", end="")

        overall_distances[name] = cluster_distances

    # Rewrite The File. Go to beginning, overwrite, delete leftover characters.
    file.seek(0)
    json.dump(loaded_data, file, indent=2)
    file.truncate()
file_name = file.name.split("/")
file_name[-1] = "complete_" + file_name[-1]
file_name = "/".join(file_name)
# os.replace(file.name, file_name)

print("\n")
for name, cluster_dist in overall_distances.items():
    print(f"Name: {name}\t\t\tTotal: {sum(cluster_dist)}\t\t\t{cluster_dist}")

# print(sources)
# distance_matrix = osrm_distance_matrix(sources, osrm_server_address="http://WolfyHost:5000", osrm_batch_size=50)
# xopt, fopt = solve_tsp_local_search(distance_matrix, verbose=False)
# print(xopt)
# print(fopt)

# Set Up Plot And Show
fig.canvas.mpl_connect("pick_event", on_pick)
img = plt.imread("data/area_map.png")
for ax in [student_loc_ax, cluster_ax]:
    ax.imshow(img, extent=bounding_box)
    ax.set_xlim(bounding_box[0], bounding_box[1])
    ax.set_ylim(bounding_box[2], bounding_box[3])

# Box plots - Outliers For Each School
box_ax.boxplot([i[1] for i in box_distances])
box_ax.set_xticklabels([i[0] for i in box_distances], rotation=45)

# Histogram - Show Amount Of Students At A Distance
histo_distances = []
for student in students:
    histo_distances.append(distance(student.house, student.school))
histo_ax.hist(histo_distances, bins=250)

plt.show()
print(bounding_box)
