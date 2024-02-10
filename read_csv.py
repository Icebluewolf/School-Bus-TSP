import csv
import matplotlib.pyplot as plt
from matplotlib import colors
from project_dataclasses import Location, School, Student
from utils import interpolate


def get_schools(filepath: str) -> list[School]:
    schools = []
    with open(filepath) as schools_csv:
        reader = csv.reader(schools_csv)
        headers = next(reader)
        for row in reader:
            schools.append(School(row[2], float(row[7]), float(row[8]), access_street=row[3]))

    return schools


def get_students(filepath: str) -> list[Student]:
    if not School.name_to_obj:
        raise RuntimeWarning("Schools Are Not Filled Results Will Contain No Information About Schools")
    students = []
    with open(filepath) as student_csv:
        reader = csv.reader(student_csv)
        headers = next(reader)
        for row in reader:
            students.append(
                Student(
                    Location(float(row[2]), float(row[3])),
                    School.name_to_obj.get(row[0], School.name_to_obj["UNKNOWN"]),
                    row[1]))

    return students
