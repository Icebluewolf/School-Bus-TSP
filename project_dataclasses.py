import datetime
import math
import copy
import typing
from datetime import time


class Location:
    def __init__(self, long: float, lat: float, access_street: str | None = None, street_node: int = None):
        self.long: float = long
        self.lat: float = lat
        self.access_street: str | None = access_street
        self.street_node: int | None = street_node
        self._connections = []

    def __repr__(self):
        return f"({self.long}, {self.lat})"

    # def __eq__(self, other):
    #     # Coordinates Are Precise To 15 Decimal Places This Is Adjusted For The Tolerance To Work
    #     return math.isclose(self.long, other.long, rel_tol=1e-17) and math.isclose(self.lat, other.lat, rel_tol=1e-17)

    def __hash__(self):
        return hash((self.lat, self.long))

    def get_coord(self) -> tuple[float, float]:
        return self.long, self.lat

    def create_connections(self, locations: list):
        self._connections = []
        for i in locations:
            dist = math.dist(self.get_coord(), i.get_coord())
            if dist > 0:
                self._connections.append((i, dist))

    def get_shortest_link(self, remove: bool = False):
        result = min(self._connections, key=lambda x: x[1])
        if remove:
            self._connections.remove(result)
        return result

    def __copy__(self):
        l = Location(self.long, self.lat, self.access_street)
        l._connections = copy.deepcopy(self._connections)
        return l


class School(Location):
    name_to_obj = {}

    def __init__(self,
                 name: str,
                 long: float,
                 lat: float,
                 access_street: str | None = None,
                 start_time: datetime.time = None,
                 end_time: datetime.time = None):
        super().__init__(long, lat, access_street)

        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        School.name_to_obj[self.name] = self


# Set Up A UNKNOWN For Schools I Do Not Have Info On
School.name_to_obj["UNKNOWN"] = School("UNKNOWN", -92.3, 44.0)


class Student:
    def __init__(self, house: Location, school: School, level: str):
        if level not in ["E", "M", "H"]:
            raise TypeError("Level Should Be One Of 'E', 'M', Or 'H'")
        self.house = house
        self.school = school
        self.level = level
        self.distance_to_school = None


class Stop(Location):
    def __init__(self, pickup_time: time, students: list[Student]):
        self.time = pickup_time
        self.students = students
        super().__init__(1, 1)


class Route:
    def __init__(self):
        self.stops = []


class Link:
    def __init__(self):
        self.time_to_drive = 0


class Bus:
    def __init__(self, start: Stop):
        self.capacity = 72  # TODO: Get A Value For This Number
        self.stops = [start]
