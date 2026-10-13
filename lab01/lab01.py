"""Lab 1: Python and numerical libraries.

Run one exercise: python lab01.py EXERCISE INPUT_FILE
Exercise 7 does not need an input file.
"""

import argparse
import calendar
import math

import numpy as np


# Exercise 1: Gymnastics


def read_competitors(filename):
    competitors = []
    with open(filename, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            fields = line.split()
            name = fields[0] + " " + fields[1]
            country = fields[2]
            scores = [float(value) for value in fields[3:]]
            # Remove one lowest score and one highest score.
            final_score = sum(sorted(scores)[1:-1])
            competitors.append((name, country, final_score))
    return competitors


def rank_competitors(competitors):
    return sorted(competitors, key=lambda competitor: competitor[2], reverse=True)


def country_totals(competitors):
    totals = {}
    for name, country, score in competitors:
        totals[country] = totals.get(country, 0.0) + score
    return totals


def exercise_1(filename):
    competitors = read_competitors(filename)
    if not competitors:
        raise ValueError("The file contains no competitors.")

    print("Final ranking:")
    for position, (name, country, score) in enumerate(rank_competitors(competitors)[:3], 1):
        print(f"{position}: {name} - Score: {score:.1f}")

    totals = country_totals(competitors)
    best_country = max(totals, key=totals.get)
    print("\nBest Country:")
    print(f"{best_country} - Total score: {totals[best_country]:.1f}")


# Exercise 2: Bus distance and route speed


def read_buses(filename):
    buses = {}
    with open(filename, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            bus_id, route_id, x, y, time = line.split()
            if bus_id not in buses:
                buses[bus_id] = {"route": route_id, "positions": []}
            buses[bus_id]["positions"].append((float(x), float(y), int(time)))

    for bus in buses.values():
        bus["positions"].sort(key=lambda position: position[2])
    return buses


def distance_and_time(positions):
    distance = 0.0
    for i in range(1, len(positions)):
        x1, y1, time1 = positions[i - 1]
        x2, y2, time2 = positions[i]
        distance += math.hypot(x2 - x1, y2 - y1)
    elapsed_time = positions[-1][2] - positions[0][2]
    return distance, elapsed_time


def route_speed(buses, route_id):
    total_distance = 0.0
    total_time = 0
    for bus in buses.values():
        if bus["route"] == route_id:
            distance, elapsed_time = distance_and_time(bus["positions"])
            total_distance += distance
            total_time += elapsed_time
    if total_time == 0:
        raise ValueError("This route has no positive elapsed time.")
    return total_distance / total_time


def exercise_2(filename, bus_id=None, route_id=None):
    buses = read_buses(filename)

    if bus_id is not None:
        if bus_id not in buses:
            raise ValueError(f"Bus {bus_id} was not found.")
        distance, elapsed_time = distance_and_time(buses[bus_id]["positions"])
        print(f"{bus_id} - Total Distance: {distance}")
    else:
        if not any(bus["route"] == route_id for bus in buses.values()):
            raise ValueError(f"Route {route_id} was not found.")
        speed = route_speed(buses, route_id)
        print(f"{route_id} - Avg Speed: {speed}")


# Exercise 3: Birth statistics


def count_births(filename):
    cities = {}
    months = {}
    with open(filename, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            name, surname, city, birthdate = line.split()
            month = int(birthdate.split("/")[1])
            cities[city] = cities.get(city, 0) + 1
            months[month] = months.get(month, 0) + 1
    return cities, months


def exercise_3(filename):
    cities, months = count_births(filename)

    print("Births per city:")
    for city, count in cities.items():
        print(f"  {city}: {count}")
    print("Births per month:")
    for month in sorted(months):
        print(f"  {calendar.month_name[month]}: {months[month]}")
    average = sum(cities.values()) / len(cities) if cities else 0.0
    print(f"Average number of births: {average:.2f}")


# Exercise 4: Bookstore


def read_transactions(filename):
    books = {}
    monthly_sales = {}
    with open(filename, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            isbn, operation, date, copies, price = line.split()
            copies = int(copies)
            price = float(price)
            if isbn not in books:
                books[isbn] = {"bought": 0, "sold": 0, "cost": 0.0, "revenue": 0.0}
            book = books[isbn]
            if operation == "B":
                book["bought"] += copies
                book["cost"] += copies * price
            elif operation == "S":
                book["sold"] += copies
                book["revenue"] += copies * price
                day, month, year = date.split("/")
                key = (int(year), int(month))
                monthly_sales[key] = monthly_sales.get(key, 0) + copies
            else:
                raise ValueError(f"Unknown operation: {operation}")
    return books, monthly_sales


def book_gain(book):
    if book["sold"] == 0:
        return 0.0, 0.0
    # Weight the purchase price by the number of copies in each purchase.
    average_cost = book["cost"] / book["bought"]
    gain = book["revenue"] - book["sold"] * average_cost
    return gain, gain / book["sold"]


def exercise_4(filename):
    books, monthly_sales = read_transactions(filename)

    print("Available copies:")
    for isbn in sorted(books):
        book = books[isbn]
        print(f"  {isbn}: {book['bought'] - book['sold']}")
    print("Sold copies:")
    for isbn in sorted(books):
        print(f"  {isbn}: {books[isbn]['sold']}")
    print("Sold books per month:")
    for year, month in sorted(monthly_sales):
        print(f"  {calendar.month_name[month]}, {year}: {monthly_sales[(year, month)]}")
    print("Gain per book:")
    for isbn in sorted(books):
        book = books[isbn]
        if book["sold"] > 0:
            gain, average_gain = book_gain(book)
            print(f"  {isbn}: {gain:.1f} (avg {average_gain:.1f}, sold {book['sold']})")


# Exercise 5: Lighting with lists and dictionaries


def read_spotlights(filename):
    with open(filename, encoding="utf-8") as file:
        size = int(file.readline())
        spots = []
        for line in file:
            if line.strip():
                row, column = map(int, line.split())
                spots.append((row, column))
    return size, spots


def light_strength(row_offset, column_offset):
    # Square rings: the largest offset tells us which ring contains the tile.
    distance = max(abs(row_offset), abs(column_offset))
    if distance == 0:
        return 1.0
    if distance == 1:
        return 0.5
    return 0.2


def light_with_lists(size, spots):
    room = [[0.0 for column in range(size)] for row in range(size)]
    for spot_row, spot_column in spots:
        for row_offset in range(-2, 3):
            for column_offset in range(-2, 3):
                row = spot_row + row_offset
                column = spot_column + column_offset
                if 0 <= row < size and 0 <= column < size:
                    room[row][column] += light_strength(row_offset, column_offset)
    return room


def light_with_dictionary(size, spots):
    room = {(row, column): 0.0 for row in range(size) for column in range(size)}
    for spot_row, spot_column in spots:
        for row_offset in range(-2, 3):
            for column_offset in range(-2, 3):
                row = spot_row + row_offset
                column = spot_column + column_offset
                if 0 <= row < size and 0 <= column < size:
                    room[(row, column)] += light_strength(row_offset, column_offset)
    return room


def exercise_5(filename, method="list"):
    size, spots = read_spotlights(filename)
    if method == "list":
        room = light_with_lists(size, spots)
        for row in room:
            print(" ".join(f"{value:.1f}" for value in row))
    else:
        room = light_with_dictionary(size, spots)
        for row in range(size):
            print(" ".join(f"{room[(row, column)]:.1f}" for column in range(size)))


# Exercise 6: Lighting with NumPy


def light_with_numpy(size, spots):
    room = np.zeros((size, size), dtype=np.float64)
    for spot_row, spot_column in spots:
        for row_offset in range(-2, 3):
            for column_offset in range(-2, 3):
                row = spot_row + row_offset
                column = spot_column + column_offset
                if 0 <= row < size and 0 <= column < size:
                    room[row, column] += light_strength(row_offset, column_offset)
    return room


def exercise_6(filename):
    size, spots = read_spotlights(filename)
    room = light_with_numpy(size, spots)
    for row in room:
        print(" ".join(f"{value:.1f}" for value in row))


# Exercise 7: NumPy functions


def multiplication_table(m, n):
    """Part a: create a float64 matrix with element i * j."""
    matrix = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            matrix[i, j] = i * j
    return matrix


def normalize_columns(matrix):
    """Part b: return a new matrix whose columns each sum to one."""
    column_sums = matrix.sum(axis=0)
    return matrix / column_sums


def normalize_rows(matrix):
    """Part c: keep row sums as a column so division works row by row."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sums


def remove_negatives(array):
    """Part d: replace negative values in a copy, leaving the input alone."""
    result = array.copy()
    result[result < 0] = 0
    return result


def sum_matrix_product(a, b):
    """Part e: sum all elements of the matrix product AB."""
    product = a @ b
    return product.sum()


def exercise_7():
    matrix = np.array([[1.0, 2.0, 6.0, 4.0],
                       [3.0, 4.0, 3.0, 7.0],
                       [1.0, 4.0, 6.0, 9.0]])
    print("a. Multiplication table (3 x 4):")
    print(multiplication_table(3, 4))
    print("\nb. Normalized columns:")
    print(normalize_columns(matrix))
    print("\nc. Normalized rows:")
    print(normalize_rows(matrix.T))
    print("\nd. Negative values replaced with zero:")
    values = np.array([[-2, 3, -1], [0, -5, 4]])
    print(remove_negatives(values))
    print("Original array:")
    print(values)
    print("\ne. Sum of the matrix product:")
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    print(sum_matrix_product(a, b))


# Exercise 8: Gymnastics with NumPy


def read_scores(filename):
    with open(filename, encoding="utf-8") as file:
        count = int(file.readline())
        names = []
        scores = np.zeros((count, 5), dtype=np.float64)
        for i in range(count):
            fields = file.readline().split()
            names.append(fields[0] + " " + fields[1])
            scores[i] = [float(value) for value in fields[3:]]
    return names, scores


def final_scores(scores):
    # Sort within each competitor's row and add only the middle three scores.
    sorted_scores = np.sort(scores, axis=1)
    return sorted_scores[:, 1:-1].sum(axis=1)


def best_three(scores):
    totals = final_scores(scores)
    # Negative totals put the largest score first; stable keeps input order on ties.
    indices = np.argsort(-totals, kind="stable")[:3]
    return indices, totals


def exercise_8(filename):
    names, scores = read_scores(filename)
    indices, totals = best_three(scores)
    print("Final ranking:")
    for position, index in enumerate(indices, 1):
        print(f"{position}: {names[index]} - Score: {totals[index]:.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exercise", type=int, choices=range(1, 9))
    parser.add_argument("filename", nargs="?")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-b", metavar="BUS_ID", help="Exercise 2: bus distance")
    group.add_argument("-l", metavar="ROUTE_ID", help="Exercise 2: route speed")
    parser.add_argument("--method", choices=["list", "dict"], default="list",
                        help="Exercise 5: room representation")
    args = parser.parse_args()

    if args.exercise != 7 and args.filename is None:
        parser.error("This exercise needs an input filename.")
    if args.exercise == 2 and args.b is None and args.l is None:
        parser.error("Exercise 2 needs either -b BUS_ID or -l ROUTE_ID.")
    if args.exercise != 2 and (args.b is not None or args.l is not None):
        parser.error("The -b and -l flags are only used in Exercise 2.")

    try:
        if args.exercise == 1:
            exercise_1(args.filename)
        elif args.exercise == 2:
            exercise_2(args.filename, args.b, args.l)
        elif args.exercise == 3:
            exercise_3(args.filename)
        elif args.exercise == 4:
            exercise_4(args.filename)
        elif args.exercise == 5:
            exercise_5(args.filename, args.method)
        elif args.exercise == 6:
            exercise_6(args.filename)
        elif args.exercise == 7:
            exercise_7()
        elif args.exercise == 8:
            exercise_8(args.filename)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
