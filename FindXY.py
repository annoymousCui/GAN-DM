import csv
import numpy as np

def findXY(path):
    path_name = path.split(".")[0]+".csv"
    with open('./input/recard/'+path_name, 'r') as file:
        csv_reader = csv.reader(file)

        data_list = []

        for i, row in enumerate(csv_reader):
            if i != 0:
                if row:
                    row_list = [int(row[0]), int(row[1])]
                    data_list.append(row_list)

        arr = np.full((512, 512), False, dtype=bool)

        for coord in data_list:
            row, col = coord
            arr[row, col] = True

    return arr