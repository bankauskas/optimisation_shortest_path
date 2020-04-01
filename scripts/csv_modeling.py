import csv
import os
import io


def save_csv(data_to_write, columns, file):
    """
    """
    with io.open(file, 'a', encoding="utf-8", newline='') as myfile:
        writer = csv.DictWriter(myfile, fieldnames=columns)
        writer.writerow(data_to_write)

    