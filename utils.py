import csv


def read_file_as_dict(file_path):
    data = {}
    with open(file_path, "r") as file:
        reader = csv.reader(file, delimiter="\n")
        for row in reader:
            try:
                value, key = row[0].split("\t")
                data[key] = float(value)
            except:
                pass
    return data


if __name__ == "__main__":
    data = read_file_as_dict("Letter2_Freq.txt")
    assert data["ZE"] == 0.0003, f"bad value:"  # {data['ZE']}'
