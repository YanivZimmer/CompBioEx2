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


def read_file_as_set(file_path):
    data = set()
    with open(file_path, "r") as file:
        reader = csv.reader(file, delimiter="\n")
        for row in reader:
            try:
                data.add(row[0])
            except:
                pass
    return data


if __name__ == "__main__":
    data = read_file_as_dict("Letter2_Freq.txt")
    assert data["ZE"] == 0.0003, f"bad value"
    words = read_file_as_set("dict.txt")
    assert "active" in words, f"bad words set"
