import csv


class EnglishDictionary:
    """
    This class represents the english language and its characteristics and language model (letter & letter-pairs
    frequency as well as common english words).
    """
    def __init__(self, words_path: str, pair_dict_path: str, letter_dict_path: str):
        self.words = self._read_file_as_set(words_path)
        self.letter_pairs_to_freq = self._read_file_as_dict(pair_dict_path)
        self.letter_to_freq = self._read_file_as_dict(letter_dict_path)

    @staticmethod
    def _read_file_as_dict(file_path) -> dict[str, float]:
        data = {}
        with open(file_path, "r") as file:
            reader = csv.reader(file, delimiter="\n")
            for row in reader:
                try:
                    value, key = row[0].lower().split("\t")
                    data[key] = float(value)
                except:
                    pass
        return data

    @staticmethod
    def _read_file_as_set(file_path) -> set:
        data = set()
        with open(file_path, "r") as file:
            reader = csv.reader(file, delimiter="\n")
            for row in reader:
                try:
                    data.add(row[0].lower())
                except:
                    pass
        return data