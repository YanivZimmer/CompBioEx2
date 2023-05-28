from typing import Tuple, Dict
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
        self.letter_trio_to_freq = self._read_words_as_trigram_dict(self.words)

    @staticmethod
    def _read_words_as_trigram_dict(words) -> Tuple[Dict[str, float], Dict[str, float]]:
        trios = {}
        duos = {}
        cnt = 0
        for word in words:
            for first, second, third in zip(word, word[1:], word[2:]):
                duo = f"{first}{second}"
                trio = f"{first}{second}{third}"
                if trio in trios:
                    trios[trio] += 1
                else:
                    trios[trio] = 1
                if duo in duos:
                    duos[duo] += 1
                else:
                    duos[duo] = 1
            cnt += 1

        if cnt > 0:
            for trio in trios:
                duo = trio[:2]
                trios[trio] = trios[trio] / duos[duo]

        return trios

    @staticmethod
    def _read_file_as_dict(file_path) -> Dict[str, float]:
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


if __name__ == "__main__":
    dictionary = EnglishDictionary('dict.txt', 'Letter2_Freq.txt', 'Letter_Freq.txt')

    # total = 0
    # freq_one_letter = 0
    # for letter, freq in dictionary.letter_to_freq.items():
    #     freq_one_letter += freq
    #
    # freq_letter_pairs = 0
    # for letter, freq in dictionary.letter_pairs_to_freq.items():
    #     freq_letter_pairs += dictionary.letter_pairs_to_freq[letter]

    # for letter in dictionary.letter_to_freq:
    #     cnt = 0
    #     for second_letter in dictionary.letter_to_freq:
    #         cnt = cnt + dictionary.bigram_to_freq[f"{letter}{second_letter}"]
    #     total += cnt
    # print(f"total bigrams:{total}")
