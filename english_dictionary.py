import csv
from enum import Enum


class WordsPart(Enum):
    FULL = 0
    PREFIX = 1
    SUFFIX = 2
class EnglishDictionary:
    """
    This class represents the english language and its characteristics and language model (letter & letter-pairs
    frequency as well as common english words).
    """
    def __init__(self, words_path: str, pair_dict_path: str, letter_dict_path: str, prefix_size=3, suffix_size=3):
        self.words = self._read_file_as_set(words_path,WordsPart.FULL)
        self.letter_pairs_to_freq = self._read_file_as_dict(pair_dict_path)
        self.letter_to_freq = self._read_file_as_dict(letter_dict_path)
        self.prefix_size = prefix_size
        self.prefix = self._read_file_as_set(words_path, WordsPart.PREFIX, size=prefix_size)
        self.suffix_size = suffix_size
        self.suffix = self._read_file_as_set(words_path, WordsPart.SUFFIX, size=suffix_size)
        # letter_freq = {}
        # for first_letter, second_letter in self.letter_pairs_to_freq:
        #     letter_pairs = f"{first_letter}{second_letter}"
        #     if first_letter in letter_freq:
        #         letter_freq[first_letter] += self.letter_pairs_to_freq[letter_pairs]
        #     else:
        #         letter_freq[first_letter] = self.letter_pairs_to_freq[letter_pairs]

        # self.bigram_to_freq = {}
        # for first_letter, second_letter in self.letter_pairs_to_freq:
        #     bigram_txt = f"{first_letter}{second_letter}"
        #     self.bigram_to_freq[bigram_txt] = self.letter_pairs_to_freq[bigram_txt] / letter_freq[first_letter]

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
    def _read_file_as_set(file_path: str, what_part: WordsPart, size: int = 3) -> set:
        data = set()
        with open(file_path, "r") as file:
            reader = csv.reader(file, delimiter="\n")
            for row in reader:
                try:
                    data.add(EnglishDictionary.word_to_partial(row[0], what_part, size).lower())
                except:
                    pass
        return data

    @staticmethod
    def word_to_partial(word, what_part: WordsPart, size: int):
        if what_part is WordsPart.SUFFIX:
            return word[-size:]
        elif what_part is WordsPart.PREFIX:
            return word[:size]
        elif what_part is WordsPart.FULL:
            return word
        return word

    @staticmethod
    def _read_file_as_prefix_set(file_path, prefix_size=3) -> set:
        data = set()
        with open(file_path, "r") as file:
            reader = csv.reader(file, delimiter="\n")
            for row in reader:
                try:
                    data.add(row[0][:prefix_size].lower())
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
