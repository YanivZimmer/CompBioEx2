from utils import read_file_as_dict, read_file_as_set
import re


class Fitness:
    def __init__(self, word_set_path, pair_dict_path, letter_dict_path):
        self.letter_to_freq = read_file_as_dict(letter_dict_path)
        self.pair_to_freq = read_file_as_dict(pair_dict_path)
        self.word_freq = read_file_as_set(word_set_path)

    def calc_freq(self, symbol, freq_dict):
        if symbol in freq_dict:
            return freq_dict[symbol]
        return 0

    def fitness(self, text, lamda_arr=(0.1, 1, 1)):
        sum_letter = 0
        sum_pairs = 0
        sum_words = 0
        for l in text:
            sum_letter += self.calc_freq(l, self.letter_to_freq)

        for i in range(len(text) - 2):
            sum_pairs += self.calc_freq(text[i, i + 1], self.pair_to_freq)

        words = re.split(" |, |\t|\n", text)
        for word in words:
            if word in self.word_freq:
                sum_words += 1

        return (
            lamda_arr[2] * sum_letter
            + lamda_arr[1] * sum_pairs
            + lamda_arr[0] * sum_words
        )
