from utils import read_file_as_dict


class Fitness:
    def __init__(self, word_dict_path, pair_dict_path, letter_dict_path):
        self.letter_to_freq = read_file_as_dict(letter_dict_path)
        self.pair_to_freq = read_file_as_dict(pair_dict_path)

    def calc_freq(self, symble, freq_dict):
        if symble in freq_dict:
            return freq_dict[symble]
        return 0

    def fitness(self, text, lamda_arr=(1, 1, 1)):
        sum_letter = 0
        for l in text:
            sum_letter += self.calc_freq(l, self.letter_to_freq)
        # TODO:iterate over words
        sum_words = 0
        # TODO:iterate over pairs
        sum_pairs = 0
        return (
            lamda_arr[2] * sum_letter
            + lamda_arr[1] * sum_pairs
            + lamda_arr[0] * sum_words
        )
