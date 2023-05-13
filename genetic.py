from __future__ import annotations
from typing import Dict, List
import random
import re

from english_dictionary import EnglishDictionary


class Permutation:
    # TODO: maybe separate between the new line and punctuations
    EXEMPT_FROM_PERMUTATION = [r'\s', r'\.', ',', ';', r"\n"]

    def __init__(
            self,
            english_dictionary: EnglishDictionary,
            permutation_dict: Dict[str, str] = None,
            exempt_from_permutation: list[str] = None
    ):
        self._english_dictionary = english_dictionary
        self._letters = list(self._english_dictionary.letter_to_freq.keys())
        self._n_letters = len(self._letters)
        self._not_in_permutation = exempt_from_permutation or Permutation.EXEMPT_FROM_PERMUTATION
        permutation_dict = permutation_dict or self._generate_solution()
        if not self.is_valid_permutation_dict(permutation_dict):
            print(f"not valid permutation:{permutation_dict}")
        self._permutation = permutation_dict

    def is_valid_permutation_dict(self, permutation_dict: Dict[str, str] = None) -> bool:
        language_letters = self._english_dictionary.letter_to_freq.keys()
        permutation_keys = permutation_dict.keys()
        permutation_values = permutation_dict.values()
        if not (len(language_letters) == len(permutation_keys) == len(permutation_values)):
            return False
        for letter in language_letters:
            if letter not in permutation_keys or letter not in permutation_values:
                return False
        return True

    def _generate_solution(self) -> Dict[str, str]:
        """
        :return: returns a dict that maps key and its randomized permutation value
        """
        solution = {}
        permutation = random.sample(self._letters, len(self._letters))
        for letter, permutation_letter in zip(self._letters, permutation):
            solution[letter] = permutation_letter
        return solution

    def __repr__(self):
        return str(self._permutation)

    def _tokenize(self, txt: str) -> List[str]:
        """
        :param txt: str
        :return: list of strings, each string is a word (token) from the txt-input
        """
        regex = '|'.join(self._not_in_permutation)
        split_txt = re.split(regex, txt)
        return [txt.upper() for txt in split_txt if len(txt) != 0]

    @staticmethod
    def _number_of_tokens(tokens: List[str]) -> int:
        """
        :param tokens: list of strings (tokens)
        :return: returns the length all the tokens
        """
        return sum([len(token) for token in tokens])

    def translate(self, token: str) -> str:
        """
        :param token: str
        :return: returns the translated text from the token using the permutation
        """
        total_str = ""
        for letter in token:
            if letter in self._permutation:
                total_str += self._permutation[letter]
            else:
                print(f"not in permutation :{letter}")
                total_str += letter
        return total_str

    @staticmethod
    def crossover(first_parent: Permutation, second_parent: Permutation) -> Permutation:
        """
        :param first_parent: Permutation
        :param second_parent: Permutation
        :return: return a new permutation using crossover rule
        """
        index = random.randint(0, first_parent._n_letters)

        permutation_dict: Dict[str, str] = {}

        # take from first
        letters_from_first_parent = set(first_parent._letters[:index])
        allocated_target_letters = set()
        for letter in letters_from_first_parent:
            permutation_dict[letter] = first_parent._permutation[letter]
            allocated_target_letters.add(first_parent._permutation[letter])

        # take from second
        letters_from_second_parent = set(first_parent._letters) - letters_from_first_parent

        not_allocated_source_letters = set()
        for letter in letters_from_second_parent:
            target_letter = second_parent._permutation[letter]
            if target_letter in allocated_target_letters:
                # errors
                not_allocated_source_letters.add(letter)
            else:
                permutation_dict[letter] = target_letter
                allocated_target_letters.add(target_letter)

        not_allocated_target_letters = list(set(first_parent._letters) - allocated_target_letters)

        # fix errors
        assert len(not_allocated_source_letters) == len(not_allocated_target_letters), "not the same length"
        for not_allocated_source_letter in not_allocated_source_letters:
            n_targets = len(not_allocated_target_letters)
            idx = random.randint(0, n_targets - 1)
            permutation_dict[not_allocated_source_letter] = not_allocated_target_letters[idx]
            not_allocated_target_letters.pop(idx)

        # return
        return Permutation(
            permutation_dict=permutation_dict,
            english_dictionary=first_parent._english_dictionary
        )

    def fitness(self, txt: str) -> float:
        """
        :param txt: str
        :return: parameter's fitness score
        """
        tokens = self._tokenize(txt=txt)

        cnt_correct_token = 0
        cnt_total_pairs = 0
        cnt_letters = 0

        for token in tokens:
            translated_token = self.translate(token=token)
            # token is a word in dictionary
            if translated_token in self._english_dictionary.words:
                cnt_correct_token += len(translated_token)
            else:
                # count the pairs of unsolved tokens
                for first_letter, second_letter in zip(translated_token, translated_token[1:]):
                    pair = f"{first_letter}{second_letter}"
                    pair_freq = self._english_dictionary.letter_pairs_to_freq[pair]
                    cnt_total_pairs += pair_freq
                # count the unsolved token frequency
                for letter in translated_token:
                    if letter in self._english_dictionary.letter_to_freq:
                        cnt_letters += self._english_dictionary.letter_to_freq[letter]
                    else:
                        print(f"letter {letter} in token {token} (or translated{translated_token}) is not part of the english letters")

        return cnt_correct_token + cnt_total_pairs + cnt_letters * 0.1


class Solver:
    def __init__(self, population_size: int, english_dictionary: EnglishDictionary):
        self._population_size = population_size  # number of solution in each generation
        self._english_dictionary = english_dictionary
        self._permutations = []

        self._generate_permutation()

    def _generate_permutation(self) -> None:
        """
        This function is responsible for generating permutations
        """
        self._permutations = []
        for i in range(self._population_size):
            self._permutations.append(Permutation(english_dictionary=self._english_dictionary))

    def solve(self):
        pass


class NormalSolver(Solver):
    pass


"""
The main difference between the regular solver and the Lamark & Darwin oriented solutions it the way the genome 
they pass to the next episode.
"""


class LamarkSolver(Solver):
    pass


class DarwinSolver(Solver):
    pass


if __name__ == "__main__":
    dictionary = EnglishDictionary('dict.txt', 'Letter2_Freq.txt', 'Letter_Freq.txt')

    solver = Solver(population_size=10, english_dictionary=dictionary)
    txt = """Pm ol ohk hufaopun jvumpkluaphs av zhf, ol dyval pa pu jpwoly, aoha pz, if zv johunpun aol vykly vm aol slaalyz vm aol hswohila, aoha uva h dvyk jvbsk il thkl vba."""
    perm1 = solver._permutations[0]
    perm2 = solver._permutations[1]

    perm3 = Permutation.crossover(perm1, perm2)

    print(f"perm3 fitness:{perm3.fitness(txt)}")

    # for permutation in solver._permutations:
    #     print(permutation)
        # print(permutation.fitness(txt))
