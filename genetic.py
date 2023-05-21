from __future__ import annotations

import functools
import statistics
from typing import Dict, List, Tuple, Optional
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
        self._set_letters = set(self._letters)
        self._n_letters = len(self._letters)
        self._not_in_permutation = exempt_from_permutation or Permutation.EXEMPT_FROM_PERMUTATION
        permutation_dict = permutation_dict or self._generate_solution()
        if not self.is_valid_permutation_dict(permutation_dict):
            print(f"not valid permutation:{permutation_dict}")
        self._permutation = permutation_dict

    def clone_permutation(self,permutation) -> Permutation:
        return Permutation(english_dictionary=self._english_dictionary
                           ,permutation_dict=permutation
                           ,exempt_from_permutation=self._not_in_permutation)

    @staticmethod
    def swap_permute(permute: Dict, letter1: str, letter2: str):
        y1 = permute[letter1]
        permute[letter1] = permute[letter2]
        permute[letter2] = y1

    def local_optimize(self, n:int):
        candidate = dict(self._permutation)
        for _ in range(n):
            letter1, letter2 = random.sample(candidate.keys(), 2)
            self.swap_permute(candidate, letter1, letter2)
        return self.clone_permutation(permutation=candidate)



    def is_valid_permutation_dict(self, permutation_dict: Dict[str, str] = None) -> bool:
        """
        :param permutation_dict: dictionary
        :return: returns whether the permutation dictionary is valid (meaning that it correlates
         with the english-dictionary)
        """
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
        return [txt.lower() for txt in split_txt if len(txt) != 0]

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
        for letter in token.lower():
            if letter in self._permutation:
                value = self._permutation[letter]
            else:
                value = letter
            total_str += value
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
        letters_from_second_parent = first_parent._set_letters - letters_from_first_parent

        not_allocated_source_letters = set()
        for letter in letters_from_second_parent:
            target_letter = second_parent._permutation[letter]
            if target_letter in allocated_target_letters:
                # errors
                not_allocated_source_letters.add(letter)
            else:
                permutation_dict[letter] = target_letter
                allocated_target_letters.add(target_letter)

        not_allocated_target_letters = list(first_parent._set_letters - allocated_target_letters)

        # fix errors
        assert len(not_allocated_source_letters) == len(not_allocated_target_letters), "not the same length"
        for not_allocated_source_letter in not_allocated_source_letters:
            n_targets = len(not_allocated_target_letters)
            idx = random.randint(0, n_targets - 1)
            permutation_dict[not_allocated_source_letter] = not_allocated_target_letters[idx]
            not_allocated_target_letters.pop(idx)

        return Permutation(
            permutation_dict=permutation_dict,
            english_dictionary=first_parent._english_dictionary
        )

    @staticmethod
    def mutation(permutation: Permutation, probability: float):
        if probability < 0 or probability > 1:
            print(f"invalid probability to mutate:{probability}")
            return

        new_permutation = Permutation(english_dictionary=permutation._english_dictionary,
                                      permutation_dict=permutation._permutation.copy())
        n_mutations = int(permutation._n_letters * probability)
        for index_mutation in range(n_mutations):
            letter_1, letter_2 = random.sample(permutation._letters, k=2)
            target_letter_1 = new_permutation._permutation[letter_1]
            target_letter_2 = new_permutation._permutation[letter_2]

            new_permutation._permutation[letter_1] = target_letter_2
            new_permutation._permutation[letter_2] = target_letter_1
        return new_permutation

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
                cnt_letters += self._english_dictionary.letter_to_freq[translated_token[0]]
                for first_letter, second_letter in zip(translated_token, translated_token[1:]):
                    # count the pairs of unsolved tokens
                    pair = f"{first_letter}{second_letter}"
                    pair_freq = self._english_dictionary.letter_pairs_to_freq[pair]
                    cnt_total_pairs += pair_freq
                    # count the unsolved token frequency
                    cnt_letters += self._english_dictionary.letter_to_freq[second_letter]

        return 2 * cnt_correct_token + 15 * cnt_total_pairs + cnt_letters


class Solver:
    def __init__(self, population_size: int, text: str, english_dictionary: EnglishDictionary):
        self._population_size = population_size  # number of solution in each generation
        self._text = text
        self._english_dictionary = english_dictionary
        self._permutations = []

        # generation statistics
        self._best_score = -100.0
        self._worst_score = 100.0
        self._gen_score = 0
        self._best_sol: Optional[Permutation] = None

        self._best_score_in_all_executions = -100
        self._best_sol_in_all_executions: Optional[Permutation] = None

        # population settings
        self._keep_old_percentage = 0.15  # elite
        self._mutation_percentage = 0.05
        self._crossover_percentage = 0.5

        self._n_keep_old = int(self._population_size * self._keep_old_percentage)
        self._n_mutations = int(self._population_size * self._mutation_percentage)
        self._n_crossovers = int(self._population_size * self._crossover_percentage)

        self._n_random = max(self._population_size - (self._n_keep_old + self._n_mutations + self._n_crossovers), 0)

        self._generation = self._generate_generation_solutions_fitness(self._population_size)

    def _generate_generation(self, size: int) -> List[Permutation]:
        """
        This function is responsible for generating permutations
        """
        generation: List[Permutation] = []
        for i in range(size):
            generation.append(Permutation(english_dictionary=self._english_dictionary))
        return generation

    def _generate_generation_solutions_fitness(self, size: int) -> List[Tuple[Permutation, float]]:
        solutions = self._generate_generation(size)
        return sorted(
            [(solution, solution.fitness(self._text)) for solution in solutions],
            reverse=True,
            key=lambda x: x[1]
        )

    def evaluate_generation(self, solutions: List[Permutation]) -> List[Tuple[Permutation, float]]:
        """
        :param solutions: list of permutations
        :return: returns a list that is constructed of the solutions fitness
        """
        return sorted(
            [(solution, solution.fitness(txt=self._text)) for solution in solutions],
            reverse=True,
            key=lambda x: x[1]
        )

    @staticmethod
    def sort_solutions(solutions: List[Tuple[Permutation, float]]) -> List[Tuple[Permutation, float]]:
        return sorted(solutions, reverse=True, key=lambda x: x[1])

    def update_solution(self):
        raise NotImplementedError()

    def execution_stat(self, solutions: List[Tuple[Permutation, float]]):
        generation_fitness_list = [fitness_score for _, fitness_score in solutions]

        best_permutation_in_this_turn, best_fitness_in_this_turn = solutions[0]

        # list is ordered id a descending fitness order
        self._gen_score = sum(generation_fitness_list) / self._population_size

        if best_fitness_in_this_turn > self._best_score:
            self._best_sol = best_permutation_in_this_turn
            self._best_score = best_fitness_in_this_turn
            print(f"best score found:{best_fitness_in_this_turn}")

        if best_fitness_in_this_turn > self._best_score_in_all_executions:
            self._best_sol_in_all_executions = best_permutation_in_this_turn
            self._best_score_in_all_executions = best_fitness_in_this_turn
            print(f"best score in all executions found:{best_fitness_in_this_turn}")

        self._worst_score = generation_fitness_list[-1]

    def solve(self, num_of_generations: int = 200, n_stuck: int = 100):
        run = True
        n_generation = 0
        n_turns_with_best_score = 0
        last_best_score = self._best_score

        while run:
            self.update_solution()
            print(f"mean gen {n_generation} score:{self._gen_score}, best score:{self._best_score}, best total score:{self._best_score_in_all_executions}")

            #  update number of turns with best score
            if last_best_score == self._best_score:
                n_turns_with_best_score += 1
            if self._best_score > last_best_score:
                n_turns_with_best_score = 0

            # is in local maximum
            if n_turns_with_best_score == n_stuck:
                print(f"start over after {n_generation} generations")
                self.start_over()
                n_turns_with_best_score = 0

            if n_generation >= num_of_generations:
                run = False

            last_best_score = self._best_score
            n_generation += 1

    def start_over(self):
        """
        This function starts the generation from the beginning again (probably because it stuck)
        """
        self._generation = self._generate_generation_solutions_fitness(self._population_size)
        self._best_score = -100.0
        self._best_sol = None
        print(
            (
                f"Although starting over the best solution in "
                f"all execution is with fitness: {self._best_score_in_all_executions}"
            )
        )

    def next_generation(
            self,
            generation_fitness_tuples_list: List[Tuple[Permutation, float]]
    ) -> List[Tuple[Permutation, float]]:
        """
        This function is responsible for creating the next generation given the current one.
        :param generation_fitness_tuples_list: list of tuples, each tuple contains a Permutation and its fitness score
        :return: the next generation. list of tuples, each tuple contains a Permutation and its fitness score
        """
        next_gen: List[Tuple[Permutation, float]] = []

        # add random generations
        next_gen.extend(self._generate_generation_solutions_fitness(self._n_random))

        # keep elitism
        next_gen.extend(generation_fitness_tuples_list[:self._n_keep_old])

        # calculate chances to better select crossover couples
        all_fitness = [fit for _, fit in generation_fitness_tuples_list]
        sum_fitness = sum(all_fitness)
        chances = [fit / sum_fitness for fit in all_fitness]

        # crossovers
        crossovers: List[Permutation] = []
        for i in range(self._n_crossovers):
            (permutation_1, fitness_1), (permutation_2, fitness_2) = random.choices(
                generation_fitness_tuples_list,
                k=2,
                weights=chances
            )
            crossovers.append(Permutation.crossover(permutation_1, permutation_2))
        next_gen.extend(self.evaluate_generation(crossovers))

        # mutations
        mutations: List[Permutation] = []
        for i in range(self._n_mutations):
            permutation, fitness = random.choice(generation_fitness_tuples_list)
            mutations.append(Permutation.mutation(permutation, 0.05))
        next_gen.extend(self.evaluate_generation(mutations))

        return sorted(next_gen, reverse=True, key=lambda x: x[1])


class NormalSolver(Solver):
    def update_solution(self):
        self.execution_stat(self._generation)
        self._generation = self.next_generation(self._generation)


"""
The main difference between the regular solver and the Lamark & Darwin oriented solutions it the way the genome 
they pass to the next episode.
"""


class LamarkSolver(Solver):
    pass



class DarwinSolver(Solver):
    def __init__(self, population_size: int, text: str, english_dictionary: EnglishDictionary, local_optimization_n: int):
        super().__init__(population_size, text, english_dictionary)
        self.local_optimization_n = local_optimization_n


    def _generate_generation_solutions_fitness(self, size: int) -> List[Tuple[Permutation, float]]:
        solutions = self._generate_generation(size)
        return sorted(
            [(solution, solution.local_optimize(self.local_optimization_n).fitness(self._text)) for solution in solutions],
            reverse=True,
            key=lambda x: x[1]
        )


if __name__ == "__main__":
    dictionary = EnglishDictionary('dict.txt', 'Letter2_Freq.txt', 'Letter_Freq.txt')
    with open(r"enc.txt", "r") as f:
        txt = f.read()
    # solver = NormalSolver(population_size=1000, text=txt, english_dictionary=dictionary)
    solver = NormalSolver(population_size=750, text=txt, english_dictionary=dictionary)

    solver.solve(num_of_generations=400, n_stuck=100)
    print(solver._best_sol_in_all_executions.translate(txt))
