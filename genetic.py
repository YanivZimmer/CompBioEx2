from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional, Callable
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

    def local_optimize(self, n: int):
        candidate = self._permutation.copy()
        for _ in range(n):
            letter1, letter2 = random.sample(sorted(candidate.keys()), 2)
            self.swap_permute(candidate, letter1, letter2)
        return self.clone_permutation(permutation=candidate)

    def clone_permutation(self, permutation) -> Permutation:
        return Permutation(english_dictionary=self._english_dictionary
                           , permutation_dict=permutation
                           , exempt_from_permutation=self._not_in_permutation)

    @staticmethod
    def swap_permute(permute: Dict, letter1: str, letter2: str):
        y1 = permute[letter1]
        permute[letter1] = permute[letter2]
        permute[letter2] = y1

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
        cnt_letter_trios = 0
        cnt_letter_pairs = 0
        cnt_letters = 0

        for token in tokens:
            translated_token = self.translate(token=token)
            token_len = len(token)

            # token is a word in dictionary
            if translated_token in self._english_dictionary.words:
                cnt_correct_token += token_len
            cnt_letters += self._english_dictionary.letter_to_freq[translated_token[0]]
            for first_letter, second_letter in zip(translated_token, translated_token[1:]):
                # count the pairs of unsolved tokens
                pair = f"{first_letter}{second_letter}"
                pair_freq = self._english_dictionary.letter_pairs_to_freq[pair] #/ self._english_dictionary.letter_to_freq[first_letter]
                cnt_letter_pairs += pair_freq
                # count the unsolved token frequency
                cnt_letters += self._english_dictionary.letter_to_freq[second_letter]

            for first_letter, second_letter, third_letter in zip(translated_token, translated_token[1:], translated_token[2:]):
                letters_trio = f"{first_letter}{second_letter}{third_letter}"
                if letters_trio in self._english_dictionary.letter_trio_to_freq:
                    trio_freq = self._english_dictionary.letter_trio_to_freq[letters_trio]
                    cnt_letter_trios += trio_freq
        # print(f"n words:{20 * cnt_correct_token}, trios:{cnt_letter_trios * 10}, letter pairs:{cnt_letter_pairs * 2}, letters:{cnt_letters * 0.5}")
        return 20 * cnt_correct_token + cnt_letter_trios * 10 + cnt_letter_pairs * 2 + cnt_letters * 0.5


class Solver:
    def __init__(
            self,
            population_size: int,
            text: str,
            english_dictionary: EnglishDictionary,
            crossover_choose_func: str,
            tournament_winner_probability: float,
            tournament_size: int,
    ):
        self._population_size = population_size  # number of solution in each generation
        self._text = text
        self._english_dictionary = english_dictionary
        self._crossover_choose_func = crossover_choose_func
        self._tournament_winner_probability = tournament_winner_probability
        self._tournament_size = tournament_size

        self._permutations = []

        # generation statistics
        self._best_score = -100.0
        self._worst_score = 100.0
        self._gen_score = 0
        self._best_sol: Optional[Permutation] = None

        self._number_of_fitness_executions_in_all_executions = 0
        self._best_score_in_all_executions = -100
        self._best_sol_in_all_executions: Optional[Permutation] = None

        self._fitness_counter = 0
        # population settings
        self._keep_old_percentage = 0.1  # elite
        self._mutation_percentage = 0.4
        self._crossover_percentage = 0.9

        self._n_keep_old = int(self._population_size * self._keep_old_percentage)
        self._n_mutations = int(self._population_size * self._mutation_percentage)
        self._n_crossovers = int(self._population_size * self._crossover_percentage)

        self._n_random = max(self._population_size - (self._n_keep_old + self._n_mutations + self._n_crossovers), 0)
        self._generation = self._generate_generation_solutions_fitness(self._population_size)

        self._tournament_prob = [self._tournament_winner_probability]
        for i in range(1, self._tournament_size):
            self._tournament_prob.append(self._tournament_prob[i - 1] * (1.0 - self._tournament_winner_probability))

        self._selection_func = {
            "WeightedFitness": self.weighted_fitness_index_choose,
            "Tournament": self.tournament_selection,
            "Rank": self.ranks_fitness_index_choose
        }

    def crossover_func(self, func_name: str) -> Optional[Callable]:
        if func_name in self._selection_func:
            return self._selection_func[func_name]
        return None

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
        self._fitness_counter += size
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
        self._fitness_counter += len(solutions)
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

        if self._gen_score > best_fitness_in_this_turn or best_fitness_in_this_turn != max(generation_fitness_list):
            print(f"generation max:{best_fitness_in_this_turn}, {max(generation_fitness_list)}")
            print(f"len {self._population_size}, {len(solutions)}")
            raise Exception()

        if best_fitness_in_this_turn > self._best_score:
            self._best_sol = best_permutation_in_this_turn
            self._best_score = best_fitness_in_this_turn
            print(f"best score found:{best_fitness_in_this_turn}")

        if best_fitness_in_this_turn > self._best_score_in_all_executions:
            self._best_sol_in_all_executions = best_permutation_in_this_turn
            self._best_score_in_all_executions = best_fitness_in_this_turn
            self._number_of_fitness_executions_in_all_executions = self._fitness_counter
            print(
                f"best score in all executions found:{best_fitness_in_this_turn},"
                f" fitness executions:{self._number_of_fitness_executions_in_all_executions}"
            )

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
        self._fitness_counter = 0
        self._best_sol = None
        print(
            (
                f"Although starting over the best solution in "
                f"all execution is with fitness: {self._best_score_in_all_executions}"
            )
        )

    def weighted_fitness_index_choose(self, generation_fitness_tuples_list: List[Tuple[Permutation, float]], size: int):
        all_fitness = [fit for _, fit in generation_fitness_tuples_list]
        sum_fitness = sum(all_fitness)
        chances = [fit / sum_fitness for fit in all_fitness]
        for i in range(size):
            yield random.choices(
                generation_fitness_tuples_list,
                k=2,
                weights=chances
            )

    def ranks_fitness_index_choose(self, generation_fitness_tuples_list: List[Tuple[Permutation, float]], size: int):
        population_size = len(generation_fitness_tuples_list)
        chances = [math.sqrt(population_size - i) for i in range(population_size)]

        assert len(chances) == population_size, f"chances len:{len(chances)}, pop size:{population_size}"

        for i in range(size):
            yield random.choices(
                generation_fitness_tuples_list,
                k=2,
                weights=chances
            )

    def tournament_selection(self,
                             generation_fitness_tuples_list: List[Tuple[Permutation, float]],
                             size: int):

        for i in range(size):
            # randomly select indexes
            indexes_1 = random.choices(range(len(generation_fitness_tuples_list)), k=self._tournament_size)
            indexes_2 = random.choices(range(len(generation_fitness_tuples_list)), k=self._tournament_size)
            # take the randomized indexes
            gen_1 = [generation_fitness_tuples_list[idx] for idx in indexes_1]
            gen_2 = [generation_fitness_tuples_list[idx2] for idx2 in indexes_2]
            # sort by fitness
            sorted_gen_1 = sorted(gen_1, reverse=True, key=lambda x: x[1])
            sorted_gen_2 = sorted(gen_2, reverse=True, key=lambda x: x[1])
            # randomize by prob list
            idx1 = random.choices(sorted_gen_1, k=1, weights=self._tournament_prob)[0]
            idx2 = random.choices(sorted_gen_2, k=1, weights=self._tournament_prob)[0]
            yield idx1, idx2

    def tournament(self, permutations: List[Permutation]) -> Tuple[Permutation, float]:
        solutions_with_fitness = self.evaluate_generation(permutations)
        return solutions_with_fitness[0]

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
        next_gen = self.sort_solutions(next_gen)
        # crossovers
        crossovers: List[Permutation] = []
        selection_function = self.crossover_func(self._crossover_choose_func)
        for (permutation_1, fitness_1), (permutation_2, fitness_2) in selection_function(
                generation_fitness_tuples_list=generation_fitness_tuples_list,
                size=self._n_crossovers):
            crossovers.append(Permutation.crossover(permutation_1, permutation_2))
        next_gen.extend(self.evaluate_generation(crossovers))

        next_gen = sorted(next_gen, reverse=True, key=lambda x: x[1])
        next_gen_len = len(next_gen)
        # mutations
        mutations: List[Permutation] = []

        number_of_mutations = 1
        for i in range(self._n_mutations):
            idx = random.randint(int(next_gen_len * 0.05) + 1, next_gen_len - number_of_mutations)
            permutation, fitness = next_gen[idx]
            # remove sampled permutation
            next_gen.pop(idx)
            mutations.append(Permutation.mutation(permutation, 0.05))
            number_of_mutations += 1
        next_gen.extend(self.evaluate_generation(mutations))

        return self.sort_solutions(next_gen)


class NormalSolver(Solver):
    def update_solution(self):
        self._generation = Solver.sort_solutions(self._generation)
        self.execution_stat(self._generation)
        self._generation = self.next_generation(self._generation)


"""
The main difference between the regular solver and the Lamark & Darwin oriented solutions it the way the genome 
they pass to the next episode.
"""


class LamarkSolver(Solver):
    def __init__(
            self,
            population_size: int,
            text: str,
            english_dictionary: EnglishDictionary,
            crossover_choose_func: str,
            tournament_winner_probability: float,
            tournament_size: int,
            n_local_optimization: int):
        self.n_local_optimization = n_local_optimization
        super().__init__(population_size=population_size,
                         text=text,
                         english_dictionary=english_dictionary,
                         crossover_choose_func=crossover_choose_func,
                         tournament_winner_probability=tournament_winner_probability,
                         tournament_size=tournament_size)

    def optimize(self, gen):
        solutions = [t[0] for t in gen]
        candidates = [solution.local_optimize(self.n_local_optimization) for solution in solutions]
        best_after_local_opt = [self.get_the_better_with_fitness(x, y) for x, y in zip(solutions, candidates)]

        return sorted(
            best_after_local_opt,
            reverse=True,
            key=lambda x: x[1]
        )

    def update_solution(self):
        self.execution_stat(self._generation)
        self._generation = self.next_generation(self._generation)

    def get_the_better_with_fitness(self, candidate1: Permutation, candidate2: Permutation) -> Tuple[Permutation, float]:
        fitness1 = candidate1.fitness(self._text)
        fitness2 = candidate2.fitness(self._text)
        if fitness1 > fitness2:
            return candidate1, fitness1
        return candidate2, fitness2

    def _generate_generation_solutions_fitness(self, size: int) -> List[Tuple[Permutation, float]]:
        solutions = self._generate_generation(size)
        candidates = [solution.local_optimize(self.n_local_optimization) for solution in solutions]
        best_after_local_opt = [self.get_the_better_with_fitness(x, y) for x, y in zip(solutions, candidates)]

        return sorted(
            best_after_local_opt,
            reverse=True,
            key=lambda x: x[1]
        )


class DarwinSolver(Solver):
    def __init__(
            self,
            population_size: int,
            text: str,
            english_dictionary: EnglishDictionary,
            crossover_choose_func: str,
            tournament_winner_probability: float,
            tournament_size: int,
            n_local_optimization: int):
        self.n_local_optimization = n_local_optimization
        super().__init__(
            population_size=population_size,
            text=text,
            english_dictionary=english_dictionary,
            crossover_choose_func=crossover_choose_func,
            tournament_winner_probability=tournament_winner_probability,
            tournament_size=tournament_size
        )

    def optimize(self, gen):
        solutions = [t[0] for t in gen]
        return sorted(
            [(solution,
              max(solution.local_optimize(self.n_local_optimization).fitness(self._text),
                  solution.fitness(self._text))) for solution in
             solutions],
            reverse=True,
            key=lambda x: x[1]
        )

    def update_solution(self):
        self.execution_stat(self._generation)
        self._generation = self.next_generation(self._generation)

    def _generate_generation_solutions_fitness(self, size: int) -> List[Tuple[Permutation, float]]:
        solutions = self._generate_generation(size)
        return sorted(
            [(solution, solution.local_optimize(self.n_local_optimization).fitness(self._text)) for solution in solutions],
            reverse=True,
            key=lambda x: x[1]
        )



if __name__ == "__main__":
    dictionary = EnglishDictionary('dict.txt', 'Letter2_Freq.txt', 'Letter_Freq.txt')
    with open(r"enc.txt", "r") as f:
        txt = f.read()
    """
    solver = NormalSolver(
        population_size=200,
        text=txt,
        english_dictionary=dictionary,
        crossover_choose_func="Tournament",
        # crossover_choose_func="Rank",
        # crossover_choose_func="WeightedFitness",
        tournament_winner_probability=0.3,
        tournament_size=7
    )
    solver.solve(num_of_generations=200, n_stuck=50)
    print(solver._best_sol_in_all_executions)
    print(solver._best_sol_in_all_executions.translate(txt))
    print(f"number called to fitness:{solver._number_of_fitness_executions_in_all_executions}")
    """
    lamark_solver = LamarkSolver(
        population_size=200,
        text=txt,
        english_dictionary=dictionary,
        crossover_choose_func="Tournament",
        tournament_winner_probability=0.3,
        tournament_size=7,
        n_local_optimization=2
    )
    lamark_solver.solve(num_of_generations=200, n_stuck=50)
    print(lamark_solver._best_sol_in_all_executions)
    print(lamark_solver._best_sol_in_all_executions.translate(txt))
    print(f"number called to fitness:{lamark_solver._number_of_fitness_executions_in_all_executions}")

