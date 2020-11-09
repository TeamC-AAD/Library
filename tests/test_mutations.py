import pytest
import random

from helper.mutation import (insert_mutation,
    flip_mutation, interchanging_mutation,
    reversing_mutation, uniform_mutation,
    creep_mutation,
)

test_count = 1000
test_array_max = 100000

class TestMutations:
    def test_insert_mutation(self):
        for _ in range(test_count):
            test_array = random.sample(range(test_array_max), 10)

            mutated = insert_mutation(test_array)

            assert(all([allele in test_array for allele in mutated]))
            assert(len(mutated) == len(test_array))

    def test_flip_mutation_single(self):
        for _ in range(10):
            test_array = [(random.random() < 0.5) for _ in range(2)]
            mutated = flip_mutation(test_array, 1)

            print(test_array, mutated)

            assert(sum(mutated) == sum(test_array) + 1 or sum(mutated) == sum(test_array) - 1)


    def test_interchanging_mutation(self):
        pass

    def test_reversing_mutation(self):
        pass

    def test_uniform_mutation(self):
        pass

    def test_creep_mutation(self):
        pass