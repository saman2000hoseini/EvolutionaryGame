import copy
import random

from player import Player
import numpy as np
from config import CONFIG


def weighted_random_choice(num_players, options, total_fitness):
    choices = []
    for i in range(num_players):
        pick = random.uniform(0, total_fitness)
        current = 0
        for j in range(len(options)):
            current += options[j].fitness
            if current > pick:
                choices.append(options[j])
    return choices


def sus(num_players, options, total_fitness):
    choices = []
    current = 0
    pick = total_fitness * random.random() / num_players
    for i in range(len(options)):
        current += options[i].fitness
        if current > pick:
            choices.append(options[i])
            pick += total_fitness / num_players
    return choices


class Evolution:

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    # def calculate_fitness(self, players, delta_xs):
    #     for i, p in enumerate(players):
    #         if delta_xs[i] < 1000:
    #             p.fitness = delta_xs[i] / 10
    #         else:
    #             p.fitness = delta_xs[i] * delta_xs[i]

    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i] / 10 + p.boxes * p.distances * 10

    def mutate(self, parent):
        child = copy.copy(parent)
        if random.random() < 0.6:
            child.nn.b0 += np.random.normal(0, 1, child.nn.b0.shape) / 500
        if random.random() < 0.6:
            child.nn.b1 += np.random.normal(0, 1, child.nn.b1.shape) / 500
        if random.random() < 0.6:
            child.nn.b2 += np.random.normal(0, 1, child.nn.b2.shape) / 500
        if random.random() < 0.6:
            child.nn.w0 += np.random.normal(0, 1, child.nn.w0.shape) / 500
        if random.random() < 0.6:
            child.nn.w1 += np.random.normal(0, 1, child.nn.w1.shape) / 500
        if random.random() < 0.6:
            child.nn.w2 += np.random.normal(0, 1, child.nn.w2.shape) / 500

        return child

    def generate_new_population(self, num_players, prev_players=None):
        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:
            # print("------------------------------------------")
            # print(players_copy[0].nn.w0)

            total_fitness = 0
            for i in range(len(prev_players)):
                total_fitness += prev_players[i].fitness
                # print(prev_players[i].fitness)

            new_players = []

            # choices = sus(num_players, prev_players, total_fitness)
            choices = weighted_random_choice(num_players, prev_players, total_fitness)

            random.shuffle(choices)

            for i in range(num_players):
                new_players.append(self.mutate(choices[i]))
            # num_players example: 150
            # prev_players: an array of `Player` objects

            # TODO (additional): implementing crossover
            # np.random.shuffle(new_players)
            return new_players

    def next_population_selection(self, players, num_players):
        new_players = sorted(players, key=lambda x: x.fitness, reverse=True)

        # for i in range(5):
        #     print(new_players[i].fitness)
        # print("---------------------------------------")

        # TODO (additional): a selection method other than `top-k`
        # TODO (additional): plotting

        return new_players[: num_players]
