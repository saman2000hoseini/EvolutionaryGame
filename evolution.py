import copy
import random

from player import Player
import numpy as np
from config import CONFIG
from util import append_list_as_row


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
        while current > pick:
            choices.append(options[i])
            pick += total_fitness / num_players
    return choices


def tournament_selection(population, num):
    generation = []
    for i in range(num):
        parents = random.choices(population, k=20)
        generation.append(sorted(parents, key=lambda agent: agent.fitness, reverse=True)[0])
    return generation


def calculate_total_fitness(players):
    total_fitness = 0
    for i in range(len(players)):
        total_fitness += players[i].fitness
    return total_fitness


class Evolution:

    def __init__(self, mode):
        self.mode = mode

    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.score = delta_xs[i]
            p.fitness = delta_xs[i]

    # def calculate_fitness(self, players, delta_xs):
    #     for i, p in enumerate(players):
    #         p.fitness = delta_xs[i] * delta_xs[i] / 100 + p.distances * p.distances
    #         p.score = delta_xs[i]

    def mutate(self, parent):
        c = 1
        s = 1

        child = copy.deepcopy(parent)
        child.nn.b0 += np.random.normal(0, s, child.nn.b0.shape) / c
        child.nn.b1 += np.random.normal(0, s, child.nn.b1.shape) / c
        child.nn.w0 += np.random.normal(0, s, child.nn.w0.shape) / c
        child.nn.w1 += np.random.normal(0, s, child.nn.w1.shape) / c

        return child

    def crossover(self, p1, p2):
        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
        if random.random() < 0.5:
            pt = random.randint(1, len(p1.nn.b0) - 2)
            c1.nn.b0 = np.concatenate((p1.nn.b0[:pt], p2.nn.b0[pt:]), axis=0)
            c2.nn.b0 = np.concatenate((p2.nn.b0[:pt], p1.nn.b0[pt:]), axis=0)

            pt = random.randint(1, len(p1.nn.w0) - 2)
            c1.nn.w0 = np.concatenate((p1.nn.w0[:pt], p2.nn.w0[pt:]), axis=0)
            c2.nn.w0 = np.concatenate((p2.nn.w0[:pt], p1.nn.w0[pt:]), axis=0)

        return [c1, c2]

    def generate_new_population(self, num_players, prev_players=None):
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:
            new_players = []
            random.shuffle(prev_players)

            # for i in range(0, len(prev_players), 2):
            #     prev_players[i], prev_players[i+1] = self.crossover(prev_players[i], prev_players[i+1])

            # choices = sus(num_players, prev_players, calculate_total_fitness(prev_players))
            choices = weighted_random_choice(num_players, prev_players, calculate_total_fitness(prev_players))

            random.shuffle(choices)

            for i in range(num_players):
                if random.random() < 0.99:
                    new_players.append(self.mutate(choices[i]))
                else:
                    new_players.append(choices[i])

            random.shuffle(new_players)

            return new_players

    def next_population_selection(self, players, num_players, gen_num):
        # players = tournament_selection(players, num_players)
        players.sort(key=lambda x: x.fitness, reverse=True)

        total_fitness = calculate_total_fitness(players)
        average_fitness = total_fitness / len(players)
        max_fitness = players[0].fitness
        min_fitness = players[-1].fitness
        append_list_as_row("data.csv", [gen_num, average_fitness, max_fitness, min_fitness])

        # for i in range(5):
        #     print(new_players[i].fitness)
        # print("---------------------------------------")

        return players[: num_players]
