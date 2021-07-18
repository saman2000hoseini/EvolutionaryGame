import copy
import random

from player import Player
import numpy as np
from config import CONFIG
from util import append_list_as_row


def weighted_random_choice(num_players, players, total_fitness):
    # return np.random.choice(players, num_players, replace=True, p=lambda agent: agent.fitness)
    choices = []
    for i in range(num_players):
        pick = random.uniform(0, total_fitness)
        current = 0
        for j in range(len(players)):
            current += players[j].fitness
            if current > pick:
                choices.append(players[j])
                break
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
        parents = random.choices(population, k=10)
        generation.append(copy.deepcopy(max(parents, key=lambda x: x.fitness)))
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
            p.fitness = delta_xs[i]

    def mutate(self, parent):
        s = 0.5
        p = 0.9

        child = copy.deepcopy(parent)
        if random.random() < p:
            child.nn.b0 += np.random.normal(0, s, child.nn.b0.shape)
        if random.random() < p:
            child.nn.b1 += np.random.normal(0, s, child.nn.b1.shape)
        if random.random() < p:
            child.nn.w0 += np.random.normal(0, s, child.nn.w0.shape)
        if random.random() < p:
            child.nn.w1 += np.random.normal(0, s, child.nn.w1.shape)

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
        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:
            prev_players = tournament_selection(prev_players, num_players)

            for i in range(0, len(prev_players), 2):
                prev_players[i], prev_players[i+1] = self.crossover(prev_players[i], prev_players[i+1])

            new_players = []
            for i in range(num_players):
                if random.random() < 0.5:
                    new_players.append(self.mutate(prev_players[i]))
                else:
                    new_players.append(prev_players[i])

            return new_players

    def next_population_selection(self, players, num_players, gen_num):
        players = weighted_random_choice(num_players, players, calculate_total_fitness(players))
        players.sort(key=lambda x: x.fitness, reverse=True)

        total_fitness = calculate_total_fitness(players)
        average_fitness = total_fitness / len(players)
        max_fitness = players[0].fitness
        min_fitness = players[-1].fitness
        append_list_as_row("data.csv", [gen_num, average_fitness, max_fitness, min_fitness])

        new_players = []
        for i in range(num_players):
            new_players.append(self.mutate(players[i]))

        return new_players
