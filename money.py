import simpy
import random
import numpy as np
from deap import base, creator, tools, algorithms

def business_process(env, name, resources):
    with resources.request() as req:
        yield req
        yield env.timeout(random.uniform(1, 5)) 
        print(f"{name} completed at {env.now:.2f}")

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 1, 5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    env = simpy.Environment()
    resources = simpy.Resource(env, capacity=2)
    for i, duration in enumerate(individual):
        env.process(business_process(env, f"Task-{i}", resources))
    env.run()
    return sum(individual),

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=50)
result = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, verbose=True)

best_individual = tools.selBest(population, k=1)[0]
print(f"Best Process Plan: {best_individual}, Fitness: {best_individual.fitness.values}")