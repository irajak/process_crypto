from deap import base, creator, tools, algorithms
import random

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 1, 5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_genetic_algorithm(pop_size=50, ngen=40):
    """
    Run the genetic algorithm to find the optimal solution.
    :param pop_size: Population size.
    :param ngen: Number of generations.
    :return: Best individual found.
    """
    population = toolbox.population(n=pop_size)
    result = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=ngen, verbose=False)
    best_individual = tools.selBest(population, k=1)[0]
    return best_individual

# Example usage
if __name__ == "__main__":
    best_plan = run_genetic_algorithm()
    print(f"Best Process Plan from Genetic Algorithm: {best_plan}, Fitness: {best_plan.fitness.values}")