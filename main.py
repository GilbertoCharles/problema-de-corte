
from typing import List
import numpy
import matplotlib

class AG():

    def __init__(self) -> None:
        self.nrows, self.ncols = 5,5
        self.population = []
        self.fitness = []

    def createPopulation(self, size):
        self.population = []
        for _ in range(size):
            chromosome = Chromosome.createRandomChromosome()
            self.population.push(chromosome);

    def run(self, qtdGenerations):

        #ENTRADA: População Inicial (aleatória)
        #Função de Fitness
        #Critério de Parada
        #REPITA (até que o critério de parada seja atendido):
        #PASSO 1: Aplicar a função de fitness a cada indivíduo
        #PASSO 2: Selecionar os x melhores indivíduos
        #PASSO 3: Reprodução
        #        - Aplicar o crossover a um par (com prob = p)
        #        - Aplicar mutação (com prob = p’)
        #PASSO 4: Formar uma nova população com os filhos gerados
        #SAÍDA: Melhor indivíduo presente na geração final
 
        for _ in range(qtdGenerations):
            self.buildNextGenerarion(self);

    def buildNextGeneration(self):
        newPopulation = []
        self.setPopulationFitness()
        while len(self.population) != len(newPopulation):
            chromosomeOne = Chromosome.proportionalSelection(self.population)
            chromosomeTwo = Chromosome.proportionalSelection(self.population)
            newChromosome = Chromosome.crossing(chromosomeOne, chromosomeTwo)
            newPopulation.push(newChromosome)
    
    def setPopulationFitness(self):
        self.fitnessList = []
        for chromosome in range(self.population):
            fitnessValue = self.getFitnessValue(chromosome)
            self.fitnessList.push(fitnessValue)

    def plot(self, list) -> None:
        list = [item for sublist in list for item in sublist]
        chess = np.array(list)
        image = np.zeros(self.nrows*self.ncols)
        image = np.array(chess)
        image = image.reshape((self.nrows, self.ncols))
        plt.imshow(image, cmap='gray_r')
        plt.show()

class Chromosome():
    def __init__(self) -> None:
        pass

    def createRandomChromosome(self):
        pass

    def proportionalSelection(self):
        pass

    def crossing(self):
        pass
                
    def caltFitnessValue():
        pass

def main():
    popularionSize = 10;
    qtdGeneration = 10;
    
    ag = AG
    ag.createPopulation(popularionSize)
    ag.run(qtdGeneration)
    
    # Lista de listas, cada número representa uma cor formando o formato do corte
    list =      [
                [1,3,3,3,3],
                [1,0,0,0,3],
                [1,2,2,2,2],
                [1,1,0,0,2],
                [0,0,0,0,0]
                ]
    ag.plot(list)

if __name__ == '__main__':
    main()
    