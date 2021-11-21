
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from random import random, randint

class AG():
    def __init__(self) -> None:
        self.maxRows, self.maxColumns = 5,5
        self.population = []
        self.fitness = []
        self.chromosomeFabric = ChromosomeFabric(self.maxColumns, self.maxRows)

    def createPopulation(self, size) -> None:
        self.population = []
        for _ in range(size):
            chromosome = self.chromosomeFabric.createRandomChromosome(1)
            self.population.append(chromosome)

    def run(self, qtdGenerations) -> None:

        # ENTRADA: População Inicial (aleatória)
        # Função de Fitness
        # Critério de Parada
        # REPITA (até que o critério de parada seja atendido):
        # PASSO 1: Aplicar a função de fitness a cada indivíduo
        # PASSO 2: Selecionar os x melhores indivíduos
        # PASSO 3: Reprodução
        #        - Aplicar o crossover a um par (com prob = p)
        #        - Aplicar mutação (com prob = p’)
        # PASSO 4: Formar uma nova população com os filhos gerados
        # SAÍDA: Melhor indivíduo presente na geração final

        generation = 0

        crossingProb = 0 
        mutatingProb = 0 
        while(generation != qtdGenerations):
            fitnessList = []
            newPopulation = []
            for chromosome in self.population:
                fitnessValue = self.chromosomeFabric.caltFitnessValue(chromosome)
                fitnessList.append(fitnessValue)
            
            selectedPopulation = self.chromosomeFabric.proportionalSelection()

            chromosomeToCrossing = []
            for chromosome in selectedPopulation:
                randomValue = random()
                if ( randomValue < crossingProb):
                    if (chromosomeToCrossing == []):
                        chromosomeToCrossing = chromosome
                    else:
                        newChromosome = self.chromosomeFabric.crossing(chromosomeToCrossing,chromosome)
                
                        if ( randomValue < mutatingProb):
                            newChromosome = self.chromosomeFabric.mutating(newChromosome)

                        newPopulation.append(newChromosome)
                        chromosomeToCrossing = []

    def setPopulationFitness(self) -> None:
        self.fitnessList = []
        for chromosome in range(self.population):
            fitnessValue = self.getFitnessValue(chromosome)
            self.fitnessList.append(fitnessValue)

    def plot(self, list) -> None:
        list = [item for sublist in list for item in sublist]
        chess = np.array(list)
        image = np.zeros(10*10)
        image = np.array(chess)
        image = image.reshape((10, 10))  
        plt.imshow(image, cmap='gray_r')
        plt.show()


class ChromosomeFabric():
    def __init__(self, maxColumns, maxRows) -> None:
        self.maxColumns = maxColumns
        self.maxRows = maxRows
        self.itemStardart = [
            # rotate 0º
            [
                [0,0,1,2,3]
                ,[1,0,0,0,0]
            ],
            # rotate 90º
            [
                [0,1,1,1,1]
                ,[0,0,1,2,3]
            ],
            # rotate 180º
            [
                [0,1,2,3,3]
                ,[1,1,1,1,0]
            ],
            # rotate 270º
            [
                [0,0,0,0,1]
                ,[0,1,2,3,3]
            ]
        ]

    def mutating(self, newChromosome):
        pass

    def createRandomChromosome(self, itemQtd) -> List:
        columnItensList = []
        rowItensList = []
        for _ in range(itemQtd):
            columns,rows = self.randomItem()
            columnItensList.append(columns)
            rowItensList.append(rows)
        return [columnItensList, rowItensList]

    def randomItem(self) -> List:
        x = randint(0, self.maxColumns)
        y = randint(0, self.maxRows)
        rotateAngle = randint(0,3)
        newItem = self.itemStardart[rotateAngle].copy()


        columnsList =  []
        rowsList = []

        for column in newItem[0]:
            columnsList.append(column + x)

        for row in newItem[1]:
            rowsList.append(row + y)

        return columnsList, rowsList

    def proportionalSelection(self) -> List:
        return []

    def crossing(self) -> None:
        # trocar peças (itens)
        pass
                
    def caltFitnessValue(self) -> float:
        itemQtd = 0 # linha 17
        itemSize = 5 # linha 66
        maxX = 0 #  linha 9
        maxY = 0 # linha 9
        maxItemX = 0 # calc
        maxItemY = 0 # calc
        generation = 0 # indice do loop
        maxGeneration = 0 # linha 145
        overlap = 0 # calc

        areaMax = maxX * maxY
        idealAreaIndex = itemQtd * itemSize
        genarationIndex = (generation * 0.6 + maxGeneration * 0.4) / maxGeneration
        areaIndex = (1 - (areaMax - maxItemX * maxItemY)/( areaMax - idealAreaIndex)) ^ genarationIndex

        maxOverlap = (itemSize * itemQtd * (itemQtd - 1)) / 2
        overlapIndex = (overlap / maxOverlap) ^ genarationIndex
        
        return 1 - ((areaIndex + overlapIndex)/2)

def main():
    populationSize = 5
    qtdGeneration = 10
    
    ag = AG()
    ag.createPopulation(populationSize)
    #ag.run(qtdGeneration)
    plotList =  [[0] * 10 for i in range(10)]
    print(plotList)
    color = 1
    for chromossome in ag.population:
        color += 1
        for i in range(len(chromossome[0])):
            for j in range(len(chromossome[0][0])):
                x = chromossome[0][i][j]
                y = chromossome[1][i][j]
                print(x, y)
                plotList[y][x] = color

    ag.plot(plotList)

if __name__ == '__main__':
    main()
    
class Matriz:
    def __init__(self, mat):
        self.mat = mat
        self.lin = len(mat)
        self.col = len(mat[0])

    def getLinha(self, n):
        return [i for i in self.mat[n]]

    def getColuna(self, n):
        return [i[n] for i in self.mat]

    def __mul__(self, mat2):
        matRes = []

        for i in range(self.lin):           
            matRes.append([])

            for j in range(mat2.col):
                listMult = [x*y for x, y in zip(self.getLinha(i), mat2.getColuna(j))]
                matRes[i].append(sum(listMult))

        return matRes
# mat1 = Matriz([[2, 3], [4, 6]])
# mat2 = Matriz([[1, 3, 0], [2, 1, 1]])
# print(mat1*mat2)
