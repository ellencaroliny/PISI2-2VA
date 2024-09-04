#importar bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial

class GA_TSP:
    def __init__(self, func, n_dim, tamanho_pop, max_iter, prob_mut):
        #Inicializa os parâmetros
        self.func = func
        self.n_dim = n_dim
        self.tamanho_pop = tamanho_pop
        self.max_iter = max_iter
        self.prob_mut = prob_mut

        self.Cromossomo = None
        self.X = None
        self.Y_bruto = None
        self.Y = None
        self.Fit = None

        self.melhor_X_geracao = []
        self.melhor_Y_geracao = []
        self.historico_Y = []
        self.historico_Fit = []

        self.melhor_x, self.melhor_y = None, None

        self.tamanho_cromossomo = self.n_dim
        self.criar_populacao()

    def criar_populacao(self):
        # criar a população
        tmp = np.random.rand(self.tamanho_pop, self.tamanho_cromossomo)
        self.Cromossomo = tmp.argsort(axis=1)
        return self.Cromossomo

    def cromossomo2x(self, Cromossomo):
        return Cromossomo

    def x2y(self):
        self.Y_bruto = self.func(self.X)
        return self.Y_bruto

    def classificacao(self):
        self.Fit = -self.Y #problema de minimização

    def selecao_torneio(self, tamanho_torneio=3):
        Fit = self.Fit
        indice_selecionado = []
        for i in range(self.tamanho_pop):
            indice_aspirantes = np.random.randint(self.tamanho_pop, size=tamanho_torneio)
            indice_selecionado +=[(max(indice_aspirantes, key=lambda i: Fit[i]))]
        self.Cromossomo = self.Cromossomo[indice_selecionado, :]  
        return self.Cromossomo
    
    def reverter(self, individuo):
        n1, n2 = np.random.randint(0, individuo.shape[0] - 1, 2)
        if n1 >= n2:
            n1, n2 = n2, n1 + 1
        individuo[n1:n2] = individuo[n1:n2][::-1]
        return individuo

    def mutacao_por_inversao(self):
        for i in range(self.tamanho_pop):
            if np.random.rand() < self.prob_mut:
                self.Cromossomo[i] = self.reverter(self.Cromossomo[i])
        return self.Cromossomo

    def ox_crossover(self):
        #implementação do order crossover
        Cromossomo, tamanho_pop, tamanho_cromossomo = self.Cromossomo, self.tamanho_pop, self.tamanho_cromossomo
        for i in range(0, tamanho_pop, 2):
            pai1 = Cromossomo[i]
            pai2 = Cromossomo[i + 1]
            pos1 = np.random.randint(0, tamanho_cromossomo - 1)
            pos2 = np.random.randint(pos1, tamanho_cromossomo - 1)
            filho1 = -1 * np.ones(tamanho_cromossomo, dtype=int)
            filho2 = -1 * np.ones(tamanho_cromossomo, dtype=int)
            filho1[pos1:pos2+1] = pai1[pos1:pos2+1]
            filho2[pos1:pos2+1] = pai2[pos1:pos2+1]
            cidades_restantes1 = [cidade for cidade in pai2 if cidade not in filho1]
            cidades_restantes2 = [cidade for cidade in pai1 if cidade not in filho2]
            for j in range(tamanho_cromossomo):
                if filho1[j] == -1:
                    filho1[j] = cidades_restantes1.pop(0)
                if filho2[j] == -1:
                    filho2[j] = cidades_restantes2.pop(0)
            Cromossomo[i] = filho1
            Cromossomo[i + 1] = filho2
        return Cromossomo
            

    def executar(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            Cromossomo_antigo = self.Cromossomo.copy()
            self.X = self.cromossomo2x(self.Cromossomo)
            self.Y = self.x2y()
            self.classificacao()
            self.selecao_torneio()
            self.ox_crossover()
            self.mutacao_por_inversao()

            # juntar os pais e os filhos e selecionar os melhores
            self.Cromossomo = np.concatenate([Cromossomo_antigo, self.Cromossomo], axis=0)
            self.X = self.cromossomo2x(self.Cromossomo)
            self.Y = self.x2y()
            self.classificacao()
            indice_selecionado = np.argsort(self.Y)[:self.tamanho_pop]
            self.Cromossomo = self.Cromossomo[indice_selecionado, :]

            # registrar os melhores
            indice_melhor_geracao = self.Fit.argmax()
            self.melhor_X_geracao +=[(self.X[indice_melhor_geracao, :].copy())]
            self.melhor_Y_geracao +=[(self.Y[indice_melhor_geracao])]
            self.historico_Y +=[(self.Y.copy())]
            self.historico_Fit += [(self.Fit.copy())]

        indice_melhor_global = np.array(self.melhor_Y_geracao).argmin()
        self.melhor_x = self.melhor_X_geracao[indice_melhor_global]
        self.melhor_y = self.func(np.array([self.melhor_x]))
        return self.melhor_x, self.melhor_y

def ler_coordenadas(arquivo):
    coordenadas = []
    with open(arquivo, 'r') as f:
        for linha in f:
            _, x, y = linha.split()
            coordenadas +=[([float(x), float(y)])]
        return np.array(coordenadas)

def main():
    arquivo_coordenadas = 'dataset\\berlin.txt'
    coordenadas_pontos = ler_coordenadas(arquivo_coordenadas)
    num_pontos = len(coordenadas_pontos)
    matriz_distancia = spatial.distance.cdist(coordenadas_pontos, coordenadas_pontos, metric='euclidean')

    def calcular_distancia_total(rotina):
        num_pontos = rotina.shape[1]
        distancia_total = np.zeros(rotina.shape[0])
        for i in range(num_pontos):
            cidade_i = rotina[:, i % num_pontos]
            cidade_j = rotina[:, (i + 1) % num_pontos]
            distancia_total += matriz_distancia[cidade_i, cidade_j]
        return distancia_total

    resultados = []

    for _ in range(1):
        ga_tsp = GA_TSP(func=calcular_distancia_total, n_dim=num_pontos, tamanho_pop=50, max_iter=1000, prob_mut=0.1)
        melhores_pontos, melhor_distancia = ga_tsp.executar()
        resultados+=[((melhores_pontos, melhor_distancia))]

    for i, (melhores_pontos, melhor_distancia) in enumerate(resultados):
        print(f"Execução {i+1}: Melhor distância: {melhor_distancia}, Melhor rota: {melhores_pontos}")

    # Plotar os resultados
    fig, ax = plt.subplots(1, 2)
    for i, (melhores_pontos, _) in enumerate(resultados):
        melhores_pontos_ = np.concatenate([melhores_pontos, [melhores_pontos[0]]])
        coordenadas_melhores_pontos = coordenadas_pontos[melhores_pontos_, :]
        ax[0].plot(coordenadas_melhores_pontos[:, 0], coordenadas_melhores_pontos[:, 1], label=f"Execução {i+1}")
    ax[0].legend()
    ax[1].plot([melhor_distancia for _, melhor_distancia in resultados])
    plt.show()

if __name__ == "__main__":
    main()

