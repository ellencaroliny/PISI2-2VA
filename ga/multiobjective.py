import numpy as np
import matplotlib.pyplot as plt
import random
import json

def calcular_distancia_euclidiana(ponto1, ponto2):
    return np.sqrt(np.sum((ponto1 - ponto2) ** 2))

def calcular_matriz_distancia(coordenadas):
    num_pontos = len(coordenadas)
    matriz_distancia = np.zeros((num_pontos, num_pontos))
    for i in range(num_pontos):
        for j in range(num_pontos):
            if i != j:
                matriz_distancia[i, j] = calcular_distancia_euclidiana(coordenadas[i], coordenadas[j])
    return matriz_distancia

def ler_matriz_tempos(arquivo):
    with open(arquivo, 'r') as file:
        matriz_tempos = json.load(file)
    return matriz_tempos

def avaliar_solucao(solucao, matriz_distancias, matriz_tempos):
    distancia_total = 0
    tempo_total = 0
    num_cidades = len(solucao)

    for i in range(num_cidades - 1):
        cidade_origem = str(solucao[i] + 1)
        cidade_destino = str(solucao[i + 1] + 1)
        distancia_total += matriz_distancias[solucao[i], solucao[i + 1]]
        tempo_total += matriz_tempos[cidade_origem][cidade_destino]

    # Considerar o retorno à cidade inicial
    cidade_origem = str(solucao[-1] + 1)
    cidade_destino = str(solucao[0] + 1)
    distancia_total += matriz_distancias[solucao[-1], solucao[0]]
    tempo_total += matriz_tempos[cidade_origem][cidade_destino]

    return distancia_total, tempo_total

def domina(solucao1, solucao2):
    return (solucao1[0] <= solucao2[0] and solucao1[1] <= solucao2[1]) and (solucao1 != solucao2)

def classificar_por_dominacao_nao_dominada(populacao, avaliacoes):
    frentes = [[]]
    dominancia = [0] * len(populacao)
    dominados_por = [[] for _ in range(len(populacao))]

    for p in range(len(populacao)):
        for q in range(len(populacao)):
            if domina(avaliacoes[p], avaliacoes[q]):
                dominados_por[p]+=[(q)]
            elif domina(avaliacoes[q], avaliacoes[p]):
                dominancia[p] += 1
        if dominancia[p] == 0:
            frentes[0]+=[(p)]

    i = 0
    while len(frentes[i]) > 0:
        prox_frente = []
        for p in frentes[i]:
            for q in dominados_por[p]:
                dominancia[q] -= 1
                if dominancia[q] == 0:
                    prox_frente+=[(q)]
        i += 1
        frentes+=[(prox_frente)]

    return frentes[:-1]

def heapify(arr, n, i, key):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and key(arr[left]) > key(arr[largest]):
        largest = left

    if right < n and key(arr[right]) > key(arr[largest]):
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest, key)

def heap_sort(arr, key=lambda x: x):
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i, key)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0, key)

def calcular_distancia_crowding(populacao, avaliacoes, frentes):
    distancia_crowding = np.zeros(len(populacao))
    for frente in frentes:
        frente_avaliacoes = [avaliacoes[i] for i in frente]
        frente_avaliacoes = np.array(frente_avaliacoes)
        for m in range(frente_avaliacoes.shape[1]):
            indices = list(range(len(frente_avaliacoes)))
            heap_sort(indices, key=lambda k: frente_avaliacoes[k, m])
            distancia_crowding[frente[indices[0]]] = np.inf
            distancia_crowding[frente[indices[-1]]] = np.inf
            for k in range(1, len(frente) - 1):
                distancia_crowding[frente[indices[k]]] += (frente_avaliacoes[indices[k + 1], m] - frente_avaliacoes[indices[k - 1], m]) / (frente_avaliacoes[indices[-1], m] - frente_avaliacoes[indices[0], m])
    return distancia_crowding

class GA_TSP:
    def __init__(self, func, n_dim, tamanho_pop, max_iter, prob_mut, matriz_distancia, matriz_tempos):
        # Inicializa os parâmetros
        self.matriz_distancia = matriz_distancia
        self.matriz_tempos = matriz_tempos
        self.func = func
        self.n_dim = n_dim
        self.tamanho_pop = tamanho_pop
        self.max_iter = max_iter
        self.prob_mut = prob_mut

        self.Cromossomo = self.criar_populacao(self.tamanho_pop, self.n_dim)
        self.X = None
        self.Y_bruto = None
        self.Y = None
        self.T = None
        self.Fit = None

        self.melhor_X_geracao = []
        self.melhor_Y_geracao = []
        self.historico_Y = []
        self.historico_T = []  # Adiciona histórico de tempos
        self.historico_Fit = []

        self.melhor_x, self.melhor_y, self.melhor_t = None, None, None

        self.tamanho_cromossomo = self.n_dim

    def criar_populacao(self, tamanho_pop, n_dim):
        populacao = []
        for _ in range(tamanho_pop):
            individuo = list(range(n_dim))
            for i in range(n_dim):
                j = random.randint(0, n_dim - 1)
                individuo[i], individuo[j] = individuo[j], individuo[i]
            populacao.append(individuo)
        return np.array(populacao)

    def cromossomo2x(self, Cromossomo):
        return Cromossomo

    def x2y(self):
        self.Y_bruto = np.array([self.func(individuo, self.matriz_distancia, self.matriz_tempos) for individuo in self.X])
        self.Y = np.array([y[0] for y in self.Y_bruto])  # Distância
        self.T = np.array([y[1] for y in self.Y_bruto])  # Tempo
        return self.Y, self.T

    def classificacao(self):
        # Classificação por dominação não dominada
        avaliacoes = list(zip(self.Y, self.T))
        frentes = classificar_por_dominacao_nao_dominada(self.Cromossomo, avaliacoes)
        distancia_crowding = calcular_distancia_crowding(self.Cromossomo, avaliacoes, frentes)
        
        # Definir a aptidão com base na classificação de dominação e na distância de crowding
        self.Fit = np.zeros(len(self.Cromossomo))
        for i, frente in enumerate(frentes):
            for indice in frente:
                self.Fit[indice] = -i + distancia_crowding[indice]  # Frentes mais baixas têm maior prioridade
        
        return frentes, distancia_crowding

    def selecao_torneio(self, tamanho_torneio=3):
        Fit = self.Fit
        indice_selecionado = []
        for i in range(self.tamanho_pop):
            indice_aspirantes = np.random.randint(self.tamanho_pop, size=tamanho_torneio)
            indice_selecionado.append(max(indice_aspirantes, key=lambda i: Fit[i]))
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
        # Implementação do order crossover
        Cromossomo, tamanho_pop, tamanho_cromossomo = self.Cromossomo, self.tamanho_pop, self.tamanho_cromossomo
        for i in range(0, tamanho_pop - 1, 2):  # Ajuste para evitar índice fora dos limites
            pai1 = Cromossomo[i]
            pai2 = Cromossomo[i + 1]
            pos1 = np.random.randint(0, tamanho_cromossomo - 1)
            pos2 = np.random.randint(pos1, tamanho_cromossomo - 1)
            filho1 = -1 * np.ones(tamanho_cromossomo, dtype=int)
            filho2 = -1 * np.ones(tamanho_cromossomo, dtype=int)
            filho1[pos1:pos2 + 1] = pai1[pos1:pos2 + 1]
            filho2[pos1:pos2 + 1] = pai2[pos1:pos2 + 1]
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
            self.Y, self.T = self.x2y()
            frentes, distancia_crowding = self.classificacao()
            self.selecao_torneio()
            self.ox_crossover()
            self.mutacao_por_inversao()

            # Juntar os pais e os filhos e selecionar os melhores
            self.Cromossomo = np.concatenate([Cromossomo_antigo, self.Cromossomo], axis=0)
            self.X = self.cromossomo2x(self.Cromossomo)
            self.Y, self.T = self.x2y()
            frentes, distancia_crowding = self.classificacao()

            # Seleção da nova população
            nova_populacao = []
            for frente in frentes:
                if len(nova_populacao) + len(frente) > self.tamanho_pop:
                    frente = sorted(frente, key=lambda i: distancia_crowding[i], reverse=True)
                    nova_populacao.extend(frente[:self.tamanho_pop - len(nova_populacao)])
                    break
                nova_populacao.extend(frente)

            self.Cromossomo = self.Cromossomo[nova_populacao]

            # Registrar os melhores
            indice_melhor_geracao = self.Fit.argmax()
            self.melhor_X_geracao.append(self.X[indice_melhor_geracao, :].copy())
            self.melhor_Y_geracao.append(self.Y[indice_melhor_geracao])
            self.historico_Y.append(self.Y.copy())
            self.historico_T.append(self.T.copy())  # Armazena os tempos
            self.historico_Fit.append(self.Fit.copy())

        indice_melhor_global = np.array(self.melhor_Y_geracao).argmin()  # Melhor com base na distância
        self.melhor_x = self.melhor_X_geracao[indice_melhor_global]
        self.melhor_y, self.melhor_t = self.func(self.melhor_x, self.matriz_distancia, self.matriz_tempos)
        return self.melhor_x, self.melhor_y, self.melhor_t

def ler_coordenadas(arquivo):
    coordenadas = []
    with open(arquivo, 'r') as f:
        for linha in f:
            _, x, y = linha.split()
            coordenadas+=[([float(x), float(y)])]
    return np.array(coordenadas)

def main():
    arquivo_coordenadas = 'dataset\\berlin.txt'    
    arquivo_tempos = 'dataset\\matriz_tempos.txt'
    coordenadas_pontos = ler_coordenadas(arquivo_coordenadas)
    num_pontos = len(coordenadas_pontos)
    matriz_distancia = calcular_matriz_distancia(coordenadas_pontos)
    matriz_tempos = ler_matriz_tempos(arquivo_tempos)

    resultados = []

    for _ in range(1):
        random.seed(1)
        ga_tsp = GA_TSP(
            func=avaliar_solucao,
            n_dim=num_pontos,
            tamanho_pop=90,
            max_iter=4200,
            prob_mut=0.4,
            matriz_distancia=matriz_distancia,
            matriz_tempos=matriz_tempos
        )
        ga_tsp.executar()
        resultados.append(ga_tsp)  # Armazena a instância de GA_TSP

        print(f"Melhor distância: {ga_tsp.melhor_y}, Melhor rota: {ga_tsp.melhor_x}, Melhor tempo: {ga_tsp.melhor_t}")

    

    # Plotar os resultados
    fig, ax = plt.subplots(3, 1, figsize=(7, 19))  

    # Plotar melhor rota
    ax[0].set_title("Melhor rota")
    for i, ga_tsp in enumerate(resultados):
        melhores_pontos_ = np.concatenate([ga_tsp.melhor_x, [ga_tsp.melhor_x[0]]])
        coordenadas_melhores_pontos = coordenadas_pontos[melhores_pontos_, :]
        ax[0].plot(coordenadas_melhores_pontos[:, 0], coordenadas_melhores_pontos[:, 1], label=f"Melhor rota ")
    ax[0].legend(loc="upper right")  

    # Plotar a fronteira de pareto
    ax[1].set_title("Fronteira de Pareto")
    for i, ga_tsp in enumerate(resultados):
        for j in range(0,len(ga_tsp.historico_Y)):
            distancias = ga_tsp.historico_Y[j]
            tempos = ga_tsp.historico_T[j]
            ax[1].scatter(distancias, tempos)
    ax[1].set_xlabel("Distância")
    ax[1].set_ylabel("Tempo")
    

    # Plotar a convergência do algoritmo
    ax[2].set_title("Convergência do Algoritmo")
    for i, ga_tsp in enumerate(resultados):
        melhores_distancias = [min(geracao) for geracao in ga_tsp.historico_Y]
        ax[2].plot(melhores_distancias)
    ax[2].set_xlabel("Geração")
    ax[2].set_ylabel("Distância")
    

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)  # Ajusta o espaçamento entre os subplots

    plt.show()

if __name__ == "__main__":
    main()
