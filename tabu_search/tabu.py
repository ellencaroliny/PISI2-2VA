import random
import time
import math

def carregar_coordenadas(mapa):
    coord = {}
    with open(mapa, 'r') as file:
        linhas = file.readlines()

    for linha in linhas:
        if linha.startswith("EOF"):
            break
        partes = linha.split()
        if len(partes) >= 3:
            ponto_id = partes[0]
            x, y = map(float, partes[1:])
            coord[ponto_id] = (x, y)
    return coord

def calc_custo(percurso, coord):
    return sum(
        math.sqrt(
            (coord[percurso[i]][0] - coord[percurso[i + 1]][0]) ** 2 +
            (coord[percurso[i]][1] - coord[percurso[i + 1]][1]) ** 2
        )
        for i in range(len(percurso) - 1)
    )

def gerar_vizinhos(rota):
    vizinhos = []
    for i in range(1, len(rota) - 1):
        for j in range(i + 1, len(rota) - 1):
            vizinho = rota[:]
            vizinho[i:j] = reversed(vizinho[i:j])
            vizinhos.append(vizinho)
    return vizinhos

def gerar_solucao_inicial_aleatoria(coord, seed=None):
    if seed is not None:  
        random.seed(seed)
    solucao_inicial = list(coord.keys())
    random.shuffle(solucao_inicial) 
    solucao_inicial.append(solucao_inicial[0])
    return solucao_inicial

def buscar_melhor_vizinho(vizinhos, lista_tabu, melhor_custo_atual, coord):
    melhor_vizinho = None
    melhor_custo = float('inf')

    for vizinho in vizinhos:
        custo_vizinho = calc_custo(vizinho, coord)
        if vizinho not in lista_tabu or custo_vizinho < melhor_custo_atual: 
            if custo_vizinho < melhor_custo:
                melhor_custo = custo_vizinho
                melhor_vizinho = vizinho

    return melhor_vizinho

def busca_tabu(coord, tam_lista, max_ite, seed=None):
    solucao_inicial = gerar_solucao_inicial_aleatoria(coord, seed)
    melhor_solucao = solucao_inicial[:]
    melhor_custo = calc_custo(melhor_solucao, coord)
    lista_tabu = []

    for _ in range(max_ite):
        vizinhos = gerar_vizinhos(melhor_solucao)
        melhor_vizinho = buscar_melhor_vizinho(vizinhos, lista_tabu, melhor_custo, coord)

        if melhor_vizinho:
            custo_vizinho = calc_custo(melhor_vizinho, coord)
            if custo_vizinho < melhor_custo:
                melhor_solucao = melhor_vizinho
                melhor_custo = custo_vizinho

            lista_tabu.append(melhor_vizinho)

            if len(lista_tabu) > tam_lista:
                lista_tabu.pop(0)

    return melhor_solucao, melhor_custo

def executar_multiplo(coord, num_execucoes, tam_lista, max_ite, seeds=None):
    resultados = []

    if seeds is None:
        seeds = [random.randint(0, 2**32 - 1) for _ in range(num_execucoes)]

    for execucao in range(num_execucoes):
        seed = seeds[execucao] 
        inicio_execucao = time.time()

        solucao, custo = busca_tabu(coord, tam_lista, max_ite, seed)
        tempo_execucao = time.time() - inicio_execucao  

        resultados.append({
            "execucao": execucao + 1,
            "seed": seed,
            "solucao": solucao,
            "custo": custo,
            "tempo": tempo_execucao
        })

        print(f"Execução {execucao + 1}:")
        print(f"  Seed = {seed}")
        print(f"  Custo = {custo}")
        print(f"  Solução = {' - '.join(solucao)}")
        print(f"  Tempo = {tempo_execucao:.4f} segundos\n")

    return resultados, seeds

mapa = "cidades/berlin52.txt"
coord = carregar_coordenadas(mapa)
#seed = 
tam_lista = 300
max_ite = 5000
num_execucoes = 5

tempo_ini = time.time()
resultados, seeds = executar_multiplo(coord, num_execucoes, tam_lista, max_ite ) #seeds=[seed]
tempo_final = time.time()

print(f"Tempo total de execução: {tempo_final - tempo_ini:.4f} segundos")
