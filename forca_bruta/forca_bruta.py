from itertools import permutations #biblioteca para fazer as permutações
import time

def ler_arquivo(nome_arquivo):
    coordenadas = {}
    pontos_de_entrega = []
    
    with open(nome_arquivo, 'r') as arquivo:
        linhas = arquivo.read().splitlines()
        
        for linha in linhas:
            partes = linha.split()
            id_ponto = partes[0]
            x = float(partes[1])
            y = float(partes[2])
            coordenadas[id_ponto] = (x, y)
            pontos_de_entrega.append(id_ponto)
    
    return coordenadas, pontos_de_entrega

def calcular_rota(pontos_de_entrega, coordenadas):
    menor_custo = float('inf')
    melhor_rota = []
    
    for permutation in permutations(pontos_de_entrega):
        permutation = list(permutation)
        custo_atual = 0
        for c in range(len(permutation) - 1):
            x1, y1 = coordenadas[permutation[c]]
            x2, y2 = coordenadas[permutation[c + 1]]
            custo_atual += abs(x2 - x1) + abs(y2 - y1)
        if custo_atual < menor_custo:
            menor_custo = custo_atual
            melhor_rota = permutation
    return melhor_rota, menor_custo

def imprimir_rota(rota, custo):
    print('Rota mais curta:', ' '.join(rota))
    print('Custo:', custo)

def main():
    start_time = time.time()
    
    coordenadas, pontos_de_entrega = ler_arquivo('berlin.txt')
    melhor_rota, menor_custo = calcular_rota(pontos_de_entrega, coordenadas)
    imprimir_rota(melhor_rota, menor_custo)
    
    end_time = time.time()
    print(f"Tempo de execução: {end_time - start_time:.2f} segundos")
    
if __name__ == '__main__':
    main()