# PISI2

Este repositório contém a implementação de três algoritmos diferentes para resolver o Problema do Caixeiro Viajante, utilizando o dataset Berlin52, da disciplina de PISI2. Os algoritmos implementados são:

1. **Busca Tabu**
2. **Algoritmo Genético Multiobjetivo**
3. **Força Bruta**

## Descrição do Projeto

O objetivo deste projeto é encontrar a melhor rota para o Problema do Caixeiro Viajante (TSP) usando diferentes abordagens. O dataset Berlin52 contém coordenadas de 52 cidades, e as soluções são avaliadas com base na distância percorrida e no tempo, que são otimizados no algoritmo genético.

### Algoritmos Implementados

1. **Busca Tabu**
   - Utiliza uma abordagem heurística para evitar ciclos e explorar o espaço de soluções de maneira eficiente.
   
2. **Algoritmo Genético Multiobjetivo**
   - Este algoritmo não só minimiza a distância percorrida, mas também otimiza o tempo de percurso. Os tempos são extraídos de um arquivo JSON que contém um dicionário com as durações para cada percurso.

3. **Força Bruta**
   - Um método simples que explora todas as possíveis permutações das rotas para encontrar a solução ótima. Utiliza as bibliotecas `itertools` para gerar as combinações.

## Bibliotecas Utilizadas

- `matplotlib`: Para visualização dos resultados.
- `numpy`: Para operações numéricas.
- `random`: Para gerar números aleatórios, utilizado no algoritmo genético.
- `itertools`: Para gerar permutações no algoritmo de força bruta.

## Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/nome-do-repositorio.git
   cd nome-do-repositorio
   ```

2. Instale as dependências necessárias:
   ```bash
   pip install matplotlib numpy
   ```

3. Execute os algoritmos:
   - Para a Busca Tabu:
     ```bash
     python busca_tabu.py
     ```
   - Para o Algoritmo Genético:
     ```bash
     python algoritmo_genetico.py
     ```
   - Para o Força Bruta:
     ```bash
     python forca_bruta.py
     ```

## Resultados

Os resultados de cada algoritmo são salvos em arquivos de saída, que incluem gráficos gerados pelo `matplotlib` e detalhes sobre as melhores rotas encontradas.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue ou enviar um pull request.


