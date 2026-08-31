import numpy as np
import pandas as pd
import time
import os

rho = 7850
F = 10000
sigma_adm = 250e6

def fitness_func(x):
    A = x[:, 0]
    L = x[:, 1]
    custo = rho * A * L
    sigma = F / A
    penal = np.maximum(0, sigma - sigma_adm)
    return custo + 1e10 * penal**2

tam_populacao = 30
num_geracoes = 100
tx_cruzamento = 0.8
tx_mutacao = 0.1
lim_inf = np.array([0.0001, 0.5])
lim_sup = np.array([0.01, 5.0])

def selecao_torneio(pop, aptidoes, k=3):
    indices = np.random.choice(len(pop), k, replace=False)
    melhor_idx = indices[np.argmin(aptidoes[indices])]
    return pop[melhor_idx].copy()

def cruzamento_aritmetico(pai1, pai2):
    if np.random.rand() < tx_cruzamento:
        alpha = np.random.rand()
        filho1 = alpha * pai1 + (1 - alpha) * pai2
        filho2 = (1 - alpha) * pai1 + alpha * pai2
        return filho1, filho2
    return pai1.copy(), pai2.copy()

def mutacao(indiv):
    for i in range(len(indiv)):
        if np.random.rand() < tx_mutacao:
            ruido = np.random.normal(0, 0.1 * (lim_sup[i] - lim_inf[i]))
            indiv[i] += ruido
    return np.clip(indiv, lim_inf, lim_sup)


NUM_RODADAS = 30
linhas_historico = []   # rodada, iteracao, melhor_global
linhas_resultados = []  # rodada, massa_final
linhas_tempos = []      # rodada, tempo

for rodada in range(1, NUM_RODADAS + 1):

    populacao = np.random.uniform(lim_inf, lim_sup, (tam_populacao, 2))

    # Estruturas de Armazenamento
    historico_melhor_aptidao = []
    historico_media_aptidao = []
    tempos_iteracoes = []
    melhor_solucao_global = None
    melhor_aptidao_global = float('inf')
    tempo_inicio_total = time.perf_counter()

    for geracao in range(num_geracoes):
        t_inicio_iter = time.perf_counter()
        aptidoes = fitness_func(populacao)
        idx_melhor = np.argmin(aptidoes)
        aptidao_atual_melhor = aptidoes[idx_melhor]

        if aptidao_atual_melhor < melhor_aptidao_global:
            melhor_aptidao_global = aptidao_atual_melhor
            melhor_solucao_global = populacao[idx_melhor].copy()
        historico_melhor_aptidao.append(melhor_aptidao_global)
        historico_media_aptidao.append(np.mean(aptidoes))
        nova_populacao = [populacao[idx_melhor].copy()]
        while len(nova_populacao) < tam_populacao:
            pai1 = selecao_torneio(populacao, aptidoes)
            pai2 = selecao_torneio(populacao, aptidoes)
            filho1, filho2 = cruzamento_aritmetico(pai1, pai2)
            filho1 = mutacao(filho1)
            filho2 = mutacao(filho2)
            nova_populacao.append(filho1)
            if len(nova_populacao) < tam_populacao:
                nova_populacao.append(filho2)
        populacao = np.array(nova_populacao)
        t_fim_iter = time.perf_counter()
        tempos_iteracoes.append(t_fim_iter - t_inicio_iter)

    tempo_total = time.perf_counter() - tempo_inicio_total

    print(f"[AG] Rodada {rodada}/{NUM_RODADAS} - massa final: {melhor_aptidao_global:.4f} kg "
          f"- tempo: {tempo_total:.4f} s")


    for i, melhor in enumerate(historico_melhor_aptidao, start=1):
        linhas_historico.append({"rodada": rodada, "iteracao": i, "melhor_global": melhor})

    linhas_resultados.append({"rodada": rodada, "massa_final": melhor_aptidao_global})
    linhas_tempos.append({"rodada": rodada, "tempo": tempo_total})


pd.DataFrame(linhas_historico).to_csv("historico_iteracoes_ag.csv", index=False)
pd.DataFrame(linhas_resultados).to_csv("resultados_finais_ag.csv", index=False)
pd.DataFrame(linhas_tempos).to_csv("tempos_ag.csv", index=False)

print("\nArquivos salvos: historico_iteracoes_ag.csv, resultados_finais_ag.csv, tempos_ag.csv")
