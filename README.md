import numpy as np
import time

rho = 7850
F = 10000
sigma_adm = 250e6

# Função de Aptidão/Custo
def fitness_func(x):
    A = x[:, 0]
    L = x[:, 1]
    custo = rho * A * L
    sigma = F / A
    penal = np.maximum(0, sigma - sigma_adm)
    return custo + 1e10 * penal**2

# Parâmetros do Algoritmo Genético
tam_populacao = 30
num_geracoes = 100
tx_cruzamento = 0.8
tx_mutacao = 0.1
lim_inf = np.array([0.0001, 0.5])
lim_sup = np.array([0.01, 5.0])

# População Inicial
populacao = np.random.uniform(lim_inf, lim_sup, (tam_populacao, 2))

# Funções de Suporte do AG
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

# Estruturas de Armazenamento
historico_melhor_aptidao = []
historico_media_aptidao = []
tempos_iteracoes = []

melhor_solucao_global = None
melhor_aptidao_global = float('inf')

tempo_inicio_total = time.perf_counter()

# Laço Principal (Evolução)
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

# Estatísticas do Tempo
media_tempo_iter = np.mean(tempos_iteracoes)
desvio_tempo_iter = np.std(tempos_iteracoes)

# Exibição dos DadosColetados (Iteração x Valores x Tempo)
print("Geração | Melhor Aptidão (kg) | Média Aptidão (kg) | Tempo (s)")
print("-" * 65)
for i in range(num_geracoes):
    print(f"{i+1:7d} | {historico_melhor_aptidao[i]:19.4f} | {historico_media_aptidao[i]:18.4f} | {tempos_iteracoes[i]:.6f}")

print("=" * 65)
print("RESUMO FINAL:")
print(f"Melhor Área (A): {melhor_solucao_global[0]:.6f} m²")
print(f"Melhor Comprimento (L): {melhor_solucao_global[1]:.4f} m")
print(f"Massa mínima (Aptidão): {melhor_aptidao_global:.4f} kg")
print("-" * 65)
print(f"Tempo Total: {tempo_total:.4f} s")
print(f"Média de tempo por iteração: {media_tempo_iter:.6f} s ({media_tempo_iter*1000:.3f} ms)")
print(f"Desvio Padrão do tempo: {desvio_tempo_iter:.6f} s ({desvio_tempo_iter*1000:.3f} ms)")
print("=" * 65)
