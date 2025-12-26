import numpy as np
from sklearn.decomposition import PCA

# Nossos tokens na mesma ordem que o script bash gerou
tokens = ["o", "menino", "e", "a", "menina", "foram", "brincar", "rei", "rainha", "estao", "jantando"]

# A matriz de coocorrência que o script bash calculou (copiada manualmente para o estudo)
# Cada linha é o vetor esparso de uma palavra
co_occurrence_matrix = np.array([
    [0, 1, 2, 2, 1, 1, 1, 1, 1, 1, 0], # o
    [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], # menino
    [2, 1, 0, 2, 1, 0, 0, 1, 1, 0, 0], # e
    [2, 1, 2, 0, 1, 0, 0, 1, 1, 1, 0], # a
    [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0], # menina
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # foram
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], # brincar
    [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0], # rei
    [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1], # rainha
    [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1], # estao
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  # jantando
])

# --- O Passo de Redução de Dimensionalidade ---

# Queremos reduzir os vetores de 11 dimensões para apenas 2 dimensões
# n_components é a dimensão do nosso embedding final!
pca = PCA(n_components=2)

# O PCA "aprende" as melhores direções a partir da matriz
# e "transforma" nossos vetores originais para o novo espaço de 2D.
dense_embeddings = pca.fit_transform(co_occurrence_matrix)


# --- O Resultado Final: Embeddings Densos ---

print("Embeddings Densos (vetores de 2 dimensões):")
for word, vector in zip(tokens, dense_embeddings):
    # Formata o vetor para melhor visualização
    formatted_vector = [f"{val:.2f}" for val in vector]
    print(f"{word:<12} -> {formatted_vector}")
