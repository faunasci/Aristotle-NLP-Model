import matplotlib.pyplot as plt
import numpy as np

# Os mesmos dados do script anterior
tokens = ["o", "menino", "e", "a", "menina", "foram", "brincar", "rei", "rainha", "estao", "jantando"]

# Os vetores que o PCA gerou
dense_embeddings = np.array([
    [2.15, -1.02],
    [-0.26, -0.07],
    [0.59, 1.70],
    [1.07, 0.28],
    [0.23, -0.75],
    [-1.26, -0.33],
    [-1.26, -0.33],
    [0.09, -0.11],
    [0.17, 0.25],
    [-0.45, 0.99],
    [-1.07, -0.60]
])

# --- Criação do Gráfico ---

# Prepara a figura e os eixos
fig, ax = plt.subplots(figsize=(10, 8))

# Extrai as coordenadas x e y
x = dense_embeddings[:, 0]
y = dense_embeddings[:, 1]

# Plota os pontos no gráfico
ax.scatter(x, y)

# Adiciona o nome de cada palavra como um rótulo no gráfico
for i, word in enumerate(tokens):
    ax.annotate(word, (x[i], y[i]), xytext=(5, 5), textcoords='offset points')

# Adiciona títulos e uma grade para facilitar a leitura
ax.set_title('Visualização dos Embeddings de Palavras em 2D')
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.grid(True)

# Salva o gráfico em um arquivo de imagem
output_filename = 'embeddings_plot.png'
plt.savefig('plots/embeddings_plot.png')

print(f"Gráfico salvo com sucesso como '{output_filename}'!")
