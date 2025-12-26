import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Os mesmos dados dos scripts anteriores
co_occurrence_matrix = np.array([
    [0, 1, 2, 2, 1, 1, 1, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [2, 1, 0, 2, 1, 0, 0, 1, 1, 0, 0],
    [2, 1, 2, 0, 1, 0, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
])

# --- Análise de Variância Explicada ---

# Instancia o PCA SEM definir o n_components. 
# Assim, ele calculará todos os componentes possíveis.
pca = PCA()
pca.fit(co_occurrence_matrix)

# A variância explicada por CADA componente
explained_variance_ratio = pca.explained_variance_ratio_

# A soma CUMULATIVA da variância
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# --- Geração dos Gráficos de Análise ---

# 1. Gráfico de Cotovelo (Scree Plot)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
plt.title('Gráfico de Cotovelo (Scree Plot)')
plt.xlabel('Número de Componentes')
plt.ylabel('Porcentagem de Variância Explicada')
plt.grid(True)

# 2. Gráfico de Variância Cumulativa
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='--')
plt.axhline(y=0.95, color='r', linestyle='-') # Linha de corte em 95%
plt.text(1.5, 0.96, 'Limiar de 95%', color = 'red')
plt.title('Variância Explicada Cumulativa')
plt.xlabel('Número de Componentes')
plt.ylabel('Porcentagem Cumulativa da Variância')
plt.grid(True)

plt.tight_layout()

# Salva os gráficos em um arquivo
output_filename = 'variance_analysis_plots.png'
plt.savefig('plots/variance_analysis_plots.png')

print(f"Gráficos de análise salvos como '{output_filename}'!")
