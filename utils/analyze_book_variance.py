import re
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

try:
    from bs4 import BeautifulSoup
    from scipy.sparse import lil_matrix
except ImportError:
    print("Por favor, instale as bibliotecas necessárias:")
    print("pip install beautifulsoup4 scikit-learn numpy scipy matplotlib")
    exit()

# --- Parâmetros de Configuração ---
FILENAME = 'data/Mahabharata Volume 1_djvu.txt'
VOCAB_SIZE = 5000
MAX_TOKENS = 4000
# Vamos analisar os primeiros 200 componentes para encontrar um bom número.
N_COMPONENTS_TO_ANALYZE = 100

STOP_WORDS = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'did', 'do',
    'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having',
    'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it',
    'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'o', 'of', 'off',
    'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 's', 'same', 'she',
    'should', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
    'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
    'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'you', 'your',
    'yours', 'yourself', 'yourselves'])

def analyze_variance():
    print(f"1. Lendo e processando {FILENAME}...")
    with open(FILENAME, 'r', encoding='utf-8') as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    full_text = soup.get_text()
    tokens = re.findall(r'\b\w+\b', full_text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in STOP_WORDS]
    tokens = tokens[:MAX_TOKENS]

    print(f"2. Construindo vocabulário e matriz de coocorrência...")
    counts = Counter(tokens)
    vocab = [word for word, _ in counts.most_common(VOCAB_SIZE)]
    word_to_id = {word: i for i, word in enumerate(vocab)}
    numeric_tokens = [word_to_id[word] for word in tokens if word in word_to_id]
    
    co_occurrence_matrix = lil_matrix((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float32)
    context_window = 2
    for i, target_word_id in enumerate(numeric_tokens):
        for j in range(max(0, i - context_window), min(len(numeric_tokens), i + context_window + 1)):
            if i != j:
                context_word_id = numeric_tokens[j]
                co_occurrence_matrix[target_word_id, context_word_id] += 1

    print(f"3. Executando PCA para {N_COMPONENTS_TO_ANALYZE} componentes...")
    dense_matrix = co_occurrence_matrix.toarray()
    pca = PCA(n_components=N_COMPONENTS_TO_ANALYZE)
    pca.fit(dense_matrix)

    # A variância explicada por CADA componente
    explained_variance_ratio = pca.explained_variance_ratio_
    # A soma CUMULATIVA da variância
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    print("4. Gerando gráficos de análise de variância...")
    plt.figure(figsize=(12, 6))

    # Gráfico de Cotovelo (Scree Plot)
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
    plt.title('Gráfico de Cotovelo (Scree Plot)')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Porcentagem de Variância Explicada')
    plt.grid(True)

    # Gráfico de Variância Cumulativa
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='--')
    plt.axhline(y=0.90, color='r', linestyle='-')
    plt.text(5, 0.91, 'Limiar de 90%', color='red')
    plt.title('Variância Explicada Cumulativa')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Porcentagem Cumulativa da Variância')
    plt.grid(True)

    plt.tight_layout()
    output_filename = 'mahabharata_variance_analysis.png'
    plt.savefig('plots/mahabharata_variance_analysis.png')
    print(f"\nProcesso concluído! Gráfico salvo como '{output_filename}'.")

if __name__ == '__main__':
    analyze_variance()
