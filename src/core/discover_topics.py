import re
import numpy as np
from collections import Counter, defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

try:
    from bs4 import BeautifulSoup
    from scipy.sparse import lil_matrix
except ImportError:
    print("Por favor, instale as bibliotecas necessárias:")
    print("pip install beautifulsoup4 scikit-learn numpy scipy")
    exit()

# --- Parâmetros de Configuração ---
FILENAME = 'data/Mahabharata Volume 1_djvu.txt'
VOCAB_SIZE = 10000
MAX_TOKENS = 100000
EMBEDDING_DIM = 15
N_CLUSTERS = 400  # O número de tópicos que queremos encontrar

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


def main():
    # --- PASSO 1: Processar o livro e gerar os embeddings ---
    print(f"Processando {FILENAME} para gerar embeddings de {EMBEDDING_DIM} dimensões...")
    
    with open(FILENAME, 'r', encoding='utf-8') as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    full_text = soup.get_text()
    tokens = re.findall(r'\b\w+\b', full_text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in STOP_WORDS]
    tokens = tokens[:MAX_TOKENS]

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

    dense_matrix = co_occurrence_matrix.toarray()
    pca = PCA(n_components=EMBEDDING_DIM)
    embeddings = pca.fit_transform(dense_matrix)
    print("Embeddings gerados.\n")

    # --- PASSO 2: Agrupar os embeddings em clusters com K-Means ---
    print(f"Agrupando as {VOCAB_SIZE} palavras em {N_CLUSTERS} tópicos (clusters)...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(embeddings)

    # --- PASSO 3: Exibir os clusters encontrados ---
    clusters = defaultdict(list)
    for i, word in enumerate(vocab):
        cluster_id = kmeans.labels_[i]
        clusters[cluster_id].append(word)
    
    print("\n--- Tópicos Encontrados no Mahabharata ---")
    for i in range(N_CLUSTERS):
        # Mostra o número do tópico e as primeiras 20 palavras dele
        words_in_cluster = clusters[i]
        print(f"\n--- Tópico {i+1} ---")
        print(", ".join(words_in_cluster[:20]))

if __name__ == '__main__':
    main()
