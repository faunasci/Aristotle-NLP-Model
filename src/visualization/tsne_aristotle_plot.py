import re
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
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
MAX_TOKENS = 100000
EMBEDDING_DIM = 20

# --- Definição dos Tópicos Aristotélicos ---
ARISTOTLE_TOPICS = {
    "1. Substância": ['homem', 'deus', 'rei', 'mulher', 'cavalo', 'filho', 'homens', 'espírito', 'senhor', 'chefe', 'pai', 'mãe'],
    "2. Quantidade": ['um', 'dois', 'três', 'dez', 'cem', 'mil', 'muitos', 'grande', 'todo', 'todos', 'único'],
    "3. Qualidade": ['bom', 'grande', 'belo', 'poderoso', 'divino', 'sábio', 'alto', 'forte', 'branco', 'justo', 'excelente'],
    "4. Relação": ['filho', 'pai', 'irmão', 'mãe', 'esposa', 'amigo', 'chefe', 'senhor', 'mestre', 'marido', 'irmã'],
    "5. Lugar": ['cidade', 'floresta', 'terra', 'mundo', 'céu', 'região', 'portão', 'palácio', 'casa', 'lugar', 'chão'],
    "6. Tempo": ['dia', 'noite', 'tempo', 'manhã', 'momento', 'ano', 'longo', 'sempre', 'primeiro', 'nunca'],
    "7. Posição": ['pé', 'sentado', 'deitado', 'caiu', 'ficou', 'sentou', 'foi', 'veio', 'entrou', 'montado'],
    "8. Estado": ['vivo', 'morto', 'silêncio', 'feliz', 'pronto', 'medo', 'sozinho', 'vestido', 'armado', 'satisfeito', 'zangado'],
    "9. Ação": ['falou', 'disse', 'foi', 'veio', 'viu', 'tomou', 'fez', 'deu', 'começou', 'pensou', 'pediu', 'respondeu'],
    "10. Paixão": ['satisfeito', 'encantado', 'aflito', 'zangado', 'medo', 'atingido', 'ferido', 'contemplou', 'abençoado', 'adornado']
}

STOP_WORDS = set(['a', 'à', 'adeus', 'agora', 'ainda', 'além', 'algo', 'algumas', 'alguns', 'ali', 'com', 'como', 'da', 'das', 'de', 'do', 'dos', 'e', 'é', 'em', 'foi', 'mas', 'não', 'no', 'na', 'o', 'os', 'ou', 'para', 'por', 'que', 'se', 'um', 'uma'])

def main():
    # --- PASSO 1: Gerar Embeddings ---
    print(f"Processando {FILENAME}...")
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

    pca = PCA(n_components=EMBEDDING_DIM)
    embeddings = pca.fit_transform(co_occurrence_matrix.toarray())
    print("Embeddings gerados.")

    # --- PASSO 2: Criar vetores de tópico e atribuir cada palavra a um tópico ---
    print("Atribuindo cada palavra a uma categoria de Aristóteles...")
    topic_vectors = {}
    for topic_name, seed_words in ARISTOTLE_TOPICS.items():
        seed_vectors = [embeddings[word_to_id[word]] for word in seed_words if word in word_to_id]
        if seed_vectors:
            topic_vectors[topic_name] = np.mean(seed_vectors, axis=0)

    # Para cada palavra, encontrar o tópico mais próximo
    dominant_topic_labels = []
    topic_names = list(topic_vectors.keys())
    topic_matrix = np.array([topic_vectors[name] for name in topic_names])
    
    similarities = cosine_similarity(embeddings, topic_matrix)
    dominant_topic_labels = np.argmax(similarities, axis=1)

    # --- PASSO 3: Executar t-SNE ---
    print("Executando t-SNE... (Isso pode demorar alguns minutos)")
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42, max_iter=1000, n_iter_without_progress=200)
    tsne_embeddings = tsne.fit_transform(embeddings)
    print("t-SNE concluído.")

    # --- PASSO 4: Gerar o gráfico com legenda ---
    print("Gerando o gráfico final com legenda...")
    fig, ax = plt.subplots(figsize=(20, 20))

    # Plota os pontos de cada categoria com uma cor e rótulo diferentes
    for i, topic_name in enumerate(topic_names):
        # Encontra os pontos que pertencem a este tópico
        points = tsne_embeddings[dominant_topic_labels == i]
        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], alpha=0.6, label=topic_name)

    # Adiciona rótulos para as 100 palavras mais comuns
    for i in range(100):
        ax.annotate(vocab[i], (tsne_embeddings[i, 0], tsne_embeddings[i, 1]), alpha=0.85, fontsize=9)

    ax.set_title('Mapa Semântico do Mahabharata por Categoria Aristotélica (t-SNE)', fontsize=20)
    ax.axis('off')
    ax.legend(loc="upper right", title="Categorias")

    output_filename = 'plots/tsne_aristotle_plot.png'
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
    print(f"\nGráfico salvo com sucesso como '{output_filename}'!")

if __name__ == '__main__':
    main()
