import re
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Tenta importar as bibliotecas necessárias e dá uma mensagem de erro útil se não encontrar
try:
    from bs4 import BeautifulSoup
    from scipy.sparse import lil_matrix
except ImportError:
    print("Por favor, instale as bibliotecas necessárias:")
    print("pip install beautifulsoup4 scikit-learn numpy scipy matplotlib")
    exit()

# --- Parâmetros de Configuração ---
FILENAME = 'data/Mahabharata Volume 1_djvu.txt'
VOCAB_SIZE = 5000  # Analisar as 5000 palavras mais comuns
MAX_TOKENS = 100000 # Usar os primeiros 100.000 tokens do livro para economizar tempo/memória
EMBEDDING_DIM = 50 # Queremos vetores de 50 dimensões

# Lista de stop words em inglês para serem removidas
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


def process_book():
    print(f"1. Lendo e limpando o arquivo: {FILENAME}...")
    with open(FILENAME, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    full_text = soup.get_text()
    
    # Tokeniza o texto: minúsculas, apenas palavras, sem stop words
    tokens = re.findall(r'\b\w+\b', full_text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in STOP_WORDS]
    
    if len(tokens) > MAX_TOKENS:
        print(f"Limitando a análise aos primeiros {MAX_TOKENS} tokens.")
        tokens = tokens[:MAX_TOKENS]

    print(f"2. Construindo vocabulário de {VOCAB_SIZE} palavras...")
    # Conta a frequência e pega as palavras mais comuns
    counts = Counter(tokens)
    vocab = [word for word, _ in counts.most_common(VOCAB_SIZE)]
    word_to_id = {word: i for i, word in enumerate(vocab)}
    
    # Filtra os tokens para conter apenas palavras do nosso vocabulário
    numeric_tokens = [word_to_id[word] for word in tokens if word in word_to_id]

    print("3. Construindo a matriz de coocorrência...")
    # Usa uma matriz esparsa para economizar memória
    co_occurrence_matrix = lil_matrix((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float32)
    context_window = 2
    for i, target_word_id in enumerate(numeric_tokens):
        for j in range(max(0, i - context_window), min(len(numeric_tokens), i + context_window + 1)):
            if i != j:
                context_word_id = numeric_tokens[j]
                co_occurrence_matrix[target_word_id, context_word_id] += 1

    print("4. Executando PCA para reduzir a dimensionalidade...")
    # Converte a matriz para um formato denso para o PCA
    # Cuidado: isso pode consumir muita memória se VOCAB_SIZE for muito grande
    dense_matrix = co_occurrence_matrix.toarray()
    pca = PCA(n_components=EMBEDDING_DIM)
    embeddings = pca.fit_transform(dense_matrix)

    print(f"5. Gerando gráfico com os {EMBEDDING_DIM} embeddings...")
    # Pega apenas as 2 primeiras dimensões para o gráfico
    plot_embeddings = embeddings[:, :2]
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(plot_embeddings[:, 0], plot_embeddings[:, 1], alpha=0.1)

    # Adiciona rótulos para algumas das palavras mais interessantes (as 100 mais comuns)
    for i in range(min(100, VOCAB_SIZE)):
        ax.annotate(vocab[i], (plot_embeddings[i, 0], plot_embeddings[i, 1]))

    ax.set_title(f'Embeddings de Palavras do Mahabharata (PCA, {EMBEDDING_DIM}D)')
    output_filename = 'mahabharata_embeddings_plot.png'
    plt.savefig(output_filename)
    print(f"\nProcesso concluído! Gráfico salvo como '{output_filename}'.")

if __name__ == '__main__':
    process_book()
