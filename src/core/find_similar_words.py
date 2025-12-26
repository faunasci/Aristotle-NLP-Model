import re
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

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
EMBEDDING_DIM = 20  # Usando o número que decidimos!

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
    id_to_word = {i: word for word, i in word_to_id.items()} # Dicionário inverso
    
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
    print("Processamento concluído!\n")

    # --- PASSO 2: Loop interativo para encontrar palavras similares ---
    while True:
        try:
            query_word = input("Digite uma palavra para encontrar similares (ou 'SAIR' para terminar): ").lower()
            if query_word.upper() == 'SAIR':
                break
            
            if query_word not in word_to_id:
                print(f"A palavra '{query_word}' não está no vocabulário analisado.")
                continue

            # Pega o vetor da palavra consultada
            query_vector = embeddings[word_to_id[query_word]].reshape(1, -1)
            
            # Calcula a similaridade de cosseno entre o vetor da palavra e todos os outros
            similarities = cosine_similarity(query_vector, embeddings)[0]
            
            # Obtém os índices das palavras mais similares (em ordem decrescente)
            # Usamos [::-1] para inverter a ordem de argsort que é crescente
            top_indices = np.argsort(similarities)[::-1]
            
            print(f"\n--- Palavras mais similares a '{query_word}' ---")
            # Mostra as 10 palavras mais similares (começando do índice 1 para pular a própria palavra)
            for i in range(1, 11):
                similar_word_id = top_indices[i]
                similar_word = id_to_word[similar_word_id]
                similarity_score = similarities[similar_word_id]
                print(f"{i}. {similar_word} (Similaridade: {similarity_score:.2f})")
            print("\n")

        except (KeyboardInterrupt, EOFError):
            print("\nFinalizando...")
            break

if __name__ == '__main__':
    main()