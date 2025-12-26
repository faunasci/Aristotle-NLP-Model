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
VOCAB_SIZE = 5000
MAX_TOKENS = 500000
EMBEDDING_DIM = 100

# --- Definição dos Tópicos Aristotélicos ---
ARISTOTLE_TOPICS = {
    "1. Substância": ['homem', 'deus', 'rei', 'mulher', 'cavalo', 'filho', 'homens', 'espírito', 'senhor', 'chefe', 'pai', 'mãe'],
    "2. Quantidade": ['um', 'dois', 'três', 'dez', 'cem', 'mil', 'muitos', 'grande', 'todo', 'todos', 'único'],
    "3. Qualidade": ['bom', 'grande', 'belo', 'poderoso', 'divino', 'sábio', 'alto', 'forte', 'branco', 'justo', 'excelente'],
    "4. Relação": ['filho', 'pai', 'irmão', 'mãe', 'esposa', 'amigo', 'chefe', 'senhor', 'mestre', 'marido', 'irmã'],
    "5. Lugar": ['cidade', 'floresta', 'terra', 'mundo', 'céu', 'região', 'portão', 'palácio', 'casa', 'lugar', 'chão'],
    "6. Tempo": ['dia', 'noite', 'tempo', 'manhã', 'momento', 'ano', 'longo', 'sempre', 'primeiro', 'nunca','depois'],
    "7. Posição": ['pé', 'sentado', 'deitado', 'caiu', 'ficou', 'sentou', 'foi', 'veio', 'entrou', 'montado'],
    "8. Estado": ['vivo', 'morto', 'silêncio', 'feliz', 'pronto', 'medo', 'sozinho', 'vestido', 'armado', 'satisfeito', 'zangado'],
    "9. Ação": ['falou', 'disse', 'foi', 'veio', 'viu', 'tomou', 'fez', 'deu', 'começou', 'pensou', 'pediu', 'respondeu', 'comer'],
    "10. Paixão": ['satisfeito', 'encantado', 'aflito', 'contemplou', 'abençoado', 'adornado', 'apaixonado']
}

STOP_WORDS = set(['a', 'à', 'adeus', 'agora', 'ainda', 'além', 'algo', 'algumas', 'alguns', 'ali', 'com', 'como', 'da', 'das', 'de', 'do', 'dos', 'e', 'é', 'em', 'foi', 'mas', 'não', 'no', 'na', 'o', 'os', 'ou', 'para', 'por', 'que', 'se', 'um', 'uma'])

def main():
    # --- PASSO 1: Gerar Embeddings e Vetores de Tópico ---
    print(f"Processando {FILENAME} para gerar embeddings e vetores de tópico...")
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
    
    topic_vectors = {}
    for topic_name, seed_words in ARISTOTLE_TOPICS.items():
        seed_vectors = [embeddings[word_to_id[word]] for word in seed_words if word in word_to_id]
        if seed_vectors:
            topic_vectors[topic_name] = np.mean(seed_vectors, axis=0)
    print("Processamento concluído!\n")

    # --- PASSO 2: Loop interativo para analisar o perfil de uma palavra ---
    while True:
        try:
            query_word = input("Digite uma palavra para ver seu Perfil Aristotélico (ou 'SAIR' para terminar): ").lower()
            if query_word.upper() == 'SAIR':
                break
            
            if query_word not in word_to_id:
                print(f"A palavra '{query_word}' não está no vocabulário analisado.")
                continue

            query_vector = embeddings[word_to_id[query_word]].reshape(1, -1)
            
            # Calcula a similaridade com cada tópico
            profile = {}
            for topic_name, topic_vector in topic_vectors.items():
                similarity = cosine_similarity(query_vector, topic_vector.reshape(1, -1))[0][0]
                profile[topic_name] = similarity
            
            # Ordena o perfil por pontuação (do maior para o menor)
            sorted_profile = sorted(profile.items(), key=lambda item: item[1], reverse=True)
            
            print(f"\n--- Perfil Aristotélico para '{query_word}' ---")
            for topic_name, score in sorted_profile:
                print(f"{topic_name:<15} | Afinidade: {score:.2f}")
            print("\n")

        except (KeyboardInterrupt, EOFError):
            print("\nFinalizando...")
            break

if __name__ == '__main__':
    main()
