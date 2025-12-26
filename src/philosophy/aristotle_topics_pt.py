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
MAX_TOKENS = 10000
EMBEDDING_DIM = 40

# --- Definição dos Tópicos Aristotélicos (AGORA EM PORTUGUÊS) ---
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
    "10. Paixão (ser afetado)": ['satisfeito', 'encantado', 'aflito', 'zangado', 'medo', 'atingido', 'ferido', 'contemplou', 'abençoado', 'adornado']
}

# Lista de Stop Words em Português
STOP_WORDS = set([
    'a', 'à', 'adeus', 'agora', 'ainda', 'além', 'algo', 'algumas', 'alguns', 'ali', 'ambas', 'ambos', 'antes', 'ao', 'aonde', 'aos', 'apenas', 'apoio', 'após', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aqui', 'aquilo', 'as', 'assim', 'até', 'atrás', 'através', 'cada', 'coisa', 'coisas', 'com', 'como', 'contra', 'contudo', 'da', 'daquele', 'daqueles', 'das', 'de', 'dela', 'delas', 'dele', 'deles', 'depois', 'dessa', 'dessas', 'desse', 'desses', 'desta', 'destas', 'deste', 'destes', 'deve', 'devem', 'devendo', 'dever', 'deverá', 'deverão', 'deveria', 'deveriam', 'devia', 'deviam', 'disse', 'disso', 'disto', 'dito', 'diz', 'dizem', 'dizer', 'do', 'dois', 'dos', 'e', 'é', 'ela', 'elas', 'ele', 'eles', 'em', 'enquanto', 'entre', 'era', 'eram', 'essa', 'essas', 'esse', 'esses', 'esta', 'está', 'estamos', 'estão', 'estar', 'estas', 'estava', 'estavam', 'este', 'esteja', 'estejam', 'estejamos', 'estes', 'esteve', 'estive', 'estivemos', 'estiver', 'estivera', 'estiveram', 'estivéramos', 'estiverem', 'estivermos', 'estivesse', 'estivessem', 'estivéssemos', 'estou', 'eu', 'fazendo', 'fazer', 'feita', 'feitas', 'feito', 'feitos', 'foi', 'for', 'fora', 'foram', 'fôramos', 'forem', 'formos', 'fosse', 'fossem', 'fôssemos', 'fui', 'geral', 'grande', 'grandes', 'há', 'hoje', 'hora', 'horas', 'houve', 'houvera', 'houveram', 'houvéramos', 'houverem', 'houveremos', 'houveria', 'houveriam', 'houvermos', 'houvesse', 'houvessem', 'houvéssemos', 'isso', 'isto', 'já', 'la', 'lá', 'lhe', 'lhes', 'lo', 'local', 'logo', 'longe', 'lugar', 'maior', 'mais', 'mas', 'me', 'mesma', 'mesmas', 'mesmo', 'mesmos', 'meu', 'meus', 'minha', 'minhas', 'muita', 'muitas', 'muito', 'muitos', 'na', 'nada', 'não', 'nas', 'nem', 'nenhum', 'nenhuma', 'nessa', 'nessas', 'nesta', 'nestas', 'ninguém', 'no', 'nos', 'nós', 'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa', 'nunca', 'o', 'os', 'ou', 'outra', 'outras', 'outro', 'outros', 'para', 'pela', 'pelas', 'pelo', 'pelos', 'pequena', 'pequenas', 'pequeno', 'pequenos', 'per', 'perante', 'pode', 'pude', 'podendo', 'poder', 'poderia', 'poderiam', 'podia', 'podiam', 'pois', 'ponto', 'pontos', 'por', 'porém', 'porque', 'posso', 'pouca', 'poucas', 'pouco', 'poucos', 'primeiro', 'primeiros', 'própria', 'próprias', 'próprio', 'próprios', 'quais', 'qual', 'quando', 'quanto', 'quantos', 'que', 'quem', 'quer', 'quero', 'se', 'seja', 'sejam', 'sejamos', 'sem', 'sempre', 'sendo', 'será', 'serão', 'serei', 'seremos', 'seria', 'seriam', 'seríamos', 'seu', 'seus', 'só', 'sob', 'sobre', 'somos', 'sou', 'sua', 'suas', 'tal', 'talvez', 'também', 'tampouco', 'tanta', 'tantas', 'tanto', 'tão', 'te', 'tem', 'tém', 'tendo', 'tenha', 'tenham', 'tenhamos', 'tenho', 'terá', 'terão', 'terei', 'teremos', 'teria', 'teriam', 'teríamos', 'teu', 'teus', 'teve', 'ti', 'tido', 'tinha', 'tinham', 'tínhamos', 'tive', 'tivemos', 'tiver', 'tivera', 'tiveram', 'tivéramos', 'tiverem', 'tivermos', 'tivesse', 'tivessem', 'tivéssemos', 'toda', 'todas', 'todavia', 'todo', 'todos', 'três', 'tu', 'tua', 'tuas', 'tudo', 'última', 'últimas', 'último', 'últimos', 'um', 'uma', 'umas', 'uns', 'vai', 'vamos', 'vão', 'vários', 'vem', 'vêm', 'vendo', 'ver', 'vez', 'vindo', 'vir', 'você', 'vocês', 'vos', 'às', 'àquele', 'àqueles', 'éramos'])

def main():
    # --- PASSO 1: Gerar os embeddings (lógica já conhecida) ---
    print(f"Processando {FILENAME} para gerar embeddings...")
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
    print("Embeddings gerados.\n")

    # --- PASSO 2: Criar os vetores de tópico sintéticos ---
    print("Criando vetores de tópico a partir das categorias de Aristóteles...")
    topic_vectors = {}
    for topic_name, seed_words in ARISTOTLE_TOPICS.items():
        seed_vectors = []
        for word in seed_words:
            if word in word_to_id:
                seed_vectors.append(embeddings[word_to_id[word]])
        
        if seed_vectors:
            topic_vectors[topic_name] = np.mean(seed_vectors, axis=0)

    # --- PASSO 3: Encontrar e exibir as palavras mais próximas de cada tópico ---
    print("\n--- Palavras do Mahabharata por Categoria Aristotélica ---")
    for topic_name, topic_vector in topic_vectors.items():
        # Calcula a similaridade do vetor do tópico com todos os embeddings de palavras
        similarities = cosine_similarity(topic_vector.reshape(1, -1), embeddings)[0]
        
        # Obtém os 20 mais similares
        top_indices = np.argsort(similarities)[::-1][:20]
        
        top_words = [vocab[i] for i in top_indices]
        
        print(f"\n--- {topic_name} ---")
        print(", ".join(top_words))

if __name__ == '__main__':
    main()
