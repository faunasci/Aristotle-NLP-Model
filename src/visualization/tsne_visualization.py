import re
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
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
N_CLUSTERS = 15

# Lista de Stop Words em Português
STOP_WORDS = set([
    'a', 'à', 'adeus', 'agora', 'ainda', 'além', 'algo', 'algumas', 'alguns', 'ali', 'ambas', 'ambos', 'antes', 'ao', 'aonde', 'aos', 'apenas', 'apoio', 'após', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aqui', 'aquilo', 'as', 'assim', 'até', 'atrás', 'através', 'cada', 'coisa', 'coisas', 'com', 'como', 'contra', 'contudo', 'da', 'daquele', 'daqueles', 'das', 'de', 'dela', 'delas', 'dele', 'deles', 'depois', 'dessa', 'dessas', 'desse', 'desses', 'desta', 'destas', 'deste', 'destes', 'deve', 'devem', 'devendo', 'dever', 'deverá', 'deverão', 'deveria', 'deveriam', 'devia', 'deviam', 'disse', 'disso', 'disto', 'dito', 'diz', 'dizem', 'dizer', 'do', 'dois', 'dos', 'e', 'é', 'ela', 'elas', 'ele', 'eles', 'em', 'enquanto', 'entre', 'era', 'eram', 'essa', 'essas', 'esse', 'esses', 'esta', 'está', 'estamos', 'estão', 'estar', 'estas', 'estava', 'estavam', 'este', 'esteja', 'estejam', 'estejamos', 'estes', 'esteve', 'estive', 'estivemos', 'estiver', 'estivera', 'estiveram', 'estivéramos', 'estiverem', 'estivermos', 'estivesse', 'estivessem', 'estivéssemos', 'estou', 'eu', 'fazendo', 'fazer', 'feita', 'feitas', 'feito', 'feitos', 'foi', 'for', 'fora', 'foram', 'fôramos', 'forem', 'formos', 'fosse', 'fossem', 'fôssemos', 'fui', 'geral', 'grande', 'grandes', 'há', 'hoje', 'hora', 'horas', 'houve', 'houvera', 'houveram', 'houvéramos', 'houverem', 'houveremos', 'houveria', 'houveriam', 'houvermos', 'houvesse', 'houvessem', 'houvéssemos', 'isso', 'isto', 'já', 'la', 'lá', 'lhe', 'lhes', 'lo', 'local', 'logo', 'longe', 'lugar', 'maior', 'mais', 'mas', 'me', 'mesma', 'mesmas', 'mesmo', 'mesmos', 'meu', 'meus', 'minha', 'minhas', 'muita', 'muitas', 'muito', 'muitos', 'na', 'nada', 'não', 'nas', 'nem', 'nenhum', 'nenhuma', 'nessa', 'nessas', 'nesta', 'nestas', 'ninguém', 'no', 'nos', 'nós', 'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa', 'nunca', 'o', 'os', 'ou', 'outra', 'outras', 'outro', 'outros', 'para', 'pela', 'pelas', 'pelo', 'pelos', 'pequena', 'pequenas', 'pequeno', 'pequenos', 'per', 'perante', 'pode', 'pude', 'podendo', 'poder', 'poderia', 'poderiam', 'podia', 'podiam', 'pois', 'ponto', 'pontos', 'por', 'porém', 'porque', 'posso', 'pouca', 'poucas', 'pouco', 'poucos', 'primeiro', 'primeiros', 'própria', 'próprias', 'próprio', 'próprios', 'quais', 'qual', 'quando', 'quanto', 'quantos', 'que', 'quem', 'quer', 'quero', 'se', 'seja', 'sejam', 'sejamos', 'sem', 'sempre', 'sendo', 'será', 'serão', 'serei', 'seremos', 'seria', 'seriam', 'seríamos', 'seu', 'seus', 'só', 'sob', 'sobre', 'somos', 'sou', 'sua', 'suas', 'tal', 'talvez', 'também', 'tampouco', 'tanta', 'tantas', 'tanto', 'tão', 'te', 'tem', 'tém', 'tendo', 'tenha', 'tenham', 'tenhamos', 'tenho', 'terá', 'terão', 'terei', 'teremos', 'teria', 'teriam', 'teríamos', 'teu', 'teus', 'teve', 'ti', 'tido', 'tinha', 'tinham', 'tínhamos', 'tive', 'tivemos', 'tiver', 'tivera', 'tiveram', 'tivéramos', 'tiverem', 'tivermos', 'tivesse', 'tivessem', 'tivéssemos', 'toda', 'todas', 'todavia', 'todo', 'todos', 'três', 'tu', 'tua', 'tuas', 'tudo', 'última', 'últimas', 'último', 'últimos', 'um', 'uma', 'umas', 'uns', 'vai', 'vamos', 'vão', 'vários', 'vem', 'vêm', 'vendo', 'ver', 'vez', 'vindo', 'vir', 'você', 'vocês', 'vos', 'às', 'àquele', 'àqueles', 'éramos'])

def main():
    # --- PASSO 1: Gerar Embeddings e Clusters ---
    print(f"Processando {FILENAME} para gerar embeddings e clusters...")
    with open(FILENAME, 'r', encoding='utf-8') as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    full_text = soup.get_text()
    tokens = re.findall(r'\b\w+\b', full_text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in STOP_WORDS]
    tokens = tokens[:MAX_TOKENS]

    counts = Counter(tokens)
    vocab = [word for word, _ in counts.most_common(VOCAB_SIZE)]
    
    numeric_tokens = [Counter(vocab)[word] for word in tokens if word in vocab]
    co_occurrence_matrix = lil_matrix((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float32)
    context_window = 2
    for i, target_word_id in enumerate(numeric_tokens):
        for j in range(max(0, i - context_window), min(len(numeric_tokens), i + context_window + 1)):
            if i != j:
                context_word_id = numeric_tokens[j]
                co_occurrence_matrix[target_word_id, context_word_id] += 1

    pca = PCA(n_components=EMBEDDING_DIM)
    embeddings = pca.fit_transform(co_occurrence_matrix.toarray())
    
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    print("Processamento inicial concluído.\n")

    # --- PASSO 2: Redução de dimensionalidade com t-SNE ---
    print("Executando t-SNE... (Isso pode demorar alguns minutos)")
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42, max_iter=1000, n_iter_without_progress=200)
    tsne_embeddings = tsne.fit_transform(embeddings)
    print("t-SNE concluído.\n")

    # --- PASSO 3: Gerar o gráfico final ---
    print("Gerando o gráfico de visualização...")
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Plota os pontos, colorindo-os de acordo com o rótulo do cluster
    scatter = ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)

    # Adiciona rótulos para as 100 palavras mais comuns para dar contexto
    for i in range(100):
        ax.annotate(vocab[i], 
                    (tsne_embeddings[i, 0], tsne_embeddings[i, 1]),
                    alpha=0.8,
                    fontsize=9)

    ax.set_title(f'Mapa Semântico do Mahabharata (t-SNE, {N_CLUSTERS} clusters)', fontsize=20)
    ax.axis('off') # Remove os eixos para um visual mais limpo
    
    output_filename = 'plots/tsne_mahabharata_plot.png'
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
    print(f"Gráfico salvo com sucesso como '{output_filename}'!")

if __name__ == '__main__':
    main()
