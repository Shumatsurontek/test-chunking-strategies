import os
from dotenv import load_dotenv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.logger import get_logger, logger_info, logger_debug, logger_warning
from test.text import TEXT
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
import numpy as np

load_dotenv()

def splitter_token_transformers(text, metadata, tokens_per_chunk=300, chunk_overlap=50):
    splitter = SentenceTransformersTokenTextSplitter(
        model_name="sentence-transformers/all-mpnet-base-v2",
        tokens_per_chunk=tokens_per_chunk,
        chunk_overlap=chunk_overlap
    )
    docs = splitter.create_documents([text], metadatas=[metadata])
    return [{"text": d.page_content, "metadata": d.metadata, "token_count": len(d.page_content.split())} for d in docs]

def splitter_recursive(text, metadata, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", r"(?<=[.?!])\s+", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(x.split()),
        is_separator_regex=True
    )
    docs = splitter.create_documents([text], metadatas=[metadata])
    return [{"text": d.page_content, "metadata": d.metadata, "token_count": len(d.page_content.split())} for d in docs]

def splitter_char(text, metadata, chunk_size=300, chunk_overlap=50):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    chunks = splitter.split_text(text)
    return [{"text": chunk, "metadata": metadata, "token_count": len(chunk.split())} for chunk in chunks]

def get_stats(chunks, method_name):
    sizes_words = [c['token_count'] for c in chunks]
    sizes_chars = [len(c['text']) for c in chunks]
    return {
        'method': method_name,
        'num_chunks': len(chunks),
        'min_words': min(sizes_words) if sizes_words else 0,
        'max_words': max(sizes_words) if sizes_words else 0,
        'mean_words': sum(sizes_words)/len(sizes_words) if sizes_words else 0,
        'min_chars': min(sizes_chars) if sizes_chars else 0,
        'max_chars': max(sizes_chars) if sizes_chars else 0,
        'mean_chars': sum(sizes_chars)/len(sizes_chars) if sizes_chars else 0,
    }

#Arbitraire
def score_chunking(row):
    score = 0
    # Critère 1 : nombre de chunks raisonnable
    if 10 <= row['num_chunks'] <= 100:
        score += 1
    # Critère 2 : taille moyenne des chunks (mots)
    if 30 <= row['mean_words'] <= 200:
        score += 1
    # Critère 3 : homogénéité (écart-type < 50% de la moyenne)
    if row['std_words'] < 0.5 * row['mean_words']:
        score += 1
    # Critère 4 : chunk max pas trop gros
    if row['max_words'] < 300:
        score += 1
    # Critère 5 : chunk min pas trop petit
    if row['min_words'] < 3:
        score -= 1
    return score

def main(save_results=False, save_dir=None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    logger = get_logger("chunking", save_dir)
    test_text = TEXT
    test_metadata = {"source": "test", "langue": "français"}

    logger_info(logger, "🚀 Test du chunking")
    logger_debug(logger, f"Texte original: {test_text[:100]}...")  # Affiche seulement le début

    methods = [
        ("SentenceTransformersTokenTextSplitter", splitter_token_transformers, dict(tokens_per_chunk=300, chunk_overlap=50)),
        ("RecursiveCharacterTextSplitter", splitter_recursive, dict(chunk_size=3, chunk_overlap=1)),
        ("CharacterTextSplitter", splitter_char, dict(chunk_size=10, chunk_overlap=5)),
    ]

    stats = []
    for name, splitter, params in methods:
        logger_info(logger, f"==> {name}")
        chunks = splitter(test_text, test_metadata, **params)
        logger_info(logger, f"Chunks: {len(chunks)} | min/max/mean mots: {min([c['token_count'] for c in chunks]) if chunks else 0}/"
                            f"{max([c['token_count'] for c in chunks]) if chunks else 0}/"
                            f"{round(sum([c['token_count'] for c in chunks])/len(chunks), 2) if chunks else 0}")
        stats.append(get_stats(chunks, name))

    df = pd.DataFrame(stats)
    logger_info(logger, f"Résumé stats:\n{df[['method','num_chunks','mean_words','mean_chars']]}")

    # Calcul de l'écart-type des tailles de chunks (mots)
    df['std_words'] = [np.std([c['token_count'] for c in splitter(TEXT, {"source": "test"}, **params)])
                   for (_, splitter, params) in [
                       ("SentenceTransformersTokenTextSplitter", splitter_token_transformers, dict(tokens_per_chunk=300, chunk_overlap=50)),
                       ("RecursiveCharacterTextSplitter", splitter_recursive, dict(chunk_size=3, chunk_overlap=1)),
                       ("CharacterTextSplitter", splitter_char, dict(chunk_size=10, chunk_overlap=5)),
                   ]]

    df['score'] = df.apply(score_chunking, axis=1)
    df = df.sort_values('score', ascending=False)

    logger_info(logger, f"\n===== SCORING DES MÉTHODES (RAG/Agent) =====\n{df[['method','score','num_chunks','mean_words','std_words','max_words','min_words']]}")
    logger_info(logger, f"🏆 Meilleure méthode (score): {df.iloc[0]['method']} (score={df.iloc[0]['score']})")

    # Visualisation épurée
    variables = ['mean_words', 'mean_chars', 'num_chunks']
    fig, axes = plt.subplots(1, len(variables), figsize=(5 * len(variables), 4))
    if len(variables) == 1:
        axes = [axes]
    palette = sns.color_palette("Set2", len(df))
    for ax, var in zip(axes, variables):
        sns.barplot(data=df, x='method', y=var, ax=ax, palette=palette)
        ax.set_title(var.replace('_', ' ').capitalize())
        ax.set_ylabel('')
        ax.set_xlabel('')
        for label in ax.get_xticklabels():
            label.set_rotation(20)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
    plt.suptitle('Comparaison des méthodes de chunking', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

    # Optionnel : sur le graphique, surligner la meilleure méthode
    best_method = df.iloc[0]['method']
    for ax in axes:
        for bar, label in zip(ax.patches, df['method']):
            if label == best_method:
                bar.set_edgecolor('red')
                bar.set_linewidth(3)

    if save_results:
        save_dir = save_dir or "."
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "chunking_stats.csv")
        df.to_csv(save_path, index=False)
        logger_info(logger, f"Résultats sauvegardés dans {save_path}")

if __name__ == "__main__":
    save_dir = os.environ.get("SAVE_DIR")
    save_results_env = os.environ.get("SAVE_RESULTS", "False").lower()
    save_results = save_results_env in ("1", "true", "yes")
    main(save_results=save_results, save_dir=save_dir)
