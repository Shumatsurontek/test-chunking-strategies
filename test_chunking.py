from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
import json

def splitter_token_transformers(text, metadata, tokens_per_chunk=300, chunk_overlap=50):
    """
    Découpe un texte en chunks à l'aide de SentenceTransformers
    
    Args:
        text: Le texte à découper
        metadata: Les métadonnées à associer aux chunks
        tokens_per_chunk: Nombre de tokens par chunk
        chunk_overlap: Nombre de tokens de chevauchement entre chunks
    
    Returns:
        Liste de dictionnaires contenant le texte, métadonnées et nombre de tokens
    """
    splitter = SentenceTransformersTokenTextSplitter(
        model_name="sentence-transformers/all-mpnet-base-v2",
        tokens_per_chunk=tokens_per_chunk,
        chunk_overlap=chunk_overlap
    )
    docs = splitter.create_documents([text], metadatas=[metadata])
    return [{"text": d.page_content, "metadata": d.metadata, "token_count": len(d.page_content.split())} for d in docs]

def splitter_recursive(text, metadata, chunk_size=300, chunk_overlap=50):
    """
    Découpe un texte en chunks à l'aide de RecursiveCharacterTextSplitter
    
    Args:
        text: Le texte à découper
        metadata: Les métadonnées à associer aux chunks
        chunk_size: Taille des chunks (en nombre de mots)
        chunk_overlap: Chevauchement entre chunks (en nombre de mots)
    
    Returns:
        Liste de dictionnaires contenant le texte, métadonnées et nombre de tokens
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", r"(?<=[.?!])\s+", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(x.split()),
        is_separator_regex=True
    )
    docs = splitter.create_documents([text], metadatas=[metadata])
    return [{"text": d.page_content, "metadata": d.metadata, "token_count": len(d.page_content.split())} for d in docs]

def splitter_char(text, metadata, chunk_size=1000, chunk_overlap=200):
    """
    Découpe un texte en chunks à l'aide de CharacterTextSplitter
    
    Args:
        text: Le texte à découper
        metadata: Les métadonnées à associer aux chunks
        chunk_size: Taille des chunks (en caractères)
        chunk_overlap: Chevauchement entre chunks (en caractères)
    
    Returns:
        Liste de dictionnaires contenant le texte, métadonnées et nombre de tokens
    """
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    chunks = splitter.split_text(text)
    return [{"text": chunk, "metadata": metadata, "token_count": len(chunk.split())} for chunk in chunks]

def visualize_chunks(chunks, method_name):
    """
    Affiche les chunks de manière formatée
    """
    print(f"\n" + "="*70)
    print(f"RÉSULTATS DU CHUNKING - {method_name}")
    print("="*70)
    
    if not chunks:
        print("❌ Aucun chunk généré")
        return
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n📄 CHUNK {i}:")
        print(f"   Texte: '{chunk['text']}'")
        print(f"   Nombre de mots: {chunk['token_count']}")
        print(f"   Longueur (caractères): {len(chunk['text'])}")
        print("-" * 50)
    
    print(f"\n📊 RÉSUMÉ:")
    print(f"   Nombre total de chunks: {len(chunks)}")
    reconstructed = ' '.join([c['text'] for c in chunks])
    print(f"   Texte reconstitué: '{reconstructed}'")
    print(f"   Préservation du texte: {'✅ OUI' if reconstructed.replace(' ', '').lower() == 'lavoitureestrouge' else '❌ NON'}")

def main():
    # Texte à tester
    test_text = "La voiture est rouge"
    test_metadata = {"source": "test", "langue": "français"}
    
    print("🚀 Test du chunking avec SentenceTransformers")
    print(f"📝 Texte original: '{test_text}'")
    print(f"📋 Métadonnées: {test_metadata}")
    
    # Tests pour chaque méthode de chunking
    print("\n\n===== TESTS SENTENCE TRANSFORMERS TOKEN SPLITTER =====")
    chunks = splitter_token_transformers(test_text, test_metadata, tokens_per_chunk=300, chunk_overlap=50)
    visualize_chunks(chunks, "SentenceTransformersTokenTextSplitter")
    
    print("\n\n===== TESTS RECURSIVE CHARACTER SPLITTER =====")
    chunks = splitter_recursive(test_text, test_metadata, chunk_size=3, chunk_overlap=1)
    visualize_chunks(chunks, "RecursiveCharacterTextSplitter")

    print("\n\n===== TESTS CHARACTER TEXT SPLITTER =====")
    chunks = splitter_char(test_text, test_metadata, chunk_size=10, chunk_overlap=5)
    visualize_chunks(chunks, "CharacterTextSplitter")

if __name__ == "__main__":
    main()
