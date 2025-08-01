# Test de Chunking avec SentenceTransformers

Ce projet teste la fonction de découpage de texte (chunking) utilisant SentenceTransformers sur la phrase "La voiture est rouge".

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python test_chunking.py
```

## Description

Le script teste différentes configurations de chunking :
- **Paramètres par défaut** : tokens_per_chunk=300, chunk_overlap=50
- **Chunks très petits** : tokens_per_chunk=2, chunk_overlap=1  
- **Chunks petits** : tokens_per_chunk=10, chunk_overlap=2

## Fonction principale

```python
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
```

## Résultats attendus

Pour une phrase courte comme "La voiture est rouge", les différents paramètres permettront de voir :
- Comment le texte est découpé en fonction de la taille des chunks
- L'effet du chevauchement entre chunks
- La préservation des métadonnées dans chaque chunk
