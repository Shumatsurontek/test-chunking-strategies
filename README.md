# 🧩 Benchmark Chunking pour RAG & Agent IA

Ce projet permet de **tester, comparer et scorer automatiquement** différentes méthodes de découpage de texte ("chunking") pour des usages de type RAG (Retrieval-Augmented Generation) ou d'agents IA.

## 🚀 Objectif

- **Comparer** plusieurs techniques de chunking (SentenceTransformers, RecursiveCharacter, CharacterTextSplitter…)
- **Mesurer** leurs performances sur un texte de test (nombre de chunks, taille moyenne, homogénéité…)
- **Scorer** chaque méthode selon des critères pertinents pour le RAG/agent (granularité, homogénéité, taille…)
- **Visualiser** les résultats et le classement via des graphiques Seaborn
- **Exporter** les résultats pour analyse ou reporting

## 📦 Installation

```bash
pip install -r requirements.txt
```

## ⚙️ Utilisation

```bash
python test_chunking.py
```

- Pour sauvegarder les résultats (CSV/logs) dans un dossier :
  ```bash
  SAVE_RESULTS=1 SAVE_DIR=results python test_chunking.py
  ```

## 📝 Fonctionnement

- Le script applique plusieurs méthodes de chunking sur un texte de test (modifiable dans `test/text.py`).
- Pour chaque méthode, il calcule :
  - Nombre de chunks
  - Taille moyenne/min/max/écart-type (en mots et caractères)
- Un **score** automatique est attribué à chaque méthode selon des critères adaptés au RAG/agent (voir code).
- Un graphique Seaborn affiche la comparaison.
- Le classement/scoring est loggé et exporté si demandé.

## 🏆 Critères de scoring (exemple)

- Nombre de chunks raisonnable (10–100)
- Taille moyenne des chunks adaptée (30–200 mots)
- Homogénéité (écart-type faible)
- Chunk max < 300 mots
- Chunk min pas trop petit

*(Les critères sont ajustables dans le code.)*

## 📊 Exemple de sortie

- Logs détaillés dans le terminal et/ou un fichier log
- Fichier CSV des stats dans le dossier choisi
- Graphique comparatif généré automatiquement

## 🔧 Personnalisation

- Modifiez le texte de test dans `test/text.py`
- Ajoutez vos propres méthodes de chunking dans `test_chunking.py`
- Adaptez les critères de scoring selon vos besoins métier

## 📁 Structure

- `test_chunking.py` : script principal (tests, scoring, visualisation)
- `test/text.py` : texte de test
- `utils/logger.py` : gestion des logs
- `requirements.txt` : dépendances Python

## 🤝 Contribuer

Suggestions, issues et PR bienvenus !

---

**Ce projet vous aide à choisir la meilleure stratégie de chunking pour vos pipelines RAG ou vos agents IA, de façon reproductible et visuelle.**
