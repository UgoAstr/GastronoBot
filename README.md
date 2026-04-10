# 🍳 GastronoBot

## 🇫🇷 Français

### Description

GastronoBot est un assistant culinaire intelligent développé lors d'un stage, basé sur un livre de recettes du monde. Le projet met en œuvre une architecture **RAG (Retrieval-Augmented Generation)** entièrement codée à la main, enrichie d'une couche **agentique avec tool calling**, permettant au LLM de choisir dynamiquement les outils à utiliser pour répondre aux questions.

### Architecture

Le pipeline RAG se décompose en trois briques modulaires :

- **R (Retrieval)** — Recherche hybride combinant la similarité cosinus sur des embeddings (SentenceTransformers) et le score BM25 pour retrouver les extraits les plus pertinents du livre.
- **A (Augmentation)** — Construction d'un prompt structuré intégrant les extraits récupérés comme contexte pour le LLM.
- **G (Generation)** — Génération de la réponse via un LLM (Ollama en local ou Groq en cloud).

La couche agentique ajoute le **tool calling** : le LLM analyse la question de l'utilisateur et décide quel outil appeler parmi ceux disponibles (recherche dans le livre, conversion d'unités de cuisine), sans jamais inventer de réponse.

### Fonctionnalités

- Recherche hybride (sémantique + mots-clés) dans un livre de recettes
- Génération de réponses basées uniquement sur le contenu du livre
- Traduction automatique français ↔ anglais pour les requêtes et réponses
- Tool calling agentique : le LLM choisit dynamiquement entre la recherche RAG et la conversion d'unités
- Interface Streamlit pour interagir avec le chatbot
- Support multi-modèles : Ollama (local) et Groq (cloud)

### Stack technique

| Composant | Technologie |
|---|---|
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Recherche mots-clés | BM25 (rank_bm25) |
| LLM local | Llama 3.2 via Ollama |
| LLM cloud | Qwen 32B / Llama 3.1 8B via Groq |
| Traduction | Llama 3.1 8B via Groq |
| Interface | Streamlit |
| Données | Pandas, NumPy, PyTorch |

### Installation

```bash
# Cloner le repo
git clone https://github.com/votre-username/GastronoBot.git
cd GastronoBot

# Configurer la clé API Groq
# Créer un fichier api_groq.py avec : API_KEYS = "votre_clé"

# (Optionnel) Installer Ollama pour l'utilisation locale
# https://ollama.ai
ollama pull llama3.2
```

### Utilisation

```bash
# Lancer l'interface Streamlit
streamlit run app.py
```

---

## 🇬🇧 English

### Description

GastronoBot is an intelligent cooking assistant developed during an internship, built on a world cookbook compiled by Georg-August-Universität Göttingen. The project implements a fully hand-coded **RAG (Retrieval-Augmented Generation)** architecture, enhanced with an **agentic layer using tool calling**, allowing the LLM to dynamically choose which tools to use when answering questions.

### Architecture

The RAG pipeline is broken down into three modular components:

- **R (Retrieval)** — Hybrid search combining cosine similarity on embeddings (SentenceTransformers) and BM25 scoring to find the most relevant excerpts from the book.
- **A (Augmentation)** — Construction of a structured prompt incorporating the retrieved excerpts as context for the LLM.
- **G (Generation)** — Response generation via an LLM (Ollama locally or Groq in the cloud).

The agentic layer adds **tool calling**: the LLM analyzes the user's question and decides which tool to call among those available (book search, cooking unit conversion), without ever making up an answer.

### Features

- Hybrid search (semantic + keywords) across a world cookbook
- Responses generated strictly from book content
- Automatic French ↔ English translation for queries and answers
- Agentic tool calling: the LLM dynamically chooses between RAG search and unit conversion
- Streamlit interface to interact with the chatbot
- Multi-model support: Ollama (local) and Groq (cloud)

### Tech Stack

| Component | Technology |
|---|---|
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Keyword search | BM25 (rank_bm25) |
| Local LLM | Llama 3.2 via Ollama |
| Cloud LLM | Qwen 32B / Llama 3.1 8B via Groq |
| Translation | Llama 3.1 8B via Groq |
| Interface | Streamlit |
| Data | Pandas, NumPy, PyTorch |

### Installation

```bash
# Clone the repo
git clone https://github.com/your-username/GastronoBot.git
cd GastronoBot


# Configure Groq API key
# Create a file api_groq.py with: API_KEYS = "your_key"

# (Optional) Install Ollama for local usage
# https://ollama.ai
ollama pull llama3.2
```

### Usage

```bash
# Launch the Streamlit interface
streamlit run app.py
```

---

*Projet réalisé dans le cadre d'un stage — Project developed during an internship*