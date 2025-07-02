[![Open in Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
_Interaktive Web-App direkt im Browser öffnen (via Streamlit Cloud)_

> **Hinweis:**
> Im Standard-Demo-Modus werden die kostenlosen HF-Inference-APIs genutzt.
> Bei Bedarf kann der Nutzer eigene API-Keys (Hugging Face oder OpenAI) eingeben, um höhere Limits oder bessere Performance zu erzielen.

# NLP Multi-Agent RAG (Wahlfachprojekt)

**Wahlfachprojekt** im Rahmen des Studiengangs
**BSc Systemtechnik – Vertiefung Computational Engineering**
**Frühjahr 2025** – OST – Ostschweizer Fachhochschule
**Autor:** Rino Albertin

---

## 📌 Projektbeschreibung

Dieses Projekt implementiert ein Multi-Agent RAG-System.

Die Anwendung:
- lädt PDF-Dokumente hoch
- verarbeitet sie zu semantischen Chunks
- sucht relevante Inhalte über eine Vektor-Datenbank (FAISS)
- generiert Antworten mit Hilfe eines LLM
- bietet verschiedene Agenten-Logiken

---

## ⚙️ Lokale Ausführung

<details>
<summary><strong>Ausführung mit Poetry</strong></summary>

**Voraussetzungen:**

- [Poetry](https://python-poetry.org/) ist installiert

**Vorgehen:**

1. Repository klonen:
    ```bash
    git clone https://github.com/<dein-username>/nlp_multiagent_rag.git
    cd nlp_multiagent_rag
    ```

2. Abhängigkeiten installieren:
    ```bash
    poetry install
    ```

3. `.env` Datei anlegen (basierend auf `.env.template`) und eigene Keys eintragen.

4. Streamlit App starten:
    ```bash
    poetry run streamlit run ui/app.py
    ```
</details>

---

## 📂 Projektstruktur
<details>
<summary><strong>Projektstruktur anzeigen</strong></summary>

```bash
.
├── .github/                              # GitHub-spezifische Workflows und Aktionen
│   └── workflows/                        # Enthält CI/CD-Workflows für GitHub Actions
│       └── lint.yml                      # Linter-Workflow, der bei jedem Push/Pull Request ausgeführt wird
│
├── docs/                                 # Dokumentation für das Projekt
│
├── src/                                  # Quellcode des Projekts
│   ├── agents/                           # Enthält verschiedene Agents (Retriever, Generator etc.)
│   │   ├── __init__.py
│   │   ├── generator_agent.py            # Agent, der Antworten generiert (LLM)
│   │   ├── retriever_agent.py            # Agent, der relevante Dokument-Abschnitte sucht
│   │   ├── summarizer_agent.py           # Agent, der Inhalte zusammenfasst
│   │
│   ├── ingestion/                        # Alles zum Einlesen und Vorverarbeiten von Daten
│   │   ├── __init__.py
│   │   ├── chunker.py                    # Zerlegt Texte in Chunks für Embeddings
│   │   ├── embedder.py                   # Berechnet Embeddings (Hugging Face oder OpenAI)
│   │   ├── loader.py                     # PDF-Loader
│   │
│   ├── memory/                           # Speichert Chat-Verlauf, um Kontext zu behalten
│   │   ├── __init__.py
│   │   ├── memory.py                     # Speichert Conversation Memory
│   │
│   ├── utils/                            # Hilfsfunktionen
│   │   ├── __init__.py
│   │   ├── logging.py                    # Logging-Konfiguration und Utilities
│   │
│   ├── vectorstore/                      # Speichert Embeddings für semantische Suche
│   │   ├── __init__.py
│   │   ├── faiss_store.py                # Wrapper für FAISS-Vektor-DB
│   │
│   ├── __init__.py                       # Initialisierungsdatei für das Gesamtmodul
│   │
│   ├── main.py                           # Einstiegspunkt für das Projekt (z.B. Pipeline starten)
│   │
│   ├── pipeline.py                       # Orchestriert den LangGraph-Flow zwischen Agents
│
├── tests/                                # Tests für das Projekt
│
├── ui/                                   # Frontend/UI für Streamlit
│   ├── app.py                            # Streamlit-App
│
├── .env.template                         # Vorlage für Environment-Variablen
├── .gitignore                            # Regeln für Dateien, die nicht ins Repo gehören
├── .pre-commit-config.yaml               # Config für pre-commit Hooks (Black, Ruff etc.)
├── LICENSE                               # Lizenzdatei für das Projekt (z. B. MIT License)
├── poetry.lock                           # Fixierte Dependency-Versionen für das Projekt
├── pyproject.toml                        # Poetry-Projektdefinition (Dependencies, Config etc.)
├── README.md                             # Dokumentation und Anleitung zum Projekt
```
</details>

---

## 📄 Lizenz

Dieses Projekt steht unter der [MIT-Lizenz](LICENSE).

---
## 📚 Quellen

- Lehrunterlagen „Natural Language Processing“ – OST – Ostschweizer Fachhochschule
