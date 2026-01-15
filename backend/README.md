# Veritas AI - Integrity Defense Backend

The backend system for **Veritas AI**, a Hybrid Plagiarism and Synthetic Text Detection platform. This service uses state-of-the-art NLP models to analyze documents for content integrity.

## 🚀 Key Features

### 1. Hybrid Analysis Engine
*   **Plagiarism Detection**: Uses **SBERT (Sentence-BERT)** to generate semantic embeddings for text chunks and stores them in a **FAISS** vector index for fast similarity search. It goes beyond simple keyword matching to find paraphrased content.
*   **AI Detection**: Utilizes a fine-tuned **RoBERTa** model to classify text as Human or AI-written. It also calculates statistical metrics like **Perplexity** (using GPT-2) and **Burstiness** to validate the findings.

### 2. Persistent Data Pool
*   Uploaded "reference" documents (e.g., student essays, verified articles) are ingested into a permanent **Vector Store**.
*   The FAISS index and metadata are saved to disk (`/vector_store`), ensuring the database survives server restarts.
*   New `documents` API allows retrieving the list of indexed files.

### 3. Dual-Mode Analysis
*   **File Upload**: Supports PDF, DOCX, and TXT files.
*   **Text Input**: Direct text pasting for quick analysis.

---

## 🛠️ Architecture

*   **Framework**: FastAPI (Python 3.10+)
*   **Vector DB**: FAISS (Facebook AI Similarity Search) - CPU Version
*   **Models**:
    *   *Plagiarism*: `all-MiniLM-L6-v2` (SBERT)
    *   *AI Detection*: `roberta-base-openai-detector` (HuggingFace)
    *   *Perplexity*: `gpt2`
*   **Storage**: Local filesystem (`/vector_store`) for indices; In-memory for active processing.

---

## 📂 Directory Structure

```
backend/
├── app/
│   ├── engine/
│   │   ├── detector.py      # AI Detection Logic (RoBERTa, GPT-2)
│   │   ├── plagiarism.py    # Plagiarism Logic (SBERT, FAISS, Persistence)
│   │   └── ...
│   ├── routers/
│   │   ├── analyze.py       # Detection Endpoints
│   │   ├── data.py          # Data Pool Management
│   │   └── ingest.py        # Reference Document Ingestion
│   ├── utils/
│   │   ├── chunking.py      # Text segmentation
│   │   └── text_extract.py  # File parsing
│   ├── config.py            # Global settings
│   └── models.py            # Pydantic Schemas
├── vector_store/            # Persisted FAISS index & metadata
├── main.py                  # Entry point
└── requirements.txt         # Dependencies
```

---

## 🔌 API Endpoints

### prefix: `/api/v1`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **POST** | `/ingest` | Upload a **Reference Document** to the Data Pool. Indexing is immediate and persistent. |
| **GET** | `/documents` | List all unique documents currently in the Data Pool. |
| **GET** | `/documents/{filename}/chunks` | Retrieve raw text chunks and IDs for a specific document. |
| **POST** | `/analyze` | Upload a **Suspect Document** (File) for integrity checking. |
| **POST** | `/analyze/text` | Submit raw text body for integrity checking. |

---

## ⚡ Setup & Running

### Prerequisites
*   Python 3.10 or higher
*   Virtual Environment (recommended)

### Installation

1.  **Clone & Enter Directory**
    ```bash
    cd backend
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv venv
    .\venv\Scripts\Activate  # Windows
    # source venv/bin/activate  # Mac/Linux
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The first run will download ~1GB of model weights from HuggingFace.*

### Running the Server

```bash
uvicorn main:app --reload --port 8000
```
*   The API will be available at `http://localhost:8000`
*   Swagger UI (Docs): `http://localhost:8000/docs`

---

## 💾 Persistence

The system automatically saves the vector index when new documents are ingested.
*   **Index File**: `vector_store/plagiarism.index`
*   **Metadata**: `vector_store/metadata.pkl`

To reset the database, simply stop the server and delete the `vector_store/` directory.
