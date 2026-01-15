# Backend Architecture & Integration Analysis Report

## Executive Summary

This backend system is a **Hybrid AI Detection & Plagiarism Analysis Platform** built with FastAPI. It combines multiple state-of-the-art detection methods to analyze documents for:

- **AI-Generated Content Detection** (using 4 different approaches)
- **Plagiarism Detection** (semantic similarity + re-ranking)
- **Stylometric Analysis** (AI phrase fingerprinting)

The system uses a sophisticated ensemble approach that combines model-based detection, statistical analysis, and linguistic pattern matching to provide comprehensive integrity scoring.

---

## 1. Core Framework & Architecture

### 1.1 Web Framework

- **Framework**: FastAPI (v0.115.0+)
- **Server**: Uvicorn with ASGI standard support
- **Architecture**: RESTful API with modular router-based design
- **CORS**: Configured for frontend integration (localhost:3000)

### 1.2 Entry Point

**File**: [`main.py`](backend/main.py)

```python
FastAPI App Configuration:
- Title: "Hybrid AI & Plagiarism Detection System"
- Version: 1.0.0
- Middleware: CORS (allows frontend communication)
- Routers:
  ├── /api/v1/ingest  (Document ingestion)
  ├── /api/v1/analyze (Analysis endpoints)
  └── /api/v1/data    (Data pool management)
```

**Health Check**: `/health` endpoint for service monitoring

---

## 2. Configuration System

### 2.1 Settings Management

**File**: [`app/config.py`](backend/app/config.py)

**Technology**: Pydantic Settings with environment variable support

**Key Configurations**:

```python
- APP_NAME: "Hybrid Integrity System"
- PLAGIARISM_MODEL_NAME: "all-mpnet-base-v2" (Sentence-BERT)
- AI_DETECTOR_MODEL_NAME: "PirateXX/AI-Content-Detector" (RoBERTa)
- DEVICE: Auto-detection (CUDA if available, else CPU)
```

**Dynamic Device Selection**: Automatically uses GPU acceleration if available, otherwise falls back to CPU.

---

## 3. Data Models & Schema

### 3.1 Pydantic Models

**File**: [`app/models.py`](backend/app/models.py)

**Purpose**: Type-safe request/response schemas for API validation

#### Key Models:

1. **IngestResponse**

   - Returned after document ingestion
   - Fields: `id`, `message`, `chunk_count`

2. **SentenceScore**

   - Sentence-level AI detection result
   - Fields: `text`, `index`, `start_char`, `end_char`, `ai_probability`, `is_ai_generated`, `perplexity`, `window_count`

3. **StylemetryReport**

   - AI phrase analysis results
   - Fields: `total_ai_phrases`, `unique_phrases`, `phrase_density`, `category_breakdown`, `top_phrases`, `stylometry_score`

4. **AnalysisSegment**

   - Individual text segment analysis
   - Fields: `text`, `start_index`, `end_index`, `ai_probability`, `plagiarism_score`, `is_plagiarized`, `source_metadata`

5. **AnalysisMetrics**

   - Aggregated detection metrics
   - Includes: AI scores, plagiarism scores, stylometry scores, statistical metrics, SuperAnnotate scores

6. **AnalysisReport** (Main Output)
   - Complete analysis with separate scores
   - Separate scores for: AI detection, plagiarism, stylometry
   - Legacy field: `overall_integrity_score` (backward compatibility)

---

## 4. API Endpoints (Routers)

### 4.1 Ingest Router

**File**: [`app/routers/ingest.py`](backend/app/routers/ingest.py)

**Endpoint**: `POST /api/v1/ingest`

**Purpose**: Ingests reference documents into the plagiarism database

**Process Flow**:

```
1. Upload file (PDF/DOCX/TXT)
2. Extract text using text_extract utility
3. Split into sentences using spaCy
4. Generate embeddings using SBERT (all-mpnet-base-v2)
5. Store in FAISS vector index
6. Save metadata (filename, chunk_index, text_snippet, positions)
7. Persist to disk (vector_store/)
```

**Technology Used**:

- Sentence-level chunking (spaCy for precise segmentation)
- SBERT embeddings for semantic representation
- FAISS for efficient similarity search

---

### 4.2 Analysis Router

**File**: [`app/routers/analyze.py`](backend/app/routers/analyze.py)

**Endpoints**:

- `POST /api/v1/analyze` (file upload)
- `POST /api/v1/analyze/text` (direct text input)

**Purpose**: Analyzes documents for AI generation and plagiarism

#### 4.2.1 Analysis Pipeline (Unified v2.0)

The analysis pipeline combines **4 independent detection methods**:

##### **Method 1: Sentence-Based AI Detection (RoBERTa)**

```
Process:
1. Split text into sentences using spaCy
2. Create sliding windows (4 sentences per window, stride=1)
3. Score each window with RoBERTa classifier
4. Aggregate window scores to sentence-level
5. Calculate local perplexity for each sentence
```

**Model**: `PirateXX/AI-Content-Detector` (Fine-tuned RoBERTa)

- Output: 0.0 (Human) to 1.0 (AI)
- Threshold: 0.5 for binary classification

##### **Method 2: Stylometric Analysis (N-gram Phrase Matching)**

```
Process:
1. Load AI phrase patterns from ai_phrases.json
2. Search text for characteristic AI phrases
3. Count occurrences by category (transitions, verbs, adjectives, etc.)
4. Calculate phrase density (phrases per 100 words)
5. Generate stylometry score (0-1)
```

**Categories**:

- Transitions: "In conclusion", "Furthermore", "Moreover"
- Abstract nouns: "tapestry", "landscape", "realm", "paradigm"
- Verbs: "delve", "underscore", "leverage", "harness"
- Adjectives: "crucial", "pivotal", "seamless"
- Filler phrases: "It is essential to", "plays a vital role"

**Scoring**:

- Density threshold: >2% suspicious, >5% highly suspicious
- Considers both density and diversity of phrases

##### **Method 3: SuperAnnotate AI Detector (RoBERTa Large)**

```
Process:
1. Separate code blocks from text (code gets default 0.5 score)
2. Preprocess text (whitespace normalization)
3. Split into chunks (max 512 tokens)
4. Run inference with fine-tuned RoBERTa Large
5. Weighted average by chunk length
```

**Model**: `SuperAnnotate/ai-detector` (RoBERTa Large ~1.5GB)

- Fine-tuned specifically for AI detection
- Uses custom classifier architecture
- GPU-optimized with FP16 support
- Lazy loading to save memory

**Score Boosting**:

- Conservative scores (0.7-0.8) are boosted to 0.9-1.0
- Recognizes SuperAnnotate's calibration characteristics

##### **Method 4: Plagiarism Detection (Bi-Encoder + Cross-Encoder)**

```
Process:
1. Retrieve Stage: SBERT embeddings + FAISS search (Top-5)
2. Re-Rank Stage: Cross-Encoder (stsb-roberta-large) for paraphrase detection
3. Threshold: >0.80 for plagiarism flag
4. Track source metadata and match positions
```

**Two-Stage Approach**:

- **Stage 1 (Retrieve)**: Fast vector similarity search

  - Model: `all-mpnet-base-v2` (SBERT)
  - Index: FAISS (Inner Product similarity)
  - Top-K: 5 candidates per sentence

- **Stage 2 (Re-Rank)**: Precise paraphrase scoring
  - Model: `cross-encoder/stsb-roberta-large`
  - Batch inference for efficiency
  - Handles paraphrased content

**Persistence**:

- Vector store saved to disk (`vector_store/`)
- Survives server restarts
- Metadata includes: source filename, chunk indices, text snippets, character positions

---

#### 4.2.2 Statistical Metrics

**Calculated by**: [`app/engine/detector.py`](backend/app/engine/detector.py)

##### **Perplexity** (GPT-2)

- **Purpose**: Measures text predictability
- **Model**: GPT-2 language model
- **Interpretation**: Lower = more predictable = potentially AI
- **Reliability**: Partial (see statistical_metrics_report.md)
- **Threshold**: <12 strongly suggests AI (adjusted from <30)

##### **Burstiness** (Sentence Length Variation)

- **Purpose**: Measures variation in sentence structure
- **Calculation**: Coefficient of Variation (StdDev / Mean)
- **Interpretation**: Lower = uniform = potentially AI
- **Reliability**: High (strong signal)
- **Thresholds**:
  - <0.3: Strong AI signal
  - <0.5: Moderate AI signal
  - > 0.5: Human-like variation

##### **Perplexity Flux** (Variance in Predictability)

- **Purpose**: Measures consistency of text predictability
- **Calculation**: Coefficient of variation across sentence perplexities
- **Interpretation**:
  - Low flux (<0.3): Consistently predictable = AI
  - High flux (>0.5): Erratic predictability = Human
- **Reliability**: High (strongest statistical signal)
- **Why it works**: AI maintains consistent predictability; humans have spikes

---

#### 4.2.3 Ensemble Scoring Strategy

**Final AI Score Calculation**:

```python
# Weighted average (baseline)
weighted_score = (
    model_score * 0.40 +           # RoBERTa sentence scores
    statistical_penalty * 0.25 +   # Burstiness + Flux
    stylometry_score * 0.35 +      # Phrase matching
    superannotate_score * 0.40     # Specialized detector
)

# Use MAXIMUM of all signals
final_ai_score = max(
    weighted_score,
    stylometry_score,      # Trust strong phrase saturation (1.0)
    model_score,           # Trust strong model confidence (1.0)
    boosted_sa_score       # Trust specialized detector
)
```

**Rationale for MAX strategy**:

- If ANY strong signal is detected, report it
- Prevents averaging out clear signals
- Example: Stylometry at 1.0 should not be diluted by lower scores

---

#### 4.2.4 Sentence Score Contextual Boosting

**Problem**: Individual sentences appear "human" but overall document is clearly AI

**Solution**: Blend local sentence scores with global AI pattern

```python
if document_ai_score > 0.4:
    for sentence in sentences:
        # Blend: Local 20%, Global 80%
        sentence.score = (local_score * 0.2) + (global_ai_score * 0.8)
```

**Why**: SuperAnnotate sees full document patterns that RoBERTa misses on short sentences

---

### 4.3 Data Pool Router

**File**: [`app/routers/data.py`](backend/app/routers/data.py)

**Endpoints**:

- `GET /api/v1/documents` - List all indexed documents
- `GET /api/v1/documents/{filename}/chunks` - Get chunks for specific document
- `DELETE /api/v1/documents/clear` - Clear entire data pool
- `DELETE /api/v1/documents/{filename}` - Delete specific document

**Purpose**: Manage the plagiarism reference database

**Functionality**:

- Aggregates chunks by source filename
- Provides metadata (chunk counts, timestamps)
- Soft deletion (removes from metadata, not from FAISS index)

---

## 5. Detection Engines

### 5.1 AI Detector Engine

**File**: [`app/engine/detector.py`](backend/app/engine/detector.py)

**Models Loaded**:

1. **RoBERTa Classifier**: `PirateXX/AI-Content-Detector`

   - Sequence classification
   - Label_0 = Fake (AI), Label_1 = Real (Human)
   - Batch processing support (batch_size=8)

2. **GPT-2 Language Model**: For perplexity calculation
   - Causal language model
   - Sliding window approach for long texts
   - Stride: 512 tokens

**Key Methods**:

- `detect_probability(texts: List[str])`: Batch AI detection
- `calculate_perplexity(text: str)`: GPT-2 perplexity
- `calculate_perplexity_flux(text: str)`: Perplexity variance
- `analyze_burstiness(text: str)`: Sentence length variation

**Optimizations**:

- Batch processing for GPU efficiency
- Max length: 512 tokens (truncation)
- Inference mode (no gradients)

---

### 5.2 Plagiarism Engine

**File**: [`app/engine/plagiarism.py`](backend/app/engine/plagiarism.py)

**Models Loaded**:

1. **Bi-Encoder (SBERT)**: `all-mpnet-base-v2`

   - Embedding dimension: 768
   - Normalized embeddings
   - Fast vector search

2. **Cross-Encoder**: `cross-encoder/stsb-roberta-large`
   - Re-ranking model
   - Paraphrase detection
   - Score range: 0.0 to 1.0

**Vector Store**:

- **Technology**: FAISS (Facebook AI Similarity Search)
- **Index Type**: `IndexFlatIP` (Inner Product similarity)
- **Persistence**: Saves to `vector_store/plagiarism.index` and `metadata.pkl`

**Key Methods**:

- `encode(texts)`: Generate SBERT embeddings
- `add_to_index(texts, metadatas)`: Ingest documents
- `search(query_texts, top_k)`: Retrieve candidates
- `compute_cross_scores(pairs)`: Re-rank with Cross-Encoder
- `delete_document(filename)`: Soft delete by source
- `clear_index()`: Reset entire database

**Persistence Strategy**:

- Auto-save after every write
- Metadata stored separately (pickle)
- Tracks `current_id` for sequential indexing

---

### 5.3 Stylometry Engine

**File**: [`app/engine/stylometry.py`](backend/app/engine/stylometry.py)

**Purpose**: Detect AI-characteristic phrases and linguistic patterns

**Phrase Database**: [`app/data/ai_phrases.json`](backend/app/data/ai_phrases.json)

- 5 categories of AI-typical phrases
- ~100+ patterns total
- Includes transitions, abstract nouns, verbs, adjectives, filler phrases

**Analysis Process**:

```python
1. Load phrase patterns from JSON
2. Search text with regex (case-insensitive)
3. Track positions and frequencies
4. Calculate density (phrases per 100 words)
5. Generate score (0-1) based on density and diversity
```

**Scoring Formula**:

```python
density_score = min(1.0, density / 5.0)      # Cap at 5%
diversity_score = min(1.0, unique_phrases / 15.0)  # Cap at 15 unique
stylometry_score = (density_score * 0.7 + diversity_score * 0.3)
```

**Output**:

- Total phrase count
- Unique phrases found
- Category breakdown
- Top N most frequent phrases
- Highlighted text (Markdown bold)

**Singleton Pattern**: Single instance (`phrase_fingerprint`) for efficiency

---

### 5.4 SuperAnnotate Detector

**File**: [`app/engine/superannotate_detector.py`](backend/app/engine/superannotate_detector.py)

**Architecture**: Custom RoBERTa Large classifier

**Model Components**:

```python
class RobertaClassifier:
    - RoBERTa base model (no pooling layer)
    - Dropout layer
    - Linear classifier (hidden_size -> 1)
```

**Features**:

- **Lazy Loading**: Only initializes when first used (saves memory)
- **Code Block Handling**: Separates code from text (assigns default 0.5 score)
- **Text Chunking**: Splits long texts into 512-token chunks
- **Weighted Averaging**: Final score weighted by chunk length
- **GPU Optimization**: FP16 (half precision) for faster inference

**Preprocessing**:

- Newline to space conversion
- Whitespace normalization
- Code block extraction (markdown triple backticks)

**Inference**:

- Model returns `(loss, logits)` tuple
- Sigmoid activation for probability
- Batch processing support

**Singleton Pattern**: Global instance with lazy initialization

---

## 6. Utility Modules

### 6.1 Text Extraction

**File**: [`app/utils/text_extract.py`](backend/app/utils/text_extract.py)

**Supported Formats**:

- **PDF**: Using `pypdf` library
- **DOCX**: Using `python-docx` library
- **TXT**: Direct UTF-8 decoding

**Process**:

```python
1. Detect file type (content-type or extension)
2. Route to appropriate extractor
3. Extract all text content
4. Return as single string
```

**Error Handling**: HTTPException with descriptive messages

---

### 6.2 Text Chunking (Legacy)

**File**: [`app/utils/chunking.py`](backend/app/utils/chunking.py)

**Algorithm**: Recursive Character Text Splitter

**Features**:

- Hierarchical splitting (paragraph -> sentence -> word -> character)
- Chunk overlap for context preservation
- Configurable chunk size and overlap

**Default Settings**:

- Chunk size: 512 characters
- Overlap: 128 characters

**Note**: Now mostly replaced by sentence-based chunking for better semantic preservation

---

### 6.3 Sentence-Based Chunking

**File**: [`app/utils/sentence_chunking.py`](backend/app/utils/sentence_chunking.py)

**Purpose**: Context-aware sentence segmentation and windowing

**Technology**: spaCy (`en_core_web_sm` model)

**Key Concepts**:

#### Sliding Window Approach

```
Why?
- AI models leave "fingerprints" in long-range coherence
- Analyzing 3-5 sentences together captures these patterns
- Overlapping windows allow sentence-level score aggregation
```

**Configuration**:

- Window size: 4 sentences
- Stride: 1 sentence (full overlap)
- Min sentence length: 10 characters

**Process**:

```python
1. Split text into sentences (spaCy)
2. Create overlapping windows (4 sentences each)
3. Track sentence indices in each window
4. Store character positions (start, end)
5. Aggregate window scores to sentence-level
```

**Data Structures**:

- `SentenceWindow`: Window with sentence indices and positions
- `SentenceWithScore`: Sentence with aggregated score and metadata

**Aggregation Strategy**:

- Each sentence appears in multiple windows
- Final score = average of all window scores
- Tracks `window_count` for transparency

---

## 7. Dependencies & Libraries

### 7.1 Core Dependencies

**File**: [`requirements.txt`](backend/requirements.txt)

```
Framework:
- fastapi>=0.115.0          # Web framework
- uvicorn[standard]>=0.30.0 # ASGI server
- python-multipart>=0.0.9   # File upload support

ML/NLP:
- torch>=2.4.0              # PyTorch (CUDA support)
- transformers>=4.44.0      # HuggingFace models
- sentence-transformers>=3.0.0  # SBERT for embeddings
- spacy (en_core_web_sm)    # Sentence segmentation

Vector Store:
- faiss-cpu>=1.7.2          # Vector similarity search

Document Processing:
- pypdf>=4.0.0              # PDF extraction
- python-docx>=1.1.0        # DOCX extraction

Utilities:
- numpy>=1.26.0             # Numerical operations
- scipy>=1.13.0             # Statistical functions
- pydantic-settings>=2.0.0  # Configuration management
```

### 7.2 Model Downloads

Models are downloaded from HuggingFace Hub on first run:

1. `all-mpnet-base-v2` (~420MB)
2. `PirateXX/AI-Content-Detector` (~500MB)
3. `gpt2` (~500MB)
4. `cross-encoder/stsb-roberta-large` (~1.4GB)
5. `SuperAnnotate/ai-detector` (~1.5GB)
6. `en_core_web_sm` (spaCy, ~12MB)

**Total**: ~4-5GB of model weights

---

## 8. Testing & Verification

### 8.1 Test Files

#### `test_api.py`

- Tests health check endpoint
- Tests document ingestion
- Tests analysis endpoint
- Basic integration testing

#### `test_statistical_metrics.py`

- Validates perplexity calculation
- Validates burstiness calculation
- Compares human vs AI text samples
- Provides threshold recommendations

#### `verify_superannotate.py`

- Tests SuperAnnotate integration
- Validates scoring for human text
- Validates scoring for AI text
- Checks response schema

#### `verify_ghostbuster.py`

- Legacy test (Ghostbuster features not currently integrated)
- Checks for Ghostbuster-specific metrics

### 8.2 Test Utilities

#### `check_import.py`

- Verifies Python import paths
- Debugs module loading issues

#### `desklib.py`

- Contains custom Desklib AI detection model
- Alternative detector implementation (not currently integrated)
- Uses custom RoBERTa architecture with mean pooling

---

## 9. Data Storage & Persistence

### 9.1 Vector Store

**Location**: `vector_store/`

**Files**:

- `plagiarism.index` - FAISS binary index file
- `metadata.pkl` - Python pickle file with document metadata

**Persistence Strategy**:

- Save after every write operation
- Load on server startup
- Survives server restarts

### 9.2 Metadata Structure

```python
{
    0: {
        "source": "document.pdf",
        "chunk_index": 0,
        "text_snippet": "First 100 chars...",
        "text": "Full sentence text",
        "start_char": 0,
        "end_char": 150
    },
    1: {...},
    ...
}
```

---

## 10. Analysis Report Structure

### 10.1 Response Schema

**Top-Level Fields**:

```python
{
    "report_id": "uuid",
    "timestamp": "ISO 8601",

    # MAIN SCORES (Separate, not averaged)
    "ai_score": 0.0-1.0,           # Higher = more AI
    "plagiarism_score": 0.0-1.0,   # Higher = more plagiarized
    "stylometry_score": 0.0-1.0,   # Higher = more AI phrases

    # Legacy (backward compatibility)
    "overall_integrity_score": 0.0-1.0,  # 1 - max(all scores)

    "metrics": {...},
    "segments": [...],
    "sentence_scores": [...],
    "stylometry": {...}
}
```

### 10.2 Metrics Object

```python
{
    # AI Detection
    "ai_score": float,
    "roberta_score": float,
    "ensemble_method": "roberta_stats",

    # Plagiarism
    "plagiarism_score": float,
    "plagiarism_percentage": float,

    # Stylometry
    "stylometry_score": float,
    "ai_phrase_count": int,
    "phrase_density": float,

    # Statistical
    "burstiness_score": float,
    "perplexity_avg": float,
    "perplexity_flux": float,

    # SuperAnnotate
    "superannotate_score": float,

    # Sentence stats
    "sentence_count": int,
    "ai_sentence_count": int
}
```

### 10.3 Segments Array

```python
[
    {
        "text": "Sentence text",
        "start_index": int,
        "end_index": int,
        "ai_probability": 0.0-1.0,
        "plagiarism_score": 0.0-1.0,
        "is_plagiarized": bool,
        "source_id": int | null,
        "source_metadata": {...} | null
    },
    ...
]
```

### 10.4 Sentence Scores Array

```python
[
    {
        "text": "Sentence text",
        "index": int,
        "start_char": int,
        "end_char": int,
        "ai_probability": 0.0-1.0,
        "is_ai_generated": bool,
        "perplexity": float | null,
        "window_count": int
    },
    ...
]
```

---

## 11. Key Design Decisions

### 11.1 Why Multiple Detection Methods?

**Rationale**: No single method is 100% accurate. Ensemble approach provides:

- Higher accuracy through cross-validation
- Explainability (multiple evidence sources)
- Robustness against adversarial attacks
- Different strengths complement each other

### 11.2 Why Sentence-Level Analysis?

**Rationale**:

- Preserves semantic boundaries (better than character chunking)
- Enables granular highlighting in UI
- Captures long-range coherence patterns
- More interpretable results for users

### 11.3 Why Two-Stage Plagiarism Detection?

**Rationale**:

- Stage 1 (SBERT): Fast retrieval from millions of documents
- Stage 2 (Cross-Encoder): Precise paraphrase detection
- Combined: Speed + Accuracy
- Handles both exact matches and paraphrased content

### 11.4 Why Separate Scores Instead of Overall Score?

**Rationale**:

- AI and plagiarism are independent issues
- Users need to understand specific problems
- Averaging dilutes strong signals
- Better for decision-making and UI display

### 11.5 Why Statistical Metrics?

**Rationale**:

- Provides evidence beyond black-box models
- Interpretable and explainable
- Catches patterns models might miss
- Burstiness and Flux are strong signals

---

## 12. Performance Considerations

### 12.1 GPU vs CPU

**Auto-Detection**: System automatically uses CUDA if available

**Performance Impact**:

- GPU: ~10-50x faster for model inference
- SuperAnnotate (RoBERTa Large) especially benefits from GPU
- FAISS search is CPU-bound (uses SIMD)

### 12.2 Batch Processing

**Implemented for**:

- RoBERTa AI detection (batch_size=8)
- Cross-Encoder re-ranking (all pairs in one call)
- SBERT embedding generation (automatic batching)

### 12.3 Memory Optimization

**Strategies**:

- Lazy loading for SuperAnnotate (loads on first use)
- FP16 precision for GPU models
- Streaming file uploads
- FAISS index in memory (fast search)

### 12.4 Bottlenecks

**Potential Slow Points**:

1. Model loading (first request after restart)
2. Long document perplexity calculation
3. Large FAISS index search (many documents)
4. Cross-Encoder inference (many pairs)

**Mitigation**:

- Keep models in memory (singleton pattern)
- Limit perplexity to first 2000-3000 chars
- Top-K limiting for FAISS (5 candidates)
- Batch inference for Cross-Encoder

---

## 13. API Usage Examples

### 13.1 Ingest Document

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@reference.pdf"
```

**Response**:

```json
{
  "id": 42,
  "message": "Successfully ingested reference.pdf",
  "chunk_count": 15
}
```

### 13.2 Analyze Document (File)

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@document.pdf"
```

### 13.3 Analyze Text (Direct)

```bash
curl -X POST "http://localhost:8000/api/v1/analyze/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here..."}'
```

### 13.4 List Documents

```bash
curl "http://localhost:8000/api/v1/documents"
```

**Response**:

```json
[
  {
    "filename": "reference1.pdf",
    "chunk_count": 15,
    "upload_timestamp": "N/A"
  },
  {
    "filename": "reference2.docx",
    "chunk_count": 23,
    "upload_timestamp": "N/A"
  }
]
```

### 13.5 Delete Document

```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/reference1.pdf"
```

### 13.6 Clear Data Pool

```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/clear"
```

---

## 14. Error Handling

### 14.1 HTTP Status Codes

- `200`: Success
- `400`: Bad request (empty file, unsupported format, invalid input)
- `404`: Resource not found
- `500`: Internal server error (model failure, processing error)

### 14.2 Exception Handling

**Patterns**:

- Try-catch blocks around model inference
- Fallback values on failure (e.g., 0.0 scores)
- Detailed error messages in HTTPException
- Stack trace printing for debugging
- Graceful degradation (if one model fails, others continue)

---

## 15. Security Considerations

### 15.1 Current Security Features

- CORS restrictions (only localhost:3000 allowed)
- File type validation
- File size limits (implicit from FastAPI)
- Input sanitization (via Pydantic)

### 15.2 Potential Security Improvements

**Not Currently Implemented**:

- Authentication/Authorization
- Rate limiting
- Input size limits (very large documents)
- API key protection
- HTTPS enforcement
- File content scanning (malware)

---

## 16. Monitoring & Observability

### 16.1 Current Monitoring

- Health check endpoint (`/health`)
- Console logging for model loading
- Error stack traces

### 16.2 Potential Improvements

**Not Currently Implemented**:

- Structured logging (JSON format)
- Request/response logging
- Performance metrics (latency, throughput)
- Error rate tracking
- Model performance monitoring

---

## 17. Scalability Considerations

### 17.1 Current Architecture

- Single-process application
- In-memory models (not shared across workers)
- Local file storage (vector store)

### 17.2 Scaling Strategies (Future)

**Horizontal Scaling**:

- Load balancer
- Multiple Uvicorn workers
- Shared vector store (Redis, PostgreSQL with pgvector)
- Model serving infrastructure (TorchServe, Triton)

**Vertical Scaling**:

- Larger GPU for faster inference
- More RAM for larger vector stores
- SSD for faster model loading

**Optimization**:

- Model quantization (INT8)
- ONNX runtime
- TensorRT compilation
- Caching frequent requests

---

## 18. Integration Points

### 18.1 Frontend Integration

**Expected Frontend**: React/Next.js app at localhost:3000

**Communication**: REST API over HTTP

**File Upload**: FormData multipart/form-data

**JSON Responses**: Structured Pydantic schemas

### 18.2 External Services

**HuggingFace Hub**: Model downloads

- Automatic on first run
- Cached locally after download

**spaCy**: Language model download

- `en_core_web_sm` model
- Auto-download with subprocess if missing

---

## 19. Documentation Files

### 19.1 README.md

- Setup instructions
- API endpoint documentation
- Architecture overview
- Dependencies list

### 19.2 statistical_metrics_report.md

- Detailed analysis of statistical metrics
- Reliability assessment
- Threshold recommendations
- Known limitations

---

## 20. Future Enhancements (Based on Code Comments)

### 20.1 Potential Additions

1. **Timestamp Tracking**: Store upload times in metadata
2. **Background Processing**: Async ingestion for large documents
3. **Advanced FAISS**: GPU index for massive datasets
4. **Improved Perplexity**: Use larger language models (GPT-Neo, Llama)
5. **Enhanced Stylometry**: Machine learning-based phrase detection
6. **Multi-Language Support**: Additional spaCy models
7. **Caching**: Redis for frequent queries
8. **Webhooks**: Notify clients when analysis completes

### 20.2 Alternative Models (Mentioned in Code)

1. **Desklib AI Detector**: Custom implementation in `desklib.py`
2. **Binoculars**: Falcon-7B based detector (referenced in comments)
3. **Ghostbuster**: Probability distribution analysis (legacy test exists)

---

## 21. Critical Code Paths

### 21.1 Analysis Pipeline Flow

```
1. User uploads file OR submits text
   ↓
2. Extract text (if file)
   ↓
3. Split into sentences (spaCy)
   ↓
4. [PARALLEL PROCESSING]
   ├─→ RoBERTa sentence windows → Aggregate
   ├─→ Stylometry phrase matching
   ├─→ SuperAnnotate full text
   └─→ Plagiarism (FAISS + Cross-Encoder)
   ↓
5. Calculate statistical metrics (Perplexity, Burstiness, Flux)
   ↓
6. Ensemble scoring (weighted + MAX strategy)
   ↓
7. Contextual sentence boosting
   ↓
8. Build AnalysisReport
   ↓
9. Return JSON response
```

### 21.2 Ingestion Flow

```
1. User uploads reference document
   ↓
2. Extract text
   ↓
3. Split into sentences (spaCy)
   ↓
4. Generate SBERT embeddings
   ↓
5. Add to FAISS index
   ↓
6. Save metadata
   ↓
7. Persist to disk
   ↓
8. Return success response
```

---

## 22. Key Algorithms & Formulas

### 22.1 Stylometry Score

```python
density_score = min(1.0, phrase_density / 5.0)
diversity_score = min(1.0, unique_phrases / 15.0)
stylometry_score = (density_score * 0.7) + (diversity_score * 0.3)
```

### 22.2 Burstiness

```python
sentence_lengths = [len(s.split()) for s in sentences]
mean = np.mean(sentence_lengths)
std_dev = np.std(sentence_lengths)
burstiness = std_dev / mean  # Coefficient of Variation
```

### 22.3 Perplexity Flux

```python
sentence_perplexities = [calculate_perplexity(s) for s in sentences]
mean_ppl = np.mean(sentence_perplexities)
std_ppl = np.std(sentence_perplexities)
flux = std_ppl / mean_ppl  # Coefficient of Variation
```

### 22.4 Statistical Penalty

```python
# Burstiness penalty
if burstiness < 0.3:
    burstiness_penalty = 0.8
elif burstiness < 0.5:
    burstiness_penalty = 0.4
else:
    burstiness_penalty = 0.0

# Flux penalty
if perplexity_flux < 0.2:
    flux_penalty = 0.9
elif perplexity_flux < 0.35:
    flux_penalty = 0.5
else:
    flux_penalty = 0.0

statistical_penalty = (burstiness_penalty * 0.4) + (flux_penalty * 0.6)
```

### 22.5 Ensemble AI Score

```python
weighted_score = (
    model_score * 0.40 +
    statistical_penalty * 0.25 +
    stylometry_score * 0.35 +
    superannotate_score * 0.40
)
weighted_score = min(1.0, weighted_score)

final_ai_score = max(
    weighted_score,
    stylometry_score,
    model_score,
    boosted_sa_score
)
```

### 22.6 SuperAnnotate Score Boosting

```python
if superannotate_score > 0.7:
    boosted = 0.9 + ((score - 0.7) * 0.33)  # Map 0.7→0.9, 1.0→1.0
elif superannotate_score > 0.5:
    boosted = 0.6 + ((score - 0.5) * 1.5)   # Map 0.5→0.6, 0.7→0.9
else:
    boosted = superannotate_score
```

### 22.7 Sentence Contextual Boosting

```python
if document_ai_score > 0.4:
    for sentence in sentences:
        boosted_score = (local_score * 0.2) + (document_ai_score * 0.8)
        sentence.ai_probability = boosted_score
        sentence.is_ai_generated = boosted_score > 0.5
```

---

## 23. Configuration & Environment

### 23.1 Environment Variables

**Supported** (via Pydantic Settings):

- `APP_NAME`: Application name
- `API_V1_STR`: API prefix
- `PLAGIARISM_MODEL_NAME`: SBERT model
- `AI_DETECTOR_MODEL_NAME`: RoBERTa model

**Note**: `.env` file support enabled but not required (defaults work)

### 23.2 Model Configuration

**Paths**: Models stored in HuggingFace cache (`~/.cache/huggingface/`)

**Customization**: Change model names in `config.py` to use different models

---

## 24. Known Limitations & Issues

### 24.1 Technical Limitations

1. **FAISS Soft Delete**: Deleted documents remain in index (metadata removed only)
2. **No Timestamps**: Upload times not tracked (marked as "N/A")
3. **Perplexity Unreliability**: GPT-2 flags too much as AI (see report)
4. **Memory Usage**: All models loaded in memory (~6-8GB RAM + GPU VRAM)
5. **Single-Process**: No worker parallelization

### 24.2 Model Limitations

1. **Context Length**: Models limited to 512 tokens (longer texts truncated)
2. **Language Support**: English only (spaCy model)
3. **Domain Specificity**: Models trained on general text (may not work well for specialized domains)
4. **False Positives**: Formal/Wikipedia-style writing may trigger AI detection

### 24.3 Edge Cases

1. **Very Short Texts**: Statistical metrics unreliable (<50 words)
2. **Code-Heavy Documents**: May not extract cleanly
3. **Scanned PDFs**: No OCR support (text must be embedded)
4. **Non-English Text**: Will produce incorrect results

---

## 25. Testing Coverage

### 25.1 Tested Scenarios

- Health check
- Document ingestion (TXT format)
- Analysis endpoint (file and text)
- Statistical metrics (human vs AI samples)
- SuperAnnotate integration

### 25.2 Untested Scenarios

- PDF/DOCX extraction edge cases
- Large documents (>10k words)
- Concurrent requests
- Document deletion
- Data pool management endpoints
- Error recovery
- Model failure scenarios

---

## 26. Deployment Considerations

### 26.1 System Requirements

**Minimum**:

- Python 3.10+
- 8GB RAM
- 10GB disk space (for models)
- CPU with AVX support (for PyTorch)

**Recommended**:

- Python 3.10+
- 16GB RAM
- NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3070, T4, V100)
- 20GB disk space
- Fast SSD

### 26.2 Running the Server

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

**Note**: Multiple workers not recommended (models not shared, high memory usage)

---

## 27. Maintenance & Operations

### 27.1 Regular Maintenance Tasks

1. Clear data pool periodically (if not needed)
2. Monitor disk space (vector store growth)
3. Update models (HuggingFace releases)
4. Review detection thresholds (calibration)

### 27.2 Troubleshooting

**Common Issues**:

1. **Model download fails**: Check internet connection, HuggingFace status
2. **CUDA out of memory**: Reduce batch size or use CPU
3. **Slow inference**: Check GPU utilization, consider FP16
4. **Empty results**: Check file extraction, ensure text content exists
5. **Import errors**: Verify Python path, check dependencies

---

## 28. Code Quality & Style

### 28.1 Code Organization

- **Modular**: Clear separation of concerns (routers, engines, utils)
- **Type Hints**: Extensive use of Python type annotations
- **Documentation**: Docstrings for complex functions
- **Comments**: Inline comments explaining logic

### 28.2 Design Patterns

- **Singleton**: Used for engines (ai_detector, plagiarism_engine, etc.)
- **Factory**: Model loading with lazy initialization
- **Strategy**: Multiple detection methods combined via ensemble
- **Repository**: Plagiarism engine abstracts storage

### 28.3 Best Practices Observed

- Pydantic for validation
- Exception handling with context
- Batch processing for efficiency
- Resource cleanup (model.eval(), torch.no_grad())
- Configuration via environment variables

---

## 29. Conclusion

This backend is a **production-ready, research-grade AI detection and plagiarism analysis system** with the following strengths:

### 29.1 Strengths

1. **Multi-Method Approach**: Combines 4+ detection methods for robustness
2. **State-of-the-Art Models**: Uses latest HuggingFace models (RoBERTa, SBERT, Cross-Encoders)
3. **Explainable**: Provides separate scores and evidence (phrases, statistics, segments)
4. **Efficient**: Batch processing, GPU acceleration, FAISS indexing
5. **Persistent**: Vector store survives restarts
6. **Well-Structured**: Clean code, modular architecture, type-safe

### 29.2 Unique Features

1. **Sentence-Level Analysis**: Granular highlighting capability
2. **Perplexity Flux**: Novel metric for consistency detection
3. **SuperAnnotate Integration**: Specialized AI detector with score boosting
4. **Contextual Sentence Boosting**: Aligns local and global signals
5. **Two-Stage Plagiarism**: Fast retrieval + precise re-ranking

### 29.3 Use Cases

- **Academic Integrity**: Detect student plagiarism and AI usage
- **Content Moderation**: Verify authenticity of submissions
- **Journalism**: Check article integrity
- **Legal Discovery**: Analyze document originality
- **Research**: Study AI-generated text characteristics

---

## 30. References & Resources

### 30.1 Models

- **SBERT**: https://www.sbert.net/
- **FAISS**: https://github.com/facebookresearch/faiss
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
- **spaCy**: https://spacy.io/

### 30.2 Papers

- Sentence-BERT (Reimers & Gurevych, 2019)
- Cross-Encoders for Semantic Textual Similarity
- Detecting AI-Generated Text (various approaches)

### 30.3 Related Projects

- SuperAnnotate AI Detector: https://github.com/superannotateai/generated_text_detector
- Desklib AI Detector: https://huggingface.co/desklib
- GPTZero, Originality.ai, Copyleaks (commercial alternatives)

---

## Appendix A: File-by-File Summary

| File                                   | Lines | Purpose                   | Key Technologies              |
| -------------------------------------- | ----- | ------------------------- | ----------------------------- |
| `main.py`                              | 28    | FastAPI app entry point   | FastAPI, CORS                 |
| `app/config.py`                        | 18    | Configuration management  | Pydantic Settings             |
| `app/models.py`                        | 150   | Request/response schemas  | Pydantic models               |
| `app/routers/analyze.py`               | 414   | Analysis endpoints        | RoBERTa, SBERT, SuperAnnotate |
| `app/routers/ingest.py`                | 42    | Document ingestion        | SBERT, FAISS, spaCy           |
| `app/routers/data.py`                  | 62    | Data pool management      | FAISS metadata                |
| `app/engine/detector.py`               | 156   | AI detection engine       | RoBERTa, GPT-2, PyTorch       |
| `app/engine/plagiarism.py`             | 144   | Plagiarism engine         | SBERT, FAISS, Cross-Encoder   |
| `app/engine/stylometry.py`             | 213   | Phrase fingerprinting     | Regex, JSON                   |
| `app/engine/superannotate_detector.py` | 209   | SuperAnnotate integration | RoBERTa Large, PyTorch        |
| `app/utils/text_extract.py`            | 46    | File parsing              | pypdf, python-docx            |
| `app/utils/chunking.py`                | 92    | Legacy text chunking      | Recursive splitting           |
| `app/utils/sentence_chunking.py`       | 164   | Sentence windowing        | spaCy                         |
| `app/data/ai_phrases.json`             | 107   | AI phrase database        | JSON                          |
| `test_api.py`                          | 43    | Integration tests         | requests                      |
| `test_statistical_metrics.py`          | 98    | Statistical validation    | asyncio                       |
| `verify_superannotate.py`              | 77    | SuperAnnotate tests       | requests                      |
| `check_import.py`                      | 13    | Import verification       | sys                           |
| `desklib.py`                           | 88    | Alternative detector      | Custom RoBERTa                |

---

## Appendix B: API Endpoint Reference Card

| Method | Endpoint                              | Description                | Input                  | Output                      |
| ------ | ------------------------------------- | -------------------------- | ---------------------- | --------------------------- |
| GET    | `/health`                             | Service health check       | None                   | `{"status": "ok"}`          |
| GET    | `/`                                   | API welcome message        | None                   | `{"message": "Welcome..."}` |
| POST   | `/api/v1/ingest`                      | Ingest reference document  | File (PDF/DOCX/TXT)    | `IngestResponse`            |
| GET    | `/api/v1/documents`                   | List all indexed documents | None                   | Array of documents          |
| GET    | `/api/v1/documents/{filename}/chunks` | Get chunks for document    | filename (path)        | Array of chunks             |
| DELETE | `/api/v1/documents/clear`             | Clear entire data pool     | None                   | Success message             |
| DELETE | `/api/v1/documents/{filename}`        | Delete specific document   | filename (path)        | Success message             |
| POST   | `/api/v1/analyze`                     | Analyze document file      | File (PDF/DOCX/TXT)    | `AnalysisReport`            |
| POST   | `/api/v1/analyze/text`                | Analyze raw text           | JSON `{"text": "..."}` | `AnalysisReport`            |

---

## Appendix C: Score Interpretation Guide

### AI Score (0.0 - 1.0)

- **0.0 - 0.3**: Likely human-written
- **0.3 - 0.5**: Uncertain, mixed signals
- **0.5 - 0.7**: Likely AI-generated
- **0.7 - 1.0**: Very likely AI-generated

### Plagiarism Score (0.0 - 1.0)

- **0.0 - 0.2**: Minimal overlap
- **0.2 - 0.5**: Some similarities (possibly common phrases)
- **0.5 - 0.8**: Significant overlap (paraphrasing suspected)
- **0.8 - 1.0**: High overlap (plagiarism likely)

### Stylometry Score (0.0 - 1.0)

- **0.0 - 0.2**: Few AI-characteristic phrases
- **0.2 - 0.5**: Moderate AI phrase usage
- **0.5 - 0.8**: High AI phrase density
- **0.8 - 1.0**: Saturated with AI phrases

### Burstiness

- **< 0.3**: Uniform sentence structure (AI-like)
- **0.3 - 0.5**: Moderate variation
- **> 0.5**: High variation (human-like)

### Perplexity Flux

- **< 0.2**: Robotically consistent (AI-like)
- **0.2 - 0.35**: Suspiciously consistent
- **> 0.35**: Human-like variation

---

**End of Report**
