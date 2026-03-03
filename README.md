# Construction Arbitration Document Intelligence Pipeline

A multi-agent pipeline that processes construction arbitration document segments: it extracts metadata, clusters by document type and claim heads, and produces a lawyer-ready case analysis.

**Domain:** URC Construction (contractor) vs ISRO/Government (employer) dispute over the AITF-1 extension at ISITE Bangalore.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

## Run

```bash
python main.py
```

The pipeline runs four phases. Completed phases are skipped on re-run (checkpoint/resume).

## Pipeline Phases

| Phase | Output                                                          | Description                                                                          |
| ----- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 1     | `output/index.json`                                             | Extract metadata (type, purpose, parties, dates, amounts, summary) from each segment |
| 2     | `output/type_taxonomy.json`, `output/type_classifications.json` | LLM proposes document type taxonomy and classifies each segment                      |
| 3     | `output/claim_heads.json`, `output/claim_classifications.json`  | Discover claim heads from documents; map segments with relevance and reasoning       |
| 4     | `output/case_analysis.md`                                       | Produce a narrative case analysis for lawyers                                        |

Phases 2 and 3 can be run in parallel (both depend only on Phase 1).

## Configuration

- **Segment selection:** Edit `selected_segments.json` to change which segments (folder names in `EvaluationSegments/`) are processed.
- **Paths:** `config.py` defines `SEGMENTS_DIR`, `OUTPUT_DIR`, and `SEGMENTS_FILE`.

## Project Structure

```
├── main.py                 # Entry point
├── config.py               # Paths and config
├── selected_segments.json  # Which segments to process
├── src/
│   ├── extraction.py       # Phase 1
│   ├── type_clustering.py  # Phase 2
│   ├── claim_discovery.py  # Phase 3
│   ├── case_analysis.py    # Phase 4
│   └── models.py           # Pydantic schemas
├── EvaluationSegments/     # Document segments (ocr_text.txt per folder)
└── output/                 # All pipeline outputs
```

## Requirements

- Python 3.10+
- Gemini API key (2.0 Flash for extraction; 2.5 Pro for reasoning)

## Future Scope and Architectural Decisions

### Scaling

At a scale larger than the 50 segments we built currently, we should go for an indexed database with chunked pieces of information from the segments as the context window could get overflown with a large index. Also have retrieval strategies - top K, cosine, etc. because passing the full index would not be feasible anymore.

### Parallel Execution

Yes, Phases 2 and 3 were run in parallel using the ThreadPoolExecutor function available in the basic concurrent module. It can be also be executed parallely using asyncio which requires to be installed.

### Serverless Architecture

In a serverless infrastructure, we convert each phase to be trigerred by an HTTP call, orchestrator fires Phase 1 call, then Phase 2 and 3 with the response from first. It would be either two HTTP calls or pub-sub messages as we need to wait for the responses.
The storage moves to S3 or any similar persistant storage.
Processes can be fanned out and parallelized on a segment-basis, queues can be used for waiting and delivering messages, etc.

### Handling Messy OCR

To handle the OCR text, some explicit instruction were provided with regards to the limitations of OCR text - mixed languages, broken tables, etc. The documents were also truncated to keep the context and token limit to not exceed the threshold.

The LLM clearly struggled with explicitly retruning answers for financial information, dates, other languages being skipped and context being missed from there, etc.

### AI tools usage

Used Cursor for the implementation of each phase of the system based on architecture and specified requirements.
