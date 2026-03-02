# Construction Arbitration Document Intelligence Pipeline — Architecture Plan

## Pipeline Diagram

```
Phase 1: Index and Extract (sequential)
    └── ocr_text.txt per segment → LLM → index.json

Phase 2: Document Type Clustering (parallel with Phase 3)
    └── index.json → LLM taxonomy → type_taxonomy.json
    └── index.json → LLM classify → type_classifications.json

Phase 3: Claim Head Discovery (parallel with Phase 2)
    └── index.json → LLM discover → claim_heads.json
    └── index.json → LLM map → claim_classifications.json

Phase 4: Case Analysis (sequential, depends on all)
    └── all outputs → LLM → case_analysis.md
```

## Agent Design Per Phase

### Phase 1: Index and Extract
- **Input**: `ocr_text.txt` content + segment ID (folder name)
- **Prompt**: Extract document type, purpose, parties, dates, reference numbers, amounts, summary. Handle OCR noise gracefully.
- **Output schema**: `{ segment_id, document_type, purpose, parties[], dates[], reference_numbers[], monetary_amounts[], summary }`
- **Model**: Gemini 1.5 Flash (cost-effective for extraction)

### Phase 2: Document Type Clustering
- **Input**: Full index.json
- **Step A**: LLM proposes hierarchical taxonomy from actual document types in index
- **Step B**: Classify each segment into 1–3 clusters with confidence, primary cluster
- **Output**: type_taxonomy.json, type_classifications.json
- **Model**: Gemini 1.5 Pro (reasoning for taxonomy)

### Phase 3: Claim Head Discovery
- **Input**: Full index.json
- **Step A**: LLM discovers claim heads from documents (no hardcoded list)
- **Step B**: Map each segment to claim heads with relevance type, role, confidence, reasoning
- **Output**: claim_heads.json, claim_classifications.json
- **Model**: Gemini 1.5 Pro (reasoning)

### Phase 4: Case Analysis
- **Input**: All prior outputs
- **Prompt**: Produce lawyer-ready narrative: case overview, document landscape, claim analysis, coverage map, timeline, key observations
- **Output**: case_analysis.md
- **Model**: Gemini 1.5 Pro (synthesis)

## State Management

- Each phase writes to `output/` directory
- Before running a phase: check if output file(s) exist
- If exist: log "Phase X: already completed, skipping" and continue
- No database; file-based checkpoints only
- Resume: re-run pipeline; completed phases skip automatically

## LLM Choices

| Phase | Model | Rationale |
|-------|-------|-----------|
| 1 | Gemini 2.5 Flash | Fast, cheap for structured extraction |
| 2, 3, 4 | Gemini 2.5 Pro | Better reasoning for taxonomy, claim discovery, synthesis |

## Segment Selection (50 segments)

Representative mix across categories:

| Category | Count | Segments |
|----------|-------|----------|
| Contract foundation | 6 | Agreement, Contract_Agreement, Work_Order (3), Supplementary_Work_Order |
| Payment chain | 8 | RA_Bill submission, Invoice (escalation), Certificate_Final_Bill, Bank_Letter, Claim_Abstract, Payment records |
| Correspondence | 12 | Letters: Delay, Site_Issues, Hindrance, Extension, Escalation, Payment, etc. |
| Dispute escalation | 8 | Demand_Notice, Legal_Notice, Claim_Letter, Notice (delay, COVID), Petition |
| Annexures/specs | 10 | Claim_Dispute, Billing_Details, Milestones, Construction_Work_Bills, etc. |
| Arbitration/legal | 6 | Board_Resolution, Affidavit, Court_Filing, Statement_of_Truth, Non-Starter_Report |

Selected a variety of files from the different categories identified from the filenames.