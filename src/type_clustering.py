"""Phase 2: Document Type Clustering — LLM-driven taxonomy and per-segment classification."""

import json
import os
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

from .models import (
    SegmentMetadata,
    TaxonomyNode,
    TypeClassification,
)

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

TAXONOMY_PROMPT = """You are analyzing a construction arbitration document corpus. Below is an index of document segments with their extracted metadata (document_type, purpose, summary).

Examine the actual document types and purposes present. Propose a hierarchical taxonomy of document types that makes sense for THIS specific case. Do not use generic categories — ground the taxonomy in what you observe.

Requirements:
- Create a tree: top-level categories with optional nested subcategories
- Use keys (lowercase, underscored) and human-readable names
- Each category should have a brief description of what documents fit there

Return a single JSON object:
{{
  "taxonomy": [
    {{
      "key": "contract_foundation",
      "name": "Contract & Foundation Documents",
      "description": "...",
      "children": [
        {{ "key": "work_orders", "name": "Work Orders", "description": "...", "children": [] }}
      ]
    }}
  ]
}}

Index of documents:
---
{index_summary}
---
"""


CLASSIFY_PROMPT = """Given this document type taxonomy and a list of document segments, classify each segment into 1–3 taxonomy categories.

For each segment:
- Assign 1–3 cluster keys from the taxonomy (use the "key" field from taxonomy nodes, including nested)
- Provide a confidence score between 0 and 1 for each assignment
- Mark exactly ONE assignment as primary (is_primary: true) which should be the best fit
- Include cluster_name for each

Return a single JSON object:
{{
  "classifications": [
    {{
      "segment_id": "...",
      "clusters": [
        {{ "cluster_key": "...", "cluster_name": "...", "confidence": 0.9, "is_primary": true }},
        {{ "cluster_key": "...", "cluster_name": "...", "confidence": 0.6, "is_primary": false }}
      ]
    }}
  ]
}}

Taxonomy:
---
{taxonomy_json}
---

Documents to classify (segment_id, document_type, purpose, summary):
---
{docs_batch}
---
"""


def _index_to_summary(index: list[SegmentMetadata]) -> str:
    lines = []
    for m in index:
        lines.append(
            f"[{m.segment_id}]\n"
            f"  type: {m.document_type}\n"
            f"  purpose: {m.purpose}\n"
            f"  summary: {m.summary}"
        )
    return "\n\n".join(lines)


def _generate_taxonomy(
    index: list[SegmentMetadata], model_name: str = "gemini-2.5-pro"
) -> list[TaxonomyNode]:
    """Step A: LLM proposes hierarchical taxonomy from the index."""
    summary = _index_to_summary(index)
    max_len = 25000
    if len(summary) > max_len:
        summary = summary[:max_len] + "\n\n[... truncated ...]"

    prompt = TAXONOMY_PROMPT.format(index_summary=summary)
    model = genai.GenerativeModel(model_name)

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
        taxonomy = data.get("taxonomy", data) if isinstance(data, dict) else data
        if not isinstance(taxonomy, list):
            taxonomy = [taxonomy]
        return [TaxonomyNode.model_validate(n) for n in taxonomy]
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Failed to parse taxonomy response: {e}") from e


def _classify_batch(
    segments: list[SegmentMetadata],
    taxonomy: list[TaxonomyNode],
    model_name: str = "gemini-2.5-pro",
) -> list[TypeClassification]:
    """Classify a batch of segments into taxonomy clusters."""
    taxonomy_json = json.dumps(
        [t.model_dump() for t in taxonomy], indent=2, ensure_ascii=False
    )
    docs_batch = "\n\n".join(
        f"[{m.segment_id}]\ntype: {m.document_type}\npurpose: {m.purpose}\nsummary: {m.summary[:250]}"
        for m in segments
    )

    prompt = CLASSIFY_PROMPT.format(
        taxonomy_json=taxonomy_json,
        docs_batch=docs_batch,
    )
    model = genai.GenerativeModel(model_name)

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
        classifications = data.get("classifications", data)
        if not isinstance(classifications, list):
            classifications = [classifications]
        result = []
        for c in classifications:
            if isinstance(c, dict):
                clusters = c.get("clusters", [])
                if clusters and not any(cl.get("is_primary") for cl in clusters):
                    clusters[0]["is_primary"] = True
                result.append(TypeClassification.model_validate(c))
            else:
                result.append(TypeClassification.model_validate(c))
        return result
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Failed to parse classification response: {e}") from e


def run_type_clustering(
    output_dir: Path,
    index: list[SegmentMetadata] | None = None,
    index_path: Path | None = None,
    model_name: str = "gemini-2.5-pro",
    batch_size: int = 10,
) -> tuple[list[TaxonomyNode], list[TypeClassification]]:

    output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy_path = output_dir / "type_taxonomy.json"
    classifications_path = output_dir / "type_classifications.json"

    if taxonomy_path.exists() and classifications_path.exists():
        print("Phase 2: already completed, skipping")
        taxonomy_data = json.loads(taxonomy_path.read_text())
        nodes = (
            taxonomy_data
            if isinstance(taxonomy_data, list)
            else taxonomy_data.get("taxonomy", [taxonomy_data])
        )
        taxonomy = [TaxonomyNode.model_validate(n) for n in nodes]
        classifications_data = json.loads(classifications_path.read_text())
        cl_list = (
            classifications_data
            if isinstance(classifications_data, list)
            else classifications_data.get("classifications", [])
        )
        classifications = [TypeClassification.model_validate(c) for c in cl_list]
        return taxonomy, classifications

    if index is None:
        if index_path is None:
            index_path = output_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"index.json not found at {index_path}. Run Phase 1 first."
            )
        data = json.loads(index_path.read_text())
        index = [SegmentMetadata.model_validate(d) for d in data]

    print("Phase 2: Document Type Clustering")

    # Step A: Generate taxonomy
    print("  Step A: Generating document type taxonomy...")
    taxonomy = _generate_taxonomy(index, model_name)
    taxonomy_path.write_text(
        json.dumps([t.model_dump() for t in taxonomy], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Wrote {taxonomy_path}")

    # Step B: Classify segments in batches
    print("  Step B: Classifying segments...")
    classifications: list[TypeClassification] = []
    for i in range(0, len(index), batch_size):
        batch = index[i : i + batch_size]
        batch_class = _classify_batch(batch, taxonomy, model_name)
        classifications.extend(batch_class)
        print(f"    [{i + len(batch)}/{len(index)}] classified")

    classifications_path.write_text(
        json.dumps(
            [c.model_dump() for c in classifications],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"  Wrote {classifications_path}")

    print("Phase 2 complete")
    return taxonomy, classifications
