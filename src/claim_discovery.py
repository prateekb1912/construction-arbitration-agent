import json
import os
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

from .models import (
    ClaimClassification,
    ClaimHead,
    ClaimMapping,
    SegmentMetadata,
)

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

DISCOVER_PROMPT = """
You are analyzing a construction arbitration case. Below is an index of document segments. 

Discover the claim heads - the discrete categories of financial recovery being fought over. 
These must be specific to THIS case, grounded in the actual documents. 
Do NOT use generic labels like "Delay Claims". Use concrete names with dates, amounts, or parties when visible.

For each claim head provide:
- key: lowercase_underscored
- name: descriptive, case-specific
- description: what is being claimed, by whom, with document references
- approximate_amount: if discernible from the index
- claimant: Contractor (URC) or Employer (ISRO) or both
- sub_heads: any sub-categories

A document can support one party while rebutting another on the same claim head. Be specific.

Return a single JSON object:
{{
  "claim_heads": [
    {{
      "key": "...",
      "name": "...",
      "description": "...",
      "approximate_amount": "...",
      "claimant": "...",
      "sub_heads": []
    }}
  ]
}}

Index:
---
{index_summary}
---
"""

MAP_PROMPT = """
Given these claim heads and document segments, for EACH document determine which claim heads it supports, rebuts, or relates to.
A document can map to multiple claim heads. It can support one party and rebut another on the same claim head.

For each document's mappings:
- relevance_type: direct_evidence | contains_data | contextual
- role: supports_claimant | supports_employer | rebuts | neutral
- confidence: 0-1
- reasoning: 1-2 sentences (required)

Return a single JSON object with "classifications" array — one object per document, each with segment_id and mappings:
{{
  "classifications": [
    {{
      "segment_id": "...",
      "mappings": [
        {{ "claim_key": "...", "relevance_type": "direct_evidence", "role": "supports_claimant", "confidence": 0.9, "reasoning": "..." }}
      ]
    }}
  ]
}}

Claim heads:
---
{claim_heads_json}
---

Documents:
---
{docs_batch}
---
"""


def _index_to_summary(index: list[SegmentMetadata], max_len: int = 30000) -> str:
    lines = []
    for m in index:
        lines.append(
            f"[{m.segment_id}]\ntype: {m.document_type}\npurpose: {m.purpose}\n"
            f"parties: {', '.join(m.parties[:5])}\n"
            f"amounts: {', '.join(m.monetary_amounts[:5])}\n"
            f"summary: {m.summary}"
        )
    s = "\n\n".join(lines)
    return s[:max_len] + "\n[... truncated ...]" if len(s) > max_len else s


def _discover_claim_heads(
    index: list[SegmentMetadata],
    model_name: str = "gemini-2.5-pro",
) -> list[ClaimHead]:
    summary = _index_to_summary(index)
    prompt = DISCOVER_PROMPT.format(index_summary=summary)
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

    data = json.loads(raw)
    heads = data.get("claim_heads", data)
    if not isinstance(heads, list):
        heads = [heads]
    result = []
    for h in heads:
        if not isinstance(h, dict):
            continue
        h = dict(h)
        sub = h.get("sub_heads", [])
        if sub and isinstance(sub[0], dict):
            h["sub_heads"] = [s.get("key", s.get("name", str(s))) for s in sub]
        result.append(ClaimHead.model_validate(h))
    return result


def _map_batch_to_claims(
    segments: list[SegmentMetadata],
    claim_heads: list[ClaimHead],
    model_name: str = "gemini-2.5-pro",
) -> list[ClaimClassification]:
    ch_json = json.dumps(
        [c.model_dump() for c in claim_heads], indent=2, ensure_ascii=False
    )
    docs_batch = "\n\n".join(
        f"[{s.segment_id}]\ntype: {s.document_type}\npurpose: {s.purpose}\nsummary: {s.summary[:400]}"
        for s in segments
    )
    prompt = MAP_PROMPT.format(
        claim_heads_json=ch_json,
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
            classifications = [classifications] if classifications else []
        result = []
        for i, c in enumerate(classifications):
            cdict = c if isinstance(c, dict) else {}
            sid = cdict.get("segment_id") or (
                segments[i].segment_id if i < len(segments) else "unknown"
            )
            result.append(
                ClaimClassification(
                    segment_id=sid,
                    mappings=[
                        ClaimMapping.model_validate(m)
                        for m in cdict.get("mappings", [])
                    ],
                )
            )
        while len(result) < len(segments):
            result.append(
                ClaimClassification(
                    segment_id=segments[len(result)].segment_id, mappings=[]
                )
            )
        return result[: len(segments)]
    except (json.JSONDecodeError, TypeError):
        return [
            ClaimClassification(segment_id=s.segment_id, mappings=[]) for s in segments
        ]


def run_claim_discovery(
    output_dir: Path,
    index: list[SegmentMetadata] | None = None,
    index_path: Path | None = None,
    model_name: str = "gemini-2.5-pro",
    batch_size: int = 8,
) -> tuple[list[ClaimHead], list[ClaimClassification]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    heads_path = output_dir / "claim_heads.json"
    mappings_path = output_dir / "claim_classifications.json"

    if heads_path.exists() and mappings_path.exists():
        print("Phase 3: already completed, skipping")
        heads = [
            ClaimHead.model_validate(h) for h in json.loads(heads_path.read_text())
        ]
        maps_data = json.loads(mappings_path.read_text())
        maps_list = (
            maps_data
            if isinstance(maps_data, list)
            else maps_data.get("classifications", [])
        )
        classifications = [ClaimClassification.model_validate(c) for c in maps_list]
        return heads, classifications

    print("Phase 3: Claim Head Discovery")

    print("  Step A: Discovering claim heads from documents...")
    claim_heads = _discover_claim_heads(index, model_name)
    heads_path.write_text(
        json.dumps([c.model_dump() for c in claim_heads], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Wrote {heads_path} ({len(claim_heads)} claim heads)")

    print("  Step B: Mapping segments to claim heads...")
    classifications = []
    for i in range(0, len(index), batch_size):
        batch = index[i : i + batch_size]
        batch_result = _map_batch_to_claims(batch, claim_heads, model_name)
        classifications.extend(batch_result)
        print(f"    [{min(i + batch_size, len(index))}/{len(index)}] mapped")

    mappings_path.write_text(
        json.dumps(
            [c.model_dump() for c in classifications], indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    print(f"  Wrote {mappings_path}")

    print("Phase 3 complete")
    return claim_heads, classifications
