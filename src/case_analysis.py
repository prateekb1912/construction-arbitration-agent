import json
import os
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)


REPORT_PROMPT = """
    You are a construction arbitration lawyer preparing a case analysis for a colleague.

    Using the structured data below, produce a narrative report in Markdown. 
    Write like a lawyer — precise, grounded in the documents, actionable. Surface insights a human might miss: gaps in the paper trail, unanswered notices, missing documentation, contradictions between documents, claim heads with weak documentary support, and documents that support one party while rebutting another on the same claim.

    Structure your report with these sections:

    ## 1. Case Overview
    What is this dispute about? Who are the parties? What is at stake? What is the total amount in dispute (sum across claim heads where amounts are known)?

    ## 2. Document Landscape
    How many segments? What document types (from the taxonomy)? What time period do they span? Any notable gaps in coverage?

    ## 3. Claim Head Analysis
    For each claim head: what is being claimed, by whom, what documents support it, what documents rebut it, how strong is the evidence, approximate quantum. Note where evidence is thin or one-sided.

    ## 4. Document Coverage Map
    - High-value segments (relevant to 3+ claim heads): list them and why they matter
    - Orphan segments (0–1 claim heads): list and whether they are peripheral or potentially overlooked
    - Claim heads with weak documentary support: flag them

    ## 5. Timeline
    Key events in chronological order, anchored to document references. Include contract date, delay notices, COVID impact, final bill, levy, demand notice, etc.

    ## 6. Key Observations
    Gaps, unanswered notices, missing documentation, contradictions, anything a lawyer should pay attention to. Be specific — cite segment IDs and claim heads.

    ---

    INPUT DATA:

    ### Index (segment metadata)
    {index_summary}

    ### Document Type Taxonomy
    {taxonomy_json}

    ### Claim Heads
    {claim_heads_json}

    ### Claim Classifications (per-segment mappings with reasoning)
    {claim_classifications_summary}

    ### Pre-computed Coverage
    {coverage_summary}
"""


def _load_outputs(output_dir: Path) -> tuple[dict, dict, dict, dict, dict]:
    index_path = output_dir / "index.json"
    taxonomy_path = output_dir / "type_taxonomy.json"
    heads_path = output_dir / "claim_heads.json"
    claim_class_path = output_dir / "claim_classifications.json"

    for p, name in [
        (index_path, "index"),
        (taxonomy_path, "type taxonomy"),
        (heads_path, "claim heads"),
        (claim_class_path, "claim classifications"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found at {p}. Run prior phases first.")

    return (
        json.loads(index_path.read_text()),
        json.loads(taxonomy_path.read_text()),
        json.loads(heads_path.read_text()),
        json.loads(claim_class_path.read_text()),
    )


def _build_index_summary(index: list[dict], max_chars: int = 15000) -> str:
    lines = []
    for m in index:
        lines.append(
            f"- **{m.get('segment_id', '?')}** | type: {m.get('document_type')} | "
            f"purpose: {m.get('purpose', '')[:150]} | "
            f"dates: {', '.join(m.get('dates', [])[:3])} | "
            f"amounts: {', '.join(m.get('monetary_amounts', [])[:3])}"
        )
    s = "\n".join(lines)
    return s[:max_chars] + "\n[... truncated ...]" if len(s) > max_chars else s


def _build_coverage_summary(
    claim_heads: list[dict],
    claim_classifications: list[dict],
) -> str:
    seg_to_count = {}
    claim_to_docs = {
        h["key"]: {"support": [], "rebut": [], "other": []} for h in claim_heads
    }

    for c in claim_classifications:
        seg_id = c.get("segment_id", "")
        mappings = c.get("mappings", [])
        seg_to_count[seg_id] = len(mappings)

        for m in mappings:
            ck = m.get("claim_key", "")
            if ck not in claim_to_docs:
                continue
            role = m.get("role", "neutral")
            if role == "supports_claimant":
                claim_to_docs[ck]["support"].append(
                    (seg_id, m.get("reasoning", "")[:80])
                )
            elif role in ("supports_employer", "rebuts"):
                claim_to_docs[ck]["rebut"].append((seg_id, m.get("reasoning", "")[:80]))
            else:
                claim_to_docs[ck]["other"].append((seg_id, m.get("reasoning", "")[:80]))

    high_value = [s for s, n in seg_to_count.items() if n >= 3]
    orphans = [s for s, n in seg_to_count.items() if n <= 1]

    weak_claims = []
    for h in claim_heads:
        k = h["key"]
        docs = claim_to_docs.get(k, {})
        total = len(docs["support"]) + len(docs["rebut"]) + len(docs["other"])
        direct = sum(
            1
            for c in claim_classifications
            for m in c.get("mappings", [])
            if m.get("claim_key") == k and m.get("relevance_type") == "direct_evidence"
        )
        if total < 3 or direct < 1:
            weak_claims.append(k)

    lines = [
        "**High-value segments (3+ claim heads):** " + ", ".join(high_value[:15])
        or "None",
        "**Orphan segments (0–1 claim heads):** " + ", ".join(orphans[:15]) or "None",
        "**Claim heads with weak support:** " + ", ".join(weak_claims) or "None",
    ]
    return "\n".join(lines)


def _build_claim_class_summary(
    claim_classifications: list[dict], max_chars: int = 12000
) -> str:
    lines = []
    for c in claim_classifications:
        seg = c.get("segment_id", "")
        maps = c.get("mappings", [])
        if not maps:
            lines.append(f"- {seg}: (no claim mappings)")
            continue
        parts = [f"- **{seg}**:"]
        for m in maps:
            parts.append(
                f"  {m.get('claim_key')} ({m.get('role')}, {m.get('relevance_type')}, "
                f"conf={m.get('confidence', 0)}) — {m.get('reasoning', '')[:100]}"
            )
        lines.append("\n".join(parts))
    s = "\n\n".join(lines)
    return s[:max_chars] + "\n[... truncated ...]" if len(s) > max_chars else s


def run_case_analysis(
    output_dir: Path,
    model_name: str = "gemini-2.5-pro",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "case_analysis.md"

    if report_path.exists():
        print("Phase 4: already completed, skipping")
        return report_path

    print("Phase 4: Case Reasoning and Analysis")

    index, taxonomy, claim_heads, claim_classifications = _load_outputs(output_dir)

    index_summary = _build_index_summary(index)
    taxonomy_json = json.dumps(
        taxonomy if isinstance(taxonomy, list) else taxonomy.get("taxonomy", taxonomy),
        indent=2,
        ensure_ascii=False,
    )
    claim_heads_json = json.dumps(claim_heads, indent=2, ensure_ascii=False)
    claim_class_summary = _build_claim_class_summary(claim_classifications)
    coverage_summary = _build_coverage_summary(claim_heads, claim_classifications)

    prompt = REPORT_PROMPT.format(
        index_summary=index_summary,
        taxonomy_json=taxonomy_json[:8000],
        claim_heads_json=claim_heads_json,
        claim_classifications_summary=claim_class_summary,
        coverage_summary=coverage_summary,
    )

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0,
        ),
    )

    report = response.text.strip()
    if report.startswith("```"):
        report = report.split("```")[1]
        if report.startswith("markdown"):
            report = report[8:]
        report = report.strip()

    report_path.write_text(report, encoding="utf-8")
    print(f"Phase 4 complete. Wrote {report_path}")
    return report_path
