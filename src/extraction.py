import json
import os
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

from .models import SegmentMetadata

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)

EXTRACT_PROMPT = """
        You are analyzing a construction arbitration document. The text below is from OCR and may contain scanning artifacts, mixed languages (English/Kannada/Hindi), broken tables, or formatting noise. Extract what you can; ignore unreadable parts.

        For the document with segment_id "{segment_id}", extract:

        1. document_type: Type of document (e.g. Letter, Work Order, Annexure, Notice, Invoice, Certificate)
        2. purpose: What is this document's purpose or function in one sentence
        3. parties: List of parties mentioned (e.g. URC Construction, ISRO, ISAC, Engineer-in-Charge)
        4. dates: Key dates (format as YYYY-MM-DD or DD.MM.YYYY if unclear)
        5. reference_numbers: Reference numbers, WO numbers, letter numbers, document numbers
        6. monetary_amounts: Any amounts in rupees or other currency (preserve as written)
        7. summary: 2-4 sentence summary of the document content

        Return a single JSON object (not an array). Valid JSON only, no markdown. If a field has no data, use empty array [] or empty string "".

        ---
        DOCUMENT TEXT:
        ---
        {text}
        ---
"""


def load_segment_text(segments_dir: Path, segment_id: str) -> str:
    path = segments_dir / segment_id / "ocr_text.txt"
    if not path.exists():
        raise FileNotFoundError(f"Segment not found: {segment_id}")
    return path.read_text(encoding="utf-8", errors="replace")


def extract_metadata(
    segment_id: str, text: str, model_name: str = "gemini-2.0-flash"
) -> SegmentMetadata:
    max_chars = 28000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n..."

    prompt = EXTRACT_PROMPT.format(segment_id=segment_id, text=text)
    model = genai.GenerativeModel(model_name)

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
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
        parsed = json.loads(raw)
        data = parsed[0] if isinstance(parsed, list) and parsed else parsed
        if not isinstance(data, dict):
            raise ValueError("Expected dict")
    except (json.JSONDecodeError, ValueError, IndexError):
        data = {
            "segment_id": segment_id,
            "document_type": "Unknown",
            "purpose": "Could not parse LLM response",
            "parties": [],
            "dates": [],
            "reference_numbers": [],
            "monetary_amounts": [],
            "summary": raw[:500] if raw else "No summary extracted",
        }

    data["segment_id"] = segment_id
    return SegmentMetadata.model_validate(data)


def run_extraction(
    segments_dir: Path,
    output_dir: Path,
    segment_ids: list[str],
    model_name: str = "gemini-2.5-flash",
) -> list[SegmentMetadata]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "index.json"

    if output_path.exists():
        print("Phase 1: already completed, skipping")
        data = json.loads(output_path.read_text())
        return [SegmentMetadata.model_validate(d) for d in data]

    results: list[SegmentMetadata] = []

    for i, segment_id in enumerate(segment_ids, 1):
        try:
            text = load_segment_text(segments_dir, segment_id)
            meta = extract_metadata(segment_id, text, model_name)
            results.append(meta)
            print(f"  [{i}/{len(segment_ids)}] {segment_id}")
        except Exception as e:
            print(f"  [{i}/{len(segment_ids)}] {segment_id} — ERROR: {e}")
            results.append(
                SegmentMetadata(
                    segment_id=segment_id,
                    document_type="Unknown",
                    purpose="Extraction failed",
                    parties=[],
                    dates=[],
                    reference_numbers=[],
                    monetary_amounts=[],
                    summary=str(e),
                )
            )

    output_path.write_text(
        json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Phase 1 completed.")
    return results
