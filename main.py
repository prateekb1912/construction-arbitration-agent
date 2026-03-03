import json
from concurrent.futures import ThreadPoolExecutor, wait

from config import OUTPUT_DIR, SEGMENTS_DIR, SEGMENTS_FILE
from src.case_analysis import run_case_analysis
from src.claim_discovery import run_claim_discovery
from src.extraction import run_extraction
from src.type_clustering import run_type_clustering


def main() -> None:
    config = json.load(open(SEGMENTS_FILE))
    segment_ids = config["segments"]

    # Phase 1: Index and Extract
    index = run_extraction(
        segments_dir=SEGMENTS_DIR,
        output_dir=OUTPUT_DIR,
        segment_ids=segment_ids,
    )

    # Phases 2 and 3: Run in parallel
    with ThreadPoolExecutor(max_workers=2) as ex:
        f2 = ex.submit(run_type_clustering, output_dir=OUTPUT_DIR, index=index)
        f3 = ex.submit(run_claim_discovery, output_dir=OUTPUT_DIR, index=index)
        wait([f2, f3])

    # Phase 4: Case Analysis
    run_case_analysis(output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
