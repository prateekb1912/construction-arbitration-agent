import json

from config import OUTPUT_DIR, SEGMENTS_DIR, SEGMENTS_FILE
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

    # Phase 2: Document Type Clustering
    run_type_clustering(output_dir=OUTPUT_DIR, index=index)


if __name__ == "__main__":
    main()
