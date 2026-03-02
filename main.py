import json

from config import OUTPUT_DIR, SEGMENTS_DIR, SEGMENTS_FILE
from src.extraction import run_extraction


def main() -> None:
    config = json.load(open(SEGMENTS_FILE))
    segment_ids = config["segments"]

    run_extraction(
        segments_dir=SEGMENTS_DIR,
        output_dir=OUTPUT_DIR,
        segment_ids=segment_ids,
    )


if __name__ == "__main__":
    main()
