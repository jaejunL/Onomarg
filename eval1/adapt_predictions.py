from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize existing model prediction JSONL into eval1 schema.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", choices=("firat", "rwcp"), required=True)
    parser.add_argument("--id-field", default="audio_id")
    parser.add_argument("--text-field", default="prediction")
    args = parser.parse_args()
    source = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    output = []
    for row in source:
        audio_id = str(row[args.id_field])
        if not audio_id.startswith(f"{args.dataset}:"):
            audio_id = f"{args.dataset}:{audio_id}"
        output.append({
            "eval_audio_id": audio_id,
            "eval_dataset": args.dataset,
            "beam_rank": int(row.get("beam_rank", 0)),
            "prediction": str(row[args.text_field]),
        })
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output), encoding="utf-8")


if __name__ == "__main__":
    main()

