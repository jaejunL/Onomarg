"""Score one frozen prediction JSONL with the Eval2 retrieval protocol."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from eval2.fixed_eval import detailed_mode_collapse, evaluate, load_rows, prediction_phones

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--prediction-format", required=True, choices=("english_text", "firat_romanized_korean", "rwcp_repo_phone", "ipa_tokens", "arpabet_runs"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arpabet-map", type=Path, default=Path("eval2/mappings/arpabet_to_ipa.json"))
    parser.add_argument("--firat-lexicon", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()
    rows = load_rows(args.predictions); manifest = load_rows(args.manifest)
    predictions, chosen = prediction_phones(rows, args.prediction_format, "submission", args.arpabet_map, args.firat_lexicon)
    result = evaluate(predictions, manifest, reps=args.bootstrap_replicates, seed=1337, cache_root=args.output / "cache")
    test = {row["eval_audio_id"]: row for row in manifest if row.get("split") == "test"}
    raw = [chosen[key].get("prediction", "") for key in sorted(chosen)]
    result["mode_collapse"] = detailed_mode_collapse(raw, [predictions[key] for key in sorted(predictions)], [test[key]["class_key"] for key in sorted(predictions)])
    result.update({"prediction_format": args.prediction_format, "prediction_sha256": sha256(args.predictions), "manifest_sha256": sha256(args.manifest)})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
if __name__ == "__main__": main()
