from eval1.retrieval import evaluate_retrieval


def _row(audio_id, class_key, base):
    return {
        "eval_audio_id": audio_id,
        "class_key": class_key,
        "labels": [
            {
                "label_id": f"{audio_id}:{index}",
                "phone_tokens_temporal": [base, "a"],
                "phone_tokens_segmental": [base, "a"],
            }
            for index in range(5)
        ],
    }


def test_exact_phone_prediction_retrieves_correct_audio():
    rows = [_row("a", "c", "p"), _row("b", "c", "k")]
    predictions = {"a": ["p", "a"], "b": ["k", "a"]}
    result = evaluate_retrieval(predictions, rows, view="temporal", replicates=2, labels_per_candidate=1)
    assert result["global"]["R@1"]["mean"] == 1.0
    assert result["within_class"]["R@1"]["mean"] == 1.0

