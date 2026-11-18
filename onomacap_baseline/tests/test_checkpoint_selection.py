def better(candidate, incumbent):
    return (
        candidate["METEOR"], candidate["BLEU1"], -candidate["epoch"]
    ) > (
        incumbent["METEOR"], incumbent["BLEU1"], -incumbent["epoch"]
    )


def test_validation_selection_order():
    assert better(
        {"METEOR": 0.2, "BLEU1": 0.9, "epoch": 2},
        {"METEOR": 0.1, "BLEU1": 0.99, "epoch": 1},
    )
    assert better(
        {"METEOR": 0.2, "BLEU1": 0.9, "epoch": 1},
        {"METEOR": 0.2, "BLEU1": 0.9, "epoch": 2},
    )
