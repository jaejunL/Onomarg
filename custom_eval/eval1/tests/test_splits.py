from collections import Counter

from eval1.splits import assign_stratified_80_10_10


def test_split_is_seed20_deterministic_and_80_10_10():
    rows = [
        {"eval_audio_id": f"x:{index:03d}", "class_key": str(index % 5)}
        for index in range(100)
    ]
    left = assign_stratified_80_10_10([dict(row) for row in rows], seed=20)
    right = assign_stratified_80_10_10([dict(row) for row in rows], seed=20)
    assert left == right
    assert Counter(row["split"] for row in left) == Counter(train=80, validation=10, test=10)

