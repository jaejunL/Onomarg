from eval1.phonology import (
    duration_distance,
    phone_edit_distance,
    rwcp_repo_to_phones,
    segmental_view,
    temporal_view,
)


def test_temporal_and_segmental_views_preserve_then_collapse_duration():
    temporal = temporal_view(["b", "uː", "u", "ŋ", "k"])
    assert temporal == ["b", "u", "u", "u", "ŋ", "k"]
    assert segmental_view(temporal) == ["b", "u", "ŋ", "k"]
    assert duration_distance(["a"], ["a", "a", "a"]) == 2 / 3


def test_rwcp_mapping_and_feature_distance():
    assert rwcp_repo_to_phones("b u: N") == ["b", "ɯ", "ɯ", "ɴ"]
    assert rwcp_repo_to_phones(": q s u") == ["ʔ", "s", "ɯ"]
    assert phone_edit_distance(["b"], ["p"]) < phone_edit_distance(["b"], ["a"])

