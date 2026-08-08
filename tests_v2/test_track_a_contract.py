from evidence_extension_v1.track_a.common import (
    MODE_CONFIG,
    VARIANT_ORDER,
    expected_variant_hashes,
)


def test_track_a_frozen_matrix_and_hashes():
    assert MODE_CONFIG["confirmatory"]["expected_rows"] == 1920
    assert MODE_CONFIG["confirmatory"]["instances"] == [16, 17, 18, 19, 20]
    assert MODE_CONFIG["confirmatory"]["dimensions"] == [5, 20]
    assert len(VARIANT_ORDER) == 8
    hashes = expected_variant_hashes()
    assert hashes["Full"] == "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
    assert len(set(hashes.values())) == 8
