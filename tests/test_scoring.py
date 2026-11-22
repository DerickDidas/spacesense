from pytest import approx

from src.evaluate import (
    detection_delays,
    false_alarms_per_10k,
    flags_to_sequences,
    merge_close,
    overlaps,
    score_events,
)


def test_perfect_match():
    s = score_events([[10, 20]], [[10, 20]])
    assert (s.tp, s.fp, s.fn) == (1, 0, 0)
    assert s.precision == 1.0 and s.recall == 1.0


def test_missed_anomaly():
    s = score_events([], [[10, 20]])
    assert (s.tp, s.fp, s.fn) == (0, 0, 1)
    assert s.precision == 0.0 and s.recall == 0.0


def test_false_alarm():
    s = score_events([[50, 60]], [[10, 20]])
    assert (s.tp, s.fp, s.fn) == (0, 1, 1)


def test_partial_overlap_counts_as_found():
    s = score_events([[15, 40]], [[10, 20]])
    assert (s.tp, s.fp, s.fn) == (1, 0, 0)


def test_touching_endpoints_overlap():
    assert overlaps([10, 20], [20, 30])
    assert not overlaps([10, 19], [20, 30])


def test_one_long_flag_covers_two_anomalies():
    s = score_events([[0, 100]], [[10, 20], [50, 60]])
    assert (s.tp, s.fp, s.fn) == (2, 0, 0)


def test_two_flags_inside_one_anomaly_are_not_false_alarms():
    s = score_events([[10, 12], [15, 18]], [[5, 20]])
    assert (s.tp, s.fp, s.fn) == (1, 0, 0)


def test_mixed_case_metrics():
    s = score_events([[10, 20], [100, 110]], [[12, 15], [200, 210]])
    assert (s.tp, s.fp, s.fn) == (1, 1, 1)
    assert s.precision == approx(0.5)
    assert s.recall == approx(0.5)
    assert s.f1 == approx(0.5)


def test_empty_inputs():
    s = score_events([], [])
    assert (s.tp, s.fp, s.fn) == (0, 0, 0)
    assert s.f1 == 0.0


def test_flags_to_sequences_and_offset():
    flags = [0, 1, 1, 0, 0, 1, 0]
    assert flags_to_sequences(flags) == [[1, 2], [5, 5]]
    assert flags_to_sequences(flags, offset=250) == [[251, 252], [255, 255]]
    assert flags_to_sequences([0, 0, 0]) == []


def test_detection_delays():
    assert detection_delays([[15, 40]], [[10, 20], [100, 110]]) == [5, None]


def test_false_alarms_per_10k():
    assert false_alarms_per_10k(2, 5000) == approx(4.0)


def test_merge_close():
    assert merge_close([[0, 5], [10, 15]], gap=4) == [[0, 15]]   # 4 steps between
    assert merge_close([[0, 5], [10, 15]], gap=3) == [[0, 5], [10, 15]]


def test_merge_close_unsorted_and_empty():
    assert merge_close([[10, 15], [0, 5]], gap=4) == [[0, 15]]
    assert merge_close([], gap=4) == []