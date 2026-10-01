import pytest

from vol_gate.router import RoutingError, route_question


def test_routes_median_change():
    shape, params = route_question(
        "Did median 25-delta put implied vol for SYN-A01 rise between 2025-01-02 and 2025-03-27?"
    )
    assert shape == "median_change"
    assert params == {"delta_bucket": 25, "right": "P", "ticker": "SYN-A01",
                       "start_date": "2025-01-02", "end_date": "2025-03-27"}


def test_routes_cohort_compare():
    shape, params = route_question(
        "Is there a significant difference in 10-delta call implied vol regime between cohort A and "
        "cohort B between 2025-01-02 and 2025-03-27?"
    )
    assert shape == "cohort_compare"
    assert params["right"] == "C"
    assert params["cohort_a"] == "A" and params["cohort_b"] == "B"


def test_unrecognized_question_raises():
    with pytest.raises(RoutingError):
        route_question("What's the weather like today?")
