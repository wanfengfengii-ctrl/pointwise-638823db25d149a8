from contextlib import closing
import json
import math

from analysis_tasks.requests import WindowRequest
from analysis_tasks.service import analyze, freeze
from archive import connect
from duration_inference.evidence import ZeroEvidenceError
from examples import observation_batch, single_state, two_state
from observations.sequences import revise_observations
from runtime_models.registry import register_model
from runtime_models.schema import ModelDefinition


def sample_cases():
    alternating = ModelDefinition(
        ["idle", "run"], [[0, 1], [1, 0]],
        [[(1, 1)], [(1, 1)]], [[("a", 1)], [("b", 1)]],
    )
    rare = ModelDefinition(
        ["run"], [[1]], [[(2, 1)]], [[("x", 1e-10), ("z", 0.9999999999)]],
    )
    return {
        "renewal": (single_state(), "xx", "x"),
        "supported": (alternating, "ab", "a"),
        "unsupported": (alternating, "ba", "a"),
        "tiny": (rare, "x" * 20, "x" * 20),
        "noisy": (two_state(), "ab", "ba"),
    }


def load_station(connection, sequence_id, definition, symbols, revision_id="line-model"):
    ref = register_model(connection, revision_id, definition)
    version = revise_observations(connection, sequence_id, observation_batch(symbols))
    return ref, version


def v1_report(definition, symbols, windows):
    with closing(connect()) as connection:
        ref, version = load_station(connection, "reference-source", definition, symbols)
        freeze(connection, "reference", ref, version, len(symbols), windows)
        return analyze(connection, "reference")


def continuous_v1(definition, left, right):
    split = len(left)
    return v1_report(definition, left + right, [
        WindowRequest(0, split), WindowRequest(split, split + len(right)),
    ])


def baseline_comparison():
    cases = sample_cases()
    model, left, right = cases["renewal"]
    earlier = v1_report(model, left, [WindowRequest(0, len(left))])
    later = v1_report(model, right, [WindowRequest(0, len(right))])
    joined = continuous_v1(model, left, right)
    assert earlier["mean"] == [2, 0.5]
    assert later["mean"] == [1, 0]
    assert joined["mean"] == [2, 0.5, 1, 0.5]
    assert joined["covariance"][1][3] == joined["covariance"][3][1] == -0.25

    model, left, right = cases["supported"]
    other_left = cases["unsupported"][1]
    first = v1_report(model, left, [WindowRequest(0, len(left))])
    second = v1_report(model, other_left, [WindowRequest(0, len(other_left))])
    for field in ("evidence", "mean", "covariance"):
        assert first[field] == second[field], field
    supported = continuous_v1(model, left, right)
    assert supported["evidence"]["probability"] == 0.5
    assert supported["mean"][-1] == 1
    try:
        continuous_v1(*cases["unsupported"])
    except ZeroEvidenceError:
        unsupported = 0
    else:
        raise AssertionError("the opposite seam unexpectedly has positive evidence")

    tiny = continuous_v1(*cases["tiny"])["evidence"]
    assert tiny["probability"] is None and tiny["probability_representable"] is False
    assert tiny["exponent"] == -400
    assert math.isclose(tiny["log_evidence"], -400 * math.log(10), abs_tol=1e-8)
    return {
        "separate_renewal_mean": [earlier["mean"][-1], later["mean"][-1]],
        "continuous_renewal_mean": [joined["mean"][1], joined["mean"][3]],
        "continuous_cross_covariance": joined["covariance"][1][3],
        "seam_evidence": {"ab|a": supported["evidence"]["probability"], "ba|a": unsupported},
        "tiny_evidence": tiny,
    }


if __name__ == "__main__":
    print(json.dumps(baseline_comparison(), sort_keys=True, indent=2))
