"""Tests for benchmarks/bfcl/grade.py — function-call grader.

PRELIMINARY (2026-05-03). Validates simplified BFCL grading logic against
synthetic ground-truth shapes derived from the bfcl_eval data layout.
"""

from __future__ import annotations

from benchmarks.bfcl.grade import (
    GradeResult,
    _call_matches_gt_entry,
    _dict_shape_matches,
    _normalize_math_expr,
    _value_matches,
    grade,
    grade_irrelevance,
    grade_parallel,
    grade_relevance,
    grade_simple,
)


def test_value_matches_exact():
    assert _value_matches(5, [5, 10]) is True
    assert _value_matches("hello", ["hello", ""]) is True


def test_value_matches_str_to_number():
    assert _value_matches(5, ["5"]) is True  # str gt, int actual
    assert _value_matches("5", [5]) is True  # int gt, str actual


def test_value_matches_no_match():
    assert _value_matches(5, [10, 20]) is False
    assert _value_matches("hello", ["world"]) is False


def test_call_matches_gt_entry_basic():
    gt = {"add": {"a": [1, 2], "b": [3]}}
    assert _call_matches_gt_entry("add", {"a": 1, "b": 3}, gt) is True
    assert _call_matches_gt_entry("add", {"a": 2, "b": 3}, gt) is True
    assert _call_matches_gt_entry("add", {"a": 5, "b": 3}, gt) is False
    assert _call_matches_gt_entry("subtract", {"a": 1, "b": 3}, gt) is False


def test_call_matches_optional_arg_omitted():
    gt = {"f": {"required": [1], "optional": ["", "default"]}}
    # Omitting an arg whose allowed list contains "" is OK.
    assert _call_matches_gt_entry("f", {"required": 1}, gt) is True
    # Providing it is also OK.
    assert _call_matches_gt_entry("f", {"required": 1, "optional": "default"}, gt) is True
    # But providing a wrong value isn't.
    assert _call_matches_gt_entry("f", {"required": 1, "optional": "wrong"}, gt) is False


def test_grade_simple_matches_one_call():
    gt = [{"add": {"a": [1], "b": [2]}}]
    res = grade_simple([("add", {"a": 1, "b": 2})], gt)
    assert res.passed is True
    assert res.actual_calls == 1


def test_grade_simple_no_call_fails():
    gt = [{"add": {"a": [1], "b": [2]}}]
    res = grade_simple([], gt)
    assert res.passed is False
    assert "no_tool_call" in res.reason


def test_grade_simple_too_many_calls_fails():
    gt = [{"add": {"a": [1], "b": [2]}}]
    res = grade_simple([("add", {"a": 1, "b": 2}), ("add", {"a": 1, "b": 2})], gt)
    assert res.passed is False
    assert "expected_1" in res.reason


def test_grade_simple_picks_any_gt_entry():
    """Multiple-choice GT — any entry is acceptable."""
    gt = [
        {"add": {"a": [1], "b": [2]}},
        {"sum": {"x": [3]}},
    ]
    assert grade_simple([("add", {"a": 1, "b": 2})], gt).passed is True
    assert grade_simple([("sum", {"x": 3})], gt).passed is True


def test_grade_parallel_set_match():
    gt = [
        {"f": {"x": [1]}},
        {"g": {"y": [2]}},
    ]
    # Order doesn't matter
    assert grade_parallel(
        [("g", {"y": 2}), ("f", {"x": 1})], gt
    ).passed is True


def test_grade_parallel_count_mismatch():
    gt = [{"f": {"x": [1]}}, {"g": {"y": [2]}}]
    res = grade_parallel([("f", {"x": 1})], gt)
    assert res.passed is False
    assert "expected_2" in res.reason


def test_grade_parallel_one_unmatched_call_fails():
    gt = [{"f": {"x": [1]}}, {"g": {"y": [2]}}]
    res = grade_parallel(
        [("f", {"x": 1}), ("h", {"z": 3})], gt
    )
    assert res.passed is False
    assert "unmatched" in res.reason


def test_grade_irrelevance_no_call_passes():
    res = grade_irrelevance([])
    assert res.passed is True


def test_grade_irrelevance_call_fails():
    res = grade_irrelevance([("calculator", {"expr": "1+1"})])
    assert res.passed is False
    assert "calculator" in res.reason


def test_grade_dispatch_routes_by_category():
    gt = [{"f": {"x": [1]}}]
    assert grade("simple_python", [("f", {"x": 1})], gt).passed is True
    assert grade("multiple", [("f", {"x": 1})], gt).passed is True
    assert grade("parallel", [("f", {"x": 1})], gt).passed is True
    assert grade("parallel_multiple", [("f", {"x": 1})], gt).passed is True
    assert grade("irrelevance", [], None).passed is True


def test_grade_dispatch_routes_live_categories():
    # Live BFCL categories reuse the same scoring as their non-live siblings.
    gt = [{"f": {"x": [1]}}]
    assert grade("live_simple", [("f", {"x": 1})], gt).passed is True
    assert grade("live_multiple", [("f", {"x": 1})], gt).passed is True
    assert grade("live_parallel", [("f", {"x": 1})], gt).passed is True
    assert grade("live_parallel_multiple", [("f", {"x": 1})], gt).passed is True
    # live_irrelevance: pass = no calls (same as non-live irrelevance).
    assert grade("live_irrelevance", [], None).passed is True
    assert grade("live_irrelevance", [("f", {})], None).passed is False
    # live_relevance: pass = at least one call (any tool, any args).
    assert grade("live_relevance", [("f", {})], None).passed is True
    assert grade("live_relevance", [], None).passed is False


def test_grade_relevance_helper():
    # grade_relevance ignores arg shape; only call-count matters.
    assert grade_relevance([("anything", {})]).passed is True
    assert grade_relevance([("a", {}), ("b", {})]).passed is True
    res = grade_relevance([])
    assert res.passed is False
    assert "no_tool_call_when_relevant" in res.reason


def test_grade_unsupported_category():
    res = grade("nonexistent", [], None)
    assert res.passed is False
    assert "unsupported" in res.reason


# Nested-dict allowed-list handling — BFCL v4 wraps every leaf (including
# leaves inside dict-typed args) in its own allowed-list. The grader
# treats dict allowed-entries as shape templates, not equality targets.

def test_value_matches_nested_dict_realestate_shape():
    # multiple_8 GT: budget arg has nested-dict shape with list-wrapped leaves.
    allowed = [{"min": [300000], "max": [400000]}]
    actual = {"min": 300000, "max": 400000}
    assert _value_matches(actual, allowed) is True


def test_value_matches_nested_dict_grades_shape():
    # multiple_9 GT: gradeDict arg has multi-key nested-dict shape.
    allowed = [{"math": [90], "science": [75], "history": [82], "music": [89]}]
    actual = {"math": 90, "science": 75, "history": 82, "music": 89}
    assert _value_matches(actual, allowed) is True


def test_value_matches_nested_dict_wrong_leaf_fails():
    allowed = [{"min": [300000], "max": [400000]}]
    actual = {"min": 300000, "max": 999}  # max wrong
    assert _value_matches(actual, allowed) is False


def test_value_matches_nested_dict_optional_leaf():
    # An inner key whose allowed list includes "" or None is optional.
    allowed = [{"required": [1], "optional": ["", "default"]}]
    assert _value_matches({"required": 1}, allowed) is True
    assert _value_matches({"required": 1, "optional": "default"}, allowed) is True
    assert _value_matches({"required": 1, "optional": "wrong"}, allowed) is False


def test_value_matches_nested_dict_alternatives():
    # Multiple acceptable shape templates — match any.
    allowed = [
        {"unit": ["metric"], "value": [10]},
        {"unit": ["imperial"], "value": [22]},
    ]
    assert _value_matches({"unit": "metric", "value": 10}, allowed) is True
    assert _value_matches({"unit": "imperial", "value": 22}, allowed) is True
    assert _value_matches({"unit": "metric", "value": 22}, allowed) is False


def test_value_matches_dict_vs_scalar_doesnt_recurse():
    # If the model passes a scalar where GT expects a dict shape, fail.
    allowed = [{"min": [1], "max": [2]}]
    assert _value_matches(5, allowed) is False


def test_dict_shape_matches_extra_keys_tolerated():
    # Extra keys the model emits but GT doesn't list are not penalized.
    allowed_shape = {"min": [300000], "max": [400000]}
    actual = {"min": 300000, "max": 400000, "currency": "USD"}
    assert _dict_shape_matches(actual, allowed_shape) is True


def test_call_matches_with_nested_dict_arg():
    # Full integration through _call_matches_gt_entry.
    gt = {
        "realestate.find_properties": {
            "location": ["San Diego, CA"],
            "budget": [{"min": [300000], "max": [400000]}],
        }
    }
    assert _call_matches_gt_entry(
        "realestate.find_properties",
        {"location": "San Diego, CA", "budget": {"min": 300000, "max": 400000}},
        gt,
    ) is True
    # Wrong leaf in the nested dict still fails the call.
    assert _call_matches_gt_entry(
        "realestate.find_properties",
        {"location": "San Diego, CA", "budget": {"min": 300000, "max": 999}},
        gt,
    ) is False


# Math-expression normalization — implicit vs explicit multiplication.
# BFCL ground truth uses the partially-Python form `3x**2`; many models
# emit the fully-Python form `3*x**2`. These are semantically identical
# and should match.

def test_normalize_math_expr_strips_implicit_multiplication():
    assert _normalize_math_expr("3*x**2") == "3x**2"
    assert _normalize_math_expr("3 * x**2") == "3x**2"
    assert _normalize_math_expr("3x**2") == "3x**2"


def test_normalize_math_expr_handles_polynomial():
    assert _normalize_math_expr("3*x**2 + 2*x - 1") == _normalize_math_expr("3x**2 + 2x - 1")


def test_normalize_math_expr_does_not_touch_identifier_multiplication():
    # `a*b` is NOT implicit mul — both are identifiers. Leave it.
    assert _normalize_math_expr("a*b") == "a*b"
    assert _normalize_math_expr("a * b") == "a * b"


def test_normalize_math_expr_does_not_rewrite_caret():
    # `^` is XOR in Python — emitting `x^2` is a real model failure,
    # NOT semantically equivalent to `x**2`. Do not normalize.
    assert _normalize_math_expr("x^2") != _normalize_math_expr("x**2")


def test_value_matches_implicit_multiplication():
    # Real BFCL case: simple_python_14, allowed `3x**2 + 2x - 1`.
    # qwen25-1.5b emitted `3*x**2 + 2*x - 1`. Semantically identical.
    assert _value_matches("3*x**2 + 2*x - 1", ["3x**2 + 2x - 1"]) is True


def test_value_matches_caret_still_fails():
    # `x^3` vs `x**3` — model used wrong operator. Real failure.
    assert _value_matches("x^3", ["x**3"]) is False


def test_value_matches_implicit_mul_within_lambda():
    # GT often includes a lambda alternative — `lambda x: 3x**2 + 2x - 1`.
    assert _value_matches(
        "lambda x: 3*x**2 + 2*x - 1",
        ["3x**2 + 2x - 1", "lambda x: 3x**2 + 2x - 1"],
    ) is True


def test_value_matches_wrong_polynomial_still_fails():
    # Implicit-mul normalization should not loosen unrelated comparisons.
    assert _value_matches("3*x**2 + 2*x - 99", ["3x**2 + 2x - 1"]) is False
