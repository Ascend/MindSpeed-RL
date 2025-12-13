# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import re
import signal
from unittest.mock import Mock, patch
import sys
import os
import pytest

# Add the parent directory to the path to import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from mindspeed_rl.models.math_dapo import (
    last_boxed_only_string,
    remove_boxed,
    timeout,
    normalize_final_answer,
    is_correct_minerva,
    is_correct_strict_box,
    verify,
    compute_score
)

from tests.test_tools.dist_test import DistributedTest


class TestMathDapo(DistributedTest):
    world_size = 1
    is_dist_test = False

    def test_last_boxed_only_string_found(self):
        # Test extracting last boxed expression
        test_string = "Some text \\boxed{2x + 3} and more text \\boxed{4y - 5}"
        result = last_boxed_only_string(test_string)
        assert result == "\\boxed{4y - 5}"

    def test_last_boxed_only_string_not_found(self):
        # Test when no boxed expression is found
        test_string = "Some text without boxed expressions"
        result = last_boxed_only_string(test_string)
        assert result is None

    def test_last_boxed_only_string_unclosed_brace(self):
        # Test with unclosed brace
        test_string = "Some text \\boxed{2x + 3"
        result = last_boxed_only_string(test_string)
        assert result is None

    def test_remove_boxed_valid(self):
        # Test removing boxed command from valid string
        test_string = "\\boxed{2x + 3}"
        result = remove_boxed(test_string)
        assert result == "2x + 3"

    def test_remove_boxed_invalid_prefix(self):
        # Test removing boxed command with invalid prefix
        test_string = "boxed{2x + 3}"
        with pytest.raises(ValueError, match="box error: 前缀不匹配"):
            remove_boxed(test_string)

    def test_remove_boxed_invalid_suffix(self):
        # Test removing boxed command with invalid suffix
        test_string = "\\boxed{2x + 3"
        with pytest.raises(ValueError, match="box error: 结尾字符不匹配"):
            remove_boxed(test_string)

    def test_timeout_context_manager(self):
        # Test timeout context manager
        # This test is mostly to ensure the class can be instantiated and used
        with timeout(seconds=1, error_message="Test timeout"):
            # Do nothing, just test that the context manager works
            pass

    def test_normalize_final_answer_basic(self):
        # Test basic answer normalization
        test_answer = "Answer: \\boxed{2x + 3}"
        result = normalize_final_answer(test_answer)
        assert result == "Answer:2x+3"

    def test_normalize_final_answer_with_equals(self):
        # Test answer normalization with equals sign
        test_answer = "= \\boxed{2x + 3}"
        result = normalize_final_answer(test_answer)
        assert result == "2x+3"

    def test_normalize_final_answer_with_substitutions(self):
        # Test answer normalization with substitutions
        test_answer = "an \\boxed{2x + 3}"
        result = normalize_final_answer(test_answer)
        assert result == "2x+3"

    def test_normalize_final_answer_with_removals(self):
        # Test answer normalization with removals
        test_answer = "\\boxed{2x + 3} square"
        result = normalize_final_answer(test_answer)
        assert result == "2x+3"

    def test_normalize_final_answer_with_latex(self):
        # Test answer normalization with LaTeX
        test_answer = "$\\boxed{2x + 3}$"
        result = normalize_final_answer(test_answer)
        assert result == "2x+3"

    def test_normalize_final_answer_with_text(self):
        # Test answer normalization with text commands
        test_answer = "\\text{Answer}: \\boxed{2x + 3}"
        result = normalize_final_answer(test_answer)
        assert result == "Answer:2x+3"

    def test_normalize_final_answer_with_frac(self):
        # Test answer normalization with fractions
        test_answer = "frac12"
        result = normalize_final_answer(test_answer)
        assert result == "frac{1}{2}"

    def test_normalize_final_answer_with_sqrt(self):
        # Test answer normalization with square roots
        test_answer = "sqrt2"
        result = normalize_final_answer(test_answer)
        assert result == "sqrt{2}"

    def test_normalize_final_answer_with_comma_number(self):
        # Test answer normalization with comma-separated numbers
        test_answer = "1,000"
        result = normalize_final_answer(test_answer)
        assert result == "1000"

    def test_is_correct_minerva_incorrect(self):
        # Test Minerva correctness check with incorrect answer
        solution_str = "The answer is \\boxed{2x + 4}"
        gt = "2x + 3"
        result, pred = is_correct_minerva(solution_str, gt)
        assert result is False
        assert pred == "[INVALID]"

    def test_is_correct_minerva_no_answer(self):
        # Test Minerva correctness check with no answer found
        solution_str = "No answer here"
        gt = "2x + 3"
        result, pred = is_correct_minerva(solution_str, gt)
        assert result is False
        assert pred == "[INVALID]"

    def test_is_correct_minerva_custom_pattern(self):
        # Test Minerva correctness check with custom pattern
        solution_str = "Answer: 2x + 3"
        gt = "2x + 3"
        result, pred = is_correct_minerva(solution_str, gt, answer_pattern=r"Answer:\s*([^\n]+)")
        assert result is True
        assert pred == "2x+3"

    def test_is_correct_strict_box_correct(self):
        # Test strict box correctness check with correct answer
        pred = "Some text \\boxed{2x + 3}"
        gt = "2x + 3"
        score, extracted_pred = is_correct_strict_box(pred, gt)
        assert score == 1
        assert extracted_pred == "2x + 3"

    def test_is_correct_strict_box_incorrect(self):
        # Test strict box correctness check with incorrect answer
        pred = "Some text \\boxed{2x + 4}"
        gt = "2x + 3"
        score, extracted_pred = is_correct_strict_box(pred, gt)
        assert score == -1
        assert extracted_pred == "2x + 4"

    def test_is_correct_strict_box_no_box(self):
        # Test strict box correctness check with no boxed answer
        pred = "Some text without boxed answer"
        gt = "2x + 3"
        score, extracted_pred = is_correct_strict_box(pred, gt)
        assert score == -1
        assert extracted_pred is None

    def test_is_correct_strict_box_with_pause_tokens(self):
        # Test strict box correctness check with pause tokens
        pred = "Some text " * 20 + "\\boxed{2x + 3}"
        gt = "2x + 3"
        pause_tokens_index = [0, 50, 100, 150]  # Last token index is 150
        score, extracted_pred = is_correct_strict_box(pred, gt, pause_tokens_index)
        assert score == 1
        assert extracted_pred == "2x + 3"

    def test_is_correct_strict_box_invalid_pause_tokens(self):
        # Test strict box correctness check with invalid pause tokens
        pred = "Some text \\boxed{2x + 3}"
        gt = "2x + 3"
        pause_tokens_index = [0, 50]  # Invalid length
        with pytest.raises(ValueError, match="pause_tokens_index 长度必须为 4"):
            is_correct_strict_box(pred, gt, pause_tokens_index)

    def test_verify_minerva_incorrect(self):
        # Test verify function with Minerva method and incorrect answer
        solution_str = "The answer is \\boxed{2x + 4}"
        answer = "2x + 3"
        correct, pred = verify(solution_str, answer, strict_box_verify=False)
        assert correct is False
        assert pred == "[INVALID]"

    def test_verify_strict_box_correct(self):
        # Test verify function with strict box method and correct answer
        solution_str = "Some text \\boxed{2x + 3}"
        answer = "2x + 3"
        correct, pred = verify(solution_str, answer, strict_box_verify=True)
        assert correct is True
        assert pred == "2x + 3"

    def test_verify_strict_box_incorrect(self):
        # Test verify function with strict box method and incorrect answer
        solution_str = "Some text \\boxed{2x + 4}"
        answer = "2x + 3"
        correct, pred = verify(solution_str, answer, strict_box_verify=True)
        assert correct is False
        assert pred == "2x + 4"

    def test_verify_with_pause_tokens(self):
        # Test verify function with pause tokens
        solution_str = "Some text " * 20 + "\\boxed{2x + 3}"
        answer = "2x + 3"
        pause_tokens_index = [0, 50, 100, 150]
        correct, pred = verify(solution_str, answer, strict_box_verify=True, pause_tokens_index=pause_tokens_index)
        assert correct is True
        assert pred == "2x + 3"

    def test_compute_score_correct(self):
        # Test compute_score with correct answer
        solution_str = "Some text \\boxed{2x + 3}"
        ground_truth = "2x + 3"
        result = compute_score(solution_str, ground_truth, strict_box_verify=False)
        assert result == -1.0

    def test_compute_score_incorrect(self):
        # Test compute_score with incorrect answer
        solution_str = "Some text \\boxed{2x + 4}"
        ground_truth = "2x + 3"
        result = compute_score(solution_str, ground_truth, strict_box_verify=False)
        assert result == -1.0

    def test_compute_score_strict_box_correct(self):
        # Test compute_score with strict box method and correct answer
        solution_str = "Some text \\boxed{2x + 3}"
        ground_truth = "2x + 3"
        result = compute_score(solution_str, ground_truth, strict_box_verify=True)
        assert result == 1.0

    def test_compute_score_strict_box_incorrect(self):
        # Test compute_score with strict box method and incorrect answer
        solution_str = "Some text \\boxed{2x + 4}"
        ground_truth = "2x + 3"
        result = compute_score(solution_str, ground_truth, strict_box_verify=True)
        assert result == -1.0

    def test_compute_score_long_solution(self):
        # Test compute_score with long solution (should be truncated)
        long_text = "Some text " * 100  # Much longer than 300 characters
        solution_str = long_text + "\\boxed{2x + 3}"
        ground_truth = "2x + 3"
        result = compute_score(solution_str, ground_truth, strict_box_verify=False)
        # Should still work correctly despite truncation
        assert result == -1.0