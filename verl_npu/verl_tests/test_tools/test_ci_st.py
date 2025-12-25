import warnings
import pytest
from verl_tests.test_tools.acquire_json import transfer_logs_as_json, read_json


ENTROPY = "entropy"
LOGP_DIFF = "logp_diff"
PG_LOSS = "pg_loss"
GRAD_NORM = "grad_norm"


class TestMargin:
    _MARGIN_NAME = " margin"
    entropy = 0
    grad_norm = 0
    logp_diff = 0
    pg_loss = 0

    @classmethod
    def refresh_margin_from_json(cls, json_obj): 
        cls.entropy = json_obj.get(ENTROPY + cls._MARGIN_NAME, cls.entropy)
        cls.logp_diff = json_obj.get(LOGP_DIFF + cls._MARGIN_NAME, cls.logp_diff)
        cls.pg_loss = json_obj.get(PG_LOSS + cls._MARGIN_NAME, cls.pg_loss)
        cls.grad_norm = json_obj.get(GRAD_NORM + cls._MARGIN_NAME, cls.grad_norm)
        

class TestCIST:
#  基线数据self.expected  波动范围TestMargin.refresh_margin_from_json(self.expected)
    def _get_baseline(self, baseline_json):
        # acquire expected results
        self.expected = read_json(baseline_json)
        TestMargin.refresh_margin_from_json(self.expected)

    def _get_actual(self, generate_log, generate_json):
        # acquire actual results
        transfer_logs_as_json(generate_log, generate_json)
        self.actual = read_json(generate_json)

    def _test_helper(self, test_obj):
        print("test_obj: ", test_obj)
        """
        Core test function

        Args:
            test_obj: the object we want to test compare.
            test_type: deterministic or approximate, default is None.

        Here we temperally test `lm loss`, 'throughput' , `allocated memory` and `elapsed time per iteration`
        """
        comparison_base = {
            ENTROPY: self._compare_entropy,
            LOGP_DIFF: self._compare_logp_diff,
            PG_LOSS: self._compare_pg_loss,
            GRAD_NORM: self._compare_grad_norm
        }
     
        comparison_selection = {**comparison_base}

        if test_obj in comparison_selection:
            expected_list = self.expected[test_obj]
            if not expected_list:
                return
            print(f"===================== Begin comparing {test_obj} ===================")
            actual_list = self.actual[test_obj]
            print(f"The list of expected values: {expected_list}")
            print(f"The list of actual values: {actual_list}")
            # Check if lists exist and are non-empty
            if not actual_list:
                raise ValueError(f"Actual list for {test_obj} is empty or not found. Maybe program has failed! Check it.")

            # Check if lists have the same length
            if len(expected_list) != len(actual_list):
                raise ValueError(f"Actual lengths of the lists for {test_obj} do not match. Maybe program has failed! Check it.")

            compare_func = comparison_selection[test_obj]
            compare_func(expected_list, actual_list)
        else:
            warnings.warn(f"The metric of {test_obj} is not selected and will be skipped.", RuntimeWarning, stacklevel=2)

    def _compare_entropy(self, expected_list, actual_list):
        # Because "deterministic computation" affects the throughput, so we just test
        # lm loss in case of approximation.
        for step, (expected_val, actual_val) in enumerate(zip(expected_list, actual_list)):
            print(f"Checking step {step + 1} for entropy")
            assert actual_val == pytest.approx(expected=expected_val, rel=TestMargin.entropy),\
            f"The entropy at step {step} should be approximate to {expected_val} but it is {actual_val}."

    def _compare_logp_diff(self, expected_list, actual_list):
        for step, (expected_val, actual_val) in enumerate(zip(expected_list, actual_list)):
            print(f"Checking step {step + 1} for logp diff")
            assert actual_val == pytest.approx(expected=expected_val, rel=TestMargin.logp_diff),\
            f"The grad norm at step {step} should be approximate to {expected_val} but it is {actual_val}."

    def _compare_grad_norm(self, expected_list, actual_list):
        for step, (expected_val, actual_val) in enumerate(zip(expected_list, actual_list)):
            print(f"Checking step {step + 1} for grad norm")
            assert actual_val == pytest.approx(expected=expected_val, rel=TestMargin.grad_norm),\
            f"The grad norm at step {step} should be approximate to {expected_val} but it is {actual_val}."

    def _compare_pg_loss(self, expected_list, actual_list):
        for step, (expected_val, actual_val) in enumerate(zip(expected_list, actual_list)):
            print(f"Checking step {step + 1} for pg_loss")
            assert actual_val == pytest.approx(expected=expected_val, rel=TestMargin.pg_loss),\
            f"The pg_loss at step {step} should be approximate to {expected_val} but it is {actual_val}."

    def test_entropy(self, baseline_json, generate_log, generate_json):
        self._get_baseline(baseline_json)
        self._get_actual(generate_log, generate_json)
        self._test_helper("entropy")


    def test_grad_norm(self, baseline_json, generate_log, generate_json):
        self._get_baseline(baseline_json)
        self._get_actual(generate_log, generate_json)
        self._test_helper("grad_norm")


    def test_pg_loss(self, baseline_json, generate_log, generate_json):
        self._get_baseline(baseline_json)
        self._get_actual(generate_log, generate_json)
        self._test_helper("pg_loss")

    def test_logp_diff(self, baseline_json, generate_log, generate_json):
        self._get_baseline(baseline_json)
        self._get_actual(generate_log, generate_json)
        self._test_helper("logp_diff")