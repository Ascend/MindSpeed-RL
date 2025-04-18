"""
Answer checker API that uses sympy to simplify expressions and check for equality.

Call grade_answer(given_answer: str, ground_truth: str).

FROM: https://github.com/openai/prm800k/blob/main/prm800k/grading/grader.py
"""
import re
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

from . import math_normalize
from .grader import math_equal
# import math_normalize
# from grader import math_equal

# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
        "liter",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)
    # expr = re.sub(f"\^*\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    # expr = expr.replace(" ", "")

    # # if we somehow still have latex braces here, just drop them
    # expr = expr.replace("{", "")
    # expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    """
    if given_answer is None:
        return False

    ground_truth_normalized_mathd = math_normalize.normalize_answer(ground_truth)
    given_answer_normalized_mathd = math_normalize.normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def _last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        left_brace_idx = None
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
                if left_brace_idx is None:
                    left_brace_idx = i
            elif string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break

            i += 1
        
        if left_brace_idx is None or right_brace_idx is None:
            return None

        return string[left_brace_idx + 1: right_brace_idx].strip()

def match_answer(response):
    """优先从<answer>标签中提取答案，如果没有则从</think>后提取最后一个boxed内容"""
    try:
        # 首先尝试从<answer>标签中提取
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = list(re.finditer(answer_pattern, response, re.DOTALL))
        
        if matches and matches[-1].group(1).strip():
            # 取最后一个<answer>标签内容
            answer_content = matches[-1].group(1).strip()
            
            # 尝试从answer内容中提取boxed内容
            boxed_content = _last_boxed_only_string(answer_content)
            if boxed_content:
                return True, boxed_content
            
            # 如果没有boxed内容，返回整个answer内容
            return True, answer_content
        
        # 如果没有<answer>标签或内容为空，尝试从</think>后提取
        think_end_pos = response.find('</think>')
        if think_end_pos > 0:
            # 只处理</think>后的内容
            after_think = response[think_end_pos + len('</think>'):].strip()
            
            # 从after_think中提取最后一个boxed内容
            boxed_content = _last_boxed_only_string(after_think)
            if boxed_content:
                return True, boxed_content
        
        # 如果都没有找到，返回失败
        return False, ""
        
    except Exception as e:
        print(f"Error in match_answer: {str(e)}")
        return False, ""

def normalize_math_answer(answer):
    """简化版数学答案标准化，只处理单位和基本格式"""
    if not answer:
        return answer
    
    # 格式化答案，移除多余空格
    answer = answer.strip()
    
    # 移除单位文本
    for unit in ["square feet", "feet", "foot", "meters", "meter", "cm", "mm", "km", "inches", "inch"]:
        if answer.endswith(unit):
            answer = answer[:-len(unit)].strip()
    
    return answer

import math
import threading
def limited_time_execution(timeout=10.0):
    """全局超时保护机制，不依赖于信号机制"""
    
    class TimeoutError(Exception):
        pass
    
    def check_timeout():
        thread_id = threading.current_thread().ident
        if thread_id in timeout_flags and timeout_flags[thread_id]:
            raise TimeoutError("Operation timed out")
    
    timeout_flags = {}
    thread_id = threading.current_thread().ident
    timeout_flags[thread_id] = False
    
    def set_timeout_flag():
        timeout_flags[thread_id] = True
    
    timer = threading.Timer(timeout, set_timeout_flag)
    timer.daemon = True  # 确保线程不会阻止程序退出
    timer.start()
    
    try:
        yield check_timeout
    finally:
        timer.cancel()
        if thread_id in timeout_flags:
            del timeout_flags[thread_id]

def evaluate_math(model_output: str, ground_truth: str, timeout=5.0) -> bool:
    model_output = str(model_output)
    ground_truth = str(ground_truth)
    
    # 提取答案
    try:
        is_matched, extracted_model_output = match_answer(model_output)
    except Exception as e:
        print(f"Error extracting answer: {str(e)}")
        return False, "Error extracting answer"
    
    if not is_matched or not extracted_model_output:
        return False, "No answer found"
    
    # 标准化答案进行简单比较和调试
    try:
        normalized_model_answer = normalize_math_answer(extracted_model_output)
        normalized_ground_truth = normalize_math_answer(ground_truth)
        
        # 打印标准化后的答案用于调试
        print(f"Normalized model answer: '{normalized_model_answer}'")
        print(f"Normalized ground truth: '{normalized_ground_truth}'")
    except Exception as e:
        print(f"Error in answer normalization: {str(e)}")
    
    # 使用grade_answer进行等价性判断
    try:
        if grade_answer(extracted_model_output, ground_truth):
            return True, extracted_model_output
    except Exception as e:
        print(f"Error in grade_answer: {str(e)}")
    
    # 如果grade_answer失败，返回False
    return False, extracted_model_output

