from .utils import check_correctness as apps_check_correctness
import json
import re
import traceback

# def evaluate_code(completion, test_cases):
#     # try to get code solution from completion. if the completion is pure code, this will not take effect.
#     solution = completion.split('```python')[-1].split('```')[0]
#     try:
#         try:
#             if not isinstance(test_cases, dict):
#                 test_cases = json.loads(test_cases)
#         except Exception as e:
#             print(f"Error:{e}")

#         # 先检查正确性，如果正确，则再one by one 检查test case
#         try:
#             res, metadata = apps_check_correctness(
#                 in_outs=test_cases,
#                 generation=solution,
#                 timeout=5,
#                 debug=False
#                 )
#             metadata = dict(enumerate(metadata))[0]
#             success = all(map(lambda x: x == True, res))
#             if success:
#                 return success, metadata
#         except Exception as e:
#             pass

#         test_cases_list = []
#         inputs = test_cases["inputs"]
#         outputs = test_cases["outputs"]
#         for i in range(len(inputs)):
#             test_cases_list.append({
#                 "inputs": [inputs[i]],
#                 "outputs": [outputs[i]]
#             })

#         metadata_list = []
#         res_list = []
#         for test_case_id, test_case in enumerate(test_cases_list):
#             res, metadata = apps_check_correctness(
#                 in_outs=test_case,
#                 generation=solution,
#                 timeout=5,
#                 debug=False
#             )
#             try:
#                 metadata = dict(enumerate(metadata))[0] # 运算失败时metadata有可能为空
#             except Exception as e:
#                 metadata={}
#             metadata["test_case"] = {}
#             metadata["test_case"]["input"] = str(test_case["inputs"][0])
#             metadata["test_case"]["output"] = str(test_case["outputs"][0])
#             metadata["test_case"]["res"] = str(res)
#             metadata_list.append(metadata)
#             res_list.extend(res)

#             if test_case_id>=9:
#                 break

#         success = all(map(lambda x: x == True, res_list))
#     except Exception as e:
#         traceback.print_exc(10)
#         success = False
#         metadata_list = None
#     return success, metadata_list
    
import concurrent.futures
import json
import traceback
from functools import partial

def evaluate_code(completion, test_cases):
    # 提取代码解决方案
    solution = completion.split('```python')[-1].split('```')[0].strip()
    
    try:
        # 解析测试用例
        if not isinstance(test_cases, dict):
            test_cases = json.loads(test_cases)
            
        # 预检查完整测试用例集
        try:
            res, metadata = apps_check_correctness(
                in_outs=test_cases,
                generation=solution,
                timeout=5,
                debug=False
            )
            if all(map(lambda x: x == True, res)):
                return True, dict(enumerate(metadata))[0]
        except:
            pass

        # 准备并行测试用例
        test_cases_list = [{
            "inputs": [inputs],
            "outputs": [outputs]
        } for inputs, outputs in zip(test_cases["inputs"], test_cases["outputs"])][:10]  # 保持前10个用例的限制

        # 并行处理函数
        def process_test_case(test_case, solution):
            try:
                res, metadata = apps_check_correctness(
                    in_outs=test_case,
                    generation=solution,
                    timeout=5,
                    debug=False
                )
                metadata = dict(enumerate(metadata))[0] if metadata else {}
            except Exception as e:
                return {"error": str(e), "test_case": test_case}
            
            return {
                "metadata": {
                    **metadata,
                    "test_case": {
                        "input": str(test_case["inputs"][0]),
                        "output": str(test_case["outputs"][0]),
                        "res": str(res)
                    }
                },
                "result": res
            }

        # 使用线程池并行执行
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_test_case, tc, solution) for tc in test_cases_list]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # 处理结果
        res_list = []
        metadata_list = []
        for result in results:
            if "error" in result:
                print(f"Error in test case: {result['error']}")
                continue
            res_list.extend(result["result"])
            metadata_list.append(result["metadata"])
        success = all(map(lambda x: x == True, res_list))
        return success, metadata_list

    except Exception as e:
        traceback.print_exc()
        return False, []

if __name__ == "__main__":
    pass