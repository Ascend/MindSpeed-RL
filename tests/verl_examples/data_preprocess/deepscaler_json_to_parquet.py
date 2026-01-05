import json
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def read_json_file(read_file_path):
    try:
        with open(read_file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            return json_data, True
    except FileNotFoundError:
        print(f"[ERROR] File {read_file_path} not found.")
        return None, False
    except json.JSONDecodeError:
        print(f"[ERROR] File {read_file_path} is not a valid JSON file.")
        return None, False
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        return None, False


def read_parquet_file(read_file_path):
    try:
        parquet_table = pq.read_table(read_file_path)
        parquet_df = parquet_table.to_pandas()
        parquet_result = [row for _, row in parquet_df.iterrows()]
        return parquet_result, True
    except FileNotFoundError:
        print(f"[ERROR] Parquet file {read_file_path} not found.")
        return None, False
    except Exception as e:
        print(f"[ERROR] An error occurred when reading Parquet: {e}")
        return None, False


def convert_lists_to_json(input_df):
    """Convert lists in DataFrame to JSON strings."""
    temp_df = input_df.copy()
    for column in temp_df.columns:
        if temp_df[column].apply(lambda x: isinstance(x, list)).any():
            temp_df[column] = temp_df[column].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    return temp_df


def save_parquet(save_file_path, data_to_save):
    if isinstance(data_to_save, list):
        data_to_save = pd.DataFrame(data_to_save)
    if not isinstance(data_to_save, pd.DataFrame):
        raise ValueError("[ERROR] Data must be a pandas DataFrame or a list of lists")
    # Convert lists to JSON strings before saving to Parquet
    data_to_save = convert_lists_to_json(data_to_save)
    pq.write_table(pa.Table.from_pandas(data_to_save), save_file_path)
    print(f'Save {save_file_path} is ok!')

input_file_path = '/path/to/datasets/deepscaler/deepscaler.json'
output_file_path = '/path/to/datasets/deepscaler/deepscaler.parquet'
loaded_json_data, is_success = read_json_file(input_file_path)
if not is_success:
    print("[FATAL] Failed to read JSON file, data processing aborted!")
else:
    template = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Put your final answer within \\boxed{}.\n"

    new_data = []
    for idx, item in enumerate(loaded_json_data):
        new_item = {}
        new_item['data_source'] = "deepscaler"
        new_item['ability'] = "math"
        new_item['prompt'] = []
        tmp = {}
        tmp["content"] = template + item['problem']
        tmp["role"] = "user"

        new_item['prompt'].append(tmp)
        new_item['prompt'] = np.array(new_item['prompt'])
        new_item['reward_model'] = {}
        new_item['reward_model']["style"] = "rule"
        new_item['reward_model']["ground_truth"] = item['answer']
        new_item['reward_model']["solution"] = item['solution']
        new_item['extra_info'] = {}
        new_item['extra_info']["split"] = "train"
        new_item['extra_info']["index"] = idx
        new_data.append(new_item)
    processed_df = pd.DataFrame(new_data)
    save_parquet(output_file_path, processed_df)