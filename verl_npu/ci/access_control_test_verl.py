import os
import subprocess
import shlex
from pathlib import Path


def read_files_from_txt(txt_file):
    with open(txt_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def is_markdown(file):
    return file.endswith(".md") or file.endswith(".MD")


def is_image(file):
    return file.endswith(".jpg") or file.endswith(".png")


def is_txt(file):
    return file.endswith(".txt")


def is_no_suffix(file):
    return os.path.splitext(file)[1] == ''


def in_verl(file):
    return file.startswith("verl_npu/")


def skip_ci(files, skip_conds):
    for file in files:
        # 该文件不满足任何跳过条件  而且该文件在verl-npu里
        if in_verl(file) and (not any(condition(file) for condition in skip_conds)):
            return False
    return True


def choose_skip_ci(raw_txt_file):
    if not os.path.exists(raw_txt_file):
        return False

    file_list = read_files_from_txt(raw_txt_file)
    skip_conds = [
        is_markdown,
        is_image,
        is_txt,
        is_no_suffix
    ]

    return skip_ci(file_list, skip_conds)


def acquire_exitcode(command):
    """不使用 shell 的更安全版本（推荐用于处理用户输入）"""
    args = shlex.split(command)
    process = subprocess.Popen(
        args,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output, end='', flush=True)
    
    # 等待进程结束
    return process.wait()

# ===============================================
# ST test, run with sh.
# ===============================================


class STTest:
    def __init__(self):
        self.base_dir = Path(__file__).absolute().parents[1]
        self.test_dir = os.path.join(self.base_dir, 'verl_tests')

        self.st_dir = "st"
        self.st_shell = os.path.join(
            self.test_dir, self.st_dir, "st_run.sh"
        )

    def run_st(self):
        rectify_case = f"bash {self.st_shell}"
        rectify_code = acquire_exitcode(rectify_case)
        if rectify_code != 0:
            print("rectify case failed, check it.")
            exit(1)


def run_tests(raw_txt_file):
    st = STTest()
    st.run_st()


def main():
    parent_dir = Path(__file__).absolute().parents[2]
    raw_txt_file = os.path.join(parent_dir, "modify.txt")

    skip_signal = choose_skip_ci(raw_txt_file)
    if skip_signal:
        print("Skipping CI")
    else:
        run_tests(raw_txt_file)


if __name__ == "__main__":
    main()
