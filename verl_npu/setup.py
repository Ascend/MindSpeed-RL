import os
import sys
import sysconfig
import subprocess
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


# 插件注入逻辑
def inject_verl_plugin(custom_path=None):
    """将NPU加速支持注入到verl包中"""
    print("Starting verl plugin injection...")
    
    # 优先级：环境变量 > 自定义路径 > 自动查找
    if 'VERL_PATH' in os.environ:
        verl_path = os.path.join(os.environ['VERL_PATH'], "verl")
        print(f"Using verl path from environment variable: {verl_path}")
    elif custom_path:
        verl_path = custom_path
        print(f"Using custom verl path: {verl_path}")
    else:
        print("Searching for verl package automatically...")
        # 尝试多种方式查找verl安装路径
        paths_to_try = [
            sysconfig.get_paths()["purelib"],
            sysconfig.get_paths()["platlib"],
        ] + sys.path  # 搜索所有Python路径
        
        verl_path = None
        for path in paths_to_try:
            if not path:
                continue
                
            candidate = os.path.join(path, "verl")
            if os.path.exists(candidate) and os.path.isdir(candidate):
                verl_path = candidate
                break
        
        # 使用pip show作为备用方案
        if not verl_path:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "show", "verl"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                for line in result.stdout.splitlines():
                    if line.startswith("Location:"):
                        verl_path = os.path.join(line.split(": ")[1], "verl")
                        break
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"pip show failed: {e}")
    
    if not verl_path:
        print("Error: verl package not found. Please specify with VERL_PATH environment variable.")
        return False
    
    print(f"Found verl at: {verl_path}")
    
    # 1. 修改 __init__.py 文件
    init_file = os.path.join(verl_path, "__init__.py")
    if not os.path.exists(init_file):
        print(f"Error: verl initialization file not found: {init_file}")
        return False
    
    # 检查是否已经注入过
    import_content = """
# NPU acceleration support added by MindSpeed RL plugin
from verl.utils.device import is_npu_available

if is_npu_available:
    import verl_npu
    print("NPU acceleration enabled for verl")
"""
    
    # 读取当前内容
    try:
        with open(init_file, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {init_file}: {e}")
        return False
    
    if import_content in content:
        print(f"Info: {init_file} already contains NPU acceleration import")
    else:
        # 添加注入内容
        try:
            with open(init_file, "a") as f:
                f.write(import_content)
            print(f"Successfully modified {init_file} to add NPU acceleration support")
        except Exception as e:
            print(f"Error writing to {init_file}: {e}")
            return False
    
    # 2. 修改 linear_cross_entropy.py 文件
    linear_cross_entropy_file = os.path.join(verl_path, "utils", "kernel", "linear_cross_entropy.py")
    if not os.path.exists(linear_cross_entropy_file):
        print(f"Warning: linear_cross_entropy file not found: {linear_cross_entropy_file}")
        return True
    
    # 需要注释的行
    line_to_comment = "from . import kernels"
    
    try:
        with open(linear_cross_entropy_file, "r") as f:
            lines = f.readlines()
        
        modified = False
        new_lines = []
        for line in lines:
            # 检查是否是需要注释的行（并且尚未被注释）
            if line.strip() == line_to_comment:
                new_lines.append(f"# {line}")  # 注释掉该行
                print(f"Commented out line in {linear_cross_entropy_file}: {line.strip()}")
                modified = True
            else:
                new_lines.append(line)
        
        if modified:
            # 写回修改后的内容
            with open(linear_cross_entropy_file, "w") as f:
                f.writelines(new_lines)
            print(f"Successfully modified {linear_cross_entropy_file}")
        else:
            # 检查是否已经被注释
            already_commented = any(f"# {line_to_comment}" in line for line in lines)
            if already_commented:
                print(f"Info: line already commented in {linear_cross_entropy_file}")
            else:
                print(f"Warning: line to comment not found in {linear_cross_entropy_file}: {line_to_comment}")
    
    except Exception as e:
        print(f"Error modifying {linear_cross_entropy_file}: {e}")
        return False
    
    return True


# vllm patch
def inject_vllm_plugin():
    print("Searching for vllm package automatically...")
    # 尝试多种方式查找vllm安装路径
    paths_to_try = [
        sysconfig.get_paths()["purelib"],
        sysconfig.get_paths()["platlib"],
    ] + sys.path  # 搜索所有Python路径
    
    vllm_path = None
    for path in paths_to_try:
        if not path:
            continue
            
        candidate = os.path.join(path, "vllm")
        if os.path.exists(candidate) and os.path.isdir(candidate):
            vllm_path = candidate
            break
    
    # 使用pip show作为备用方案
    if not vllm_path:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "vllm"],
                capture_output=True,
                text=True,
                check=True
            )
            for line in result.stdout.splitlines():
                if line.startswith("Location:"):
                    vllm_path = os.path.join(line.split(": ")[1], "vllm")
                    break
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"pip show failed: {e}")
    
    if not vllm_path:
        print("Error: vllm package not found. Please specify with VLLM_PATH environment variable.")
        return False
    
    print(f"Found vllm at: {vllm_path}")

    fp8_utils_file = os.path.join(vllm_path, "model_executor", "layers", "quantization", "utils", "fp8_utils.py")
    if not os.path.exists(fp8_utils_file):
        print(f"Warning: linear_cross_entropy file not found: {fp8_utils_file}")
    else:
        file_line_comment = "from typing import Any, Callable"

        try:
            with open(fp8_utils_file, "r") as f:
                lines = f.readlines()
            
            modified = False
            new_lines = []
            for line in lines:
                if file_line_comment in line.strip() and 'List' not in line.strip():
                    new_lines.append(f"{line[:-1]}, List\n")
                    print(f"Commented out line in {fp8_utils_file}: {line.strip()}")
                    modified = True
                elif 'list' in line:
                    new_lines.append(line.replace('list', 'List'))
                    modified = True
                else:
                    new_lines.append(line)
            
            if modified:
                # 写回修改后的内容
                with open(fp8_utils_file, "w") as f:
                    f.writelines(new_lines)
                print(f"Successfully modified {fp8_utils_file}")
            else:
                already_commented = any(f"List" in line for line in lines)
                if already_commented:
                    print(f"Info: line already commented in {fp8_utils_file}")
                else:
                    print(f"Warning: line to comment not found in {fp8_utils_file}: {file_line_comment}")
        except Exception as e:
            print(f"Error modifying {fp8_utils_file}: {e}")
            return False

    fused_moe_file = os.path.join(vllm_path, "model_executor", "layers", "fused_moe", "fused_moe.py")
    if not os.path.exists(fused_moe_file):
        print(f"Warning: linear_cross_entropy file not found: {fused_moe_file}")
    else:
        file_line_comment = "from typing import Any, Callable"

        try:
            with open(fused_moe_file, "r") as f:
                lines = f.readlines()
            
            modified = False
            new_lines = []
            for line in lines:
                if file_line_comment in line.strip() and 'List' not in line.strip():
                    new_lines.append(f"{line[:-1]}, List\n")
                    print(f"Commented out line in {fused_moe_file}: {line.strip()}")
                    modified = True
                elif 'list' in line:
                    new_lines.append(line.replace('list', 'List'))
                    modified = True
                else:
                    new_lines.append(line)
            
            if modified:
                # 写回修改后的内容
                with open(fused_moe_file, "w") as f:
                    f.writelines(new_lines)
                print(f"Successfully modified {fused_moe_file}")
            else:
                already_commented = any(f"List" in line for line in lines)
                if already_commented:
                    print(f"Info: line already commented in {fused_moe_file}")
                else:
                    print(f"Warning: line to comment not found in {fused_moe_file}: {file_line_comment}")
        except Exception as e:
            print(f"Error modifying {fused_moe_file}: {e}")
            return False
    return True


# 自定义安装命令
class CustomInstallCommand(install):
    """自定义安装命令"""
    def run(self):
        super().run()
        print("Running verl injection after standard install...")
        # 尝试从环境变量获取路径
        custom_path = os.environ.get('VERL_PATH', None)
        if not inject_verl_plugin(custom_path):
            print("Error: verl injection failed. Please check installation.")
        if not inject_vllm_plugin():
            print("Error: vllm injection failed. Please check installation.")


# 自定义开发模式安装命令
class CustomDevelopCommand(develop):
    """自定义开发模式安装命令"""
    def run(self):
        super().run()
        print("Running verl injection after develop install...")
        # 尝试从环境变量获取路径
        custom_path = os.environ.get('VERL_PATH', None)
        if not inject_verl_plugin(custom_path):
            print("Error: verl injection failed. Please check installation.")
        if not inject_vllm_plugin():
            print("Error: vllm injection failed. Please check installation.")


# 主安装函数
def main():
    print("Setting up verl_npu plugin...")
    
    setup(
        name="verl_npu",
        version="0.0.1",
        license="Apache 2.0",
        description="verl npu backend plugin",
        packages=find_packages(include=["verl_npu"]),
        classifiers=[
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "License :: OSI Approved :: Apache Software License",
            "Intended Audience :: Developers",
            "Intended Audience :: Information Technology",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Information Analysis",
        ],
        python_requires=">=3.9",
        cmdclass={
            'install': CustomInstallCommand,
            'develop': CustomDevelopCommand,
        },
    )


if __name__ == '__main__':
    main()