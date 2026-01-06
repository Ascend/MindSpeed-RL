from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
import subprocess
import sys
from pathlib import Path

from packaging.version import Version


def is_available(pkg_name):
    try:
        return bool(get_version(pkg_name))
    except PackageNotFoundError:
        return False


def compare_versions(version1: str, version2: str) -> int:
    """
    Compare two version strings.

    :param version1: First version string.
    :param version2: Second version string.
    :return: -1 if version1 < version2, 0 if version1 == version2, 1 if version1 > version2.
    """
    v1 = Version(version1)
    v2 = Version(version2)
    if v1 < v2:
        return -1
    elif v1 == v2:
        return 0
    else:
        return 1


def is_version_greater_or_equal(package_name: str, target_version: str) -> bool:
    """
    Check if the installed version of a package is greater than or equal to the target version.

    :param package_name: Name of the package.
    :param target_version: Target version to compare against.
    :return: True if the installed version is greater than or equal to the target version, False otherwise.
    """
    installed_version = get_version(package_name)
    return compare_versions(installed_version, target_version) >= 0


def is_version_less(package_name: str, target_version: str) -> bool:
    """
    Check if the installed version of a package is less than the target version.

    :param package_name: Name of the package.
    :param target_version: Target version to compare against.
    :return: True if the installed version is less than the target version, False otherwise.
    """
    installed_version = get_version(package_name)
    return compare_versions(installed_version, target_version) < 0


def is_version_equal(package_name: str, target_version: str) -> bool:
    """
    Check if the installed version of a package is equal to the target version.

    :param package_name: Name of the package.
    :param target_version: Target version to compare against.
    :return: True if the installed version is equal to the target version, False otherwise.
    """
    installed_version = get_version(package_name)
    return compare_versions(installed_version, target_version) == 0


def check_commit_id(target_path: str, version: str) -> bool:
    """
    Check if the installed version of a package is equal to the target version.

    :param target_path: path of the package.
    :param version: Target version to compare against.
    :return: True if the installed version is equal to the target version, False otherwise.
    """
    def get_commit_id(target_path):
        try:
            commit_id = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=target_path, timeout=5
                ).strip().decode('utf-8').replace("\n", "")
            return commit_id
        except subprocess.CalledProcessError as e:
            print(f"Info: get git commit id error, {e}")
            return None
    commit_id = get_commit_id(target_path)

    if commit_id is not None:
        return commit_id == version, commit_id
    return True, None


def get_target_path(repo):
    target_path = None
    repo_meta = subprocess.check_output(
        [sys.executable, "-m", "pip", "show", repo]
    ).decode("utf-8")
    # Prioritize editable install location, since pip show lists both locations
    # if installed in editable mode.
    for line in repo_meta.split("\n"):
        line = line.strip()
        if line.startswith("Editable project location: "):
            target_path = str(Path(line.split(": ")[1]))
            break
    else:
        for line in repo_meta.split("\n"):
            line = line.strip()
            if line.startswith("Location: "):
                target_path = str(Path(line.split(": ")[1]))

    return target_path

