# SPDX-License-Identifier: Apache-2.0

import importlib.util
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import setup


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)

# cannot import envs directly because it depends on vllm,
#  which is not installed yet
envs = load_module_from_path('envs', os.path.join(ROOT_DIR, 'vllm', 'envs.py'))

VLLM_TARGET_DEVICE = envs.VLLM_TARGET_DEVICE

if sys.platform.startswith("darwin") and VLLM_TARGET_DEVICE != "cpu":
    logger.warning(
        "VLLM_TARGET_DEVICE automatically set to `cpu` due to macOS")
    VLLM_TARGET_DEVICE = "cpu"
elif not (sys.platform.startswith("linux")
          or sys.platform.startswith("darwin")):
    logger.warning(
        "vLLM only supports Linux platform (including WSL) and MacOS."
        "Building on %s, "
        "so vLLM may not be able to run correctly", sys.platform)
    VLLM_TARGET_DEVICE = "empty"

MAIN_CUDA_VERSION = "12.8"


def _is_hpu() -> bool:
    # if VLLM_TARGET_DEVICE env var was set explicitly, skip HPU autodetection
    if os.getenv("VLLM_TARGET_DEVICE", None) == VLLM_TARGET_DEVICE:
        return VLLM_TARGET_DEVICE == "hpu"

    # if VLLM_TARGET_DEVICE was not set explicitly, check if hl-smi succeeds,
    # and if it doesn't, check if habanalabs driver is loaded
    is_hpu_available = False
    try:
        out = subprocess.run(["hl-smi"], capture_output=True, check=True)
        is_hpu_available = out.returncode == 0
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError):
        if sys.platform.startswith("linux"):
            try:
                output = subprocess.check_output(
                    'lsmod | grep habanalabs | wc -l', shell=True)
                is_hpu_available = int(output) > 0
            except (ValueError, FileNotFoundError, PermissionError,
                    subprocess.CalledProcessError):
                pass
    return is_hpu_available


def _is_neuron() -> bool:
    return VLLM_TARGET_DEVICE == "neuron"


def get_neuronxcc_version():
    import sysconfig
    site_dir = sysconfig.get_paths()["purelib"]
    version_file = os.path.join(site_dir, "neuronxcc", "version",
                                "__init__.py")

    # Check if the command was executed successfully
    with open(version_file) as fp:
        content = fp.read()

    # Extract the version using a regular expression
    match = re.search(r"__version__ = '(\S+)'", content)
    if match:
        # Return the version string
        return match.group(1)
    else:
        raise RuntimeError("Could not find Neuron version in the output")


neuron_ver = os.getenv("NEURON_VERSION", None)
neuronxcc_version = get_neuronxcc_version(
) if neuron_ver is None else neuron_ver


def get_gaudi_sw_version():
    """
    Returns the driver version.
    """
    # Enable console printing for `hl-smi` check
    output = subprocess.run("hl-smi",
                            shell=True,
                            text=True,
                            capture_output=True,
                            env={"ENABLE_CONSOLE": "true"})
    if output.returncode == 0 and output.stdout:
        return output.stdout.split("\n")[2].replace(
            " ", "").split(":")[1][:-1].split("-")[0]
    return "0.0.0"  # when hl-smi is not available


def find_version(filepath: str) -> str:
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(r"__version__ = ['\"]([^'\"]*)['\"]",
                                  fp.read().strip(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def get_vllm_version() -> str:

    version = find_version(os.path.join(ROOT_DIR, 'vllm', 'version.py'))
    if _is_neuron():
        # Get the Neuron version
        neuron_version = str(neuronxcc_version)
        neuron_version_str = neuron_version.replace(".", "")[:3]
        version += f"+neuron{neuron_version_str}"
    return version


def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""
    requirements_dir = ROOT_DIR / "requirements"

    def _read_requirements(filename: str) -> list[str]:
        with open(requirements_dir / filename) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif not line.startswith("--") and not line.startswith(
                    "#") and line.strip() != "":
                resolved_requirements.append(line)
        return resolved_requirements

    if _is_neuron():
        requirements = _read_requirements("neuron.txt")
    else:
        raise ValueError(
            "Unsupported platform, please use CUDA, ROCm, Neuron, HPU, "
            "or CPU.")
    return requirements


ext_modules = []

package_data = {
    "vllm": [
        "py.typed",
        "model_executor/layers/fused_moe/configs/*.json",
        "model_executor/layers/quantization/utils/configs/*.json",
    ]
}

cmdclass = {}

setup(
    # static metadata should rather go in pyproject.toml
    version=get_vllm_version(),
    ext_modules=ext_modules,
    install_requires=get_requirements(),
    extras_require={
        "tensorizer": ["tensorizer>=2.9.0"],
        "fastsafetensors": ["fastsafetensors >= 0.1.10"],
        "runai": ["runai-model-streamer", "runai-model-streamer-s3", "boto3"],
        "audio": ["librosa", "soundfile"],  # Required for audio processing
        "video": []  # Kept for backwards compatibility
    },
    cmdclass=cmdclass,
    package_data=package_data,
    entry_points={
        "console_scripts": [
            "vllm=vllm.scripts:main",
            "neuron-proxy-server=vllm.neuron_immediate_first_token_proxy_server:main"
        ],
    },
)
