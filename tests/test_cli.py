import os
from logging import getLogger
from subprocess import PIPE, STDOUT, Popen

import pytest

LOGGER = getLogger("test-cli")


CONFIG_NAMES = [
    config.split(".")[0]
    for config in os.listdir("tests/configs")
    if config.endswith(".yaml") and not (config.startswith("_") or config.endswith("_"))
]


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_configs(config_name):
    args = [
        "optimum-benchmark",
        "--config-dir",
        "tests/configs",
        "--config-name",
        config_name,
        "--multirun",
    ]

    popen = run_process_and_stream_output(args)
    assert popen.returncode == 0, popen.stderr


def test_exit_code():
    args_0 = [
        "optimum-benchmark",
        "--config-dir",
        "tests/configs",
        "--config-name",
        "cpu_inference_pytorch_bert_sweep",
        # compatible task and model
        "task=text-classification",
        "model=bert-base-uncased",
    ]

    popen_0 = run_process_and_stream_output(args_0)
    assert popen_0.returncode == 0, popen_0.stderr.decode("utf-8")

    args_1 = [
        "optimum-benchmark",
        "--config-dir",
        "tests/configs",
        "--config-name",
        "cpu_inference_pytorch_bert_sweep",
        # incompatible task and model to trigger error
        "task=image-classification",
        "model=bert-base-uncased",
    ]

    popen_1 = run_process_and_stream_output(args_1)
    assert popen_1.returncode == 1, popen_1.stderr.decode("utf-8")


def run_process_and_stream_output(args):
    popen = Popen(args, stdout=PIPE, stderr=STDOUT)
    for line in iter(popen.stdout.readline, b""):
        if line is not None:
            LOGGER.info(line.decode("utf-8").rstrip())

    popen.wait()
    return popen
