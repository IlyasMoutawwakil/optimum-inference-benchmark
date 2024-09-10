import os
import traceback
from itertools import product
from logging import getLogger

from llm_perf.utils import (
    CANONICAL_PRETRAINED_OPEN_LLM_LIST,
    GENERATE_KWARGS,
    INPUT_SHAPES,
    OPEN_LLM_LIST,
    PRETRAINED_OPEN_LLM_LIST,
    is_benchmark_conducted,
)
from optimum_benchmark import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkReport,
    InferenceConfig,
    ProcessConfig,
    PyTorchConfig,
)
from optimum_benchmark.logging_utils import setup_logging

SUBSET = os.getenv("SUBSET", None)
MACHINE = os.getenv("MACHINE", None)
BACKEND = "pytorch"
HARDWARE = "cpu"

if os.getenv("MACHINE", None) is None and os.getenv("SUBSET", None) is None:
    PUSH_REPO_ID = f"optimum-benchmark/llm-perf-{BACKEND}-{HARDWARE}-debug"
    CANONICAL_PRETRAINED_OPEN_LLM_LIST = ["gpt2"]
    SUBSET = "unquantized"
elif os.getenv("MACHINE", None) is not None and os.getenv("SUBSET", None) is not None:
    PUSH_REPO_ID = f"optimum-benchmark/llm-perf-{BACKEND}-{HARDWARE}-{SUBSET}-{MACHINE}"
else:
    raise ValueError("Either both MACHINE and SUBSET should be set for benchmarking or neither for debugging")

ATTENTION_CONFIGS = ["eager", "sdpa"]


if SUBSET == "unquantized":
    WEIGHTS_CONFIGS = {
        # unquantized
        "float32": {"torch_dtype": "float32", "quant_scheme": None, "quant_config": {}},
        "float16": {"torch_dtype": "float16", "quant_scheme": None, "quant_config": {}},
        "bfloat16": {"torch_dtype": "bfloat16", "quant_scheme": None, "quant_config": {}},
    }
else:
    raise ValueError(f"Subset {SUBSET} not supported")


LOGGER = getLogger("llm-perf-backend")
LOGGER.info(f"len(OPEN_LLM_LIST): {len(OPEN_LLM_LIST)}")
LOGGER.info(f"len(PRETRAINED_OPEN_LLM_LIST): {len(PRETRAINED_OPEN_LLM_LIST)}")
LOGGER.info(f"len(CANONICAL_PRETRAINED_OPEN_LLM_LIST): {len(CANONICAL_PRETRAINED_OPEN_LLM_LIST)}")


def is_benchmark_supported(weights_config, attn_implementation, hardware):
    if attn_implementation == "flash_attention_2":
        return False

    return True


def benchmark_cpu_pytorch(model, attn_implementation, weights_config):
    benchmark_name = f"{weights_config}-{attn_implementation}-{BACKEND}"
    subfolder = f"{benchmark_name}/{model.replace('/', '--')}"

    torch_dtype = WEIGHTS_CONFIGS[weights_config]["torch_dtype"]
    quant_scheme = WEIGHTS_CONFIGS[weights_config]["quant_scheme"]
    quant_config = WEIGHTS_CONFIGS[weights_config]["quant_config"]

    if not is_benchmark_supported(weights_config, attn_implementation, HARDWARE):
        LOGGER.info(f"Skipping benchmark {benchmark_name} with model {model} since it is not supported")
        return

    if is_benchmark_conducted(PUSH_REPO_ID, subfolder):
        LOGGER.info(f"Skipping benchmark {benchmark_name} with model {model} since it was already conducted")
        return

    launcher_config = ProcessConfig()
    scenario_config = InferenceConfig(
        memory=True,
        energy=True,
        latency=True,
        duration=10,
        iterations=10,
        warmup_runs=10,
        input_shapes=INPUT_SHAPES,
        generate_kwargs=GENERATE_KWARGS,
    )

    backend_config = PyTorchConfig(
        model=model,
        device="cpu",
        no_weights=True,
        library="transformers",
        task="text-generation",
        torch_dtype=torch_dtype,
        quantization_scheme=quant_scheme,
        quantization_config=quant_config,
        attn_implementation=attn_implementation,
        model_kwargs={"trust_remote_code": True},
    )

    benchmark_config = BenchmarkConfig(
        name=benchmark_name, scenario=scenario_config, launcher=launcher_config, backend=backend_config
    )

    benchmark_config.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=subfolder, private=True)

    try:
        LOGGER.info(f"Running benchmark {benchmark_name} with model {model}")
        benchmark_report = Benchmark.launch(benchmark_config)
        benchmark_report.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=subfolder, private=True)
        benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
        benchmark.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=subfolder, private=True)

    except Exception:
        LOGGER.error(f"Benchmark {benchmark_name} failed with model {model}")
        benchmark_report = BenchmarkReport.from_dict({"traceback": traceback.format_exc()})
        benchmark_report.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=subfolder, private=True)
        benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
        benchmark.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=subfolder, private=True)


if __name__ == "__main__":
    # for isolated process
    os.environ["LOG_TO_FILE"] = "0"
    os.environ["LOG_LEVEL"] = "INFO"

    # for main process
    setup_logging(level="INFO", prefix="MAIN-PROCESS")

    models_attentions_weights = list(
        product(CANONICAL_PRETRAINED_OPEN_LLM_LIST, ATTENTION_CONFIGS, WEIGHTS_CONFIGS.keys())
    )

    LOGGER.info(
        f"Running a total of {len(models_attentions_weights)} benchmarks, "
        f"with {len(CANONICAL_PRETRAINED_OPEN_LLM_LIST)} models, "
        f"{len(ATTENTION_CONFIGS)} attentions implementations "
        f"and {len(WEIGHTS_CONFIGS)} weights configurations."
    )

    for model, attn_implementation, weights_config in models_attentions_weights:
        benchmark_cpu_pytorch(model, attn_implementation, weights_config)
