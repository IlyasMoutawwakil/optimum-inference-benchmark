from dataclasses import dataclass
from logging import getLogger
from typing import Dict, List
import gc

import torch
from torch import Tensor
from optimum.exporters import TasksManager
from transformers.utils.fx import symbolic_trace
from transformers import AutoConfig, GenerationMixin
from optimum.bettertransformer import BetterTransformer

from src.backend.base import Backend, BackendConfig
from src.profiler.fx_profiler import FXProfilingWrapper
from src.utils import get_used_memory

BACKEND_NAME = "pytorch"
LOGGER = getLogger(BACKEND_NAME)


@dataclass
class PyTorchConfig(BackendConfig):
    name: str = BACKEND_NAME
    version: str = torch.__version__

    # inference options
    disable_grad: bool = False
    eval_mode: bool = False

    # graph optimization options
    fp16: bool = False
    bettertransformer: bool = False
    torch_compile: bool = False


class PyTorchBackend(Backend):
    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)
        self.model_type = AutoConfig.from_pretrained(self.model).model_type
        self.automodel_class = TasksManager.get_model_class_for_task(
            task=self.task, model_type=self.model_type
        )
        self.fp16 = False

    def configure(self, config: PyTorchConfig) -> None:
        LOGGER.info("Configuring pytorch Backend:")
        super().configure(config)

        # Torch specific environment variables
        if config.inter_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting pytorch inter_op_num_threads({config.inter_op_num_threads}))"
            )
            torch.set_num_threads(config.inter_op_num_threads)

        if config.intra_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting pytorch intra_op_num_threads({config.intra_op_num_threads}))"
            )
            torch.set_num_interop_threads(config.intra_op_num_threads)

        # Disable gradients
        if not config.disable_grad or config.eval_mode:
            LOGGER.info("\t+ Disabling gradients")
            torch.set_grad_enabled(False)

        # Load model
        LOGGER.info(f"\t+ Loading {self.model} with {self.automodel_class.__name__}")
        self.pretrained_model = self.automodel_class.from_pretrained(self.model)

        # Move model to device
        if self.pretrained_model.device.type != self.device:
            LOGGER.info(f"\t+ Moving model to {self.device}")
            self.pretrained_model.to(self.device)

        LOGGER.info(f"\t+ Device used memory: {get_used_memory(device=self.device)}")

        # Turn on eval mode
        if config.eval_mode:
            LOGGER.info("\t+ Turning on eval mode")
            self.pretrained_model.eval()

        # Turn on better transformer inference
        if config.bettertransformer:
            LOGGER.info("\t+ Using optimum.bettertransformer")
            self.pretrained_model = BetterTransformer.transform(  # type: ignore
                self.pretrained_model, keep_original_model=False
            )
            LOGGER.info(
                f"\t+ Device used memory: {get_used_memory(device=self.device)}"
            )

        # Compile model
        if config.torch_compile:
            LOGGER.info("\t+ Using torch.compile")
            self.pretrained_model.forward = torch.compile(self.pretrained_model.forward)
            LOGGER.info(
                f"\t+ Device used memory: {get_used_memory(device=self.device)}"
            )

        # Turn on fp16
        if config.fp16:
            LOGGER.info("\t+ Turning on fp16")
            self.fp16 = True

    def forward(self, input: Dict[str, Tensor]):
        with torch.cuda.amp.autocast(enabled=self.fp16):  # type: ignore
            output = self.pretrained_model(**input)
        return output

    @property
    def is_generator(self) -> bool:
        return isinstance(self.pretrained_model, GenerationMixin)

    def generate(self, input: Dict[str, Tensor]) -> None:
        with torch.cuda.amp.autocast(enabled=self.fp16):  # type: ignore
            output = self.pretrained_model.generate(  # type: ignore
                **input,
                max_new_tokens=self.pretrained_model.config.max_length,  # type: ignore
                pad_token_id=self.pretrained_model.config.eos_token_id,  # type: ignore
            )
        return output

    def prepare_for_profiling(self, input_names: List[str]) -> None:
        LOGGER.info("\t+ Symbolic tracing model")
        self.pretrained_model = symbolic_trace(
            model=self.pretrained_model,  # type: ignore
            input_names=input_names,
        )
        LOGGER.info("\t+ Wrapping model with profiler")
        self.pretrained_model = FXProfilingWrapper(self.pretrained_model)

    def clean(self) -> None:
        del self.pretrained_model
        gc.collect()
        torch.cuda.empty_cache()
