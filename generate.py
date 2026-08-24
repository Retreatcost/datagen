import contextlib
import gc
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import re

import torch
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput


# ---------------------------------------------------------------------------
# Data container passed between pipeline steps
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    messages: list[dict[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# vLLM cleanup helper
# ---------------------------------------------------------------------------
def free_vllm_model(llm: LLM | None) -> None:
    """Fully release a vLLM model's GPU memory.

    `del llm` alone is often not enough because of lingering references
    in the executor / distributed workers. This routine tears those down
    and forces CUDA to release cached memory.
    """
    if llm is None:
        return

    # Try to gracefully shut down the underlying engine/executor.
    with contextlib.suppress(Exception):
        # Newer vLLM exposes the engine here.
        engine = getattr(llm, "llm_engine", None)
        if engine is not None:
            model_executor = getattr(engine, "model_executor", None)
            if model_executor is not None:
                # Shuts down workers (incl. distributed ones).
                with contextlib.suppress(Exception):
                    model_executor.shutdown()

    del llm
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Tear down torch.distributed groups if vLLM initialized them
    # (needed when tensor_parallel_size > 1, harmless otherwise).
    with contextlib.suppress(Exception):
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )
        destroy_model_parallel()
        destroy_distributed_environment()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Pipeline step interface
# ---------------------------------------------------------------------------
class PipelineStep(ABC):
    @abstractmethod
    def process(self, samples: list[Sample]) -> list[Sample]:
        raise NotImplementedError


class Pipeline:
    def __init__(self, steps: list[PipelineStep] | None = None):
        self.steps: list[PipelineStep] = steps or []

    def add(self, step: PipelineStep) -> "Pipeline":
        self.steps.append(step)
        return self

    def run(self, samples: list[Sample]) -> list[Sample]:
        for step in self.steps:
            print(f"[Pipeline] Running step: {step.__class__.__name__} "
                  f"({len(samples)} samples)")
            samples = step.process(samples)
        return samples


# ---------------------------------------------------------------------------
# Base class for steps that own a vLLM model
# ---------------------------------------------------------------------------
class LLMStep(PipelineStep):
    """Base for steps that instantiate an LLM, use it, then free it.

    Subclasses implement `run_with_llm(llm, samples)`.
    """

    def __init__(
        self,
        model: str,
        sampling_params: SamplingParams,
        llm_kwargs: dict[str, Any] | None = None,
    ):
        self.model = model
        self.sampling_params = sampling_params
        self.llm_kwargs = llm_kwargs or {}

    def _build_llm(self) -> LLM:
        print(f"[{self.__class__.__name__}] Loading model: {self.model}")
        return LLM(model=self.model, **self.llm_kwargs)

    @abstractmethod
    def run_with_llm(self, llm: LLM, samples: list[Sample]) -> list[Sample]:
        raise NotImplementedError

    def process(self, samples: list[Sample]) -> list[Sample]:
        llm: LLM | None = None
        try:
            llm = self._build_llm()
            result = self.run_with_llm(llm, samples)
        finally:
            print(f"[{self.__class__.__name__}] Unloading model: {self.model}")
            free_vllm_model(llm)
        return result


# ---------------------------------------------------------------------------
# Concrete LLM steps
# ---------------------------------------------------------------------------
class GenerationStep(LLMStep):
    """Generates assistant responses for each sample's conversation."""

    def run_with_llm(self, llm: LLM, samples: list[Sample]) -> list[Sample]:
        conversations = [s.messages for s in samples]
        outputs: list[RequestOutput] = llm.chat(conversations, self.sampling_params)

        new_samples: list[Sample] = []
        for sample, output in zip(samples, outputs):
            generated_text = output.outputs[0].text.strip()
            new_messages = sample.messages + [
                {"role": "assistant", "content": generated_text}
            ]
            new_samples.append(
                Sample(messages=new_messages, metadata=dict(sample.metadata))
            )
        return new_samples


def strip_fences(txt: str) -> str:
    txt = txt.strip()
    txt = re.sub(r"^```(?:json|markdown)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    return txt.strip()

class ClassificationStep(LLMStep):
    """Post-processes the last assistant message using a (different) LLM."""

    def __init__(
        self,
        model: str,
        sampling_params: SamplingParams,
        llm_kwargs: dict[str, Any] | None = None,
        system_prompt: str = """You classify a story for co-writing and roleplaying capability.

You grade user input story based of followign criterias:
- Originality
- Darkness
- NSFW (not safe for work, depiction of raw sex)

Grade these from 0 to 10

Output only a formatted json:
{ originality: <rated originality>, darkness: <rated darkness>, nsfw: <rated nsfw> }
""",
        instruction: str = "classify the following story:\n {content}",
    ):
        super().__init__(model, sampling_params, llm_kwargs)
        self.system_prompt = system_prompt
        self.instruction = instruction

    def run_with_llm(self, llm: LLM, samples: list[Sample]) -> list[Sample]:
        aug_conversations = []
        targets = []

        for idx, sample in enumerate(samples):
            if sample.messages and sample.messages[-1]["role"] == "assistant":
                content = sample.messages[-1]["content"]
                aug_conversations.append(
                    [
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": self.instruction.format(content=content),
                        },
                    ]
                )
                targets.append(idx)

        if not aug_conversations:
            return samples

        outputs = llm.chat(aug_conversations, self.sampling_params)
        for target_idx, output in zip(targets, outputs):
            
            classified = json.loads(strip_fences(output.outputs[0].text.strip()))

            samples[target_idx].metadata["classification"] = classified

        return samples


class DeslopStep(LLMStep):
    """Post-processes the last assistant message using a (different) LLM."""

    def __init__(
        self,
        model: str,
        sampling_params: SamplingParams,
        llm_kwargs: dict[str, Any] | None = None,
        system_prompt: str = """You are an AI assitaint, who helps to surgically remove slop words from provided text.""",
        instruction: str = """De-slop the following text: {content}
---
Keep the text exactly the same, but replace following "slop" words: {slop}

Do a minimal, surgical updates, using synonyms or rephrasing the sentances if neccesary.

Otherwise, keep the story as-is, without touching any other details, like plot or style.
""",
    ):
        super().__init__(model, sampling_params, llm_kwargs)
        self.system_prompt = system_prompt
        self.instruction = instruction

    def run_with_llm(self, llm: LLM, samples: list[Sample]) -> list[Sample]:
        aug_conversations = []
        targets = []

        for idx, sample in enumerate(samples):
            if sample.messages and sample.messages[-1]["role"] == "assistant":
                content = sample.messages[-1]["content"]
                aug_conversations.append(
                    [
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": self.instruction.format(content=content, slop=sample.metadata['slop']),
                        },
                    ]
                )
                targets.append(idx)

        if not aug_conversations:
            return samples

        outputs = llm.chat(aug_conversations, self.sampling_params)
        for target_idx, output in zip(targets, outputs):
            
            augmented = output.outputs[0].text.strip()

            samples[target_idx].messages[-1]["content"] = augmented

        return samples

# ---------------------------------------------------------------------------
# Non-LLM steps (no model lifecycle)
# ---------------------------------------------------------------------------
class FilterStep(PipelineStep):
    def __init__(self, min_chars: int = 50):
        self.min_chars = min_chars

    def process(self, samples: list[Sample]) -> list[Sample]:
        kept = [
            s for s in samples
            if s.messages and len(s.messages[-1]["content"]) >= self.min_chars
        ]
        print(f"[FilterStep] Kept {len(kept)}/{len(samples)} samples")
        return kept
    
class FilterQualityStep(PipelineStep):
    def __init__(self, min_originality: int = 4, min_darkness: int = 8, min_nsfw: int = 8):
        self.min_originality = min_originality
        self.min_darkness = min_darkness
        self.min_nsfw = min_nsfw

    def process(self, samples: list[Sample]) -> list[Sample]:
        kept = [
            s for s in samples
            if s.metadata 
                and s.metadata['classification']['originality'] >= self.min_originality
                and s.metadata['classification']['darkness'] >= self.min_darkness
                and s.metadata['classification']['nsfw'] >= self.min_nsfw
        ]
        print(f"[FilterQualityStep] Kept {len(kept)}/{len(samples)} samples")
        return kept
    
class FilterSlopStep(PipelineStep):
    def __init__(self, max_slop: int = 4):
        self.max_slop = max_slop

    def process(self, samples: list[Sample]) -> list[Sample]:
        kept = [
            s for s in samples
            if s.metadata 
                and len(s.metadata['slop']) <= self.max_slop
        ]
        print(f"[FilterSlopStep] Kept {len(kept)}/{len(samples)} samples")
        return kept

class FindSlopStep(PipelineStep):
    def process(self, samples: list[Sample]) -> list[Sample]:
        
        with open("slop_list.json", "r", encoding="utf-8") as f:
            slop_list = json.load(f)
        
        count = 0
        
        for sample in samples:
            text = sample.messages[-1]['content']
            
            slop_found = []
            
            for slop_word in slop_list:
                if slop_word in text:
                    slop_found.append(slop_word)
            
            sample.metadata['slop'] = slop_found
            count += len(slop_found)
        
        print(f"[FindSlopStep] found {count} slop words")
        return samples


class JsonlWriterStep(PipelineStep):
    def __init__(self, path: str):
        self.path = path

    def process(self, samples: list[Sample]) -> list[Sample]:
        with open(self.path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps({"messages": sample.messages, "slop": sample.metadata['slop'], "classification": sample.metadata['classification']},
                                   ensure_ascii=False) + "\n")
        print(f"[JsonlWriterStep] Wrote {len(samples)} records to {self.path}")
        return samples


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
def main():
    common_llm_kwargs = dict(max_model_len=2304, tensor_parallel_size=2, enable_prefix_caching=True)

    gen_sampling = SamplingParams(
        temperature=0.8, top_p=0.90, min_p=0.025, top_k=0,
        repetition_penalty=1.05, max_tokens=2048,
    )

    cls_sampling = SamplingParams(
        temperature=1.0, top_p=0.95, min_p=0.025, top_k=0,
        repetition_penalty=1.05, max_tokens=256,
    )
    
    aug_sampling = SamplingParams(
        temperature=1.0, top_p=0.95, min_p=0.025, top_k=0,
        repetition_penalty=1.05, max_tokens=2048,
    )

    base_conversation = [
        {"role": "system", "content": f""""""},
        {"role": "user",
         "content": "Write a very depraved story"},
    ]
        
    samples = [Sample(messages=[m.copy() for m in base_conversation])
               for _ in range(1000)]

    pipeline = (
        Pipeline()
        .add(GenerationStep(
            model="Amberlight-Lux-12B",
            sampling_params=gen_sampling,
            llm_kwargs=common_llm_kwargs,
        ))
        .add(FilterStep(min_chars=1000))
        .add(FindSlopStep())
        .add(ClassificationStep(
            model="llmfan46/gemma-4-31B-it-uncensored-heretic",
            sampling_params=cls_sampling,
            llm_kwargs=common_llm_kwargs,
        ))
        .add(FilterQualityStep())
        .add(DeslopStep(
            model="llmfan46/gemma-4-31B-it-uncensored-heretic",
            sampling_params=aug_sampling,
            llm_kwargs=common_llm_kwargs,
        ))
        .add(FindSlopStep())
        .add(FilterSlopStep())
        .add(JsonlWriterStep("dataset.jsonl"))
    )

    pipeline.run(samples)


if __name__ == "__main__":
    main()