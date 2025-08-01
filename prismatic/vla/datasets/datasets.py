"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

########################################################
import random
import json
# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    return_image: bool = False
    use_wrist_image: bool = False
    # use_bidi_attention: bool = False
    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        # print("datasets.py __call__ rlds_batch['action']:", rlds_batch["action"].shape)
        # print(type(rlds_batch["action"])) 
        if rlds_batch["action"].shape[0] > 1:#cjy
            action = rlds_batch["action"].reshape(-1)
        else:
            action = rlds_batch["action"][0]
        dataset_name = rlds_batch["dataset_name"]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        # print("datasets.py __call__ action shape:", action.shape)

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        # if self.use_bidi_attention:
        #     input_ids[-len(action)-1:] = [0] * (len(action)+1)


        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)
        # print("pixel_values:",pixel_values)


        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        if self.return_image:# cjy
            return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name,image=img)
        else:
            return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name,image=None)
        # print("rlds_batch['observation'].keys():",rlds_batch["observation"].keys())
        if self.use_wrist_image:
            # all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    # all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = pixel_values_wrist
            # return_dict["pixel_values_wrist"] = torch.cat(pixel_values_wrist, dim=0)
        return return_dict

class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        window_size: int = 1,
        future_action_window_size: int = 0,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary","wrist"),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:#加载完数据 再batch trasformer处理
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out

def load_jsonl_files_from_directory(directory_path):
    # 获取目录下所有的.jsonl文件
    jsonl_files = [f for f in os.listdir(directory_path) if f.endswith('.jsonl')]

    all_json_data = []

    for jsonl_file in jsonl_files:
        file_path = os.path.join(directory_path, jsonl_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            # 一边读取一边解析 JSON，并加入到总列表
            for line in file:
                line = line.strip()
                if line:  # 忽略空行
                    all_json_data.append(json.loads(line))

    return all_json_data

   
class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        # base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        # prompt_builder_fn: Type[PromptBuilder],
        json_path: Path,
        # dataset_statistics: dict,
    ) -> None:
        self.action_tokenizer = action_tokenizer
        # self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        # self.prompt_builder_fn = prompt_builder_fn
        self.train_json=load_jsonl_files_from_directory(json_path)
        self.length=len(self.train_json)

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        # self.dataset_statistics = dataset_statistics
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return self.length

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values

        json_data=self.train_json[idx]
        image_path=json_data["image_path"]
        image=Image.open(image_path).convert('RGB')
        # 确保图像是 RGB 格式
        # if image.mode != 'RGB':
        #     image = image.convert('RGB')
        image = image.resize((224, 224))
        # image = np.asarray(image, dtype=np.uint8)
        
        input_ids=torch.tensor(json_data["teacher_output_ids"][0],dtype=torch.long)
        teacher_output_ids=input_ids.clone()

        labels=torch.tensor(json_data["labels_ids"][0],dtype=torch.long)
        prompt_ids=torch.tensor(json_data["prompt_ids"][0],dtype=torch.long)
        i=random.randint(0,len(json_data["answer_trajectory_ids"])-1)
        jacobi_ids=torch.cat((prompt_ids,torch.tensor(json_data["answer_trajectory_ids"][i],dtype=torch.long)),dim=0)
        pixel_values = self.image_transform(image)
        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels,teacher_output_ids=teacher_output_ids,jacobi_ids=jacobi_ids,dataset_name="libero_10_consistent")
