import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from feature import mol_to_feature


class ComplexDataset(Dataset):
    def __init__(self, keys: List[str], data_dir: str, id_to_y: Dict[str, float]):
        self.keys = keys
        self.data_dir = data_dir
        self.id_to_y = id_to_y

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.keys[idx]
        with open(self.data_dir + "/" + key, "rb") as f:
            m1, _, m2, _ = pickle.load(f)

        sample = mol_to_feature(m1, m2)
        sample["affinity"] = self.id_to_y[key] * -1.36
        sample["key"] = key

        return sample


def get_dataset_dataloader(
    keys: List[str],
    data_dir: str,
    id_to_y: Dict[str, float],
    batch_size: int,
    num_workers: int,
    train: bool = True,
) -> Tuple[Dataset, DataLoader]:

    dataset = ComplexDataset(keys, data_dir, id_to_y)
    dataloader = DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
        collate_fn=tensor_collate_fn,
        shuffle=train,
    )

    return dataset, dataloader


def check_dimension(tensors: List[Any]) -> Any:
    size = []
    for tensor in tensors:
        if isinstance(tensor, np.ndarray):
            size.append(tensor.shape)
        else:
            size.append(0)
    size = np.asarray(size)

    return np.max(size, 0)


def collate_tensor(tensor: Any, max_tensor: Any, batch_idx: int) -> Any:
    if isinstance(tensor, np.ndarray):
        dims = tensor.shape
        slice_list = tuple([slice(0, dim) for dim in dims])
        slice_list = [slice(batch_idx, batch_idx + 1), *slice_list]
        max_tensor[tuple(slice_list)] = tensor
    elif isinstance(tensor, str):
        max_tensor[batch_idx] = tensor
    else:
        max_tensor[batch_idx] = tensor

    return max_tensor


def tensor_collate_fn(batch: List[Any]) -> Dict[str, Any]:
    batch_items = [it for e in batch for it in e.items()]
    dim_dict = dict()
    total_key, total_value = list(zip(*batch_items))
    batch_size = len(batch)
    n_element = int(len(batch_items) / batch_size)
    total_key = total_key[0:n_element]
    for i, k in enumerate(total_key):
        value_list = [v for j, v in enumerate(total_value) if j % n_element == i]
        if isinstance(value_list[0], np.ndarray):
            dim_dict[k] = np.zeros(np.array([batch_size, *check_dimension(value_list)]))
        elif isinstance(value_list[0], str):
            dim_dict[k] = ["" for _ in range(batch_size)]
        else:
            dim_dict[k] = np.zeros((batch_size,))

    ret_dict = {}
    for j in range(batch_size):
        if batch[j] is None:
            continue
        for key, value in dim_dict.items():
            value = collate_tensor(batch[j][key], value, j)
            if not isinstance(value, list):
                value = torch.from_numpy(value).float()
            ret_dict[key] = value

    return ret_dict
