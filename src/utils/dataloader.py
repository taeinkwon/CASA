import numpy as np
import torch
import random


def collate_stack(batch):
    rt_dataset = {}
    for key_name in batch[0].keys():
        elem_list = []
        for item in batch:
            if type(item[key_name]) == str or type(item[key_name]) == tuple:
                elem_list.append(item[key_name])
            else:
                elem_list.append(torch.FloatTensor(item[key_name]))
        # np.shape(elem_list)
        rt_dataset[key_name] = elem_list
    return rt_dataset


def collate_fixed_len(batch, num_frames, sampling_strategy):
    if sampling_strategy == 'offset_uniform':
        # To access the dataset directly
        def _sample_random(item_len):
            steps = random.sample(
                range(1, item_len), num_frames)
            return sorted(steps)

        def _sample_all():
            return list(range(0, num_frames))

        def sampled_num(nparray, steps):
            return nparray[steps]

        rt_dataset = {}

        for key_name in batch[0].keys():
            rt_dataset[key_name] = []
    
            
        for item in batch:

            len0 = len(item['label0'])
            check0 = (num_frames <= len0)
            if check0:
                steps0 = _sample_random(len0)
            else:
                steps0 = _sample_all()

            check1 = (num_frames <= len(item['label1']))
            len1 = len(item['label1'])
            if check1:
                steps1 = _sample_random(len1)
            else:
                steps1 = _sample_all()

            elem = sampled_num(np.array(item["keypoints0"]), steps0)
            rt_dataset["keypoints0"].append(elem)
            elem = sampled_num(np.array(item["keypoints1"]), steps1)
            rt_dataset["keypoints1"].append(elem)
            elem = sampled_num(np.array(item["label0"]), steps0)
            rt_dataset["label0"].append(elem)
            elem = sampled_num(np.array(item["label1"]), steps1)
            rt_dataset["label1"].append(elem)
            rt_dataset["dataset_name"].append(item["dataset_name"])
            rt_dataset["pair_id"].append(item["pair_id"])
            rt_dataset["pair_names"].append(item["pair_names"])

        
        rt_dataset["keypoints0"] = torch.FloatTensor(np.array(rt_dataset["keypoints0"], dtype=float))
        rt_dataset["keypoints1"] = torch.FloatTensor(np.array(rt_dataset["keypoints1"], dtype=float))
        rt_dataset["label0"] = np.array(rt_dataset["label0"], dtype=int)
        rt_dataset["label1"] = np.array(rt_dataset["label1"], dtype=int)
        rt_dataset["pair_id"] = np.array(rt_dataset["pair_id"], dtype=int)

    else:
        assert()
    return rt_dataset


def get_local_split(items: list, world_size: int, rank: int, seed: int):
    """ The local rank only loads a split of the dataset. """
    n_items = len(items)
    items_permute = np.random.RandomState(seed).permutation(items)
    if n_items % world_size == 0:
        padded_items = items_permute
    else:
        padding = np.random.RandomState(seed).choice(
            items,
            world_size - (n_items % world_size),
            replace=True)
        padded_items = np.concatenate([items_permute, padding])
        assert len(padded_items) % world_size == 0, \
            f'len(padded_items): {len(padded_items)}; world_size: {world_size}; len(padding): {len(padding)}'
    n_per_rank = len(padded_items) // world_size
    local_items = padded_items[n_per_rank * rank: n_per_rank * (rank+1)]

    return local_items
