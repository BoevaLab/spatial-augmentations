import multiprocessing

import numpy as np
from torch.utils.data import RandomSampler
from torch_geometric.loader import DataLoader


class InfDataLoader(DataLoader):
    """
    A DataLoader that acts as an infinite stream by overriding the __len__ method.

    This is useful for scenarios where the number of iterations is not predetermined,
    such as in certain training loops or when using infinite sampling strategies.
    """

    def __len__(self):
        return int(1e10)


class SubgraphSampler:
    """
    A sampler for subgraphs that dynamically loads and manages regions of a dataset.

    This class is designed to handle large datasets by loading only a subset of regions
    (subgraphs) into memory at a time. It uses a queue to manage region indices and
    preloads subgraphs from disk as needed.

    Parameters
    ----------
    dataset : Dataset
        The dataset containing regions and subgraphs.
    batch_size : int, optional
        The number of samples per batch (default is 64).
    num_regions_per_segment : int, optional
        The number of regions to load per segment (default is 32).
    steps_per_segment : int, optional
        The number of steps to process before loading a new segment (default is 1000).
    num_workers : int, optional
        The number of worker threads for data loading (default is the number of CPU cores).
    seed : int, optional
        A random seed for reproducibility (default is None).
    pin_memory : bool, optional
        Whether to use pinned memory for data loading (default is True).
    """

    def __init__(
        self,
        dataset,
        batch_size=64,
        num_regions_per_segment=32,
        steps_per_segment=1000,
        num_workers=0,
        seed=None,
        pin_memory=True,
    ):
        self.dataset = dataset
        self.all_region_ids = list(dataset.regions)
        self.batch_size = batch_size
        self.num_regions_per_segment = num_regions_per_segment
        self.steps_per_segment = steps_per_segment
        self.num_workers = num_workers if num_workers else multiprocessing.cpu_count()
        self.seed = seed
        self.pin_memory = pin_memory

        # Full region index list
        self.selected_inds = list(range(len(dataset.regions)))
        self.region_inds_queue = []
        self.fill_queue()

        self.step_counter = 0
        self.data_iter = None
        self.load_new_segment()

    def fill_queue(self):
        """
        Fill the region index queue with shuffled indices of regions that have not yet been processed.

        If a seed is provided, it ensures reproducibility by setting the random seed before shuffling.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        remaining_inds = sorted(set(self.selected_inds) - set(self.region_inds_queue))
        np.random.shuffle(remaining_inds)
        self.region_inds_queue.extend(remaining_inds)

    def load_new_segment(self):
        """
        Load a new segment of regions into memory.

        This method selects a subset of regions from the queue, clears the dataset cache,
        and preloads the subgraphs for the selected regions. It also initializes a new
        infinite DataLoader for the current segment.
        """
        # choose new random regions
        region_inds = self.region_inds_queue[: self.num_regions_per_segment]
        self.region_inds_queue = self.region_inds_queue[self.num_regions_per_segment :]
        if len(self.region_inds_queue) < self.num_regions_per_segment:
            self.fill_queue()

        self.dataset.clear_cache()
        selected_region_names = [self.all_region_ids[i] for i in region_inds]
        self.dataset.regions = selected_region_names

        # pre-load subgraphs from disk (chunked)
        for i in range(len(self.dataset.regions)):
            self.dataset.load_to_cache(i, subgraphs=True)

        sampler = RandomSampler(self.dataset, replacement=True, num_samples=int(1e10))
        loader = InfDataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.data_iter = iter(loader)
        self.step_counter = 0

    def __iter__(self):
        """
        Return the iterator object.

        Returns
        -------
        SubgraphSampler
            The current instance of the SubgraphSampler.
        """
        return self

    def __next__(self):
        """
        Return the next batch of data.

        If the number of steps processed exceeds the steps per segment, a new segment
        is loaded into memory.

        Returns
        -------
        Batch
            A batch of data from the dataset.
        """
        if self.step_counter >= self.steps_per_segment:
            self.load_new_segment()
        self.step_counter += 1
        return next(self.data_iter)
