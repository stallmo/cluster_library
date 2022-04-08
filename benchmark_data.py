import os
from glob import glob

import numpy as np
from fuzzywuzzy import fuzz


class BenchmarkDataLoader:

    def __init__(self, path_to_all_data):
        """
        This class allows to quickly construct benchmark data sets from the "Clustering basic benchmark" data.
        It does assume data is available under <path_to_all_data> on the local machine, however.
        Moreover, it is assumed the folder structure follows a certain logic.
        """
        self.path_to_all_data = os.path.abspath(path_to_all_data)
        self.data_category_dirs = glob(self.path_to_all_data)
        self.category_to_dir = {d.split(os.sep)[-1]: os.path.abspath(d) for d in self.data_category_dirs}
        self.category_to_gt_dirs = {cat: os.path.join(d, 'ground_truth') for cat, d in self.category_to_dir.items()}

    def get_all_set_paths_in_category(self, category):
        """
        The basic benchmark data is categorized in six categories. Each category contains multiple datasets.
        This function returns the paths to all data in a given <category>.

        :param category: str. Defines category to find paths for.
        :return: list of str. Paths to data in category <category>.
        """

        category_dir = self.category_to_dir.get(category)
        all_sub_paths = glob(category_dir + os.sep + '*')
        return [p for p in all_sub_paths if not 'ground_truth' in p]

    def get_ground_truth_path(self, set_path):
        """
        Some cluster data sets come with ground truth cluster centers.
        This function finds the path to the ground truth data corresponding to data under <set_path>.

        :param set_path: str. Path to a cluster dataset.
        :return: str. Path to ground truth set corresponding to dataset stored und <set_path>.
        """
        category = set_path.split(os.sep)[-2]
        category_ground_truth_dir = self.category_to_gt_dirs.get(category)

        # ground_truth_paths = glob(os.path.join(category_ground_truth_dir, '*'))
        gt_fname = set_path.split('/')[-1].split('.')[0] + '-gt.txt'

        ground_truth_path = os.path.join(category_ground_truth_dir, gt_fname)

        return ground_truth_path

    def get_partitions_path(self, set_path):
        """
        Some cluster data sets come with ground truth partitions.
        This function finds the path to the ground truth partitions corresponding to data under <set_path>.

        :param set_path: str. Path to a cluster dataset.
        :return: str. Path to ground truth partition corresponding to dataset stored under <set_path>.
        """

        category = set_path.split(os.sep)[-2]
        category_ground_truth_dir = self.category_to_gt_dirs.get(category)

        # ground_truth_paths = glob(os.path.join(category_ground_truth_dir, '*'))
        part_fname = set_path.split('/')[-1].split('.')[0] + '-gt.pa'
        partitions_path = os.path.join(category_ground_truth_dir, part_fname)

        return partitions_path

    def load_ground_truth_partitions(self, set_path):
        """
        This function loads the ground truth partitions corresponding to data stored under <set_path>.
         Returns a 1D numpy array.


        :param set_path: str. Path to a cluster dataset.
        :return: np.array. 1D array with partition indices.
        """
        if not 'g2' in set_path:
            raise NotImplementedError

        partition_path = self.get_partitions_path(set_path)
        with open(partition_path) as f:
            all_pa_lines = f.readlines()

        # the first for lines are text; we start counting at zero while the file starts at 1
        partitions = np.array([int(l.strip()) - 1 for l in all_pa_lines[4:]]).reshape(-1, 1)

        # Minimal check to make sure we have correct cluster indices
        if any(partitions < 0):
            raise ValueError

        return partitions

    def load_ground_truth_data(self, set_path, normalize=False):
        """
        This function loads the ground truth data corresponding to data stored under <set_path> and returns it as numpy array.

        :param set_path: str. Path to a cluster dataset.
        :return: np.array. Numpy array with ground truth data corresponding to data stored under <set_path>.
        """
        gt_path = self.get_ground_truth_path(set_path)
        X = np.loadtxt(gt_path)

        if normalize:
            X = (X - X.min()) / (X.max() - X.min())

        return X

    def load_data(self, path, normalize=False):
        """
        Load data stored under <path> and return as numpy array.

        :param path: str. Path to a (cluster) dataset.
        :param normalize: bool. Specifies whether data X should be normalized before returning.
            According to: (X - X.min()) / (X.max() - X.min())
        :return: np.array.
        """
        X = np.loadtxt(path)

        if normalize:
            X = (X - X.min()) / (X.max() - X.min())

        return X
