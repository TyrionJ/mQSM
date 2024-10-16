import os
import numpy as np
from os.path import join
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.utilities.file_and_folder_operations import load_pickle

rndst = np.random.RandomState(1234)


class NetDataloader(DataLoader):
    def __init__(self, data_folder, selected, batch_size, patch_size, logger=None):
        super().__init__(None, batch_size, 1, None, True, False, True)

        self.foreground_percent = 0.333
        self.data_folder = data_folder
        self.indices = self.collect_indices(selected)
        self.patch_size = patch_size
        self.data_shape = [batch_size] + [3] + patch_size
        self.seg_shape = [batch_size] + [1] + patch_size
        self.lbl_shape = [batch_size] + [1] + patch_size
        self.logger = logger if logger is not None else lambda x: ''

    def must_foreground(self, sample_idx):
        return not sample_idx < round(self.batch_size * (1 - self.foreground_percent))

    def collect_indices(self, selected):
        d = [f[:-4] for f in os.listdir(self.data_folder) if f.endswith('.pkl')]
        return [i for i in d if i in selected]

    def generate_train_batch(self):
        selected_keys = self.get_indices()

        rec_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.float32)
        lbl_all = np.zeros(self.lbl_shape, dtype=np.int16)

        for i, key in enumerate(selected_keys):
            rec = np.load(join(self.data_folder, f'{key}_rec.npy')).astype(float)
            seg = np.load(join(self.data_folder, f'{key}_seg.npy')).astype(float)
            lbl = np.load(join(self.data_folder, f'{key}_lbl.npy')).astype(float)
            pkl = load_pickle(join(self.data_folder, f'{key}.pkl'))
            must_fg = self.must_foreground(i)

            shape = rec.shape[1:]
            bbox_lbs, bbox_ubs = self.get_bbox(shape, must_fg, pkl['class_locs'])

            rec_slice = tuple([slice(0, rec.shape[0])] + [slice(i, j) for i, j in zip(bbox_lbs, bbox_ubs)])
            rec = rec[rec_slice]

            seg_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(bbox_lbs, bbox_ubs)])
            seg = seg[seg_slice]

            lbl_slice = tuple([slice(0, lbl.shape[0])] + [slice(i, j) for i, j in zip(bbox_lbs, bbox_ubs)])
            lbl = lbl[lbl_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(3)]
            rec_all[i] = np.pad(rec, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[i] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=0)
            lbl_all[i] = np.pad(lbl, ((0, 0), *padding), 'constant', constant_values=-1)

        return {'recon': rec_all, 'seg_data': seg_all, 'seg_label': lbl_all}

    def get_bbox(self, shape, must_fg, class_pos):
        lbs, ubs = [0, 0, 0], [i-j for i, j in zip(shape, self.data_shape[2:])]
        if not must_fg:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(3)]
        else:
            eligible_classes_or_regions = [i for i in class_pos.keys() if len(class_pos[i]) > 0]
            selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))]
            voxels_of_that_class = class_pos[selected_class]
            selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
            bbox_lbs = [max(lbs[i], selected_voxel[i] - self.patch_size[i] // 2) for i in range(3)]
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(3)]

        return bbox_lbs, bbox_ubs
