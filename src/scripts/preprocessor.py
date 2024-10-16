import os
import shutil
import numpy as np
import nibabel as nb
from tqdm import tqdm
from typing import Any
from os.path import join, isdir, exists
from sklearn.model_selection import KFold
from batchgenerators.utilities.file_and_folder_operations import save_json, load_json, save_pickle

from utils.normalization import ZScoreNormalization


num_foreground_voxels = 10e7
rs = np.random.RandomState(1234)


class Processor:
    def __init__(self, dataset_id, raw_folder, processed_folder):
        self.raw_folder = raw_folder
        self.processed_folder = processed_folder
        self.dataset_name = self.get_dataset_name(dataset_id)

    def get_dataset_name(self, dataset_id):
        assert isdir(self.raw_folder), "The requested raw data folder could not be found"
        for dataset in os.listdir(self.raw_folder):
            if f'{dataset_id:03d}' in dataset:
                return dataset
        raise f'The requested dataset {dataset_id} could not be found in mQSM_raw'

    def run(self):
        print(f'Processing {self.dataset_name} ...')
        data_folder = join(self.raw_folder, self.dataset_name)
        imagesTr = join(data_folder, 'imagesTr')
        labelsTr = join(data_folder, 'labelsTr')
        procedFdr = join(self.processed_folder, self.dataset_name)
        assert isdir(imagesTr) and isdir(labelsTr), "The requested dataset could not be found in mQSM_raw"
        if exists(join(procedFdr, 'data')):
            shutil.rmtree(join(procedFdr, 'data'))
        os.makedirs(join(procedFdr, 'data'))

        img_keys = []
        for img_file in os.listdir(imagesTr):
            img_key = img_file[:-12]
            if img_key not in img_keys:
                img_keys.append(img_key)
        img_keys.sort()
        self.split_dataset(img_keys)

        data_json = load_json(join(data_folder, 'dataset.json'))
        regions = [v for k, v in data_json['labels'].items() if k.upper() != 'BACKGROUND']
        modalities: dict = data_json['modalities']
        with tqdm(desc='Preprocessing', total=len(img_keys)) as p:
            for img_key in img_keys:
                lbl_nii: Any = nb.load(join(labelsTr, f'{img_key}.nii.gz'))
                lbl_data = lbl_nii.get_fdata()

                m0_nii = nb.load(join(imagesTr, f'{img_key}_{0:04d}.nii.gz'))
                m1_nii = nb.load(join(imagesTr, f'{img_key}_{1:04d}.nii.gz'))
                m0_data = m0_nii.get_fdata()
                m1_data = m1_nii.get_fdata()

                m0_data = np.transpose(m0_data, (3, 0, 1, 2))
                m1_data = ZScoreNormalization.run(m1_data, lbl_data)

                np.save(join(procedFdr, 'data', f'{img_key}_rec.npy'), m0_data.astype(np.float32))
                np.save(join(procedFdr, 'data', f'{img_key}_seg.npy'), m1_data[None].astype(np.float32))
                np.save(join(procedFdr, 'data', f'{img_key}_lbl.npy'), lbl_data[None].astype(np.int8))

                class_locs = _sample_foreground_locations(lbl_data, regions)
                save_pickle({'class_locs': class_locs, 'affine': lbl_nii.affine},
                            join(procedFdr, 'data', f'{img_key}.pkl'))
                p.set_postfix(**{'key': img_key})
                p.update()

    def split_dataset(self, img_keys):
        if exists(join(self.processed_folder, self.dataset_name, 'splits.json')):
            return
        splits = []
        kfold = KFold(n_splits=5, shuffle=True, random_state=20184)
        for i, (train_idx, test_idx) in enumerate(kfold.split(img_keys)):
            train_keys = np.array(img_keys)[train_idx]
            test_keys = np.array(img_keys)[test_idx]
            splits.append({
                'train': list(train_keys),
                'val': list(test_keys)
            })
        save_json(splits, join(self.processed_folder, self.dataset_name, 'splits.json'))


def _sample_foreground_locations(seg: np.ndarray, classes_or_regions):
    num_samples = 10000
    min_percent_coverage = 0.01

    class_locs = {}
    for c in classes_or_regions:
        k = c if not isinstance(c, list) else tuple(c)
        if isinstance(c, (tuple, list)):
            mask = seg == c[0]
            for cc in c[1:]:
                mask = mask | (seg == cc)
            all_locs = np.argwhere(mask)
        else:
            all_locs = np.argwhere(seg == c)
        if len(all_locs) == 0:
            class_locs[k] = []
            continue
        target_num_samples = min(num_samples, len(all_locs))
        target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

        selected = all_locs[rs.choice(len(all_locs), target_num_samples, replace=False)]
        class_locs[k] = selected
    return class_locs
