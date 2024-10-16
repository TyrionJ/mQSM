import os
import shutil
import numpy as np
import nibabel as nb
from typing import Any
from scipy.io import loadmat
from os.path import exists, join, isdir
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json

from utils.TKD import create_mix

gyro = 2 * np.pi * 42.5857
index_json = load_json(r'E:\Data\Researches\MRI\UAI_Segm\buss\ext-label_index.json')
labels = load_json(r'E:\Data\Researches\MRI\UAI_Segm\buss\ext-label.json')


def process_label(label_file, mask_file=None):
    nii: Any = nb.load(label_file)
    data = nii.get_fdata()
    if mask_file is not None:
        mask = nb.load(mask_file).get_fdata()
        data *= mask
    label = np.zeros_like(data, dtype=float)
    for k in index_json:
        for idx in index_json[k]:
            label[data == int(idx)] = int(k)
    return nb.Nifti1Image(label, nii.affine)


def process_recon_data(phs_file, msk_file, hdr_file):
    nii: Any = nb.load(phs_file)
    phs = nii.get_fdata()
    msk = nb.load(msk_file).get_fdata()
    hdr = loadmat(hdr_file)

    TE = np.mean(hdr['TEs'])
    fld = phs / (TE * hdr['B0'][0, 0] * gyro)
    qsm0 = create_mix(fld, hdr['B0_vector'][0], hdr['voxelsize'][0], msk)

    data = np.concatenate([qsm0[None], fld[None], msk[None]], axis=0)
    return nb.Nifti1Image(np.transpose(data, (1, 2, 3, 0)), nii.affine)


def main():
    imagesTr, labelsTr = 'imagesTr', 'labelsTr'

    jdata = load_json('./dataset.json')
    jdata['modalities'] = {0: 'QSM0', 1: 'T1'}
    jdata['labels'] = labels
    data_info = {}

    sources = ['F:/Data/Researches/MRI/drug_addition',
               'F:/Data/Researches/MRI/CSVD',
               'F:/Data/Researches/MRI/ASL&QSM']
    to_folder = 'F:/Data/runtime/mQSM/mQSM_raw/Dataset003_QSMT1-EXT'

    if exists(f'{to_folder}/{imagesTr}'):
        shutil.rmtree(f'{to_folder}/{imagesTr}')
        shutil.rmtree(f'{to_folder}/{labelsTr}')
    os.makedirs(f'{to_folder}/{imagesTr}')
    os.makedirs(f'{to_folder}/{labelsTr}')

    numTraining = 0
    for fr_folder in sources:
        for d_type in os.listdir(fr_folder):
            if not isdir(join(fr_folder, d_type)):
                continue
            for sub in os.listdir(join(fr_folder, d_type)):
                roi_fdr = join(fr_folder, 'roi_revision', d_type, sub)
                if not exists(join(roi_fdr, 'done')):
                    continue

                sub_fdr = join(fr_folder, d_type, sub)
                print(sub_fdr)

                phs_file = join(sub_fdr, 'QSM', 'tissue_phase_T1.nii.gz')
                msk_file = join(sub_fdr, 'QSM', 'eroded_mask_T1.nii.gz')
                hdr_file = join(sub_fdr, 'QSM', 'header.mat')
                t1_file = join(sub_fdr, 'T1', 'T1_QSM.nii.gz')
                label_file = join(roi_fdr, 'brain_roi.nii.gz')

                numTraining += 1
                data_id = f'QT_{numTraining:03d}'
                rec_nii = process_recon_data(phs_file, msk_file, hdr_file)
                rec_nii.to_filename(join(to_folder, imagesTr, f'{data_id}_0000.nii.gz'))
                shutil.copy(t1_file, join(to_folder, imagesTr, f'{data_id}_0001.nii.gz'))
                label_nii = process_label(label_file)
                label_nii.to_filename(join(to_folder, labelsTr, f'{data_id}.nii.gz'))

                data_info[data_id] = f'{fr_folder.split("/")[-1]}-{d_type}-{sub}'

    jdata['numTraining'] = numTraining
    save_json(jdata, f'{to_folder}/dataset.json')
    save_json(data_info, f'{to_folder}/data_info.json')


if __name__ == '__main__':
    main()
