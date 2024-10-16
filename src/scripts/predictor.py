import torch
import numpy as np
import nibabel as nb
from tqdm import tqdm
from typing import Any
from acvl_utils.cropping_and_padding.padding import pad_nd_image

from utils.TKD import create_mix
from utils.softmax import softmax_helper
from utils.normalization import ZScoreNormalization

gyro = 2 * np.pi * 42.5857


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    else:
        pass


def compute_steps_for_sliding_window(image_size, tile_size, tile_step_size):
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]
    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999
        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
        steps.append(steps_here)
    return steps


class Predictor:
    def __init__(self, network, device, patch_size):
        self.network = network
        self.device = device
        self.patch_size = patch_size
        self.tile_step_size = 0.5

    def predict_one(self, phs_file, phs_mask, t1_file, t1_mask, TE, B0, B0_vector, to_file=None):
        nii: Any = nb.load(phs_file)
        phs_data = nii.get_fdata()
        phs_mask = nb.load(phs_mask).get_fdata()
        t1_data = nb.load(t1_file).get_fdata()
        t1_mask = nb.load(t1_mask).get_fdata()

        fld = phs_data / (TE * B0 * gyro)
        vx_sz = [abs(nii.affine[i][i]) for i in range(3)]
        qsm0 = create_mix(fld, B0_vector, vx_sz, phs_mask)

        a_qsm0 = ZScoreNormalization.run(qsm0, phs_mask)
        t1_data = ZScoreNormalization.run(t1_data, t1_mask)
        in_data = np.concatenate([qsm0[None], a_qsm0[None], t1_data[None]], axis=0)

        susc, segm = self.predict_sliding_window(in_data)
        if to_file is not None:
            nb.Nifti1Image(susc, nii.affine).to_filename(f'{to_file}_susc.nii.gz')
            nb.Nifti1Image(segm, nii.affine).to_filename(f'{to_file}_segm.nii.gz')

        return susc, segm

    def predict_sliding_window(self, input_image):
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()
        self.network.set_deep_supervision(False)
        seg_chs = self.network.seg_net.seg_layers[0].out_channels

        empty_cache(self.device)

        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                assert len(input_image.shape) == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

                data, slicer_revert_padding = pad_nd_image(input_image, self.patch_size, 'constant', {'value': 0}, True, None)
                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
                results_device = torch.device('cpu')

                data = data.to(self.device)
                predicted_segm = torch.zeros((seg_chs, *data.shape[1:]), dtype=torch.float32, device=results_device)
                n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

                empty_cache(self.device)

                qsm0 = input_image[:1][None].to(self.device)
                predicted_susc, _ = self.network.recons_net(qsm0)
                predicted_susc = predicted_susc[0, 0].cpu().numpy()

                for sl in tqdm(slicers, desc='  ', disable=False):
                    workon = data[sl][None]
                    workon = workon.to(self.device, non_blocking=False)
                    _, segm = self.network(workon)
                    segm[torch.where(torch.isnan(segm))] = 0
                    predicted_segm[sl] += segm[0].cpu()
                    n_predictions[sl[1:]] += 1
                predicted_segm /= n_predictions

        empty_cache(self.device)

        predicted_segm = predicted_segm[tuple([slice(None), *slicer_revert_padding[1:]])]
        predicted_segm = softmax_helper(predicted_segm.unsqueeze(0))
        predicted_segm = predicted_segm.argmax(1)
        predicted_segm = predicted_segm.numpy()[0]

        return predicted_susc.astype(float), predicted_segm.astype(float)

    def _internal_get_sliding_window_slicers(self, image_size):
        slicers = []
        steps = compute_steps_for_sliding_window(image_size, self.patch_size, self.tile_step_size)
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicer = tuple([slice(None), *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), self.patch_size)]])
                    slicers.append(slicer)
        return slicers
