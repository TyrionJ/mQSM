import numpy as np
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor, RemoveLabelTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform

from transforms.deep_supervision_donwsampling import DownsampleSegForDSTransform


class TrTransform(AbstractTransform):
    def __init__(self, ds_scales):
        super().__init__()
        self.transforms1 = Compose([
            GaussianNoiseTransform(p_per_sample=0.1, data_key='seg_data'),
            GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True,
                                  p_per_sample=0.2, p_per_channel=0.5, data_key='seg_data'),
            BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15, data_key='seg_data'),
            ContrastAugmentationTransform(p_per_sample=0.15, data_key='seg_data'),
            SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                           p_per_channel=0.5, order_downsample=0, order_upsample=3,
                                           p_per_sample=0.25, ignore_axes=None, data_key='seg_data'),
            GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1, data_key='seg_data'),
            GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3, data_key='seg_data'),
        ])

        self.transforms2 = Compose([
            MirrorTransform((0, 1, 2), data_key='data', label_key='seg_label'),
            RemoveLabelTransform(-1, 0, input_key='seg_label', output_key='seg_label'),
            DownsampleSegForDSTransform(ds_scales, 0, input_key='seg_label', output_key='seg_label'),
            NumpyToTensor(['data', 'seg_label'], 'float')
        ])

    def __call__(self, recon, seg_data, seg_label):
        rs = recon.shape[1]

        data = {'seg_data': seg_data}
        del seg_data
        data = self.transforms1(**data)

        data = np.concatenate([recon, data['seg_data']], axis=1)
        data = {'data': data, 'seg_label': seg_label}
        del recon, seg_label
        data = self.transforms2(**data)

        data2 = data['data']
        return {'recon': data2[:, :rs], 'seg_data': data2[:, rs:], 'seg_label': data['seg_label']}


class VdTransform(AbstractTransform):
    def __init__(self, ds_scales):
        super().__init__()
        transforms = [
            RemoveLabelTransform(-1, 0, input_key='seg_label', output_key='seg_label'),
            DownsampleSegForDSTransform(ds_scales, input_key='seg_label', output_key='seg_label'),
            NumpyToTensor(['data', 'seg_label'], 'float')
        ]
        self.transforms = Compose(transforms)

    def __call__(self, recon, seg_data, seg_label):
        rs = recon.shape[1]

        data = np.concatenate([recon, seg_data], axis=1)
        del recon, seg_data
        data = {'data': data, 'seg_label': seg_label}
        data = self.transforms(**data)

        data2 = data['data']
        return {'recon': data2[:, :rs], 'seg_data': data2[:, rs:], 'seg_label': data['seg_label']}
