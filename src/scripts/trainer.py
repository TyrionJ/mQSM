import time
import torch
import os.path
import warnings
import numpy as np
import nibabel as nb
from tqdm import tqdm
from typing import List
from datetime import datetime
from os.path import join, isdir
from torch.cuda.amp import GradScaler
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p, load_pickle

from network import MQSMNet
from net_loss import NetLoss
from scripts.predictor import Predictor
from utils.dataloader import NetDataloader
from utils.polyrescheduler import PolyLRScheduler
from utils.collate_outputs import collate_outputs
from utils.evaluation import get_tp_fp_fn_tn, ssim
from utils.default_n_proc import get_allowed_n_proc
from transforms.train_transform import TrTransform, VdTransform
from transforms.limited_length_multithreaded_augmenter import LimitedLenWrapper

warnings.filterwarnings('ignore')


class NetTrainer:
    processed_folder = fold_fdr = network = optimizer = lr_scheduler = final_valid_fdr = None

    def __init__(self, in_channels, num_classes, patch_size, batch_size, dataset_id, processed_folder,
                 result_folder, fold, go_on, epochs, device, validation=False, logger=print):

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fold = fold
        self.go_on = go_on
        self.epochs = epochs
        self.validation = validation
        self.device = torch.device(f'cuda:{device}') if device != 'cpu' else torch.device(device)

        self.install_folder(dataset_id, processed_folder, result_folder, fold)
        self.logger = self.build_logger(logger)
        self.wellcome()

        self.ds_scales = [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25],
                          [0.125, 0.125, 0.125], [0.0625, 0.0625, 0.0625]]
        self.cur_epoch = 0
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.train_iters = 250
        self.valid_iters = 50
        self.save_interval = 1
        self.best_dice = 0
        self.best_ssim = 0

        self.train_loader, self.valid_loader = self.get_tr_vd_loader()
        self.grad_scaler = GradScaler() if device != 'cpu' else None
        self.loss_fn = NetLoss(self.ds_scales)

    def wellcome(self):
        self.logger("\n#######################################################################\n"
                    "Please cite the following paper when using mQSM:\n\n"
                    "#######################################################################\n")

    def build_logger(self, logger):
        now = datetime.now()
        prefix = 'training' if not self.validation else 'validation'
        log_file = join(self.fold_fdr, f'{prefix}_log_{now.strftime("%Y-%m-%d_%H-%M-%S")}.txt')
        fw = open(log_file, 'a', encoding='utf-8')

        def log_fn(content):
            logger(content)
            fw.write(f'{content}\n')
            fw.flush()

        return log_fn

    def install_folder(self, dataset_id, processed_folder, result_folder, fold):
        assert 0 <= fold < 5, 'only support 5-fold training, and fold should belong to [0, 5)'
        assert isdir(processed_folder), "The requested processed data folder could not be found"

        d_name = None
        for dataset in os.listdir(processed_folder):
            if f'{dataset_id:03d}' in dataset:
                d_name = dataset
                break
        assert d_name is not None, f'The requested dataset {dataset_id} could not be found in processed_folder'
        self.processed_folder = join(processed_folder, d_name)
        self.fold_fdr = join(result_folder, d_name, f'fold_{fold}')
        self.final_valid_fdr = join(self.fold_fdr, 'validation')
        maybe_mkdir_p(self.final_valid_fdr)

    def get_tr_vd_indices(self, verbose=True):
        s_file = join(self.processed_folder, 'splits.json')
        splits = load_json(s_file)
        fold = splits[self.fold]

        if verbose:
            self.logger(f'Use splits: {s_file}')
            self.logger(f'The file contains {len(splits)} splits.')
            self.logger(f'Fold for training: {self.fold}')
        return fold['train'], fold['val']

    def get_tr_vd_loader(self):
        train_indices, valid_indices = self.get_tr_vd_indices()
        self.logger(f"tr_set size={len(train_indices)}, val_set size={len(valid_indices)}")
        data_fdr = join(self.processed_folder, 'data')
        tr_loader = NetDataloader(data_fdr, train_indices, self.batch_size, self.patch_size, self.logger)
        vd_loader = NetDataloader(data_fdr, valid_indices, max(2, self.batch_size // 2), self.patch_size, self.logger)
        tr_transforms, val_transforms = TrTransform(self.ds_scales), VdTransform(self.ds_scales)

        allowed_num_processes = get_allowed_n_proc()
        if allowed_num_processes == 0 or self.device.type == 'cpu':
            mt_gen_train = SingleThreadedAugmenter(tr_loader, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(vd_loader, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(self.train_iters, data_loader=tr_loader,
                                             transform=tr_transforms,
                                             num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                             pin_memory=self.device.type == 'cuda')
            mt_gen_val = LimitedLenWrapper(self.valid_iters, data_loader=vd_loader,
                                           transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
                                           num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda')
            time.sleep(0.5)
        return mt_gen_train, mt_gen_val

    def initialize(self):
        empty_cache(self.device)
        self.network = MQSMNet(self.in_channels, self.num_classes).to(self.device)
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = PolyLRScheduler(self.optimizer, self.initial_lr, self.epochs)
        self.load_states()

    def load_states(self):
        check_file = join(self.fold_fdr, 'model_latest.pt')
        if self.go_on and os.path.isfile(check_file):
            self.logger(f'Use checkpoint: {check_file}')
            weights = torch.load(join(self.fold_fdr, 'model_latest.pt'), map_location=torch.device('cpu'))
            checkpoint = torch.load(join(self.fold_fdr, 'check_latest.pth'), map_location=torch.device('cpu'))

            if 'cur_epoch' in weights:
                del weights['cur_epoch']
            self.network.load_state_dict(weights)
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.cur_epoch = checkpoint['cur_epoch']
            self.best_dice = checkpoint['best_dice']
            self.best_ssim = checkpoint['best_ssim']
            if self.grad_scaler is not None and checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

    def save_states(self, val_dice, val_ssim):
        self.cur_epoch += 1
        checkpoint = {
            'optimizer_state': self.optimizer.state_dict(),
            'cur_epoch': self.cur_epoch,
            'best_dice': self.best_dice,
            'best_ssim': self.best_ssim,
            'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None
        }
        if self.grad_scaler is not None:
            checkpoint['grad_scaler_state'] = self.grad_scaler.state_dict()

        if self.best_dice + self.best_ssim < val_dice + val_ssim:
            self.best_dice = val_dice
            self.best_ssim = val_ssim
            self.logger(f'Eureka!!! Best dice or ssim: dice={self.best_dice:.4f}, ssim={self.best_ssim:.4f}')
            torch.save(self.network.state_dict(), join(self.fold_fdr, 'model_best.pt'))

        if self.cur_epoch % self.save_interval == 0 or self.cur_epoch == self.epochs:
            torch.save(checkpoint, join(self.fold_fdr, 'check_latest.pth'))
            torch.save(self.network.state_dict(), join(self.fold_fdr, 'model_latest.pt'))
        self.logger('')

    def train_step(self, batch: dict) -> dict:
        rec_data = batch['recon']
        seg_data = batch['seg_data']
        tgt_data = batch['seg_label']

        rec_data = rec_data.to(self.device)
        seg_data = seg_data.to(self.device)
        if isinstance(tgt_data, list):
            tgt_data = [i.to(self.device, ) for i in tgt_data]
        else:
            tgt_data = tgt_data.to(self.device)

        self.optimizer.zero_grad()

        susc, segm = self.network(torch.cat([rec_data[:, :1], seg_data], dim=1))
        t_loss, r_loss, s_loss = self.loss_fn((susc, rec_data), (segm, tgt_data))

        if self.grad_scaler is not None:
            self.grad_scaler.scale(t_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            t_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': t_loss.detach().cpu().numpy(),
                'r_ls': r_loss.detach().cpu().numpy(),
                's_ls': s_loss.detach().cpu().numpy()}

    def valid_step(self, batch: dict) -> dict:
        rec_data = batch['recon']
        seg_data = batch['seg_data']
        tgt_data = batch['seg_label']

        rec_data = rec_data.to(self.device)
        seg_data = seg_data.to(self.device)
        if isinstance(tgt_data, list):
            tgt_data = [i.to(self.device) for i in tgt_data]
        else:
            tgt_data = tgt_data.to(self.device)

        susc, segm = self.network(torch.cat([rec_data[:, :1], seg_data], dim=1))
        t_loss, _, _ = self.loss_fn((susc, rec_data), (segm, tgt_data))

        seg_out = segm[0]
        target = tgt_data[0]

        axes = [0] + list(range(2, len(seg_out.shape)))
        output_seg = seg_out.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(seg_out.shape, device=seg_out.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes)

        tp_hard = tp.detach().cpu().numpy()[1:]
        fp_hard = fp.detach().cpu().numpy()[1:]
        fn_hard = fn.detach().cpu().numpy()[1:]

        s = susc[:, 0].detach().cpu().numpy()
        f = rec_data[:, 0].detach().cpu().numpy()
        SSIM = np.mean([ssim(s[i], f[i]) for i in range(susc.shape[0])])
        return {'loss': t_loss.detach().cpu().numpy(), 'ssim': SSIM,
                'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def on_train_epoch_end(self, epoch, train_loss, lr):
        self.logger(f'Epoch: {epoch} / {self.epochs}')
        self.logger(f'current lr: {np.round(lr, decimals=5)}')
        self.logger(f'train loss: {np.round(train_loss, decimals=6)}')

    def on_valid_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        loss_here = np.mean(outputs_collated['loss']).astype(np.float64)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k + 1e-8) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        mean_susc_SSIM = np.average(outputs_collated['ssim'], 0)

        self.logger(f'validation loss: {np.round(loss_here, decimals=6)}')
        self.logger(f'valid mean SSIM: {np.round(mean_susc_SSIM, decimals=6)}')
        self.logger(f'valid mean Dice: {np.round(mean_fg_dice, decimals=6)}')
        self.logger(f'Dice per  class: {[np.round(i, decimals=2) for i in global_dc_per_class]}')

        return mean_fg_dice, mean_susc_SSIM

    def conduct_final_validation(self):
        self.logger('\nFinal Validation')

        self.network.set_deep_supervision(False)
        predictor = Predictor(self.network, self.device, self.patch_size)
        train_indices, valid_indices = self.get_tr_vd_indices(False)
        mean_Dice, mean_SSIM = 0, 0
        for N, key in enumerate(valid_indices):
            self.logger(f'Validating {key}:')
            rec_data = np.load(join(self.processed_folder, 'data', f'{key}_rec.npy'))
            seg_data = np.load(join(self.processed_folder, 'data', f'{key}_seg.npy'))
            lbl_data = np.load(join(self.processed_folder, 'data', f'{key}_lbl.npy'))
            pkl_info = load_pickle(join(self.processed_folder, 'data', f'{key}.pkl'))

            in_data = np.concatenate([rec_data[:1], seg_data], axis=0)
            in_data = torch.from_numpy(in_data)
            susc, segm = predictor.predict_sliding_window(in_data)
            susc *= rec_data[2]
            segm *= rec_data[2]

            SSIM = ssim(susc, rec_data[0])
            segm_onehot = torch.zeros((1, self.num_classes) + segm.shape, dtype=torch.float32)
            segm_onehot.scatter_(1, torch.from_numpy(segm[None, None].astype(np.int64)), 1)
            tp, fp, fn, _ = get_tp_fp_fn_tn(segm_onehot, torch.from_numpy(lbl_data[None]), axes=(0, 2, 3, 4))
            tp = tp.numpy()[1:]
            fp = fp.numpy()[1:]
            fn = fn.numpy()[1:]
            DCpC = [i for i in [2 * i / (2 * i + j + k + 1e-8) for i, j, k in zip(tp, fp, fn)]]
            m_DC = np.mean(DCpC)

            mean_Dice = (mean_Dice * N + m_DC) / (N + 1)
            mean_SSIM = (mean_SSIM * N + SSIM) / (N + 1)

            self.logger(f'  SSIM={np.round(SSIM, decimals=6)}')
            self.logger(f'  AvDC={np.round(m_DC, decimals=6)}')
            self.logger(f'  DCpC={[np.round(i, decimals=2) for i in DCpC]}\n')

            nb.Nifti1Image(susc, pkl_info['affine']).to_filename(join(self.final_valid_fdr, f'{key}_susc.nii.gz'))
            nb.Nifti1Image(segm, pkl_info['affine']).to_filename(join(self.final_valid_fdr, f'{key}_segm.nii.gz'))

        self.logger('Final Validation complete')
        self.logger(f'  Mean Validation SSIM: {np.round(mean_SSIM, decimals=6)}')
        self.logger(f'  Mean Validation Dice: {np.round(mean_Dice, decimals=6)}')

    def run(self):
        self.initialize()

        if not self.validation:
            self.logger('\nBegin training ...')
            for epoch in range(self.cur_epoch, self.epochs):
                avg_loss = 0
                self.network.train(self.cur_epoch)
                self.lr_scheduler.step(self.cur_epoch)
                lr = self.optimizer.param_groups[0]['lr']
                with tqdm(desc=f'[{epoch + 1}/{self.epochs}]Training', total=self.train_iters) as p:
                    for batch_id in range(self.train_iters):
                        train_loss = self.train_step(next(self.train_loader))['loss']
                        avg_loss = (avg_loss * batch_id + train_loss) / (batch_id + 1)
                        p.set_postfix(**{'avg': '%.4f' % avg_loss, 'bat': '%.4f' % train_loss, 'lr': '%.6f' % lr})
                        p.update()

                self.network.eval()
                with torch.no_grad():
                    with tqdm(desc='~~Validation', total=self.valid_iters, colour='green') as p:
                        val_outputs = []
                        for batch_id in range(self.valid_iters):
                            val_outputs.append(self.valid_step(next(self.valid_loader)))
                            p.update()

                self.on_train_epoch_end(epoch+1, avg_loss, lr)
                val_dice, val_ssim = self.on_valid_epoch_end(val_outputs)
                self.save_states(val_dice, val_ssim)
            self.logger('Training end!')

        self.conduct_final_validation()


def empty_cache(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    else:
        pass
