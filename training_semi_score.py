from typing import Tuple, Union, List
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
import numpy as np
from batchgeneratorsv2.transforms.utils.random import RandomTransform
import os, sys
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from dataloading.dataset import defaultDataset
from dataloading.data_loader_3d import DataLoader3D
from dataloading.utils import get_allowed_n_proc_DA, Logger, dummy_context,empty_cache,collate_outputs
import torch
from torch.cuda.amp import GradScaler
from datetime import datetime
from time import time, sleep
from loss.losse import DC_and_CE_loss,PolyLRScheduler,DC_and_CE_lossWeight,WeightedL1Loss
from torch import autocast, nn
from torch import distributed as dist
from loss.dice import get_tp_fp_fn_tn
# from torch._dynamo import OptimizedModule
from predictor import Predictor,export_prediction_from_logits,compute_metrics_on_folder
import warnings,multiprocessing
from torch.autograd import Variable
import copy
from loss.ramps import sigmoid_rampup
import csv
from models_pytorch.model.model_ours import S2CAC

class trainingPlanner(object):
    def __init__(self, plans: dict, fold: int,
                 device: torch.device = torch.device('cuda')):
        
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()

        self.device = device
        self.fold = fold
        self.plans = plans
        self.alpha = 0.999
        self.beta = 0.03
        self.tau = 300

        self.ignore_label = None  #None

        self.normal_weight = 0

        self.preprocessed_dataset_folder_base = join(self.plans['data_preprocessed'], self.plans['dataset_name'])

        self.output_folder_base = join(self.plans['exp_results'], self.plans['dataset_name'],
                                       self.__class__.__name__ + '__' + self.plans['plans_name']) 
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')

        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                self.plans['data_identifier'])
        self.orininal_data_folder = '/media/bo/Elements_HERA/jinkui/dataset/nnUNet_raw/Dataset011_CAC/imagesTr'
        #self.orininal_data_folder = '/media/imedHDD/imed/personal/wzr/dataset/raw_data/Dataset011_CAC/imagesTr'

        ### Some hyperparameters for you to fiddle with
        self.Augmented = bool(self.plans['augmentated'])
        self.labeled_ratio = float(self.plans['labeled_ratio'])
        self.unlabeled_batch = int(self.plans['unlabeled_batch'])  
        self.consistency = float(self.plans['consistency'])
        self.consistency_rampup = int(self.plans['consistency_rampup'])

        self.batch_size = int(self.plans['batch_size'])

        
        if self.plans['with_normal']:
            self.labeled_batch = int(self.plans['batch_size'])+1
        else:
            self.labeled_batch = int(self.plans['batch_size'])

        self.initial_lr = float(self.plans['initial_lr'])  
        self.weight_decay = float(self.plans['weight_decay'])  
        self.oversample_foreground_percent = float(self.plans['oversample_foreground_percent'])  
        self.num_iterations_per_epoch = int(self.plans['num_iterations_per_epoch']) 
        self.num_val_iterations_per_epoch = int(self.plans['num_val_iterations_per_epoch'])  
        self.num_epochs = int(self.plans['num_epochs'])  

        self.score_w = self.plans['score_weight']

        self.current_epoch = 0
        self.enable_deep_supervision = False

        ### Dealing with labels/regions
        # self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
        # needed for predictions. We do sigmoid in case of (overlapping) regions

        self.num_input_channels = None  # -> self.initialize()
        self.network = None  # -> self.build_network_architecture()
        self.network_ema = None
        self.optimizer = self.lr_scheduler = None  # -> self.initialize
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
        self.loss = None  # -> self.initialize
        self.loss_score = None

        ### Simple logging. Don't take that away from me!
        # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
        # logging
        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        self.logger = Logger()

        ### placeholders
        self.dataloader_train_label = self.dataloader_train_unlabel = self.dataloader_val = None  # see on_train_start

        ### initializing stuff for remembering things and such
        self._best_ema = None

        ### inference things
        self.inference_allowed_mirroring_axes = None  # this variable is set in
        # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints

        ### checkpoint saving stuff
        self.save_every = 30
        self.disable_checkpointing = False

        ## DDP batch size and oversampling can differ between workers and needs adaptation
        # we need to change the batch size in DDP because we don't use any of those distributed samplers

        self.was_initialized = False

        self.print_to_log_file("\n#######################################################################\n"
                               "Trainer has been bulit."
                               "#######################################################################\n",
                               also_print_to_console=True, add_timestamp=False)
    
    def initialize(self):
        if not self.was_initialized:

            empty_cache(self.device)

            self.num_input_channels = 1

           
            reg_branch = {
                    "classes": len(self.plans['foreground_labels']),          
                    "pooling": "avg",       
                    "dropout": 0.2,         
                    "activation": None 
                }
                #
            self.network = S2CAC(encoder_name=self.plans['encoder'],in_channels=self.num_input_channels,classes=len(self.plans['annotated_classes_key']),encoder_depth=4).to(self.device)
            # self.network = smp.Unet_3D(encoder_name=self.plans['encoder'],in_channels=self.num_input_channels, classes=len(self.plans['annotated_classes_key']), aux_params=reg_branch, encoder_depth=4).to(self.device)

            self.network_ema = copy.deepcopy(self.network)

            # for param,param2 in zip(self.network_ema.parameters(),self.network.parameters()):
            #     param.data=param2.data
                
            for param in self.network_ema.parameters():
                param.detach_()


            #UNet3D(in_channels=1, num_classes=2).to(self.device)
            #
            self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
            self.lr_scheduler = PolyLRScheduler(self.optimizer, self.initial_lr, self.num_epochs)
            
            #torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
            # if ddp, wrap in DDP wrapper
            # if self.is_ddp:
            #     self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            #     self.network = DDP(self.network, device_ids=[self.local_rank])

            if self.plans['with_normal']:
                print('Training involving normal samples.')

            if self.plans['dynamic_loss']:
                print('Using dynamic loss.')
                self.loss = DC_and_CE_lossWeight({'batch_dice': False,
                                    'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1, ignore_label=self.ignore_label)
                self.loss_score = WeightedL1Loss()
            else:
                self.loss = DC_and_CE_loss({'batch_dice': False,
                                    'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1, ignore_label=self.ignore_label)
            
                self.loss_score = nn.L1Loss()
            
            self.was_initialized = True
            
    def weighted_l1_loss(prediction, label, weights):
        l1_loss = torch.abs(prediction - label)
        weighted_loss = l1_loss * weights
        return weighted_loss.mean() 

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if False:#isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.plans,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                torch.save(checkpoint, filename)

                if 'Semi' in self.plans['plans_name'] or 'semi' in self.plans['plans_name']:
                    if self.is_ddp:
                        mod = self.network_ema.module
                    else:
                        mod = self.network_ema
                    if False:#isinstance(mod, OptimizedModule):
                        mod = mod._orig_mod

                    checkpoint = {
                        'network_weights': mod.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                        'logging': self.logger.get_checkpoint(),
                        '_best_ema': self._best_ema,
                        'current_epoch': self.current_epoch + 1,
                        'init_args': self.plans,
                        'trainer_name': self.__class__.__name__,
                        'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                    }
                    directory, filename = os.path.split(filename)

                    new_filename = "EMA_" + filename

                    new_file_path = os.path.join(directory, new_filename)
                    torch.save(checkpoint, new_file_path)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def plot_network_architecture(self):

        if self.local_rank == 0:
            try:
                # raise NotImplementedError('hiddenlayer no longer works and we do not have a viable alternative :-(')
                # pip install git+https://github.com/saugatkandel/hiddenlayer.git

                # from torchviz import make_dot
                # # not viable.
                # make_dot(tuple(self.network(torch.rand((1, self.num_input_channels,
                #                                         *self.configuration_manager.patch_size),
                #                                        device=self.device)))).render(
                #     join(self.output_folder, "network_architecture.pdf"), format='pdf')
                # self.optimizer.zero_grad()

                # broken.

                import hiddenlayer as hl
                g = hl.build_graph(self.network,
                                   torch.rand((1, self.num_input_channels,
                                               *self.configuration_manager.patch_size),
                                              device=self.device),
                                   transforms=None)
                g.save(join(self.output_folder, "network_architecture.pdf"))
                del g
            except Exception as e:
                self.print_to_log_file("Unable to plot network architecture:")
                self.print_to_log_file(e)

                # self.print_to_log_file("\nprinting the network instead:\n")
                # self.print_to_log_file(self.network)
                # self.print_to_log_file("\n")
            finally:
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device.type == 'mps':
                    from torch import mps
                    mps.empty_cache()
                else:
                    pass

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = (f"{dt_object}:", *args)

            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
                        for a in args:
                            f.write(str(a))
                            f.write(" ")
                        f.write("\n")
                    successful = True
                except IOError:
                    print(f"{datetime.fromtimestamp(timestamp)}: failed to log: ", sys.exc_info())
                    sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)
        elif also_print_to_console:
            print(*args)
  
    def my_transforms(self, patch_size: Union[np.ndarray, Tuple[int]], Aug=False) :
        transforms = []
        
        patch_size_spatial = patch_size
        rotation_for_DA = (-0.5235987755982988, 0.5235987755982988)
        ignore_axes = False
        mirror_axes = (0,1,2)

        if Aug:
            transforms.append(
                SpatialTransform(
                    patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                    p_rotation=0.2,
                    rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                    bg_style_seg_sampling=False  # , mode_seg='nearest'
                )
            )
            transforms.append(RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0, 0.1),
                    p_per_channel=1,
                    synchronize_channels=True
                ), apply_probability=0.1
            ))
            transforms.append(RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.),
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=0.5, benchmark=True
                ), apply_probability=0.2
            ))
            transforms.append(RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.75, 1.25)),
                    synchronize_channels=False,
                    p_per_channel=1
                ), apply_probability=0.15
            ))
            transforms.append(RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast((0.75, 1.25)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1
                ), apply_probability=0.15
            ))
            transforms.append(RandomTransform(
                SimulateLowResolutionTransform(
                    scale=(0.5, 1),
                    synchronize_channels=False,
                    synchronize_axes=True,
                    ignore_axes=ignore_axes,
                    allowed_channels=None,
                    p_per_channel=0.5
                ), apply_probability=0.25
            ))
            if mirror_axes is not None and len(mirror_axes) > 0:
                transforms.append(
                    MirrorTransform(
                        allowed_axes=mirror_axes
                    )
                )
        else:
            transforms.append(
                SpatialTransform(
                    patch_size_spatial, patch_center_dist_from_border=0, random_crop=False
                    )
            )
            
        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        return ComposeTransforms(transforms)    
        
    def my_transforms_val(self) :
            transforms = []

            transforms.append(
                RemoveLabelTansform(-1, 0)
            )

            return ComposeTransforms(transforms)

    def get_tr_and_val_datasets(self, ratio_labeled=0.5):
        splits_file = os.path.join(self.preprocessed_dataset_folder_base, "splits_final.json")
        normal_file = os.path.join(self.preprocessed_dataset_folder_base, "normal.json")
        
        splits = load_json(splits_file)
        tr_keys_all = splits[self.fold]['train']
        val_keys = splits[self.fold]['val']

        normal = load_json(normal_file)
        normal_keys = normal[0]['train']

        label_num = int(len(tr_keys_all)*ratio_labeled)
        tr_labeled = tr_keys_all[:label_num]
        tr_unlabeled = tr_keys_all[label_num:]

        # dataset_tr = defaultDataset(self.preprocessed_dataset_folder, tr_keys)

        dataset_tr_labeled = defaultDataset(self.preprocessed_dataset_folder, tr_labeled)
        dataset_tr_unlabeled = defaultDataset(self.preprocessed_dataset_folder, tr_unlabeled)
        dataset_tr_normal = defaultDataset(self.preprocessed_dataset_folder, normal_keys)



        dataset_val = defaultDataset(self.preprocessed_dataset_folder, val_keys)

        return dataset_tr_labeled, dataset_tr_unlabeled, dataset_val,dataset_tr_normal

    def get_dataloaders(self):
        patch_size = self.plans['patch_size']
        dim = len(patch_size)
        initial_patch_size = self.plans['initial_patch_size']
       
        # training pipeline
        tr_transforms_label = self.my_transforms(patch_size, True)

        tr_transforms_unlabel = self.my_transforms(patch_size, False)

        # validation pipeline
        val_transforms = self.my_transforms_val()

        dataset_tr_label, dataset_tr_unlabel, dataset_val, dataset_tr_normal = self.get_tr_and_val_datasets(ratio_labeled=self.labeled_ratio)


        dl_tr_label = DataLoader3D(dataset_tr_label, self.batch_size,
                                    initial_patch_size,
                                    patch_size,
                                    self.plans,
                                    sampling_probabilities=None, pad_sides=None, transforms=tr_transforms_unlabel)
        
        dl_tr_unlabel = DataLoader3D(dataset_tr_unlabel, self.unlabeled_batch,
                                    initial_patch_size,
                                    patch_size,
                                    self.plans,
                                    sampling_probabilities=None, pad_sides=None, transforms=tr_transforms_unlabel)
        
        dl_val = DataLoader3D(dataset_val, 2,
                                    patch_size,
                                    patch_size,
                                    self.plans,
                                    sampling_probabilities=None, pad_sides=None, transforms=val_transforms)
        dl_tr_normal = DataLoader3D(dataset_tr_normal, 1,
                                    initial_patch_size,
                                    patch_size,
                                    self.plans,
                                    sampling_probabilities=None, pad_sides=None, transforms=tr_transforms_unlabel)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train_label = SingleThreadedAugmenter(dl_tr_label, None)
            mt_gen_train_unlabel = SingleThreadedAugmenter(dl_tr_unlabel, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
            mt_gen_train_normal = SingleThreadedAugmenter(dl_tr_normal, None)

        else:
            mt_gen_train_label = NonDetMultiThreadedAugmenter(data_loader=dl_tr_label, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_train_unlabel = NonDetMultiThreadedAugmenter(data_loader=dl_tr_unlabel, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                        transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                        num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                        pin_memory=self.device.type == 'cuda',
                                                        wait_time=0.002)
            mt_gen_train_normal = NonDetMultiThreadedAugmenter(data_loader=dl_tr_normal, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train_label)
        _ = next(mt_gen_train_unlabel)
        _ = next(mt_gen_val)
        _ = next(mt_gen_train_normal)

        return mt_gen_train_label, mt_gen_train_unlabel, mt_gen_val,mt_gen_train_normal
    

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        label_cal = batch['mark']

        w_sample = (1 - self.alpha * torch.exp(torch.tensor(-self.beta * self.current_epoch))) * (1 - label_cal) + label_cal

        w_sample = w_sample.to(self.device, non_blocking=True)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            self.network_ema.eval()
            output = self.network_ema(data)
            del data
            if self.plans['dynamic_loss']:
                l = self.loss(output, target,w_sample)
            else:
                l = self.loss(output, target)

       
        self.network_ema.train()
        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        # if self.label_manager.has_regions:
        #     predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        # else:
        # no need for softmax
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg

     
        mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        # if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
        tp_hard = tp_hard[1:]
        fp_hard = fp_hard[1:]
        fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    def get_current_consistency_weight(self, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.consistency * sigmoid_rampup(epoch, self.consistency_rampup)

    def update_ema_variables(self, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(self.network_ema.parameters(), self.network.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def load_checkpoint_restart(self, filefold: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()
        
        filename_or_checkpoint = join(filefold, 'checkpoint_latest.pth')

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if False:#isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if False:#isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])


        #EMA model
        filename_or_checkpoint_EMA = join(filefold, 'EMA_checkpoint_latest.pth')

        if isinstance(filename_or_checkpoint_EMA, str):
            checkpoint = torch.load(filename_or_checkpoint_EMA, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if False:#isinstance(self.network.module, OptimizedModule):
                self.network_ema.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network_ema.module.load_state_dict(new_state_dict)
        else:
            if False:#isinstance(self.network, OptimizedModule):
                self.network_ema._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network_ema.load_state_dict(new_state_dict)

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if False:#isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if False:#isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

    def perform_actual_validation(self, save_probabilities=False,epoch=None,):
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        predictor = Predictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans,  None,
                                        self.inference_allowed_mirroring_axes)
        
        default_num_processes = 8

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            if epoch is not None:
                validation_output_folder = join(self.output_folder, f'validation_epoch_{epoch}')
            else:
                validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            splits_file = os.path.join(self.preprocessed_dataset_folder_base, "splits_final.json")
            splits = load_json(splits_file)
            val_keys = splits[self.fold]['val']

            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

                val_keys = val_keys[self.local_rank:: dist.get_world_size()]
                # we cannot just have barriers all over the place because the number of keys each GPU receives can be
                # different

            dataset_val = defaultDataset(self.preprocessed_dataset_folder, val_keys)

            results = []

            for i, k in enumerate(dataset_val.keys()):

                self.print_to_log_file(f"predicting {k}")
                data, seg, properties = dataset_val.load_case(k)

                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()

                # this needs to go into background processes
                
                export_prediction_from_logits(prediction, properties, self.plans,
                        output_filename_truncated, save_probabilities)

                # for debug purposes
                # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                #              output_filename_truncated, save_probabilities)

                # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(self.orininal_data_folder, join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                '.nii.gz',
                                                self.plans['foreground_labels'],
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_ddp else default_num_processes)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)
            self.print_to_log_file("Mean Validation HD95: ", (metrics['foreground_mean']["HD95"]),
                                   also_print_to_console=True)
            self.print_to_log_file("Mean Validation MSD: ", (metrics['foreground_mean']["MSD"]),
                                   also_print_to_console=True)
            self.print_to_log_file("metric_per_case: ", (metrics['metric_per_case']),
                                   also_print_to_console=False)
            
            # print('Dice, HD95, MSD, ScoreRef, ScorePred:', results['metrics'][r]['Dice'], results['metrics'][r]['HD95'], results['metrics'][r]['MSD'], results['metrics'][r]['Agatston_Ref'],results['metrics'][r]['Agatston_Pred'])
            self.save_results_to_csv(metrics['metric_per_case'], join(self.output_folder,'score.csv'))

            print("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]))
            print("Mean Validation HD95: ", (metrics['foreground_mean']["HD95"]))  
            print("Mean Validation MSD: ", (metrics['foreground_mean']["MSD"]))

    def calculate_Matrics_from_folder(self):
        validation_output_folder = join(self.output_folder, 'validation')
        default_num_processes = 8
        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(self.orininal_data_folder, join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                '.nii.gz',
                                                self.plans['foreground_labels'],
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_ddp else default_num_processes)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)
            self.print_to_log_file("Mean Validation HD95: ", (metrics['foreground_mean']["HD95"]),
                                   also_print_to_console=True)
            self.print_to_log_file("Mean Validation MSD: ", (metrics['foreground_mean']["MSD"]),
                                   also_print_to_console=True)
            
            # print('Dice, HD95, MSD, ScoreRef, ScorePred:', results['metrics'][r]['Dice'], results['metrics'][r]['HD95'], results['metrics'][r]['MSD'], results['metrics'][r]['Agatston_Ref'],results['metrics'][r]['Agatston_Pred'])
            self.save_results_to_csv(metrics['metric_per_case'], join(self.output_folder,'score.csv'))

            print("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]))
            print("Mean Validation HD95: ", (metrics['foreground_mean']["HD95"]))  
            print("Mean Validation MSD: ", (metrics['foreground_mean']["MSD"]))

    def save_results_to_csv(self, results, output_file):
        headers = ['reference_file', 'Agatston_Ref', 'Agatston_Pred']

        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            writer.writerow(headers)
            
            for item in results:
                reference_file = item['reference_file']
                agatston_ref = item['metrics'][1]['Agatston_Ref']
                agatston_pred = item['metrics'][1]['Agatston_Pred']
                
                writer.writerow([os.path.basename(reference_file), agatston_ref, agatston_pred])

    def train_step(self, batch: dict, batch2: dict, batchN: dict, n_iter, loss_weight) -> dict:

        if self.plans['with_normal']:
            # print('Training involving normal samples.')
            data_lable = torch.cat([batch['data'],batchN['data']], dim=0)
            target = torch.cat([batch['target'],batchN['target']], dim=0)
            label_cal = torch.cat([batch['mark'],batchN['mark']], dim=0)
        else:
            data_lable = batch['data']
            target = batch['target']
            label_cal = batch['mark']

        self.normal_weight =  1 - self.alpha * (1 - self.sigmoid(self.beta * (self.current_epoch - self.tau)))
        w_sample = self.normal_weight* (1 - label_cal) + label_cal


        w_sample = w_sample.to(self.device, non_blocking=True)

        # data_lable = batch['data']
        data_unlabel = batch2['data']
        data = torch.cat([data_lable,data_unlabel],dim=0)
        # target = batch['target']
        
        score_label = batch['score']
        score_unlabel = batch2['score']
        score = torch.cat([score_label,score_unlabel],dim=0)

        score = score.to(self.device, non_blocking=True)

        noise = torch.clamp(torch.randn_like(
                data_unlabel) * 0.1, -0.2, 0.2)
        ema_inputs = data_unlabel + noise
        ema_inputs = ema_inputs.to(self.device, non_blocking=True)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
 
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output, pre_score = self.network(data)
            outputs_soft = torch.softmax(output, dim=1)

            outputs_soft_label = outputs_soft[:self.labeled_batch]
            outputs_soft_unlabel = outputs_soft[self.labeled_batch:]

            
            with torch.no_grad():
                output_ema, pre_score_ema = self.network_ema(ema_inputs)

            consistency_weight = self.get_current_consistency_weight(n_iter//150)
            if self.plans['dynamic_loss']:
                supervise_loss_seg = self.loss(output[:self.labeled_batch], target,w_sample)
                supervise_loss_score = self.score_w*self.loss_score(pre_score[:self.labeled_batch], score[:self.labeled_batch],w_sample)  
                consistency_loss_score = self.score_w*self.loss_score(torch.sum(torch.sum(outputs_soft_unlabel[:,1:],dim=1), dim=(1,2,3)).unsqueeze(-1), pre_score_ema, None) + \
                     self.score_w*self.loss_score(torch.sum(torch.sum(outputs_soft_label[:,1:],dim=1), dim=(1,2,3)).unsqueeze(-1), score[:self.labeled_batch],w_sample)
                consistency_loss_seg = torch.mean((output[self.labeled_batch:] - output_ema)**2) + \
                    self.score_w*self.loss_score(pre_score[self.labeled_batch:], pre_score_ema, None)

            else:
                supervise_loss_seg = self.loss(output[:self.labeled_batch], target)

                supervise_loss_score = self.score_w*self.loss_score(pre_score[:self.labeled_batch], score[:self.labeled_batch])  
                consistency_loss_score = self.score_w*self.loss_score(torch.sum(torch.sum(outputs_soft_unlabel[:,1:],dim=1), dim=(1,2,3)).unsqueeze(-1), pre_score_ema) + \
                    self.score_w*self.loss_score(torch.sum(torch.sum(outputs_soft_label[:,1:],dim=1), dim=(1,2,3)).unsqueeze(-1), score[:self.labeled_batch])
                consistency_loss_seg = torch.mean((output[self.labeled_batch:] - output_ema)**2) + \
                    self.score_w*self.loss_score(pre_score[self.labeled_batch:], pre_score_ema)

            
            l = loss_weight[0]*supervise_loss_seg + loss_weight[1]*supervise_loss_score + consistency_weight * (loss_weight[2]*consistency_loss_seg + loss_weight[3]*consistency_loss_score)

            cost = [supervise_loss_seg.item(), supervise_loss_score.item(), consistency_loss_seg.item(), consistency_loss_score.item()]

            # print(f"Supervised Loss: {supervise_loss.item()}, Consistency Loss: {consistency_loss.item()}")


        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        self.update_ema_variables(0.99, n_iter)
        return {'loss': l.detach().cpu().numpy()}, cost
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def start_train(self):
        self.dataloader_train_label, self.dataloader_train_unlabel, self.dataloader_val, self.dataloader_train_normal = self.get_dataloaders()

        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        print(self.plans)

        save_json(self.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)

        self.plot_network_architecture()
        n_iter = 0
        avg_cost = np.zeros([self.num_epochs, 4], dtype=np.float32)
        lambda_weight = np.ones([4, self.num_epochs])
        T = 2

        #early stopping
        best_dice = -np.inf
        epochs_without_improvement = 0
        early_stop_patience = 50 
        for epoch in range(self.current_epoch, self.num_epochs):
            index = epoch
            # cost = np.zeros(4, dtype=np.float32)
            if self.plans['loss_weight'] == 'dwa':
                if index == 0 or index == 1:
                    lambda_weight[:, epoch] = 1.0
                else:
                    w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
                    w_2 = avg_cost[index - 1, 1] / avg_cost[index - 2, 1]
                    w_3 = avg_cost[index - 1, 2] / avg_cost[index - 2, 2]
                    w_4 = avg_cost[index - 1, 3] / avg_cost[index - 2, 3]
                    lambda_weight[0, index] = 4 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T) + np.exp(w_4 / T))
                    lambda_weight[1, index] = 4 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T) + np.exp(w_4 / T))
                    lambda_weight[2, index] = 4 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T) + np.exp(w_4 / T))
                    lambda_weight[3, index] = 4 * np.exp(w_4 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T) + np.exp(w_4 / T))

            self.network.train()
            self.lr_scheduler.step(self.current_epoch)

            self.logger.log('epoch_start_timestamps', time(), self.current_epoch)
            self.print_to_log_file('')
            self.print_to_log_file(f'Epoch {self.current_epoch}')
            self.print_to_log_file(
                f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
            # lrs are the same for all workers so we don't need to gather them in case of DDP training
            loss_weight = float(self.normal_weight) 
            self.print_to_log_file(f"Current loss weight for Ctrl: {loss_weight:.5f}")
            
            self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                infor, cost = self.train_step(next(self.dataloader_train_label),next(self.dataloader_train_unlabel), next(self.dataloader_train_normal), n_iter, lambda_weight[:,index])
                train_outputs.append(infor)
                n_iter += 1

                if self.plans['loss_weight'] == 'dwa':
                    avg_cost[index, 0] += cost[0]/self.labeled_batch
                    avg_cost[index, 1] += cost[1]/self.labeled_batch
                    avg_cost[index, 2] += cost[2]/self.unlabeled_batch
                    avg_cost[index, 3] += cost[3]/self.unlabeled_batch
                

            outputs = collate_outputs(train_outputs)
            loss_here = np.mean(outputs['loss'])

            self.logger.log('train_losses', loss_here, self.current_epoch)

            #evalation
            with torch.no_grad():
                self.network.eval()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                outputs_val = collate_outputs(val_outputs)
                tp = np.sum(outputs_val['tp_hard'], 0)
                fp = np.sum(outputs_val['fp_hard'], 0)
                fn = np.sum(outputs_val['fn_hard'], 0)
                loss_here = np.mean(outputs_val['loss'])

                global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]

                mean_fg_dice = np.nanmean(global_dc_per_class)
                self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
                self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
                self.logger.log('val_losses', loss_here, self.current_epoch)

                if self.current_epoch > 200:
                    if mean_fg_dice > best_dice:
                        best_dice = mean_fg_dice
                        epochs_without_improvement = 0
                        self.print_to_log_file(f"New best validation dice: {best_dice:.4f}")
                    else:
                        epochs_without_improvement += 1
                        self.print_to_log_file(f"No improvement in validation dice for {epochs_without_improvement} epochs.")

                    if epochs_without_improvement >= early_stop_patience:
                        self.print_to_log_file(f"Early stopping triggered after {early_stop_patience} epochs without improvement.")
                        break  

            self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

            self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
            self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
            self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                                self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
            self.print_to_log_file(
                f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

            # handling periodic checkpointing
            current_epoch = self.current_epoch
            if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
                if current_epoch < 200:
                    self.save_checkpoint(join(self.output_folder, f'checkpoint_epoch_{current_epoch}.pth'))
                self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

            # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
            if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
                self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
                self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
                self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

            if self.local_rank == 0:
                self.logger.plot_progress_png(self.output_folder)

            self.current_epoch += 1

        # shut down dataloaders
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train_label is not None and \
                    isinstance(self.dataloader_train_label, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_train_label._finish()
                self.dataloader_train_unlabel._finish()
            if self.dataloader_val is not None and \
                    isinstance(self.dataloader_train_label, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training done.")    



if __name__ =='__main__':
    # os.environ['TORCHDYNAMO_DISABLE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    

# def __init__(self, plans: dict, fold: int, dataset_json: dict,
#                  device: torch.device = torch.device('cuda')):

    plans = load_json('config.json')

    
    folds = [1,2,3,4]
    for item in folds:
        fold = item
        myTrainer = trainingPlanner(plans, fold)

        if os.path.exists(join(myTrainer.output_folder, 'EMA_checkpoint_latest.pth')):
            myTrainer.load_checkpoint_restart(myTrainer.output_folder)
        myTrainer.start_train()
        myTrainer.load_checkpoint(join(myTrainer.output_folder, 'EMA_checkpoint_best.pth'))
        myTrainer.perform_actual_validation(False)

    # hyper = [[0.999,0.03,300],[0.5,0.03,300],[0.999,0.1,300],[0.999,0.005,300],[0.999,0.03,100]]
    # for i,heper_NSB in enumerate(hyper):
    #     folds = [0]
    #     for item in folds:
    #         fold = item
    #         plans['plans_name'] = 'CAC_SemiScore_V20_'+str(i)
    #         myTrainer = trainingPlanner(plans, fold)
    #         myTrainer.alpha = heper_NSB[0]
    #         myTrainer.beta = heper_NSB[1]
    #         myTrainer.tau = heper_NSB[2]
    #         # if os.path.exists(join(myTrainer.output_folder, 'checkpoint_best.pth')):
    #         #     myTrainer.load_checkpoint_restart(myTrainer.output_folder)
    #         # myTrainer.start_train()
    #         myTrainer.load_checkpoint(join(myTrainer.output_folder, 'checkpoint_latest.pth'))
    #         myTrainer.perform_actual_validation(False)
            # myTrainer.calculate_Matrics_from_folder()


    # epochs = range(19,499,20)
    # for epoch_temp in epochs:
    #     folds = [0,1,2,3,4]
    #     for item in folds:
    #         fold = item
    #         myTrainer = trainingPlanner(plans, fold)

    #         # if os.path.exists(join(myTrainer.output_folder, 'EMA_checkpoint_latest.pth')):
    #         #     myTrainer.load_checkpoint_restart(myTrainer.output_folder)
    #         # myTrainer.start_train()
    #         if os.path.exists(join(myTrainer.output_folder, f'EMA_checkpoint_epoch_{epoch_temp}.pth')):
    #             myTrainer.print_to_log_file(f"Validation epoch {epoch_temp}.") 
    #             myTrainer.load_checkpoint(join(myTrainer.output_folder, f'EMA_checkpoint_epoch_{epoch_temp}.pth'))
    #             myTrainer.perform_actual_validation(epoch=epoch_temp)


