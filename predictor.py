import inspect
import itertools
import multiprocessing
import os
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subfiles, \
    save_json
from torch import nn
from tqdm import tqdm
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from dataloading.utils import compute_gaussian, compute_steps_for_sliding_window, empty_cache, dummy_context
from dataloading.label_handling import LabelManager
from sampling import resample_data_or_seg_to_shape
from batchgenerators.utilities.file_and_folder_operations import load_json, isfile, save_pickle
from dataloading.dataset import SimpleITKIO
from collections.abc import Iterable

import numpy as np
from scipy.spatial.distance import directed_hausdorff
# from medpy.metric.binary import hd
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy import ndimage
from scipy.ndimage import measurements

def softmax_helper_dim0(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 0)

class Predictor(object):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        self.plans, self.list_of_parameters, self.network,  self.allowed_mirroring_axes = None, None, None, None

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        else:
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device

    
    def manual_initialization(self, network: nn.Module, plans: dict,
                              parameters: Optional[List[dict]],
                              inference_allowed_mirroring_axes: Optional[Tuple[int, ...]]):
        """
        This is used by the nnUNetTrainer to initialize nnUNetPredictor for the final validation
        """
        self.plans = plans
        self.list_of_parameters = parameters
        self.network = network
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        # self.label_manager = plans_manager.get_label_manager(dataset_json)
        # allow_compile = True
        # allow_compile = allow_compile and ('nnUNet_compile' in os.environ.keys()) and (
        #             os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))
        # allow_compile = allow_compile and not isinstance(self.network, OptimizedModule)
        # if isinstance(self.network, DistributedDataParallel):
        #     allow_compile = allow_compile and isinstance(self.network.module, OptimizedModule)
        # if allow_compile:
        #     print('Using torch.compile')
        #     self.network = torch.compile(self.network)
    

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        if len(self.plans['patch_size']) < len(image_size):
            assert len(self.plans['patch_size']) == len(
                image_size) - 1, 'if tile_size has less entries than image_size, ' \
                                 'len(tile_size) ' \
                                 'must be one shorter than len(image_size) ' \
                                 '(only dimension ' \
                                 'discrepancy of 1 allowed).'
            steps = compute_steps_for_sliding_window(image_size[1:], self.plans['patch_size'],
                                                     self.tile_step_size)
            if self.verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is'
                                   f' {image_size}, tile_size {1}, '
                                   f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.plans['patch_size'])]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.plans['patch_size'],
                                                     self.tile_step_size)
            if self.verbose: print(
                f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {1}, '
                f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.plans['patch_size'])]]))
        return slicers

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
            prediction /= (len(axes_combinations) + 1)
        return prediction

    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            num_segmentation_heads = len(self.plans['annotated_classes_key'])
            predicted_logits = torch.zeros((num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.plans['patch_size']), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')
            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                workon = data[sl][None]
                workon = workon.to(self.device)

                prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                if self.use_gaussian:
                    prediction *= gaussian
                # print(predicted_logits.shape, prediction.shape)

                predicted_logits[sl] += prediction.squeeze()
                n_predictions[sl[1:]] += gaussian

            predicted_logits /= n_predictions
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        with torch.no_grad():
            assert isinstance(input_image, torch.Tensor)
            self.network = self.network.to(self.device)
            self.network.eval()

            empty_cache(self.device)

            # Autocast can be annoying
            # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
            # and needs to be disabled.
            # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
            # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
            # So autocast will only be active if we have a cuda device.
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

                if self.verbose:
                    print(f'Input shape: {input_image.shape}')
                    print("step_size:", self.tile_step_size)
                    print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

                # if input_image is smaller than tile_size we need to pad it to tile_size.
                data, slicer_revert_padding = pad_nd_image(input_image, self.plans['patch_size'],
                                                           'constant', {'value': 0}, True,
                                                           None)

                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                if self.perform_everything_on_device and self.device != 'cpu':
                    # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                    try:
                        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                               self.perform_everything_on_device)
                    except RuntimeError:
                        print(
                            'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                        empty_cache(self.device)
                        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
                else:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                           self.perform_everything_on_device)

                empty_cache(self.device)
                # revert padding
                predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits


def fix_types_iterable(iterable, output_type):
    # this sh!t is hacky as hell and will break if you use it for anything outside nnunet. Keep your hands off of this.
    out = []
    for i in iterable:
        if type(i) in (np.int64, np.int32, np.int8, np.uint8):
            out.append(int(i))
        elif isinstance(i, dict):
            recursive_fix_for_json_export(i)
            out.append(i)
        elif type(i) in (np.float32, np.float64, np.float16):
            out.append(float(i))
        elif type(i) in (np.bool_,):
            out.append(bool(i))
        elif isinstance(i, str):
            out.append(i)
        elif isinstance(i, Iterable):
            # print('recursive call on', i, type(i))
            out.append(fix_types_iterable(i, type(i)))
        else:
            out.append(i)
    return output_type(out)

def recursive_fix_for_json_export(my_dict: dict):
    # json is ... a very nice thing to have
    # 'cannot serialize object of type bool_/int64/float64'. Apart from that of course...
    keys = list(my_dict.keys())  # cannot iterate over keys() if we change keys....
    for k in keys:
        if isinstance(k, (np.int64, np.int32, np.int8, np.uint8)):
            tmp = my_dict[k]
            del my_dict[k]
            my_dict[int(k)] = tmp
            del tmp
            k = int(k)

        if isinstance(my_dict[k], dict):
            recursive_fix_for_json_export(my_dict[k])
        elif isinstance(my_dict[k], np.ndarray):
            assert my_dict[k].ndim == 1, 'only 1d arrays are supported'
            my_dict[k] = fix_types_iterable(my_dict[k], output_type=list)
        elif isinstance(my_dict[k], (np.bool_,)):
            my_dict[k] = bool(my_dict[k])
        elif isinstance(my_dict[k], (np.int64, np.int32, np.int8, np.uint8)):
            my_dict[k] = int(my_dict[k])
        elif isinstance(my_dict[k], (np.float32, np.float64, np.float16)):
            my_dict[k] = float(my_dict[k])
        elif isinstance(my_dict[k], list):
            my_dict[k] = fix_types_iterable(my_dict[k], output_type=type(my_dict[k]))
        elif isinstance(my_dict[k], tuple):
            my_dict[k] = fix_types_iterable(my_dict[k], output_type=tuple)
        elif isinstance(my_dict[k], torch.device):
            my_dict[k] = str(my_dict[k])
        else:
            pass  # pray it can be serialized

def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask

def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn

# def compute_metrics(reference_file: str, prediction_file: str, image_reader_writer: SimpleITKIO,
#                     labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
#                     ignore_label: int = None) -> dict:
#     # load images
#     seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)
#     seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)

#     ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

#     results = {}
#     results['reference_file'] = reference_file
#     results['prediction_file'] = prediction_file
#     results['metrics'] = {}
#     for r in labels_or_regions:
#         results['metrics'][r] = {}
#         mask_ref = region_or_label_to_mask(seg_ref, r)
#         mask_pred = region_or_label_to_mask(seg_pred, r)
#         tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
#         if tp + fp + fn == 0:
#             results['metrics'][r]['Dice'] = np.nan
#             results['metrics'][r]['IoU'] = np.nan
#         else:
#             results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
#             results['metrics'][r]['IoU'] = tp / (tp + fp + fn)
#         results['metrics'][r]['FP'] = fp
#         results['metrics'][r]['TP'] = tp
#         results['metrics'][r]['FN'] = fn
#         results['metrics'][r]['TN'] = tn
#         results['metrics'][r]['n_pred'] = fp + tp
#         results['metrics'][r]['n_ref'] = fn + tp
#     return results

def compute_surface_distances(mask_ref, mask_pred):
    """
    Compute the surface distances from mask_ref to mask_pred and vice versa.

    Parameters:
        mask_ref (ndarray): Reference segmentation mask (binary).
        mask_pred (ndarray): Predicted segmentation mask (binary).

    Returns:
        tuple: Distances from reference to prediction and prediction to reference.
    """
    # Get the binary edges (surface) of the masks
    ref_surface = mask_ref ^ binary_erosion(mask_ref)  # Use XOR instead of subtraction
    pred_surface = mask_pred ^ binary_erosion(mask_pred)  # Use XOR instead of subtraction

    # Compute the distance transform for each mask
    dist_transform_ref = distance_transform_edt(~mask_ref)  # Invert mask for background distances
    dist_transform_pred = distance_transform_edt(~mask_pred)

    # Extract surface distances
    distances_ref_to_pred = dist_transform_pred[ref_surface]
    distances_pred_to_ref = dist_transform_ref[pred_surface]

    return distances_ref_to_pred, distances_pred_to_ref

def compute_mean_surface_distance(mask_ref, mask_pred):
    """
    Compute the Mean Surface Distance (MSD) between two binary masks.

    Parameters:
        mask_ref (ndarray): Reference segmentation mask (binary).
        mask_pred (ndarray): Predicted segmentation mask (binary).

    Returns:
        float: Mean Surface Distance.
    """
    distances_ref_to_pred, distances_pred_to_ref = compute_surface_distances(mask_ref, mask_pred)

    if len(distances_ref_to_pred) == 0 or len(distances_pred_to_ref) == 0:
        return np.nan

    # Compute the mean of both distances
    msd = (np.mean(distances_ref_to_pred) + np.mean(distances_pred_to_ref)) / 2.0
    return msd

def get_object_agatston(calc_object: np.ndarray, calc_pixel_count: int):
    object_max = np.max(calc_object)
    object_agatston = 0
    if 130 <= object_max < 200:
        object_agatston = calc_pixel_count * 1
    elif 200 <= object_max < 300:
        object_agatston = calc_pixel_count * 2
    elif 300 <= object_max < 400:
        object_agatston = calc_pixel_count * 3
    elif object_max >= 400:
        object_agatston = calc_pixel_count * 4
    return object_agatston

def compute_agatston_for_slice(X, Y, min_calc_object_pixels=3, dicom_attributes=None) -> int:
    if dicom_attributes is None:
        dicom_attributes = {
            'slice_thickness': 5,
            'pixel_spacing': (0.404296875, 0.404296875),
            'rescale_intercept': -1024.,
            'rescale_slope': 1.0
        }
    dicom_attributes = {
            'slice_thickness': 5,
            'pixel_spacing': (0.404296875, 0.404296875),
            'rescale_intercept': -1024.,
            'rescale_slope': 1.0
        }

    def create_hu_image(X):
        norm_const = np.array(2 ** 16 - 1).astype('float32')
        return X * norm_const * dicom_attributes['rescale_slope'] - dicom_attributes['rescale_intercept']

    mask = Y[:, :]
    if np.sum(mask) == 0:
        return 0
    
    slice_agatston = 0
    pixel_volume = (dicom_attributes['slice_thickness']
                    * dicom_attributes['pixel_spacing'][0]
                    * dicom_attributes['pixel_spacing'][1]) / 3

    # pixel_volume = (dicom_attributes['slice_thickness']
    #                 * dicom_attributes['pixel_spacing'][0]
    #                 * dicom_attributes['pixel_spacing'][1])

    hu_image = create_hu_image(X)
    # hu_image = X
    labeled_mask, num_labels = measurements.label(mask, structure=np.ones((3, 3)))
    for calc_idx in range(1, num_labels + 1):
        label = np.zeros(mask.shape)
        label[labeled_mask == calc_idx] = 1
        calc_object = hu_image * label

        calc_pixel_count = np.sum(label)
        # print('calc_pixel_count', calc_pixel_count,min_calc_object_pixels)
        if calc_pixel_count <= min_calc_object_pixels:
            continue
        calc_volume = calc_pixel_count * pixel_volume
        object_agatston = round(get_object_agatston(calc_object, calc_volume))
        slice_agatston += object_agatston
    return slice_agatston

def compute_metrics(original_file: str, reference_file: str, prediction_file: str, image_reader_writer: SimpleITKIO,
                    labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                    ignore_label: int = None) -> dict:
    # Load images
    # print(reference_file)
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)
    ori_img, dict_attribute = image_reader_writer.read_seg(original_file)

    print(os.path.basename(reference_file),os.path.basename(prediction_file),os.path.basename(original_file),seg_ref.shape, seg_pred.shape, ori_img.shape)

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}

    for r in labels_or_regions:
        results['metrics'][r] = {}
        mask_ref = region_or_label_to_mask(seg_ref, r)
        mask_pred = region_or_label_to_mask(seg_pred, r)

        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)

        if tp + fp + fn == 0:
            results['metrics'][r]['Dice'] = np.nan
            results['metrics'][r]['IoU'] = np.nan
            results['metrics'][r]['HD95'] = np.nan
            results['metrics'][r]['MSD'] = np.nan
        else:
            results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
            results['metrics'][r]['IoU'] = tp / (tp + fp + fn)

            # Compute the 95% Hausdorff Distance
            results['metrics'][r]['HD95'] = compute_hausdorff_95(mask_ref, mask_pred)

            # Compute Mean Surface Distance (MSD)
            results['metrics'][r]['MSD'] = compute_mean_surface_distance(mask_ref, mask_pred)
        spacing = dict_attribute.get('spacing')
        print('spacing', spacing)
        dicom_attributes = {
            'slice_thickness': spacing[0],
            'pixel_spacing': (spacing[1], spacing[2]),
            'rescale_intercept': -1024.,
            'rescale_slope': 1.0
        }
        results['metrics'][r]['Agatston_Ref'] = sum(
            compute_agatston_for_slice(ori_img[0, i], mask_ref[0, i], dicom_attributes=dicom_attributes) for i in range(seg_ref.shape[1])
        )
        results['metrics'][r]['Agatston_Pred'] = sum(
            compute_agatston_for_slice(ori_img[0, i], mask_pred[0, i], dicom_attributes=dicom_attributes) for i in range(seg_pred.shape[1])
        )
        results['metrics'][r]['FP'] = fp
        results['metrics'][r]['TP'] = tp
        results['metrics'][r]['FN'] = fn
        results['metrics'][r]['TN'] = tn
        results['metrics'][r]['n_pred'] = fp + tp
        results['metrics'][r]['n_ref'] = fn + tp

        # print('Dice, HD95, MSD, ScoreRef, ScorePred:', results['metrics'][r]['Dice'], results['metrics'][r]['HD95'], results['metrics'][r]['MSD'], results['metrics'][r]['Agatston_Ref'],results['metrics'][r]['Agatston_Pred'])

    return results

def compute_hausdorff_95(mask_ref, mask_pred):
    """
    Compute the 95% Hausdorff Distance between two binary masks.

    Parameters:
        mask_ref (ndarray): Reference segmentation mask (binary).
        mask_pred (ndarray): Predicted segmentation mask (binary).
    
    Returns:
        float: 95% Hausdorff Distance.
    """
    # Get coordinates of non-zero pixels (points of the segmentation)
    ref_points = np.argwhere(mask_ref)
    pred_points = np.argwhere(mask_pred)

    if len(ref_points) == 0 or len(pred_points) == 0:
        return np.nan  # Return NaN if either segmentation is empty

    # Compute pairwise distances between reference and predicted points
    distances_ref_to_pred = cdist(ref_points, pred_points)
    distances_pred_to_ref = cdist(pred_points, ref_points)

    # Get the 95th percentile of the minimum distances in both directions
    hd95_ref_to_pred = np.percentile(np.min(distances_ref_to_pred, axis=1), 95)
    hd95_pred_to_ref = np.percentile(np.min(distances_pred_to_ref, axis=1), 95)

    # The final 95% Hausdorff Distance is the maximum of the two values
    hd95 = max(hd95_ref_to_pred, hd95_pred_to_ref)
    return hd95


def label_or_region_to_key(label_or_region: Union[int, Tuple[int]]):
    return str(label_or_region)

def save_summary_json(results: dict, output_file: str):
    """
    json does not support tuples as keys (why does it have to be so shitty) so we need to convert that shit
    ourselves
    """
    results_converted = deepcopy(results)
    # convert keys in mean metrics
    results_converted['mean'] = {label_or_region_to_key(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results_converted["metric_per_case"])):
        results_converted["metric_per_case"][i]['metrics'] = \
            {label_or_region_to_key(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    # sort_keys=True will make foreground_mean the first entry and thus easy to spot
    save_json(results_converted, output_file, sort_keys=True)


def compute_metrics_on_folder(folder_ori: str, folder_ref: str, folder_pred: str, output_file: str,
                              file_ending: str,
                              regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                              ignore_label: int = None,
                              num_processes: int = 8,
                              chill: bool = True) -> dict:
    """
    output_file must end with .json; can be None
    """
    image_reader_writer = SimpleITKIO()
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    file_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    file_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    file_ori = subfiles(folder_ori, suffix=file_ending, join=False)
    # files_pred = sorted(subfiles(folder_pred, suffix=file_ending, join=False))
    # files_ref  = sorted(subfiles(folder_ref, suffix=file_ending, join=False))
    # files_ori  = sorted(subfiles(folder_ori, suffix=file_ending, join=False))

    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_ref exist in folder_pred"
    files_ref = [join(folder_ref, i) for i in file_pred]
    files_pred = [join(folder_pred, i) for i in file_pred]
    files_ori = [join(folder_ori, i.replace('.nii.gz', '_0000.nii.gz')) for i in file_pred]

    # files_ref = [join(folder_ref, i) for i in files_ref]
    # files_pred = [join(folder_pred, i) for i in files_pred]
    # files_ori = [join(folder_ori, i) for i in files_ori]

    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        # for i in list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred), [ignore_label] * len(files_pred))):
        #     compute_metrics(*i)
        results = pool.starmap(
            compute_metrics,
            list(zip(files_ori, files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred),
                     [ignore_label] * len(files_pred)))
        )


    # mean metric per class
    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())
    means = {}
    for r in regions_or_labels:
        means[r] = {}
        for m in metric_list:
            means[r][m] = np.nanmean([i['metrics'][r][m] for i in results])

    # foreground mean
    foreground_mean = {}
    for m in metric_list:
        values = []
        for k in means.keys():
            if k == 0 or k == '0':
                continue
            values.append(means[k][m])
        foreground_mean[m] = np.mean(values)

    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    recursive_fix_for_json_export(foreground_mean)
    result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}
    if output_file is not None:
        save_summary_json(result, output_file)
    return result

def apply_inference_nonlin( logits: Union[np.ndarray, torch.Tensor]) -> \
            Union[np.ndarray, torch.Tensor]:
        """
        logits has to have shape (c, x, y(, z)) where c is the number of classes/regions
        """
        inference_nonlin = softmax_helper_dim0
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)

        with torch.no_grad():
            # softmax etc is not implemented for half
            logits = logits.float()
            probabilities = inference_nonlin(logits)

        return probabilities

def convert_probabilities_to_segmentation(predicted_probabilities: Union[np.ndarray, torch.Tensor]) -> \
            Union[np.ndarray, torch.Tensor]:
        """
        assumes that inference_nonlinearity was already applied!

        predicted_probabilities has to have shape (c, x, y(, z)) where c is the number of classes/regions
        """
        if not isinstance(predicted_probabilities, (np.ndarray, torch.Tensor)):
            raise RuntimeError(f"Unexpected input type. Expected np.ndarray or torch.Tensor,"
                               f" got {type(predicted_probabilities)}")
       
        segmentation = predicted_probabilities.argmax(0)

        return segmentation

def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans: dict,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = 8):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans['transpose_forward']]
    current_spacing = plans['spacing'] if \
        len(plans['spacing']) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *plans['spacing']]
    predicted_logits = resample_data_or_seg_to_shape(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            [properties_dict['spacing'][i] for i in plans['transpose_forward']])
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    predicted_probabilities = apply_inference_nonlin(predicted_logits)
    del predicted_logits
    segmentation = predicted_probabilities.argmax(0)

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(plans['foreground_labels']) < 255 else np.uint16)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans['transpose_backward'])
    # if return_probabilities:
    #     # revert cropping
    #     predicted_probabilities = label_manager.revert_cropping_on_probabilities(predicted_probabilities,
    #                                                                              properties_dict[
    #                                                                                  'bbox_used_for_cropping'],
    #                                                                              properties_dict[
    #                                                                                  'shape_before_cropping'])
    #     predicted_probabilities = predicted_probabilities.cpu().numpy()
    #     # revert transpose
    #     predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in
    #                                                                        plans['transpose_backward']])
    #     torch.set_num_threads(old_threads)
    #     return segmentation_reverted_cropping, predicted_probabilities
    # else:
    torch.set_num_threads(old_threads)
    return segmentation_reverted_cropping
    
def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  plans: dict,
                                  output_file_truncated: str,
                                  save_probabilities: bool = False):
    # if isinstance(predicted_array_or_file, str):
    #     tmp = deepcopy(predicted_array_or_file)
    #     if predicted_array_or_file.endswith('.npy'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)
    #     elif predicted_array_or_file.endswith('.npz'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
    #     os.remove(tmp)

    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans, properties_dict,
        return_probabilities=save_probabilities
    )
    del predicted_array_or_file

    # save
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret

    rw =  SimpleITKIO()
    rw.write_seg(segmentation_final, output_file_truncated + '.nii.gz',
                 properties_dict)

if __name__ == '__main__':
    # predict a bunch of files
    pass
