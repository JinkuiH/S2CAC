import numpy as np
import torch
from threadpoolctl import threadpool_limits

from dataloading.base_data_loader import DataLoaderBase
from dataloading.dataset import defaultDataset
from scipy import ndimage

class DataLoader3D(DataLoaderBase):
    def get_object_agatston(self, calc_object: np.ndarray, calc_pixel_count: int):
        """Applies standard categorization: https://radiopaedia.org/articles/agatston-score"""

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
        # print(f'For {calc_pixel_count} with max {object_max} returning AG of {object_agatston}')
        return object_agatston

    def calculate_volume(self, seg):
        if seg is None:
            return 0
        score_volume = np.zeros(np.max((np.max(seg),1)))
        seg = seg[0]
        # 针对每个分割标签
        for label_index in range(np.max(seg)):
            # print('current label:', label_index+1)
            # 获取当前标签的分割图
            current_seg = (seg == label_index+1).astype(np.uint8)

            score_volume[label_index] = np.sum(current_seg)
            # # 连通域检测
            # labeled_array, num_features = ndimage.label(current_seg,structure=np.ones((3, 3, 3)))
            
            # # 对每个连通域计算强度
            # for feature in range(1, num_features + 1):  # feature 从 1 开始

            #     label = np.zeros(current_seg.shape)
            #     label[labeled_array == feature] = 1

            #     calc_pixel_count = np.sum(label)

            #     calc_volume = calc_pixel_count
                
            #     score_volume[label_index] += calc_volume
               
        return score_volume
                         
    def calculate_score(self, data, seg, spacing=[1,1,1], min_calc_object_pixels=3):
        """
        计算每个分割标签对应的连通域强度
        
        参数:
        - data: 原始图像，形状为 (1, x, y, z)
        - seg: 分割标签，形状为 (n, x, y, z)
        
        返回:
        - intensities: 各分割标签对应的强度值，形状为 (n,)
        """

        # 获取原始图像的灰度值
        original_image = data[0]  # 形状为 (x, y, z)
        
        # 初始化强度值数组
        score_agatston = np.zeros(np.max(seg))
        score_volume = np.zeros(np.max(seg))
        score_mass = np.zeros(np.max(seg))

        window_level  = 130
        window_width = 850

        lower_bound = window_level - (window_width / 2)
        upper_bound = window_level + (window_width / 2)
        
        seg = seg[0]
        # 针对每个分割标签
        for label_index in range(np.max(seg)):
            # print('current label:', label_index+1)
            # 获取当前标签的分割图
            current_seg = (seg == label_index+1).astype(np.uint8)
            
            # 连通域检测
            labeled_array, num_features = ndimage.label(current_seg,structure=np.ones((3, 3, 3)))
            
            # 对每个连通域计算强度
            for feature in range(1, num_features + 1):  # feature 从 1 开始

                label = np.zeros(current_seg.shape)
                label[labeled_array == feature] = 1
                

                original_image = original_image*(upper_bound-lower_bound)/255+lower_bound
                calc_object = original_image * label


                calc_pixel_count = np.sum(label)

                # pixel_volume = (spacing[0]*spacing[1]*spacing[2])/3

                calc_volume = calc_pixel_count


                # object_agatston = round(self.get_object_agatston(calc_object, calc_volume))
                object_agatston = 0
                object_mass = np.sum(calc_object)/calc_pixel_count*0.001
                
                score_agatston[label_index] += object_agatston
                score_volume[label_index] += calc_volume
                score_mass[label_index] += object_mass
               
        return score_agatston,score_volume,score_mass
    
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []
        score_all = np.zeros((self.data_shape[0], len(self.plans['foreground_labels'])), dtype=np.float32)
        marker_sample = np.zeros((self.data_shape[0],), dtype=np.float32)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = True  #understaning this

            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)

             
            # score = properties[self.plans['score_name']]
            # if len(score) > 0:
            #     score_all[j] = score
            #     marker_sample[j] = 1
            # else:
            #     score_all[j] = 0
            #     marker_sample[j] = 0

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            # print(seg.shape)
            score_all[j] = self.calculate_volume(seg)
            if score_all[j] > 0:
                marker_sample[j] = 1

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=-1)

        score_all = torch.from_numpy(score_all)
        marker_sample = torch.from_numpy(marker_sample)
        
        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list): 
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images

            return {'data': data_all, 'target': seg_all, 'keys': selected_keys, 'score':score_all, 'mark':marker_sample}

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys, 'score':score_all, 'mark':marker_sample}


if __name__ == '__main__':
    pass
    # folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    # ds = nnUNetDataset(folder, 0)  # this should not load the properties!
    # dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    # a = next(dl)
