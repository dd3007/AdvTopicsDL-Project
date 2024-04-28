import os
import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ChestX_ray14(Dataset):
    def __init__(self, data_dir, file, augment,
                 num_class=14, img_depth=3, heatmap_path=None,
                 pretraining=False):
        self.img_list = []
        self.img_label = []

        with open(file, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(data_dir, lineItems[0])
                    imageLabel = lineItems[1:num_class + 1]
                    imageLabel = [int(i) for i in imageLabel]
                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

        self.augment = augment
        self.img_depth = img_depth
        if heatmap_path is not None:
            self.heatmap = Image.open(heatmap_path).convert('RGB')
        else:
            self.heatmap = None
        self.pretraining = pretraining

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        file = self.img_list[index]
        label = self.img_label[index]

        imageData = Image.open(file).convert('RGB')
        if self.heatmap is None:
            imageData = self.augment(imageData)
            img = imageData
            label = torch.tensor(label, dtype=torch.float)
            if self.pretraining:
                label = -1
            return img, label
        else:
            heatmap = self.heatmap
            imageData, heatmap = self.augment(imageData, heatmap)
            img = imageData
            heatmap = heatmap.permute(1, 2, 0)
            label = torch.tensor(label, dtype=torch.float)
            if self.pretraining:
                label = -1
            return [img, heatmap], label
        
class CheXpert(Dataset):
    '''
    Reference:
        @inproceedings{yuan2021robust,
            title={Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
            author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
            booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
            year={2021}
            }
    '''

    def __init__(self,
                 csv_path,
                 image_root_path='',
                 class_index=0,
                 use_frontal=True,
                 use_upsampling=True,
                 flip_label=False,
                 shuffle=True,
                 seed=123,
                 verbose=True,
                 transform=None,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
                 mode='train',
                 heatmap_path=None,
                 pretraining=False
                 ):

        # load data from csv
        self.df = pd.read_csv(csv_path)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '')
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '')
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']

            # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print('Upsampling %s...' % col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        if heatmap_path is not None:
            # self.heatmap = cv2.imread(heatmap_path)
            self.heatmap = Image.open(heatmap_path).convert('RGB')

        else:
            self.heatmap = None

        # impute missing values
        for col in train_cols:
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)

        self._num_images = len(self.df)

        # 0 --> -1
        if flip_label and class_index != -1:  # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)

            # shuffle data
        if shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]

        assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        assert image_root_path != '', 'You need to pass the correct location for the dataset!'

        if class_index == -1:  # 5 classes
            print('Multi-label mode: True, Number of classes: [%d]' % len(train_cols))
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:  # 1 class
            self.select_cols = [train_cols[class_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()

        self.mode = mode
        self.class_index = class_index

        self.transform = transform

        self._images_list = [image_root_path + path for path in self.df['Path'].tolist()]
        if class_index != -1:
            self._labels_list = self.df[train_cols].values[:, class_index].tolist()
        else:
            self._labels_list = self.df[train_cols].values.tolist()

        if verbose:
            if class_index != -1:
                print('-' * 30)
                if flip_label:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[-1] + self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1]))
                    print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                else:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[0] + self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[1], self.value_counts_dict[0]))
                    print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                print('-' * 30)
            else:
                print('-' * 30)
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
                    imratio = self.value_counts_dict[class_key][1] / (
                            self.value_counts_dict[class_key][0] + self.value_counts_dict[class_key][1])
                    imratio_list.append(imratio)
                    print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0]))
                    print('%s(C%s): imbalance ratio is %.4f' % (select_col, class_key, imratio))
                    print()
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list
                print('-' * 30)
        self.pretraining = pretraining

    @property
    def class_counts(self):
        return self.value_counts_dict

    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)

    @property
    def data_size(self):
        return self._num_images

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        # image = cv2.imread(self._images_list[idx], 0)
        # image = Image.fromarray(image)
        # if self.mode == 'train':
        #     image = self.transform(image)
        # image = np.array(image)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        #
        # # resize and normalize; e.g., ToTensor()
        # image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        # image = image / 255.0
        # __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        # __std__ = np.array([[[0.229, 0.224, 0.225]]])
        # image = (image - __mean__) / __std__

        if self.heatmap is None:
            image = Image.open(self._images_list[idx]).convert('RGB')

            image = self.transform(image)

            # image = image.transpose((2, 0, 1)).astype(np.float32)

            if self.class_index != -1:  # multi-class mode
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)
                # label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
            else:
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)

            if self.pretraining:
                label = -1

            return image, label
        else:
            # heatmap = Image.open('nih_bbox_heatmap.png')
            heatmap = self.heatmap
            image = Image.open(self._images_list[idx]).convert('RGB')
            image, heatmap = self.transform(image, heatmap)
            heatmap = heatmap.permute(1, 2, 0)
            # heatmap = torchvision.transforms.functional.to_pil_image(self.heatmap)
            if self.class_index != -1:  # multi-class mode
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)
                # label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
            else:
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)

            if self.pretraining:
                label = -1

            return [image, heatmap], label

if __name__ == '__main__':

    concat_datasets = []
    mean_dict = {
        'chexpert': [0.485, 0.456, 0.406],
        'chestxray_nih': [0.5056, 0.5056, 0.5056],
    }
    std_dict = {
        'chexpert': [0.229, 0.224, 0.225],
        'chestxray_nih': [0.252, 0.252, 0.252],
    }

    args = {
        'datasets_names': ['chexpert', 'chestxray_nih'],
        'random_resize_range': None,
        'input_size': 224,
        'mask_strategy': None,
    }

    print(args['datasets_names'])

    for dataset_name in args['datasets_names']:

        dataset_mean = mean_dict[dataset_name]
        dataset_std = std_dict[dataset_name]
        
        if args['random_resize_range']:
            if args['mask_strategy'] in ['heatmap_weighted', 'heatmap_inverse_weighted']:
                resize_ratio_min, resize_ratio_max = args['random_resize_range']
                print(resize_ratio_min, resize_ratio_max)
                # transform_train = custom_train_transform(size=args['input_size'],
                                                            # scale=(resize_ratio_min, resize_ratio_max),
                                                            # mean=dataset_mean, std=dataset_std)
            else:
                resize_ratio_min, resize_ratio_max = args['random_resize_range']
                print(resize_ratio_min, resize_ratio_max)
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(args['input_size'], scale=(resize_ratio_min, resize_ratio_max),
                                                    interpolation=3),  # 3 is bicubic
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(dataset_mean, dataset_std)])

        else:
            print('Using Directly-Resize Mode. (no RandomResizedCrop)')
            transform_train = transforms.Compose([
                transforms.Resize((args['input_size'], args['input_size'])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(dataset_mean, dataset_std)]
            )

        heatmap_path = None
        if args['mask_strategy'] in ['heatmap_weighted', 'heatmap_inverse_weighted']:
            heatmap_path = 'nih_bbox_heatmap.png'

        if dataset_name == 'chexpert':
            dataset = CheXpert(csv_path="/mnt/home/mpaez/ceph/CheXpert-v1.0-small/train.csv", image_root_path='/mnt/home/mpaez/ceph/CheXpert-v1.0-small/', use_upsampling=False,
                                use_frontal=True, mode='train', class_index=-1, transform=transform_train,
                                heatmap_path=heatmap_path, pretraining=True)
        elif dataset_name == 'chestxray_nih':
            dataset = ChestX_ray14('/mnt/home/mpaez/ceph/chestxray/images', '/mnt/home/mpaez/ceph/chestxray/train_official.txt', augment=transform_train, num_class=14,
                                    heatmap_path=heatmap_path, pretraining=True)
        else:
            raise NotImplementedError

        concat_datasets.append(dataset)

    dataset_train = torch.utils.data.ConcatDataset(concat_datasets)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=4)

    for batch in train_loader:
        images, labels = batch
        print(f"Image batch dimensions: {images.shape}")
        print(f"Labels: {labels}")
        break