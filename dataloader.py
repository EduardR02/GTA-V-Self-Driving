import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import stack
import h5py
import os
import bisect
import cv2
from matplotlib import pyplot as plt
import random
import config


ADE_MEAN = (0.485, 0.456, 0.406)
ADE_STD = (0.229, 0.224, 0.225)
if config.use_dinov3:
    height = 192    # 16 * 12
    width = 240     # 16 * 15
else:
    height = 182    # 14 * 13
    width = 252     # 14 * 18
minimap_mask = np.zeros((config.height, config.width), dtype=np.uint8)
# Minimap coordinates, (before padding for patches)
x1, y1 = 3, config.height - 36  # Top-left corner of minimap
x2, y2 = 50, config.height - 3  # Bottom-right corner of minimap

minimap_mask[y1:y2, x1:x2] = 1
minimap_mask_horizontally_flipped = np.fliplr(minimap_mask)

valid_warp_label = np.array([1, 0, 0, 0], dtype=np.int8)
min_warp_shift = 50
max_warp_shift = 100
jitter_p = 0.75


class RandomSkyMask(A.ImageOnlyTransform):
    def __init__(self, height_limit=(0.3, 0.4), p=0.5):
        super().__init__(p=p)
        self.height_limit = height_limit
        
    def apply(self, img, **params):
        h, w, _ = img.shape
        mask_height = int(h * random.uniform(*self.height_limit))
        img_copy = img.copy()
        img_copy[:mask_height, :, :] = 0
        return img_copy


class H5Dataset(Dataset):
    def __init__(self, data_dirs, train_split, is_train, classifier_type, flip_prob, warp_prob, zoom_prob, shift_labels=True):
        self.train_split = train_split
        self.is_train = is_train
        self.classifier_type = classifier_type
        self.flip_prob = flip_prob
        self.warp_prob = warp_prob
        self.zoom_prob = zoom_prob
        self.label_shift = round(config.fps_at_recording_time / config.fps_at_test_time) if shift_labels else 0
        self.transform = transform
        self.data_files = [os.path.join(data_dir, f) for data_dir in data_dirs for f in sorted(os.listdir(data_dir)) if f.endswith('.h5')]
        self.lookup_table = self._create_lookup_table()

        self.train_transform_with_zoom = A.Compose([
            A.Affine(scale=(1.05, 1.2), p=self.zoom_prob),
            color_jitter_replace,
            # RandomSkyMask(height_limit=(0.3, 0.4), p=0.75),
        ], additional_targets=additional_targets)

    def _create_lookup_table(self):
        lookup = []
        total_samples = 0
        # this is dumb but I don't want to make a new function in the child class
        seq_len = 1 if not hasattr(self, 'sequence_len') else self.sequence_len
        seq_stride = 0 if not hasattr(self, 'sequence_stride') else self.sequence_stride
        effective_sequence_length = (seq_len - 1) * seq_stride + self.label_shift
        for file_path in self.data_files:
            with h5py.File(file_path, 'r', libver='latest', swmr=True) as f:
                file_samples = f['labels'].shape[0]
                file_samples -= effective_sequence_length
                if "stuck" in file_path:
                    # this is a bit confusing, but we only have to account for the shorter than amt_remove_after_pause
                    # case (otherwise -0), because we handle the case if it's longer with the sequence_len > 1 case
                    file_samples -= max(config.amt_remove_after_pause - effective_sequence_length, 0)
                if file_samples <= 0:
                    print(f"Skipping and removing {file_path} as it has {file_samples} samples (no valid samples)")
                    self.data_files.remove(file_path)
                    continue
                total_samples += file_samples
            lookup.append(total_samples)
        return lookup

    def __len__(self):
        train_split_len = self._get_split_idx()
        total_samples = self.lookup_table[-1]
        if self.is_train:
            return train_split_len
        else:
            return total_samples - train_split_len

    def __getitem__(self, idx):
        if not self.is_train:
            idx += self._get_split_idx()

        file_idx = bisect.bisect_right(self.lookup_table, idx)
        local_idx = idx - (self.lookup_table[file_idx - 1] if file_idx > 0 else 0)

        file_path = self.data_files[file_idx]

        if "stuck" in file_path:
            local_idx += config.amt_remove_after_pause - self.label_shift

        with h5py.File(file_path, 'r', libver='latest', swmr=True) as f:
            image = f['images'][local_idx]
            label = f['labels'][local_idx + self.label_shift]
            if self.classifier_type == "bce":
                label = label.astype(np.int8).flatten()
                label = self._to_wasd(label)
            else:
                label = label.flatten().astype(np.float32)
        image, label, warped = self.apply_custom_augmentations(image, label)
        image = self.augment_without_minimap([image], warped)[0]
        image = self.transform(image=image)['image']

        return image, label

    def _get_split_idx(self):
        # exclusive for train, first index of val
        return int(self.lookup_table[-1] * self.train_split)

    def _to_wasd(self, label):
        # Ensure label is at least 2D
        if label.ndim == 1:
            label = label[np.newaxis, :]

        # Create a new array with the same shape as label, but with 4 columns instead of 7
        new_label = np.zeros((*label.shape[:-1], 4), dtype=np.float32)

        # Use broadcasting to apply the conversion to all sequences at once
        new_label[..., 0] = label[..., 0] | label[..., 4] | label[..., 5]  # w
        new_label[..., 1] = label[..., 1] | label[..., 4]  # a
        new_label[..., 2] = label[..., 2]  # s
        new_label[..., 3] = label[..., 3] | label[..., 5]  # d

        return new_label.squeeze()

    def flip_samples(self, images, labels):
        if labels.ndim == 1:
            augmented = flip_image_with_minimap(images)
            labels = labels[np.newaxis, :]
        else:
            augmented = [flip_image_with_minimap(img) for img in images]
        label = self.flip_labels(labels)
        return augmented, label.squeeze()

    def flip_labels(self, labels):
        labels[:, [1, 3]] = labels[:, [3, 1]]
        # additionally swap wd and wa
        if self.classifier_type == "cce":
            labels[:, [4, 5]] = labels[:, [5, 4]]
        return labels

    def apply_custom_augmentations(self, images, labels):
        if not self.is_train:
            return images, labels, False
        warped = False
        if random.random() < self.warp_prob:
            images, labels = warp_samples(images, labels)
            warped = True
        if random.random() < self.flip_prob:
            images, labels = self.flip_samples(images, labels)
        return images, labels, warped

    def augment_without_minimap(self, images, warped):
        if not self.is_train:
            return images
        minimaps = [img[y1:y2, x1:x2] for img in images]
        # for zoom we would technically need to inpaint, becuase between edge and minimap there will be the small edge
        # of the old minimap, but the performance hit is massive for some reason
        transform_to_use = train_transform_no_zoom if warped else self.train_transform_with_zoom
        augmented_images = transform_to_use(**{'image' if i == 0 else f'image{i}': img for i, img in enumerate(images)})
        augmented_images = [augmented_images['image']] + [augmented_images[f'image{i}'] for i in range(1, len(images))]

        for img, minimap in zip(augmented_images, minimaps):
            img[y1:y2, x1:x2] = minimap
        return augmented_images


class TimeSeriesDataset(H5Dataset):
    def __init__(self, data_dirs, train_split, is_train, classifier_type, flip_prob, warp_prob, zoom_prob, sequence_len, sequence_stride, shift_labels=True):
        self.sequence_len = sequence_len
        self.sequence_stride = sequence_stride
        super().__init__(data_dirs, train_split, is_train, classifier_type, flip_prob, warp_prob, zoom_prob, shift_labels)

    def __getitem__(self, idx):
        if not self.is_train:
            idx += self._get_split_idx()

        file_idx = bisect.bisect_right(self.lookup_table, idx)
        local_idx = idx - (self.lookup_table[file_idx - 1] if file_idx > 0 else 0)

        file_path = self.data_files[file_idx]
        sequence_range = (self.sequence_len - 1) * self.sequence_stride
        if "stuck" in file_path:
            local_idx += max(config.amt_remove_after_pause - sequence_range - self.label_shift, 0)
        with h5py.File(file_path, 'r', libver='latest', swmr=True) as f:
            # Get sequence with stride
            img_indices = range(local_idx, local_idx + self.sequence_len * self.sequence_stride, self.sequence_stride)
            label_indices = range(local_idx + self.label_shift, local_idx + self.label_shift + self.sequence_len * self.sequence_stride, self.sequence_stride)
            images = f['images'][img_indices]
            labels = f['labels'][label_indices]
        if self.classifier_type == "bce":
            labels = self._to_wasd(labels.astype(np.int8))
        else:
            labels = labels.astype(np.float32)
        images, labels, warped = self.apply_custom_augmentations(images, labels)
        images = self.augment_without_minimap(images, warped)
        images = stack([self.transform(image=img)['image'] for img in images], dim=0)
        return images, labels


def warp_samples(images, labels):
    if labels.ndim == 1:
        labels = labels[np.newaxis, :]
    # if label we want to perdict is not w or "nothing", don't warp
    if np.any(labels[-1] != valid_warp_label) and np.any(labels[-1]):
        return images, labels.squeeze()

    shift = random.randint(min_warp_shift, max_warp_shift)
    direction = 'left' if random.random() < 0.5 else 'right'
    labels = handle_labels_warp(labels, direction)
    if images.ndim == 3:
        return diagonal_warp(images, shift, direction), labels
    else:
        # not sure if it makes sense to gradually decrease the warp when getting closer to the last image
        return [diagonal_warp(img, shift, direction) for img in images], labels


def handle_labels_warp(labels, direction):
    # not sure if we should adjust the non last labels, for now we don't use them though
    # if the image was shifted to the left, from car pov it looks like we
    # are more right than before, so we should correct by steering left
    if direction == 'left':
        labels[-1][1] = 1   # [w, a, s, d]
    else:
        labels[-1][3] = 1
    return labels.squeeze()


def diagonal_warp(image, shift, direction):
    """
    https://blog.comma.ai/end-to-end-lateral-planning/
    this describes the problem of not being able to self correct, so i tried recreating the solution somewhat
    this being kind of the solution (should also apply KL loss to feature vector, but I usually freeze it anyway)

    It probably makes sense to only apply this augmentation when the label is "w" (going straight), so that we
    can change it to "wa" or "wd" to correct itself
    """
    height, width = image.shape[:2]
    minimap = np.copy(image[y1:y2, x1:x2])  # extract minimap
    # This just works, no need to black out the minimap
    image = cv2.inpaint(image, minimap_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
    # Generate shift values for each row, this makes it so the shift is from the center, so both top and bottom "drift"
    if direction == 'left':
        row_shifts = np.linspace(-int(shift/2), int(shift/2), height, dtype=int)
    else:
        row_shifts = np.linspace(int(shift/2), -int(shift/2), height, dtype=int)

    # Create meshgrid for indices
    x_indices = np.tile(np.arange(width), (height, 1))
    # Apply shifts using broadcasting
    shifted_indices = (x_indices + row_shifts[:, None]) % width
    # Apply warp using advanced indexing
    image = image[np.arange(height)[:, None], shifted_indices]

    # Apply zoom using Albumentations
    zoom_transform = A.Compose([
        A.Affine(scale=1 + (shift / width), p=1),
    ])
    image = zoom_transform(image=image)['image']
    # Reinsert the minimap back into the warped image
    image[y1:y2, x1:x2] = minimap

    return image


def flip_image_with_minimap(image):
    """
    This is nice and all, esp with the minimap flip, but the problem is that this makes you
    drive on the wrong side lol... Not sure if this is a good thing to teach the model, even
    though it's kind of nice for augmenting turns,
    because each turn becomes and example for both a left and a right turn.
    """
    # Extract minimap and flip it independently (centered around itself)
    flipped_minimap = np.fliplr(np.copy(image[y1:y2, x1:x2]))
    # don't need to black out first, this just works
    image = cv2.inpaint(image, minimap_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
    # Flip the entire image horizontally
    image = np.fliplr(image)
    # Reinsert flipped minimap into its original location in the flipped image
    image[y1:y2, x1:x2] = flipped_minimap

    return image


color_jitter_replace = A.OneOf([
    # Increased range to help with darker conditions (tunnels)
    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.1), contrast_limit=(-0.1, 0.3), p=0.7),
    
    # Expanded gamma range to emphasize darker conditions
    A.RandomGamma(gamma_limit=(60, 110), p=0.6),
    
    # Slightly increased value range for lighting variance
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=(-15, 5), p=0.5),
    
    # Slight blue shift more likely (for shadow simulation)
    A.RGBShift(r_shift_limit=(-5, 10), g_shift_limit=(-5, 10), b_shift_limit=(0, 15), p=0.4),
], p=jitter_p)


# not sure about the zoom augmentation, i think it causes training to be quite a bit worse, especially
# because I don't account for if the image has been warped already to not zoom. (easy but a bit annoying to do)
# Define two different transforms so that we can dynamically zoom or not (because we need to account if we warped before)
additional_targets = {f'image{i}': 'image' for i in range(1, config.sequence_len)}
# zoom transform moved to class so that we can control the zoom param

train_transform_no_zoom = A.Compose([
    color_jitter_replace,
    # RandomSkyMask(height_limit=(0.3, 0.4), p=0.75),
], additional_targets=additional_targets)


transform = A.Compose([
    A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, fill=0),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD, max_pixel_value=255.0),
    ToTensorV2(),
])


def get_dataloader(data_dir, batch_size, train_split, is_train, classifier_type, sequence_len=1, 
                  sequence_stride=1, flip_prob=0., warp_prob=0., zoom_prob=0., shift_labels=True, shuffle=True,
                  num_workers=0, persistent_workers=True):
    if sequence_len > 1:
        dataset = TimeSeriesDataset(data_dir, train_split, is_train, classifier_type, flip_prob, warp_prob, zoom_prob, sequence_len, sequence_stride, shift_labels)
    else:
        dataset = H5Dataset(data_dir, train_split, is_train, classifier_type, flip_prob, warp_prob, zoom_prob, shift_labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        pin_memory_device="cuda",
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    return dataloader


# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
def invNormalize(x):
    return (x * np.array(ADE_STD)[None, None, :]) + np.array(ADE_MEAN)[None, None, :]


def test_dataloader():
    data_dirs = ['data/turns']
    sequence_len = 3
    # Test with workers
    train_loader = get_dataloader(data_dirs, 32, 0.95, True, "bce", sequence_len, 20, 0., 0.0, True, 
                                  num_workers=2)  # Try with 2 workers for testing
    # vizualize data with matplotlib until stopped
    for data, label in train_loader:
        for i in range(data.shape[0]):
            print(data.shape)
            if sequence_len > 1:
                for j in range(data.shape[1]):
                    print(data[i][j].max(), data[i][j].min())
                    img = invNormalize(data[i][j].permute(1, 2, 0).numpy())
                    print(label[i][j])
                    plt.imshow(img)
                    plt.show()
            else:
                print(data[i].max(), data[i].min())
                img = invNormalize(data[i].permute(1, 2, 0).numpy())
                print(label[i])
                plt.imshow(img)
                plt.show()


if __name__ == '__main__':
    test_dataloader()
