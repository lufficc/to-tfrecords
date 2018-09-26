import os
import tensorflow as tf
import abc


class BaseDataset(tf.data.Dataset):
    """simple wrapper of tf.data.TFRecordDataset
    """

    def __init__(self, tfrecord_path):
        """
        Args:
            tfrecord_path: tfrecord_path, string or list of string
        """
        super().__init__()
        if isinstance(tfrecord_path, list):
            filenames = tfrecord_path
        else:
            filenames = [tfrecord_path]
        self._dataset = tf.data.TFRecordDataset(filenames).map(self._parse)

    @abc.abstractmethod
    def _parse(self, raw_record):

        """parse a single example
        Args:
            raw_record: tf example string
        Returns:
            image, bboxes, filename
        """

    def name(self):
        return ''

    def __len__(self):
        return 0

    @property
    @abc.abstractmethod
    def num_classes(self):
        """
        Returns:
            num_classes
        """

    @property
    @abc.abstractmethod
    def class_names_map(self):
        return []

    @property
    def output_classes(self):
        return self._dataset.output_classes

    @property
    def output_shapes(self):
        return self._dataset.output_shapes

    @property
    def output_types(self):
        return self._dataset.output_types

    def _as_variant_tensor(self):
        return self._dataset._as_variant_tensor()  # pylint: disable=protected-access


counter = {
    'VOC2007': {
        'train': 2501,
        'trainval': 5011,
        'val': 2510,
        'test': 4952,
    },
    'VOC2012': {
        'train': 5717,
        'trainval': 11540,
        'val': 5823,
        'test': 10991,
    }
}

VOC_CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]
COCO_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


def _sparse_to_tensor(sparse_tensor, dtype, shape=(-1,), default_value=0):
    return tf.cast(tf.reshape(tf.sparse_tensor_to_dense(sparse_tensor, default_value=default_value), shape=shape), dtype=dtype)


class ObjectDetectionDataset(BaseDataset):
    FEATURES = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/key/sha256': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/source_id': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height': tf.FixedLenFeature((), tf.int64, 1),
        'image/width': tf.FixedLenFeature((), tf.int64, 1),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/class/label': tf.VarLenFeature(tf.int64),
        'image/object/class/text': tf.VarLenFeature(tf.string),
        'image/object/difficult': tf.VarLenFeature(tf.int64),
    }

    def __init__(self, tfrecord_path, use_difficult=False):
        """
        Args:
            use_difficult: bool, if not true, difficult object will not return
        """
        self.use_difficult = use_difficult
        super().__init__(tfrecord_path)

    @property
    def num_classes(self):
        return 0

    @property
    def class_names_map(self):
        return []

    def _parse(self, raw_record):
        examples = tf.parse_single_example(
            raw_record,
            features=self.FEATURES
        )
        # Decode image
        image_raw = tf.image.decode_image(examples['image/encoded'], channels=3)

        image = tf.cast(image_raw, tf.float32)

        filename = tf.cast(examples['image/filename'], tf.string)
        height = tf.cast(examples['image/height'], tf.int32)
        width = tf.cast(examples['image/width'], tf.int32)
        image_shape = tf.stack([height, width, 3])
        image = tf.reshape(image, image_shape)

        xmin = _sparse_to_tensor(examples['image/object/bbox/xmin'], tf.float32)
        xmax = _sparse_to_tensor(examples['image/object/bbox/xmax'], tf.float32)
        ymin = _sparse_to_tensor(examples['image/object/bbox/ymin'], tf.float32)
        ymax = _sparse_to_tensor(examples['image/object/bbox/ymax'], tf.float32)
        difficult = _sparse_to_tensor(examples['image/object/difficult'], tf.int32)
        labels = _sparse_to_tensor(examples['image/object/class/label'], tf.float32)

        if not self.use_difficult:
            difficult_mask = tf.cast(tf.subtract(1, difficult), tf.bool)
            xmin = tf.boolean_mask(xmin, difficult_mask)
            xmax = tf.boolean_mask(xmax, difficult_mask)
            ymin = tf.boolean_mask(ymin, difficult_mask)
            ymax = tf.boolean_mask(ymax, difficult_mask)
            labels = tf.boolean_mask(labels, difficult_mask)

        bboxes = tf.stack([ymin * tf.to_float(height),
                           xmin * tf.to_float(width),
                           ymax * tf.to_float(height),
                           xmax * tf.to_float(width)],
                          axis=1)

        return filename, image, height, width, bboxes, labels


class VOCDataset(ObjectDetectionDataset):

    def __init__(self, tfrecord_dir, years='VOC2007', splits='trainval', name_pattern='{}_{}.tfrecord', **kwargs):
        """

        Args:
            tfrecord_dir: dir
            num_epochs: epoch
            years: VOC2007 or VOC2012 or VOC2007+VOC2012
            splits: train, val, test, trainval, use '+' to add splits
            name_pattern: name pattern
        """
        self.count = 0
        self.years = years
        self.splits = splits
        tfrecord_path = []
        for year in years.split('+'):
            for split in splits.split('+'):
                path = os.path.join(tfrecord_dir, name_pattern.format(year, split))
                tfrecord_path.append(path)
                self.count += counter[year][split]
        super().__init__(tfrecord_path, **kwargs)

    @property
    def num_classes(self):
        return 20

    @property
    def class_names_map(self):
        return VOC_CLASS_NAMES

    def name(self):
        return '{}_{}'.format(self.years, self.splits)

    def __len__(self):
        return self.count


class COCODataset(BaseDataset):
    FEATURES = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/key/sha256': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/source_id': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height': tf.FixedLenFeature((), tf.int64, 1),
        'image/width': tf.FixedLenFeature((), tf.int64, 1),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/class/text': tf.VarLenFeature(tf.string),
        'image/object/is_crowd': tf.VarLenFeature(tf.int64),
    }

    def __init__(self, tfrecord_path, use_mask=False, use_crowd=True):
        """
        Args:
            use_mask: bool, if not true, mask will not return
            use_crowd: bool, if not true, crowd will not return
        """
        self.use_mask = use_mask
        self.use_crowd = use_crowd
        if use_mask:
            self.FEATURES['image/object/mask'] = tf.VarLenFeature(tf.string)
        super().__init__(tfrecord_path)

    @property
    def num_classes(self):
        return len(COCO_CLASS_NAMES)

    @property
    def class_names_map(self):
        return COCO_CLASS_NAMES

    def _parse(self, raw_record):
        examples = tf.parse_single_example(
            raw_record,
            features=self.FEATURES
        )
        # Decode image
        image_raw = tf.image.decode_image(examples['image/encoded'], channels=3)

        image = tf.cast(image_raw, tf.float32)

        filename = tf.cast(examples['image/filename'], tf.string)
        image_id = tf.string_to_number(examples['image/source_id'], tf.int32)
        height = tf.cast(examples['image/height'], tf.int32)
        width = tf.cast(examples['image/width'], tf.int32)
        image_shape = tf.stack([height, width, 3])
        image = tf.reshape(image, image_shape)

        xmin = _sparse_to_tensor(examples['image/object/bbox/xmin'], tf.float32)
        xmax = _sparse_to_tensor(examples['image/object/bbox/xmax'], tf.float32)
        ymin = _sparse_to_tensor(examples['image/object/bbox/ymin'], tf.float32)
        ymax = _sparse_to_tensor(examples['image/object/bbox/ymax'], tf.float32)
        crowd = _sparse_to_tensor(examples['image/object/is_crowd'], tf.int32)
        label_names = tf.sparse_tensor_to_dense(examples['image/object/class/text'], default_value='')
        labels = tf.map_fn(lambda name: tf.where(tf.equal(COCO_CLASS_NAMES, name))[0][0], label_names, dtype=tf.int64)
        labels = tf.to_float(labels)

        if not self.use_crowd:
            crowd_mask = tf.cast(tf.subtract(1, crowd), tf.bool)
            xmin = tf.boolean_mask(xmin, crowd_mask)
            xmax = tf.boolean_mask(xmax, crowd_mask)
            ymin = tf.boolean_mask(ymin, crowd_mask)
            ymax = tf.boolean_mask(ymax, crowd_mask)
            labels = tf.boolean_mask(labels, crowd_mask)

        bboxes = tf.stack([ymin * tf.to_float(height),
                           xmin * tf.to_float(width),
                           ymax * tf.to_float(height),
                           xmax * tf.to_float(width)],
                          axis=1)
        if self.use_mask:
            def decode_png_mask(image_buffer):
                image_mask = tf.squeeze(tf.image.decode_image(image_buffer, channels=1), axis=2)
                image_mask.set_shape([None, None])
                image_mask = tf.to_float(tf.greater(image_mask, 0))
                return image_mask

            png_masks = examples['image/object/mask']
            if isinstance(png_masks, tf.SparseTensor):
                png_masks = tf.sparse_tensor_to_dense(png_masks, default_value='')
            masks = tf.cond(tf.greater(tf.size(png_masks), 0),
                            lambda: tf.map_fn(decode_png_mask, png_masks, dtype=tf.float32),
                            lambda: tf.zeros(tf.to_int32(tf.stack([0, height, width]))))

            results = image_id, filename, image, height, width, masks, bboxes, labels
        else:
            results = image_id, filename, image, height, width, bboxes, labels
        return results


class FullCOCODataset(COCODataset):
    def __init__(self, tfrecord_path, split, **kwargs):
        self.split = split
        assert split in ['train', 'val', 'testdev'], split + ' is not support.'
        base_name = 'coco_{}.tfrecord'.format(split)
        if split == 'train':
            num_shards = 20
        elif split == 'val':
            num_shards = 10
        elif split == 'testdev':
            num_shards = 15
        tfrecord_path = [os.path.join(tfrecord_path, '{}-{:05d}-of-{:05d}'.format(base_name, idx, num_shards)) for idx in range(num_shards)]
        super().__init__(tfrecord_path, **kwargs)


class COCOTrainval35kDataset(COCODataset):
    def __init__(self, tfrecord_path, split, **kwargs):
        self.split = split
        assert split in ['train', 'val'], split + ' is not support.'
        if split == 'train':
            num_shards = 5
            base_name = 'coco_trainval35k.tfrecord'
        else:
            base_name = 'coco_minival.tfrecord'
            num_shards = 2
        tfrecord_path = [os.path.join(tfrecord_path, '{}-{:05d}-of-{:05d}'.format(base_name, idx, num_shards)) for idx in range(num_shards)]
        super().__init__(tfrecord_path, **kwargs)

    def __len__(self):
        return 35500 if self.split == 'train' else 5000
