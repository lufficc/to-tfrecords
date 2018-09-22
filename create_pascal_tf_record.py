from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import os

from lxml import etree
from tqdm import tqdm
import PIL.Image
import tensorflow as tf

import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.(VOCdevkit)')
flags.DEFINE_string('year', 'merged', 'Desired challenge year.')
flags.DEFINE_string('output_dir', '', 'Dir to output TFRecord')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore difficult instances')
flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite existed tfrecord')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']

CLASS_NAMES = [
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


def dict_to_tf_example(data,
                       dataset_directory,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
    """Convert XML derived dict to tf.Example proto.
    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.
    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
      image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.
    Returns:
      example: The converted tf.Example.
    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
    full_path = os.path.join(dataset_directory, img_path)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    if 'object' in data:
        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue

            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(CLASS_NAMES.index(obj['name']))
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.
    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.
    Args:
      xml: xml tree obtained by parsing XML file contents using lxml.etree
    Returns:
      Python dictionary holding XML contents.
    """
    if not xml:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def main(_):
    if FLAGS.year not in YEARS:
        raise ValueError('year must be in : {}'.format(YEARS))

    data_dir = FLAGS.data_dir
    years = ['VOC2007', 'VOC2012']
    if FLAGS.year != 'merged':
        years = [FLAGS.year]
    output_dir = FLAGS.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for year in years:
        for split in SETS:
            filename = os.path.join(output_dir, '{}_{}.tfrecord'.format(year, split))
            if os.path.exists(filename):
                if FLAGS.overwrite or os.path.getsize(filename) == 0:
                    os.remove(filename)
                else:
                    print('skipped %s.' % filename)
                    continue
            writer = tf.python_io.TFRecordWriter(filename)
            print('Reading from PASCAL %s %s dataset.' % (year, split))
            examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main', split + '.txt')
            annotations_dir = os.path.join(data_dir, year, 'Annotations')
            examples_list = dataset_util.read_examples_list(examples_path)
            for example in tqdm(examples_list):
                try:
                    path = os.path.join(annotations_dir, example + '.xml')
                    with tf.gfile.GFile(path, 'r') as fid:
                        xml_str = fid.read()
                    xml = etree.fromstring(xml_str)
                    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
                    tf_example = dict_to_tf_example(data, FLAGS.data_dir, FLAGS.ignore_difficult_instances)
                    writer.write(tf_example.SerializeToString())
                except Exception as e:
                    print(str(e))
            writer.close()


if __name__ == '__main__':
    tf.app.run()
