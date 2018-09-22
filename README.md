## Note
Convert [Pascal VOC data sets](http://host.robots.ox.ac.uk/pascal/VOC/) to tf records format. Download from
[Pascal VOC Dataset Mirror](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) for a faster speed.
## Usage
```shell
python3 create_pascal_tf_record.py  \
          --data_dir <data_dir> \
          --year <year> \
          --output_dir <output_dir> \
          --ignore_difficult_instances <ignore_difficult_instances> \
```
1. data_dir: voc data dir, `VOCdevkit` dir.
1. year: VOC2007, VOC2012, merged.
1. output_dir: dir to save tf records.
1. ignore_difficult_instances: save difficult samples or not.

```shell
python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
      --include_masks=1 \
```