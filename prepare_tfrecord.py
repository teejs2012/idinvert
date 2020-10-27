from glob import glob
import cv2
import tensorflow as tf
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def get_tfrecords(img_paths, output_dir, img_size):
    random.shuffle(
        img_paths)  # As tfrecords only can shuffle in small buffer size, we need to shuffle the total data before generate the tfrecord
    writer = tf.python_io.TFRecordWriter(output_dir)
    for img_path in tqdm(img_paths):
        # print(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose([2, 0, 1])
        example = tf.train.Example(
            features=tf.train.Features(
                feature={"data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
                		"shape": tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape))
                }))
        writer.write(example.SerializeToString())
    writer.close()

#do train
# img_dir = '/mnt/teejs/Datasets/portrait_train/*'
# tf_dir = '/mnt/teejs/Datasets/portrait_tfrecord/train.tfrecord'
# imgs_paths = glob(img_dir)
# get_tfrecords(imgs_paths,tf_dir,512)

# #do test
# img_dir = '/mnt/teejs/Datasets/portrait_test/*'
# tf_dir = '/mnt/teejs/Datasets/portrait_tfrecord/test.tfrecord'
# imgs_paths = glob(img_dir)
# get_tfrecords(imgs_paths,tf_dir,512)

img_dir = '/mnt/teejs/Datasets/portraits/*'
all_imgs = glob(img_dir)
train_imgs, validate_imgs = train_test_split (all_imgs, test_size = 0.1)

train_record = '/mnt/teejs/Datasets/portrait_tfrecord/train.tfrecord'
get_tfrecords(train_imgs,train_record,512)
test_record = '/mnt/teejs/Datasets/portrait_tfrecord/test.tfrecord'
get_tfrecords(validate_imgs,test_record,512)
