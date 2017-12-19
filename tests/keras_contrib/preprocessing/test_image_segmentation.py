from keras.preprocessing.image import img_to_array, array_to_img
from keras_contrib.preprocessing.image_segmentation import SegDataGenerator
from keras_contrib.preprocessing import image_segmentation
from PIL import Image as PILImage
import numpy as np


def test_crop(crop_function):
    arr = np.random.random(500, 800)

    img = PILImage.fromarray(arr)

    crop_width = img.width / 5
    crop_height = img.height / 5

    result = crop_function(img_to_array(img), (crop_height, crop_width), 'channels_last')
    result = array_to_img(result)

    assert result.width == crop_width
    assert result.height == crop_height


def test_pair_crop(crop_function):
    arr1 = np.random.random(500, 800)
    arr2 = np.random.random(500, 800)

    img1 = PILImage.fromarray(arr1)
    img2 = PILImage.fromarray(arr2)

    crop_width = img1.width / 5
    crop_height = img1.height / 5

    result1, result2 = crop_function(img_to_array(img1),
                                     img_to_array(img2),
                                     (crop_height, crop_width),
                                     'channels_last')
    result1 = array_to_img(result1)
    result2 = array_to_img(result2)

    assert result1.width == crop_width == result2.width
    assert result2.height == crop_height == result2.height


def test_center_crop():
    test_crop(image_segmentation.center_crop)


def test_random_crop():
    test_crop(image_segmentation.random_crop)


def test_pair_center_crop():
    test_pair_crop(image_segmentation.pair_center_crop)


def test_pair_random_crop():
    test_pair_crop(image_segmentation.pair_random_crop)


def test_seg_data_generator():
    datagen = SegDataGenerator()
