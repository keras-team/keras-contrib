import pytest
from keras_contrib.preprocessing import image_3d
import numpy as np
import os
import shutil
import tempfile


class TestImage3d:

    def setup_class(cls):
        img_w = img_h = img_d = 20
        images = []
        for n in range(8):
            bias = np.random.rand(img_w, img_h, img_d, 1) * 64
            variance = np.random.rand(img_w, img_h, img_d, 1) * (255 - 64)
            imarray = np.random.rand(img_w, img_h, img_d, 3) * variance + bias
            images.append(imarray)

        cls.all_test_images = [images]

    def teardown_class(cls):
        del cls.all_test_images

    def test_image_data_generator(self, tmpdir):
        for test_images in self.all_test_images:
            img_list = []
            img_list.append(test_images)
            images = np.vstack(img_list)
            generator = image_3d.Image3DDataGenerator(
                rotation_range=90.,
                rotation_choice_number=1,
                width_shift_range=0.1,
                height_shift_range=0.1,
                depth_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=0.,
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=False,
                vertical_flip=False,
                depth_flip=False,
                rescale=None,
                preprocessing_function=None,
                data_format='channels_last',
            )
            generator.fit(images, augment=True)

            for x, y in generator.flow(images, np.arange(images.shape[0]),
                                       shuffle=True, save_to_dir=str(tmpdir)):
                assert x.shape[1:] == images.shape[1:]
                break

    def test_image_data_generator_invalid_data(self):
        generator = image_3d.Image3DDataGenerator(
            data_format='channels_last')
        # Test fit with invalid data
        with pytest.raises(ValueError):
            x = np.random.random((3, 10, 10, 10))
            generator.fit(x)

        # Test flow with invalid data
        with pytest.raises(ValueError):
            x = np.random.random((32, 10, 10, 10))
            generator.flow(np.arange(x.shape[0]))

    def test_image_data_generator_fit(self):
        generator = image_3d.Image3DDataGenerator(
            zoom_range=(0.2, 0.2, 0.2),
            data_format='channels_last')
        # Test grayscale
        x = np.random.random((32, 10, 10, 10, 1))
        generator.fit(x)
        # Test RBG
        x = np.random.random((32, 10, 10, 10, 3))
        generator.fit(x)
        generator = image_3d.Image3DDataGenerator(
            data_format='channels_first')
        # Test grayscale
        x = np.random.random((32, 1, 10, 10, 10))
        generator.fit(x)
        # Test RBG
        x = np.random.random((32, 3, 10, 10, 10))
        generator.fit(x)

    def test_random_transforms(self):
        x = np.random.random((28, 28, 28, 2))
        # x axis
        assert image_3d.random_transform(x, rotation_range=45.0, rotation_choice_number=1).shape == (28, 28, 28, 2)
        # y axis
        assert image_3d.random_transform(x, rotation_range=45.0, rotation_choice_number=2).shape == (28, 28, 28, 2)
        # z axis
        assert image_3d.random_transform(x, rotation_range=45.0, rotation_choice_number=3).shape == (28, 28, 28, 2)
        # Shift
        assert image_3d.random_transform(x, height_shift_range=20).shape == (28, 28, 28, 2)
        assert image_3d.random_transform(x, width_shift_range=20).shape == (28, 28, 28, 2)
        assert image_3d.random_transform(x, depth_shift_range=20).shape == (28, 28, 28, 2)
        # Shear
        assert image_3d.random_transform(x, shear_range=20).shape == (28, 28, 28, 2)
        # Zoom
        assert image_3d.random_transform(x, zoom_range=[5, 5, 5]).shape == (28, 28, 28, 2)
        # Flip
        assert image_3d.random_transform(x, horizontal_flip=True, vertical_flip=True).shape == (28, 28, 28, 2)
        assert image_3d.random_transform(x, vertical_flip=True, depth_flip=True).shape == (28, 28, 28, 2)
        assert image_3d.random_transform(x, horizontal_flip=True, vertical_flip=True, depth_flip=True).shape == (28, 28, 28, 2)


if __name__ == '__main__':
    pytest.main([__file__])
