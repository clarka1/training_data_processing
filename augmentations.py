#!/usr/bin/env python3
from imgaug import augmenters as iaa
class myAffine(iaa.Affine):
    def __init__(self, scale=None, translate_percent=None, translate_px=None,
                 rotate=None, shear=None, order=1, cval=0, mode="constant",
                 fit_output=False, backend="auto",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(myAffine, self).__init__(scale=scale, translate_percent=translate_percent, translate_px=translate_px,
                                       rotate=rotate, shear=shear, order=order, cval=cval, mode=mode,
                                       fit_output=fit_output, backend=backend,
                                       seed=seed, name=name,
                                       random_state=random_state, deterministic=deterministic)
        self._mode_segmentation_maps = mode


def applyAugmentation(images, masks, seed=None):
    noiseList = [iaa.SaltAndPepper(p=(0, 0.005), per_channel=True),
                 iaa.SaltAndPepper(p=(0, 0.005), per_channel=False),
                 iaa.MultiplyElementwise(mul=(0.95, 1.05), per_channel=True),
                 iaa.MultiplyElementwise(mul=(0.95, 1.05), per_channel=False),
                 iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255), per_channel=False),
                 iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255), per_channel=True),
                 iaa.AdditivePoissonNoise(lam=(0, 10), per_channel=False),
                 iaa.AdditivePoissonNoise(lam=(0, 10), per_channel=True)]

    contrastList = [iaa.GammaContrast(gamma=(0.1, 3.0), per_channel=True),
                    iaa.GammaContrast(gamma=(0.1, 3.0), per_channel=False),
                    iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True),
                    iaa.AllChannelsCLAHE(clip_limit=(1, 5), tile_grid_size_px=(3, 12), tile_grid_size_px_min=3,
                                         per_channel=True),
                    iaa.LogContrast(gain=(0.6, 1.4), per_channel=True),
                    iaa.LinearContrast((0.4, 2.5), per_channel=True),
                    iaa.Multiply(mul=(0, 2)),
                    iaa.Multiply(mul=(0, 2), per_channel=True),
                    iaa.AllChannelsHistogramEqualization()]

    geomList = [myAffine(scale=(0.8, 1.2), translate_percent=None, translate_px=None, rotate=(-360, 360),
                         shear=(-45, 45), order=3, mode='symmetric', fit_output=False, seed=seed),
                # iaa.PerspectiveTransform(scale=(0.0, 0.1)), done by affine
                iaa.ElasticTransformation(alpha=(0, 1.0), sigma=(0.5, 1), seed=seed),
                # iaa.Rot90((0, 3)), done by affine
                iaa.Fliplr(0.5, seed=seed),
                iaa.Flipud(0.5, seed=seed)]

    otherList = [iaa.GaussianBlur(sigma=(0, 1.0)),
                 iaa.Clouds(),
                 iaa.Fog()]

    seq = iaa.Sequential([iaa.OneOf(contrastList), iaa.OneOf(noiseList), iaa.OneOf(geomList),
                          iaa.Sometimes(0.5, iaa.OneOf(otherList))], random_order=False)
    
    aug_images, aug_masks = seq(images=images, segmentation_maps=masks)

    return (aug_images, aug_masks)
