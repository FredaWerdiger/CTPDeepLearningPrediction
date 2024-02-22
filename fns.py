import os
import numpy as np
import matplotlib.pyplot as plt
import math
import glob


class BuildDataset():
    def __init__(self, directory, string):
        images = sorted(
            glob.glob(os.path.join(directory, string, 'images', '*.nii.gz'))
        )
        labels = sorted(
            glob.glob(os.path.join(directory, string, 'masks', '*.nii.gz'))
        )
        nccts = sorted(
            glob.glob(os.path.join(directory, string, 'ncct', '*.nii.gz'))
        )
        self.images_dict = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(images, labels)
        ]
        self.no_seg_dict = [
            {"image": image_name}
            for image_name in images
                            ]

        self.no_seg_ncct_dict = [
            {"image": image_name, "ncct": ncct_name}
            for image_name, ncct_name in zip(images, nccts)
                            ]
        self.ncct_dict = [
            {"image": image_name, "ncct":ncct_name, "label": label_name}
            for image_name, ncct_name, label_name in zip(images, nccts, labels)
        ]


def define_zvalues(ct_img):
    z_min = int(ct_img.shape[2] * .25)
    z_max = int(ct_img.shape[2] * .85)

    steps = int((z_max - z_min) / 18)

    if steps == 0:
        z_min = 0
        z_max = ct_img.shape[2]
        steps = 1

    z = list(range(z_min, z_max))

    rem = int(len(z) / steps) - 18

    if rem < 0:
        add_on = [z[-1] for n in range(abs(rem))]
        z.extend(add_on)
    elif rem % 2 == 0:
        z_min = z_min + int(rem / 2 * steps) + 1
        z_max = z_max - int(rem / 2 * steps) + 1

    elif rem % 2 != 0:
        z_min = z_min + math.ceil(rem / 2)
        z_max = z_max - math.ceil(rem / 2) + 1

    z = list(range(z_min, z_max, steps))

    if len(z) == 19:
        z = z[1:]
    elif len(z) == 20:
        z = z[1:]
        z = z[:18]

    return z


def create_dwi_ctp_proba_map(dwi_ct_img,
                            gt,
                            proba_50,
                            proba_70,
                            proba_90,
                            savefile,
                            z,
                            ext='png',
                            save=False,
                            dpi=250
                            ):
    dwi_ct_img, gt, proba_50, proba_70, proba_90 = [np.rot90(im) for im in [dwi_ct_img, gt, proba_50, proba_70, proba_90]]
    dwi_ct_img, gt, proba_50, proba_70, proba_90 = [np.fliplr(im) for im in [dwi_ct_img, gt, proba_50, proba_70, proba_90]]
    proba_50_mask = proba_50 == 0
    proba_70_mask = proba_70 == 0
    proba_90_mask = proba_90 == 0
    masked_dwi = np.ma.array(dwi_ct_img, mask=~proba_50_mask)
    gt_mask = gt == 0
    masked_dwi_gt = np.ma.array(dwi_ct_img, mask=~gt_mask)
    proba_50 = np.where(proba_50 == 0, np.nan, proba_50)
    proba_70 = np.where(proba_70 == 0, np.nan, proba_70)
    proba_90 = np.where(proba_90 == 0, np.nan, proba_90)
    gt = np.where(gt == 0, np.nan, gt)

    fig, axs = plt.subplots(6, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.1, wspace=-0.3)
    axs = axs.ravel()
    for ax in axs:
        ax.axis("off")
    for i in range(6):
        print(i)

        axs[i].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                      interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
        axs[i].imshow(gt[:, :, z[i]], cmap='Reds', interpolation='hanning', alpha=0.5, vmin=0, vmax=1)
        axs[i+6].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                      interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
        axs[i+6].imshow(proba_50[:, :, z[i]], cmap='gnuplot',
                      interpolation='hanning', alpha=1, vmin=0, vmax=1)
        axs[i+6].imshow(proba_70[:, :, z[i]], cmap='Wistia',
                      interpolation='hanning', alpha=1, vmin=0, vmax=1)
        axs[i+6].imshow(proba_90[:, :, z[i]], cmap='bwr',
                      interpolation='hanning', alpha=1, vmin=0, vmax=1)

    if 12 > len(z):
        max2 = len(z)
    else:
        max2 = 12
    for i in range(6, max2):
        print(i)
        axs[i + 6].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                      interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
        axs[i + 6].imshow(gt[:, :, z[i]], cmap='Reds', interpolation='hanning', alpha=0.5, vmin=0, vmax=1)
        axs[i+12].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                      interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
        im = axs[i+12].imshow(proba_50[:, :, z[i]], cmap='gnuplot',
                      interpolation='hanning', alpha=1, vmin=0, vmax=1)
        axs[i+12].imshow(proba_70[:, :, z[i]], cmap='Wistia',
                      interpolation='hanning', alpha=1, vmin=0, vmax=1)
        axs[i+12].imshow(proba_90[:, :, z[i]], cmap='bwr',
                      interpolation='hanning', alpha=1, vmin=0, vmax=1)
    if not 12 > len(z):
        if len(z) > 18:
            max3 = 18
        else:
            max3 = len(z)
        for i in range(12, max3):
            print(i)
            axs[i + 12].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                              interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
            axs[i + 12].imshow(gt[:, :, z[i]], cmap='Reds', interpolation='hanning', alpha=0.5, vmin=0, vmax=1)
            axs[i + 18].imshow(dwi_ct_img[:, :, z[i]], cmap='gray',
                               interpolation='hanning', vmin=10, vmax=dwi_ct_img.max())
            im = axs[i + 18].imshow(proba_50[:, :, z[i]], cmap='gnuplot',
                                    interpolation='hanning', alpha=1, vmin=0, vmax=1)
            axs[i + 18].imshow(proba_70[:, :, z[i]], cmap='Wistia',
                               interpolation='hanning', alpha=1, vmin=0, vmax=1)
            axs[i + 18].imshow(proba_90[:, :, z[i]], cmap='bwr',
                               interpolation='hanning', alpha=1, vmin=0, vmax=1)


    if savefile:
        plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
        plt.close()


