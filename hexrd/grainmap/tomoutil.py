import numpy as np
import scipy.ndimage as img

try:
    import imageio as imgio
except ImportError:
    from skimage import io as imgio

import skimage.transform as xformimg


def gen_bright_field(
    tbf_data_folder,
    tbf_img_start,
    tbf_num_imgs,
    nrows,
    ncols,
    stem='nf_',
    num_digits=5,
    ext='.tif',
):
    tbf_img_nums = np.arange(tbf_img_start, tbf_img_start + tbf_num_imgs, 1)
    tbf_stack = np.zeros([tbf_num_imgs, nrows, ncols])

    print('Loading data for median bright field...')
    for ii in np.arange(tbf_num_imgs):
        print(f'Image #: {ii}')
        tbf_stack[ii, :, :] = imgio.imread(
            tbf_data_folder
            + stem
            + str(tbf_img_nums[ii]).zfill(num_digits)
            + ext
        )

    print('making median...')
    return np.median(tbf_stack, axis=0)


def gen_median_image(
    data_folder: str,
    img_start: int,
    num_imgs: int,
    nrows: int,
    ncols: int,
    stem: str='nf_',
    num_digits: int=5,
    ext: str='.tif',
) -> np.ndarray:
    img_nums = np.arange(img_start, img_start + num_imgs, 1)
    stack = np.zeros([num_imgs, nrows, ncols])

    print('Loading data for median image...')
    for ii in np.arange(num_imgs):
        print(f'Image #: {ii}')
        stack[ii, :, :] = imgio.imread(
            data_folder
            + stem
            + str(img_nums[ii]).zfill(num_digits)
            + ext
        )

    print('making median...')
    return np.median(stack, axis=0)


def gen_attenuation_rads(
    tomo_data_folder,
    tbf,
    tomo_img_start,
    tomo_num_imgs,
    nrows,
    ncols,
    stem='nf_',
    num_digits=5,
    ext='.tif',
    tdf=None,
):
    # Reconstructs a single tompgrahy layer to find the extent of the sample
    tomo_img_nums = np.arange(
        tomo_img_start, tomo_img_start + tomo_num_imgs, 1
    )

    if tdf is None:
        tdf = np.zeros([nrows, ncols])

    rad_stack = np.zeros([tomo_num_imgs, nrows, ncols])

    print('Loading and Calculating Absorption Radiographs ...')
    for ii in np.arange(tomo_num_imgs):
        print(f'Image #: {ii}')
        tmp_img = imgio.imread(
            tomo_data_folder
            + stem
            + str(tomo_img_nums[ii]).zfill(num_digits)
            + ext
        ).astype(float)
        rad_stack[ii, :, :] = -np.log(
            (tmp_img - tdf) / (tbf.astype(float) - tdf)
        )

    return rad_stack


def tomo_reconstruct_layer(
    rad_stack,
    cross_sectional_dim,
    layer_row=1024,
    start_tomo_ang=0.0,
    end_tomo_ang=360.0,
    tomo_num_imgs=360,
    center=0.0,
    pixel_size=0.00148,
):
    sinogram = np.squeeze(rad_stack[:, layer_row, :])

    rotation_axis_pos = -int(np.round(center / pixel_size))
    theta = np.linspace(
        start_tomo_ang, end_tomo_ang, tomo_num_imgs, endpoint=False
    )
    max_rad = int(
        cross_sectional_dim / pixel_size / 2.0 * 1.1
    )  # 10% slack to avoid edge effects

    if rotation_axis_pos >= 0:
        sinogram_cut = sinogram[:, 2 * rotation_axis_pos :]
    else:
        sinogram_cut = sinogram[:, : (2 * rotation_axis_pos)]

    dist_from_edge = (
        np.round(sinogram_cut.shape[1] / 2.0).astype(int) - max_rad
    )

    sinogram_cut = sinogram_cut[:, dist_from_edge:-dist_from_edge]

    print('Inverting Sinogram....')
    reconstruction_fbp = xformimg.iradon(
        sinogram_cut.T, theta=theta, circle=True
    )
    # Rotation to get the result consistent with hexrd, needs to be checked
    reconstruction_fbp = np.rot90(reconstruction_fbp, 3)

    return reconstruction_fbp


def threshold_and_clean_tomo_layer(
    reconstruction_fbp,
    recon_thresh,
    noise_obj_size,
    min_hole_size,
    edge_cleaning_iter=None,
    erosion_iter=1,
    dilation_iter=4,
):
    binary_recon = reconstruction_fbp > recon_thresh

    # hard coded cleaning, grinding sausage...
    binary_recon = img.morphology.binary_dilation(
        binary_recon, iterations=dilation_iter
    )
    binary_recon = img.morphology.binary_erosion(
        binary_recon, iterations=erosion_iter
    )

    labeled_img, num_labels = img.label(binary_recon)

    print('Cleaning and removing Noise...')
    for ii in np.arange(1, num_labels):
        obj1 = np.where(labeled_img == ii)
        if obj1[0].shape[0] < noise_obj_size:
            binary_recon[obj1[0], obj1[1]] = 0

    labeled_img, num_labels = img.label(binary_recon != 1)

    print('Closing Holes...')
    for ii in np.arange(1, num_labels):
        obj1 = np.where(labeled_img == ii)
        if min_hole_size > obj1[0].shape[0] >= 1:
            binary_recon[obj1[0], obj1[1]] = 1

    if edge_cleaning_iter is not None:
        binary_recon = img.morphology.binary_erosion(
            binary_recon, iterations=edge_cleaning_iter
        )
        binary_recon = img.morphology.binary_dilation(
            binary_recon, iterations=edge_cleaning_iter
        )

    return binary_recon


def crop_and_rebin_tomo_layer(
    binary_recon,
    voxel_spacing,
    pixel_size,
    cross_sectional_dim,
    circular_mask_rad=None,
):
    scaling = voxel_spacing / pixel_size
    new_shape = np.round(np.array(binary_recon.shape) * scaling).astype(int)
    tmp_resize = xformimg.resize(binary_recon, new_shape, preserve_range=True)
    binary_recon_bin = np.floor(tmp_resize).astype(bool)
    cut_edge = int(
        np.round(
            (binary_recon_bin.shape[0] * voxel_spacing - cross_sectional_dim)
            / 2
            / voxel_spacing
        )
    )
    binary_recon_bin = binary_recon_bin[cut_edge:-cut_edge, cut_edge:-cut_edge]

    if circular_mask_rad is not None:
        center = binary_recon_bin.shape[0] / 2
        r = np.round(circular_mask_rad / voxel_spacing)
        nx, ny = binary_recon_bin.shape
        y, x = np.ogrid[-center : nx - center, -center : ny - center]

        binary_recon_bin[x * x + y * y > r * r] = 0

    return binary_recon_bin
