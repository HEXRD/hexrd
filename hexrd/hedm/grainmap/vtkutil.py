import logging
import os

import numpy as np
logger = logging.getLogger(__name__)


def output_grain_map_vtk(
    data_location, data_stems, output_stem, vol_spacing, top_down=True
):

    num_scans = len(data_stems)

    confidence_maps = [None] * num_scans
    grain_maps = [None] * num_scans
    Xss = [None] * num_scans
    Yss = [None] * num_scans
    Zss = [None] * num_scans

    for ii in np.arange(num_scans):
        logger.info(f'Loading Volume {ii}...')
        conf_data = np.load(
            os.path.join(data_location, data_stems[ii] + '_grain_map_data.npz')
        )

        confidence_maps[ii] = conf_data['confidence_map']
        grain_maps[ii] = conf_data['grain_map']
        Xss[ii] = conf_data['Xs']
        Yss[ii] = conf_data['Ys']
        Zss[ii] = conf_data['Zs']

    # assumes all volumes to be the same size
    num_layers = grain_maps[0].shape[0]

    total_layers = num_layers * num_scans

    num_rows = grain_maps[0].shape[1]
    num_cols = grain_maps[0].shape[2]

    grain_map_stitched = np.zeros((total_layers, num_rows, num_cols))
    confidence_stitched = np.zeros((total_layers, num_rows, num_cols))
    Xs_stitched = np.zeros((total_layers, num_rows, num_cols))
    Ys_stitched = np.zeros((total_layers, num_rows, num_cols))
    Zs_stitched = np.zeros((total_layers, num_rows, num_cols))

    for i in np.arange(num_scans):
        if top_down == True:
            grain_map_stitched[
                ((i) * num_layers) : ((i) * num_layers + num_layers), :, :
            ] = grain_maps[num_scans - 1 - i]
            confidence_stitched[
                ((i) * num_layers) : ((i) * num_layers + num_layers), :, :
            ] = confidence_maps[num_scans - 1 - i]
            Xs_stitched[
                ((i) * num_layers) : ((i) * num_layers + num_layers), :, :
            ] = Xss[num_scans - 1 - i]
            Zs_stitched[
                ((i) * num_layers) : ((i) * num_layers + num_layers), :, :
            ] = Zss[num_scans - 1 - i]
            Ys_stitched[
                ((i) * num_layers) : ((i) * num_layers + num_layers), :, :
            ] = (Yss[num_scans - 1 - i] + vol_spacing * i)
        else:

            grain_map_stitched[
                ((i) * num_layers) : ((i) * num_layers + num_layers), :, :
            ] = grain_maps[i]
            confidence_stitched[
                ((i) * num_layers) : ((i) * num_layers + num_layers), :, :
            ] = confidence_maps[i]
            Xs_stitched[
                ((i) * num_layers) : ((i) * num_layers + num_layers), :, :
            ] = Xss[i]
            Zs_stitched[
                ((i) * num_layers) : ((i) * num_layers + num_layers), :, :
            ] = Zss[i]
            Ys_stitched[
                ((i) * num_layers) : ((i) * num_layers + num_layers), :, :
            ] = (Yss[i] + vol_spacing * i)

    logger.info('Writing VTK data...')
    # VTK Dump
    Xslist = Xs_stitched[:, :, :].ravel()
    Yslist = Ys_stitched[:, :, :].ravel()
    Zslist = Zs_stitched[:, :, :].ravel()

    grainlist = grain_map_stitched[:, :, :].ravel()
    conflist = confidence_stitched[:, :, :].ravel()

    num_pts = Xslist.shape[0]
    num_cells = (total_layers - 1) * (num_rows - 1) * (num_cols - 1)

    f = open(os.path.join(data_location, output_stem + '_stitch.vtk'), 'w')

    f.write('# vtk DataFile Version 3.0\n')
    f.write('grainmap Data\n')
    f.write('ASCII\n')
    f.write('DATASET UNSTRUCTURED_GRID\n')
    f.write('POINTS %d double\n' % (num_pts))

    for i in np.arange(num_pts):
        f.write('%e %e %e \n' % (Xslist[i], Yslist[i], Zslist[i]))

    scale2 = num_cols * num_rows
    scale1 = num_cols

    f.write('CELLS %d %d\n' % (num_cells, 9 * num_cells))
    for k in np.arange(Xs_stitched.shape[0] - 1):
        for j in np.arange(Xs_stitched.shape[1] - 1):
            for i in np.arange(Xs_stitched.shape[2] - 1):
                base = scale2 * k + scale1 * j + i
                p1 = base
                p2 = base + 1
                p3 = base + 1 + scale1
                p4 = base + scale1
                p5 = base + scale2
                p6 = base + scale2 + 1
                p7 = base + scale2 + scale1 + 1
                p8 = base + scale2 + scale1

                f.write(
                    '8 %d %d %d %d %d %d %d %d \n'
                    % (p1, p2, p3, p4, p5, p6, p7, p8)
                )

    f.write('CELL_TYPES %d \n' % (num_cells))
    for i in np.arange(num_cells):
        f.write('12 \n')

    f.write('POINT_DATA %d \n' % (num_pts))
    f.write('SCALARS grain_id int \n')
    f.write('LOOKUP_TABLE default \n')
    for i in np.arange(num_pts):
        f.write('%d \n' % (grainlist[i]))

    f.write('FIELD FieldData 1 \n')
    f.write('confidence 1 %d float \n' % (num_pts))
    for i in np.arange(num_pts):
        f.write('%e \n' % (conflist[i]))

    f.close()
