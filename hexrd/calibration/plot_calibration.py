import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt


data_dict = pc._extract_powder_lines(fit_tth_tol=1.0)


# %% sample plot to check fit line poistions ahead of fitting
frows = int(np.ceil(np.sqrt(instr.num_panels)))
fcols = int(np.floor(np.sqrt(instr.num_panels)))
fig, ax = plt.subplots(frows, fcols)
fig_row, fig_col = np.unravel_index(np.arange(instr.num_panels),
                                    (frows, fcols))

ifig = 0
for det_key, panel in instr.detectors.items():
    all_pts = np.vstack(data_dict[det_key])
    '''
    pimg = equalize_adapthist(
            rescale_intensity(img_dict[det_key], out_range=(0, 1)),
            10, clip_limit=0.2)
    '''
    pimg = np.array(img_dict[det_key], dtype=float)
    # pimg[~panel.panel_buffer] = np.nan
    ax[fig_row[ifig], fig_col[ifig]].imshow(
        pimg,
        vmin=np.percentile(img_dict[det_key], 5),
        vmax=np.percentile(img_dict[det_key], 90),
        cmap=plt.cm.bone_r
    )
    ideal_angs, ideal_xys, tth_ranges = panel.make_powder_rings(
        plane_data, delta_eta=eta_tol
    )
    rijs = panel.cartToPixel(np.vstack(ideal_xys))
    ax[fig_row[ifig], fig_col[ifig]].plot(rijs[:, 1], rijs[:, 0], 'cx')
    ax[fig_row[ifig], fig_col[ifig]].set_title(det_key)
    rijs = panel.cartToPixel(all_pts[:, :2])
    ax[fig_row[ifig], fig_col[ifig]].plot(rijs[:, 1], rijs[:, 0], 'm+')
    ax[fig_row[ifig], fig_col[ifig]].set_title(det_key)
    ifig += 1
