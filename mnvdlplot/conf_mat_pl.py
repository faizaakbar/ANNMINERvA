from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


def compute_purity_via_row_norm(arr):
    """
    row normalize a confusion matrix and return the new matrix and a stats
    errors matrix
    """
    pur_arr = np.zeros_like(arr)
    pur_arr_err = np.zeros_like(arr)
    for i in range(np.shape(arr)[0]):
        npass = arr[i, :]
        ntotal = arr.sum(axis=1)[i]
        epsilon = npass / ntotal
        pur_arr[i, :] = epsilon
        pur_arr_err[i, :] = np.sqrt(epsilon * (1 - epsilon) / ntotal)

    return pur_arr, pur_arr_err


def compute_effic_via_col_norm(arr):
    """
    column normalize a confusion matrix and return the new matrix and a stats
    errors matrix
    """
    eff_arr = np.zeros_like(arr)
    eff_arr_err = np.zeros_like(arr)
    for i in range(np.shape(arr)[0]):
        npass = arr[:, i]
        ntotal = arr.sum(axis=0)[i]
        epsilon = npass / ntotal
        eff_arr[:, i] = epsilon
        eff_arr_err[i, :] = np.sqrt(epsilon * (1 - epsilon) / ntotal)

    return eff_arr, eff_arr_err


def make_conf_mat_plots_rowcolnormonly(
        arr, plot_type, top_title='Purity', bottom_title='Efficiency',
        colormap='Reds', print_arrays=False, print_targets=True,
        n_targets=6
):
    """
    plots and text for row and column normalized confusion matrices

    * arr - the confusion matrix
    """
    if n_targets == 5:
        target_plane_codes = {9: 1, 18: 2, 27: 3, 44: 4, 49: 5}
    elif n_targets == 6:
        target_plane_codes = {9: 1, 18: 2, 27: 3, 36: 6, 45: 4, 50: 5}
    else:
        raise ValueError('Illegal number of targets.')
    fig = plt.figure(figsize=(16, 16))
    gs = plt.GridSpec(2, 2)

    # purity calc
    pur_arr, pur_arr_err = compute_purity_via_row_norm(arr)
    eff_arr, eff_arr_err = compute_effic_via_col_norm(arr)

    if print_arrays:
        print("purity (row-normalized diagonal values)")
        for i in range(pur_arr.shape[0]):
            print('segment {:2d}: purity = {:.3f}'.format(i, pur_arr[i, i]))
        print("efficiency (column-normalized diagonal values)")
        for i in range(eff_arr.shape[0]):
            print(
                'segment {:2d}: efficiency = {:.3f}'.format(i, eff_arr[i, i])
            )

    if print_targets:
        print("purity (row-normalized diagonal values)")
        for i in target_plane_codes.keys():
            print('target {:2d}: purity = {:.3f} +/- {:.3f} (stat)'.format(
                target_plane_codes[i], pur_arr[i, i], pur_arr_err[i, i]
            ))
        print("efficiency (column-normalized diagonal values)")
        for i in target_plane_codes.keys():
            print('target {:2d}: efficiency = {:.3f} +/- {:.3f} (stat)'.format(
                target_plane_codes[i], eff_arr[i, i], eff_arr_err[i, i]
            ))

    def make_title_string(title_base, title_mod, logscale):
        title = title_base.format(title_mod)
        title = r'Log$_{10}$ ' + title if logscale else title
        return title

    def make_subplot(ax, show_arr, colormap, title):
        im = ax.imshow(
            show_arr, cmap=plt.get_cmap(colormap),
            interpolation='nearest', origin='lower'
        )
        cbar = plt.colorbar(im, fraction=0.04)
        plt.title(title)
        plt.xlabel('True z-segment')
        plt.ylabel('Reconstructed z-segment')

    # purity linear plots
    ax = plt.subplot(gs[0])
    show_arr = pur_arr
    make_subplot(
        ax, show_arr, colormap,
        make_title_string('Purity (row normalized)\n{}', top_title, False)
    )

    # purity log plots
    ax = plt.subplot(gs[1])
    show_arr = np.log10(pur_arr * 100.0)
    make_subplot(
        ax, show_arr, colormap,
        make_title_string(
            'Purity (row normalized)\n{}', top_title, True
        )
    )

    # efficiency linear plots
    ax = plt.subplot(gs[2])
    show_arr = eff_arr
    make_subplot(
        ax, show_arr, colormap,
        make_title_string(
            'Efficiency (column normalized)\n{}', bottom_title, False
        )
    )

    # efficiency log plots
    ax = plt.subplot(gs[3])
    show_arr = np.log10(eff_arr * 100.0)
    make_subplot(
        ax, show_arr, colormap,
        make_title_string(
            'Efficiency (column normalized)\n{}', bottom_title, True
        )
    )

    fig.savefig(
        'confusion_matrices_rowcolnorm_{}.pdf'.format(plot_type),
        bbox_inches='tight'
    )


def make_conf_mat_plots_raw(arr, plot_type, colormap='Reds'):
    """
    plots and text for raw confusion matrices
    """
    fig = plt.figure(figsize=(16, 16))
    gs = plt.GridSpec(1, 2)

    def make_title_string(title, logscale):
        title = r'Log$_{10}$ ' + title if logscale else title
        return title

    def make_subplot(ax, show_arr, colormap, title):
        im = ax.imshow(
            show_arr, cmap=plt.get_cmap(colormap),
            interpolation='nearest', origin='lower'
        )
        cbar = plt.colorbar(im, fraction=0.04)
        plt.title(title)
        plt.xlabel('True z-segment')
        plt.ylabel('Reconstructed z-segment')

    # linear plots
    ax = plt.subplot(gs[0])
    show_arr = arr
    make_subplot(
        ax, show_arr, colormap,
        make_title_string('Confustion matrix', False)
    )

    # log plots
    ax = plt.subplot(gs[1])
    show_arr = np.log10(arr * 100.0)
    make_subplot(
        ax, show_arr, colormap,
        make_title_string('Confustion matrix', True)
    )

    fig.savefig(
        'confusion_matrices_raw_{}.pdf'.format(plot_type),
        bbox_inches='tight'
    )
