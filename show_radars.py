__author__ = 'xiyu'
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os, re, sys

# griddata.py - 2010-07-11 ccampo
import numpy as np


def griddata(x, y, z, binsize=0.01, retbin=True, retloc=True):
    """
    Place unevenly spaced 2D data on a grid by 2D binning (nearest
    neighbor interpolation).

    Parameters
    ----------
    x : ndarray (1D)
        The idependent data x-axis of the grid.
    y : ndarray (1D)
        The idependent data y-axis of the grid.
    z : ndarray (1D)
        The dependent data in the form z = f(x,y).
    binsize : scalar, optional
        The full width and height of each bin on the grid.  If each
        bin is a cube, then this is the x and y dimension.  This is
        the step in both directions, x and y. Defaults to 0.01.
    retbin : boolean, optional
        Function returns `bins` variable (see below for description)
        if set to True.  Defaults to True.
    retloc : boolean, optional
        Function returns `wherebins` variable (see below for description)
        if set to True.  Defaults to True.

    Returns
    -------
    grid : ndarray (2D)
        The evenly gridded data.  The value of each cell is the median
        value of the contents of the bin.
    bins : ndarray (2D)
        A grid the same shape as `grid`, except the value of each cell
        is the number of points in that bin.  Returns only if
        `retbin` is set to True.
    wherebin : list (2D)
        A 2D list the same shape as `grid` and `bins` where each cell
        contains the indicies of `z` which contain the values stored
        in the particular bin.

    Revisions
    ---------
    2010-07-11  ccampo  Initial version
    """
    # get extrema values.
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # make coordinate arrays.
    xi = np.arange(xmin, xmax+binsize, binsize)
    yi = np.arange(ymin, ymax+binsize, binsize)
    xi, yi = np.meshgrid(xi, yi)

    # make the grid.
    grid = np.zeros(xi.shape, dtype=x.dtype)
    nrow, ncol = grid.shape
    if retbin: bins = np.copy(grid)

    # create list in same shape as grid to store indices
    if retloc:
        wherebin = np.copy(grid)
        wherebin = wherebin.tolist()

    # fill in the grid.
    for row in range(nrow):
        for col in range(ncol):
            xc = xi[row, col]    # x coordinate.
            yc = yi[row, col]    # y coordinate.

            # find the position that xc and yc correspond to.
            posx = np.abs(x - xc)
            posy = np.abs(y - yc)
            ibin = np.logical_and(posx < binsize/2., posy < binsize/2.)
            ind  = np.where(ibin == True)[0]

            # fill the bin.
            bin = z[ibin]
            if retloc: wherebin[row][col] = ind
            if retbin: bins[row, col] = bin.size
            if bin.size != 0:
                binval         = np.median(bin)
                grid[row, col] = binval
            else:
                grid[row, col] = np.nan   # fill empty bins with nans.

    # return the grid
    if retbin:
        if retloc:
            return grid, bins, wherebin
        else:
            return grid, bins
    else:
        if retloc:
            return grid, wherebin
        else:
            return grid


site_info = ['950', 64.85033, -146.20945]
# generate file list
file_list = []
root_dir = '/media/Seagate Expansion Drive/Data_Xy/Cloudh5/Radars2/'
for site_h5 in os.listdir(root_dir):
    if re.search(site_info[0], site_h5):
        file_list.append(site_h5)
        print(site_h5[9:17])
# read radar input
        hf = h5py.File(root_dir + site_h5, 'r')
        x1 = np.array(hf['sig_cell_lon'])  # coordinate of Radars 36*36
        y1 = np.array(hf['sig_cell_lat'])
        z1 = np.array(hf['cell_sigma0_vv_aft'])  # Radar data 36*36
        site_xy = np.array(hf['site_loc'])  # coordinate of site
        hf.close()
        # z1: from -0.01 to 10.0
        print(np.min(z1))
        print(np.max(z1))
        z1[z1 < -0.01] = -2
        print x1.shape, y1.shape, z1.shape
        # retbin = False
        grid_z, bins, bin_loc = griddata(x1, y1, z1)
        # grid_x = griddata(x1, y1, x1, retbin, True)
        # grid_y = griddata(x1, y1, y1, retbin, True)
        # print type(grid_z)
        plt.pcolormesh(x1, y1, z1, cmap='PuBu_r', vmin=0, vmax=0.3)
        # print(len(x1))
        plt.title(site_h5[9:17])
        # set the limits of the plot to the limits of the data
        # plt.axis([x1.min(), x1.max(), y1.min(), y1.max()])
        plt.colorbar()
        plt.plot(site_info[2], site_info[1], 'k*')
        plt.show()
        plt.savefig('f_b' + site_h5[9:17] + '.png', dpi=120)
        plt.close()
        sys.exit()
# the histogram of the data
# z1.shape = -1,
# n, bins, patches = plt.hist(z1, num_bins, normed=1, facecolor='green', alpha=0.5)
# plt.savefig('h1.png', dpi=120)
# plt.close()