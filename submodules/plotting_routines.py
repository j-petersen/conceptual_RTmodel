import numpy as np
import matplotlib.pyplot as plt

def plot_angle(val_array, x_array, y_array, fig, labels=None, ax=None):
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    if ax is None:
        ax = plt.gca()

    levels = MaxNLocator(nbins = 20).tick_values(200,700)
    cmap = plt.get_cmap('GnBu')
    # cmap.set_under(color = 'white')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip = False)
    x_label_place = np.arange(len(x_array))+0.5
    y_label_place = np.arange(len(y_array))+0.5

    ax.set_xticks(x_label_place)
    ax.set_yticks(y_label_place)
    ax.set_xticklabels(np.round(x_array,0))
    ax.set_yticklabels(np.round(y_array,0))
    #ax.tick_params(labelsize=14)
    im = ax.pcolormesh(range(len(x_array)+1),range(len(y_array)+1),val_array.T, cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax = ax, extend = 'min', aspect = 10)
    if labels is not None:
        ax.set_title(labels[0])
        ax.set_xlabel(labels[1])
        ax.set_ylabel(labels[2])
        cbar.ax.set_ylabel(labels[3], rotation = 90, labelpad = 5, fontsize = 14)
    if False: # True for print values
        for yy in range(len(y_array)):
            for xx in range(len(x_array)):
                text = ax.text(x_label_place[xx]-0.2,y_label_place[yy]-0.1,str(round(val_array[xx,yy],2)))
                text = ax.text(xx,yy,val_array.T[yy,xx])
    fig.tight_layout(rect=(0,0,1,1))

def plot_intensity(rad_array, height_array, ax=None, multi_way = True):
    if ax is None:
        ax = plt.gca()

    if multi_way:
        ax.plot(rad_array[0][::-1], height_array, label = 'downward')
        ax.plot(rad_array[1], height_array, label = 'upward')
    if not multi_way:
        ax.plot(rad_array, height_array, label = 'intensity')
    ax.set_xlim(0,1000)
    #ax.set_ylim(0,10)
    ax.set_xlabel(f"Intensity / a.u.")
    ax.set_ylabel(f"Height / km")
    # ax.set_title('multilayer extionction')
    ax.legend(loc='upper left')
