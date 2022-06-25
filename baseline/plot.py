import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
import numpy as np

def scatter_plot(dataset, data_range, title, axis_name, file_path):
    matplotlib.use('agg')
    plt.figure(figsize=(10, 8), dpi=300)
    for (ens10, era5, _, _) in tqdm(dataset, desc=f'[Plot]',
                                             unit ="Batch", total=len(dataset)):
        # ens10 shape: (batch_size, nsample, 10)
        # era5 shape: (batcch_size, nsample)
        era5 = era5.unsqueeze(-1).expand(ens10.shape)
        plt.scatter(era5.numpy().ravel(), ens10.numpy().ravel(), color="black", s=(data_range[1]-data_range[0])/1024*0.5)
    plt.xlabel(f"{axis_name} from ERA5")
    plt.ylabel(f"{axis_name} from ENS10 with 48h lead time")
    #plt.title(title)
    plt.savefig(file_path)

def hist2d_plot(dataset, data_range, title, axis_name, file_path):
    matplotlib.use('pdf')
    import seaborn as sns
    sns.set_theme(style="white", palette="viridis", font_scale=1.5)
    plt.figure(figsize=(10, 8)) #  dpi=300
    nbins = 1024
    edges = np.linspace(data_range[0], data_range[1], nbins+1)
    bin_area = np.square((data_range[1] - data_range[0]) / nbins)

    try:
        data = np.load(f"{file_path}.npz")
        hist2d = data["hist2d"]
    except:
        hist2d = np.zeros((nbins, nbins))
        for (ens10, era5, _, _) in tqdm(dataset, desc=f'[Plot]',
                                                unit ="Batch", total=len(dataset)):
            # ens10 shape: (batch_size, nsample, 10)
            # era5 shape: (batcch_size, nsample)
            era5 = era5.unsqueeze(-1).expand(ens10.shape)
            H, _, _ = np.histogram2d(era5.numpy().ravel(), ens10.numpy().ravel(), bins=(edges, edges), density=False)
            hist2d += H
        np.savez(f"{file_path}.npz", hist2d=hist2d)
    hist2d = hist2d / hist2d.sum() / bin_area
    hist2d[hist2d < 1e-10] = np.nan
    hist2d = hist2d.T
    plt.imshow(hist2d, interpolation='nearest', origin='lower', extent=[data_range[0], data_range[1], data_range[0], data_range[1]], norm=colors.LogNorm(vmin=np.nanmin(hist2d), vmax=np.nanmax(hist2d)), cmap="viridis")
    plt.plot(list(data_range), list(data_range), "--", linewidth=1.5, color='black', alpha=0.7)
    #plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label('Probability density')
    plt.xlabel(f"{axis_name} from ERA5")
    plt.ylabel(f"{axis_name} from ENS10 with 48h lead time")
    plt.savefig(f"{file_path}.pdf", bbox_inches='tight')