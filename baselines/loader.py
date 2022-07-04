
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr
import cf2cdm
import glob
from datetime import datetime
import glob


class ENS10GridDataset(Dataset):
    """
    data_path: the path of folder storing the preprocessed ENS10 & ERA5 data
    target_var: one of "t850", "z500", "t2m" indicating target variable to predict
    dataset_type: one of "train", "test" indicating generating train (1998-2015) or test (2016-2017) dataset
    The dataset will return the mean/std of the target_var in ERA5 +
    the mean/std of all variables in ENS10 on the same pressure level
    """

    def __init__(self, data_path, target_var, dataset_type="train", normalized=True):
        suffix = ""
        if normalized:
            suffix = "_normalized"
        if dataset_type == "train":
            time_range = slice("1998-01-01", "2015-12-31")
        elif dataset_type == "test":
            time_range = slice("2016-01-01", "2017-12-31")
        if target_var in ["t850", "z500"]:
            ds_mean = xr.open_dataset(f"{data_path}/ENS10_pl_mean{suffix}.nc", chunks={"time": 10}).sel(time=time_range)
            ds_std = xr.open_dataset(f"{data_path}/ENS10_pl_std{suffix}.nc", chunks={"time": 10}).sel(time=time_range)
            self.variables = ["Z", "T", "Q", "W", "D", "U", "V"]
            if target_var == "t850":
                self.ds_mean = ds_mean.sel(plev=85000)
                self.ds_std = ds_std.sel(plev=85000)
                self.ds_era5 = xr.open_dataset(f"{data_path}/ERA5_t850.nc", chunks={"time": 10}).sel(
                    time=time_range).isel(plev=0).T
            elif target_var == "z500":
                self.ds_mean = ds_mean.sel(plev=50000)
                self.ds_std = ds_std.sel(plev=50000)
                self.ds_era5 = xr.open_dataset(f"{data_path}/ERA5_z500.nc", chunks={"time": 10}).sel(
                    time=time_range).isel(plev=0).Z
        elif target_var in ["t2m"]:
            self.ds_mean = xr.open_dataset(f"{data_path}/ENS10_sfc_mean{suffix}.nc", chunks={"time": 10}).sel(
                time=time_range)
            self.ds_std = xr.open_dataset(f"{data_path}/ENS10_sfc_std{suffix}.nc", chunks={"time": 10}).sel(
                time=time_range)
            self.ds_era5 = xr.open_dataset(f"{data_path}/ERA5_sfc_t2m.nc", chunks={"time": 10}).sel(
                time=time_range).T2M
            self.variables = ['SSTK', 'TCW', 'TCWV', 'CP', 'MSL', 'TCC', 'U10M', 'V10M', 'T2M', 'TP', 'SKT']

        self.length = len(self.ds_era5)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, int):

            inputs = torch.zeros((len(self.variables) * 2, 361, 720))
            means = self.ds_mean.isel(time=idx).compute()
            stds = self.ds_std.isel(time=idx).compute()

            for k in range(len(self.variables)):
                variable = self.variables[k]
                inputs[2 * k, :] = torch.as_tensor(means[variable].to_numpy())
                inputs[2 * k + 1, :] = torch.as_tensor(stds[variable].to_numpy())

            targets = torch.from_numpy(self.ds_era5.compute().to_numpy()[idx])

            return inputs, targets




class ENS10PointDataset(Dataset):
    """
    data_path: the path of folder storing the preprocessed ENS10 & ERA5 data
    nbatch: number of batches (assuming batch size = 1)
    nsample: number of samples per batch
    target_var: one of "t850", "z500", "t2m" indicating target variable to predict
    dataset_type: one of "train", "test" indicating generating train (1998-2015) or test (2016-2017) dataset
    normalized: whether to use normalized data
    """
    def __init__(self, data_path, nsample, target_var, dataset_type = "train", normalized=True, location_embedding=True, return_time=False):
        suffix = ""
        if normalized:
            suffix = "_normalized"
        if dataset_type == "train":
            time_range = slice("1998-01-01", "2015-12-31")
        elif dataset_type == "test":
            time_range = slice("2016-01-01", "2017-12-31")
        if target_var in ["t850", "z500"]:
            ds_mean = xr.open_dataset(f"{data_path}/ENS10_pl_mean{suffix}.nc", chunks={"time": 1}, engine="h5netcdf").sel(time=time_range)
            ds_std = xr.open_dataset(f"{data_path}/ENS10_pl_std{suffix}.nc", chunks={"time": 1}, engine="h5netcdf").sel(time=time_range)
            self.ds_scale_mean = xr.open_dataset(f"{data_path}/ENS10_pl_mean.nc", chunks={"time": 1}, engine="h5netcdf").sel(time=time_range)
            self.ds_scale_std = xr.open_dataset(f"{data_path}/ENS10_pl_std.nc", chunks={"time": 1}, engine="h5netcdf").sel(time=time_range)
            self.variables = ["Z", "T", "Q", "W", "D", "U", "V"]
            if target_var == "t850":
                self.ds_mean = ds_mean.sel(plev=85000)
                self.ds_std = ds_std.sel(plev=85000)
                self.ds_era5 = xr.open_dataset(f"{data_path}/ERA5_t850.nc", chunks={"time": 1}, engine="h5netcdf").sel(time=time_range).isel(plev=0).T
                self.ds_scale_mean = self.ds_scale_mean.sel(plev=85000).T
                self.ds_scale_std = self.ds_scale_std.sel(plev=85000).T
            elif target_var == "z500":
                self.ds_mean = ds_mean.sel(plev=50000)
                self.ds_std = ds_std.sel(plev=50000)
                self.ds_era5 = xr.open_dataset(f"{data_path}/ERA5_z500.nc", chunks={"time": 1}, engine="h5netcdf").sel(time=time_range).isel(plev=0).Z
                self.ds_scale_mean = self.ds_scale_mean.sel(plev=50000).Z
                self.ds_scale_std = self.ds_scale_std.sel(plev=50000).Z
                
        elif target_var in ["t2m"]:
            self.ds_mean = xr.open_dataset(f"{data_path}/ENS10_sfc_mean{suffix}.nc", chunks={"time": 1}, engine="h5netcdf").sel(time=time_range)
            self.ds_std = xr.open_dataset(f"{data_path}/ENS10_sfc_std{suffix}.nc", chunks={"time": 1}, engine="h5netcdf").sel(time=time_range)
            self.ds_era5 = xr.open_dataset(f"{data_path}/ERA5_sfc_t2m.nc", chunks={"time": 1}, engine="h5netcdf").sel(time=time_range).T2M
            self.variables = ['SSTK', 'TCW', 'TCWV', 'CP', 'MSL', 'TCC', 'U10M', 'V10M', 'T2M', 'TP', 'SKT']
            self.ds_scale_mean = xr.open_dataset(f"{data_path}/ENS10_sfc_mean.nc", chunks={"time": 1}, engine="h5netcdf").sel(time=time_range).T2M
            self.ds_scale_std = xr.open_dataset(f"{data_path}/ENS10_sfc_std.nc", chunks={"time": 1}, engine="h5netcdf").sel(time=time_range).T2M
        self.nbatch = self.ds_mean.time.shape[0]
        self.nsample = nsample
        self.shape = [self.ds_mean[i].shape[0] for i in ["time", "lat", "lon"]]
        self.location_embedding = location_embedding
        self.return_time = return_time

    def __len__(self):
        return self.nbatch

    def __getitem__(self, idx):
        if isinstance(idx, int):
            tind = idx
            mean_sampled = self.ds_mean.isel(time=tind).stack(space=["lat","lon"]).compute()
            std_sampled = self.ds_std.isel(time=tind).stack(space=["lat","lon"]).compute()
            input_len = len(self.variables)*2
            if self.location_embedding:
                input_len = input_len + 3
            inputs = torch.zeros((input_len, mean_sampled.lat.shape[0]))
            for k in range(len(self.variables)):
                variable = self.variables[k]
                inputs[2*k, :] = torch.as_tensor(mean_sampled[variable].to_numpy())
                inputs[2*k+1, :] = torch.as_tensor(std_sampled[variable].to_numpy())
            if self.location_embedding:
                lat = torch.as_tensor(mean_sampled.lat.to_numpy()/180*np.pi)
                lon = torch.as_tensor(mean_sampled.lon.to_numpy()/180*np.pi)
                #inputs[-4, :] = lat.cos()
                inputs[-3, :] = lat.sin()
                inputs[-2, :] = lat.cos()*lon.sin()
                inputs[-1, :] = lat.cos()*lon.cos()
            inputs = inputs.transpose(0, 1) # shape: (#sample, #var+2)
            targets = torch.as_tensor(self.ds_era5.isel(time=tind).stack(space=["lat","lon"]).compute().to_numpy())
            scale_mean = torch.as_tensor(self.ds_scale_mean.isel(time=tind).stack(space=["lat","lon"]).compute().to_numpy())
            scale_std = torch.as_tensor(self.ds_scale_std.isel(time=tind).stack(space=["lat","lon"]).compute().to_numpy())
            if self.return_time:
                return self.ds_mean.time[tind].dt.strftime("%Y-%m-%d").item(), inputs, targets, scale_mean, scale_std
            else:
                return inputs, targets, scale_mean, scale_std


class ENS10EnsembleDataset(Dataset):
    def __init__(self, data_path, nsample, target_var, num_ensemble=10, dataset_type="train", normalized=True, return_time=False):
        self.num_ensemble = num_ensemble
        self.normalized = normalized
        self.return_time = return_time
        if dataset_type == "train":
            time_range = ("19980101", "20151231")
        elif dataset_type == "test":
            time_range = ("20160101", "20171227")
        if target_var in ["z500", "t850"]:
            level_type = "pl"
            era5_prefix = f"ERA5_{target_var}"
            self.variables = ["z", "t", "q", "w", "d", "u", "v"]
            if target_var == "z500":
                self.level = 500
                self.target_var = "z"
                self.value_range = {"z": (48200, 58000), "t": (230, 269), "q": (0., 4e-3), "w": (-0.7, 1.4), "d": (-5e-5, 8e-5), "u": (-7., 27.), "v": (-7., 7.)}
            if target_var == "t850":
                self.level = 850
                self.target_var = "t"
                self.value_range = {"z": (10000, 15500), "t":(240, 299), "q": (0., 1.5e-2), "w": (-1.2, 1.8), "d": (-1.9e-4, 1.6e-4), "u": (-16., 17.5), "v": (-10., 16.)}
        elif target_var in ["t2m"]:
            level_type = "sfc"
            era5_prefix = f"ERA5_sfc_{target_var}"
            self.variables = ['sst', 'tcw', 'tcwv', 'cp', 'msl', 'tcc', 'u10', 'v10', 't2m', 'tp', 'skt']
            self.level = 0
            self.target_var = "t2m"
            self.value_range = {"sst": (260., 305.), "tcw": (0., 60.), "tcwv": (0., 60.), "cp": (0., 0.04), "msl": (97000., 1.1e5), "tcc": (0., 1.0), "u10": (-13., 11.), "v10": (-10., 15.), "t2m":(218, 304), "tp": (0., 0.07), "skt": (210., 310.)}
        prefix = f"{data_path}/ensemble/output.{level_type}."
        self.ens10_files = [f for f in glob.glob(f"{prefix}*.grib") if (f >= f"{prefix}{time_range[0]}.grib" and f <= f"{prefix}{time_range[1]}.grib")]
        self.ds_era5 = xr.open_dataarray(f"{data_path}/{era5_prefix}.nc", chunks={"time":1}, engine="h5netcdf")

    def __len__(self):
        return len(self.ens10_files)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            ds_ens10 = xr.load_dataset(self.ens10_files[idx], engine="cfgrib", backend_kwargs={"indexpath":"", "filter_by_keys": {'level': self.level}}).fillna(9999.0)
            ds_ens10 = ds_ens10.stack(space=["latitude","longitude"])
            timestr = ds_ens10.valid_time.dt.strftime("%Y-%m-%d").item()
            ds_era5 = self.ds_era5.sel(time=timestr).compute()
            variable = self.target_var
            values_orig = torch.as_tensor(ds_ens10[variable].to_numpy()[:self.num_ensemble,:])
            if self.normalized:
                minval, maxval = self.value_range[variable]
                values = (values_orig - minval) / (maxval - minval)
            else:
                values = values_orig
            inputs = values.transpose(0, 1)
            targets = torch.as_tensor(ds_era5.to_numpy()).ravel()
            scale_std, scale_mean = torch.std_mean(values_orig, dim=0, unbiased=False)
            if self.return_time:
                return timestr, inputs, targets, scale_mean, scale_std
            else:
                return inputs, targets, scale_mean, scale_std

# dataloader = DataLoader(ENS10PointDataset(data_path, 1, 10, "t2m", "test"), batch_size=1, ...)


def loader_prepare(args):
    # Grid-wise data
    if args.model == 'UNet':
        trainloader = DataLoader(ENS10GridDataset(data_path=args.data_path,
                                                  target_var=args.target_var,
                                                  dataset_type='train'),
                                 args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

        testloader = DataLoader(ENS10GridDataset(data_path=args.data_path,
                                                 target_var=args.target_var,
                                                 dataset_type='test'),
                                args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    elif args.model == 'MLP':
        trainloader = DataLoader(ENS10PointDataset(data_path=args.data_path,
                                                   nsample=361*720,
                                                   target_var=args.target_var,
                                                   dataset_type='train', location_embedding=True),
                                 args.batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=8, persistent_workers=True)

        testloader = DataLoader(ENS10PointDataset(data_path=args.data_path,
                                                  nsample=361*720,
                                                  target_var=args.target_var,
                                                  dataset_type='test', location_embedding=True, return_time=True),
                                args.batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=False)
    elif args.model == "EMOS":
        trainloader = DataLoader(ENS10EnsembleDataset(data_path=args.data_path,
                                                      nsample=361*720,
                                                      target_var=args.target_var,
                                                      dataset_type='train', num_ensemble=args.ens_num),
                                 args.batch_size, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=4, persistent_workers=True)

        testloader = DataLoader(ENS10EnsembleDataset(data_path=args.data_path,
                                                     nsample=361*720,
                                                     target_var=args.target_var,
                                                     dataset_type='test', num_ensemble=args.ens_num, return_time=True),
                                args.batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=False)
    else:
        pass

    return trainloader, testloader