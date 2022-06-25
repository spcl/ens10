import xarray as xr
import numpy as np
import dask
from dask.diagnostics import ProgressBar
from argparse import ArgumentParser
import numba

quantiles = np.linspace(0, 1, 20+1)
dquantile = quantiles[1] - quantiles[0]
# target_var_dict: Dict[target_var, Tuple[grib_var_name, data_kind, level]]
target_var_dict = {"t850":("t", "pl", 850), "t2m":("2t", "sfc", 0), "z500":("z", "pl", 500)}
lead_time = 48 # hour, [0, 24, 48]

def preprocess_data(ens10_path, data_path, target_var):
    var_name, data_kind, level = target_var_dict[target_var]
    os.system(f"grib_copy -w,shortName={var_name},stepRange={lead_time} {ens10_path}/output.{data_kind}.2018* {data_path}/output.{target_var}_48h.grib")

def preproc_quantile(ds):
    ds = ds.assign_coords(year=ds.time.dt.year, day=(ds.time.dt.month-1)*100+(ds.time.dt.day-1)).set_index(time=["year","day"]) #.isel(step=2)
    return ds

def generate_climate_quantile(data_path, target_var, year):
    year_window_len = 18 # years
    with ProgressBar():
        ds = xr.load_dataarray(f"{data_path}/output.{target_var}_48h.grib", engine="cfgrib", backend_kwargs={"indexpath":""})
        ds = preproc_quantile(ds).unstack("time")
        q = ds.sel(year=slice(year-year_window_len, year)).rolling(day=9, center=True, min_periods=5).construct(day="window").chunk({"day":1}).quantile(quantiles, dim=["year", "number", "window"])
        q = q.transpose("day", "latitude", "longitude", "quantile")
        q.to_dataset(name=f"{target_var}_quantile").to_netcdf(f"{data_path}/climate_{target_var}_{year}.nc", encoding={f"{target_var}_quantile":{"dtype": "float32"}})

def generate_ensemble_quantile(data_path, target_var, year):
    with ProgressBar():
        ds = xr.load_dataarray(f"{data_path}/output.{target_var}_48h.grib", engine="cfgrib", backend_kwargs={"indexpath":"", "filter_by_keys": {'year': year}})
        ds = preproc_quantile(ds).sel(year=year)
        res = ds.quantile(quantiles, dim=["number"])
        res = res.transpose("day", "latitude", "longitude", "quantile")
        res.to_dataset(name=f"{target_var}_quantile").to_netcdf(f"{data_path}/ensemble_{target_var}_{year}.nc", encoding={f"{target_var}_quantile":{"dtype": "float32"}})

@numba.guvectorize(["void(float32[:], float32[:], float32[:], float32[:], float32[:])"], "(n),(m),(n),(m) ->()", target="parallel")
def efi_func(quantiles_climate, quantiles_ensemble, cbins, ebins, res):
    dq = dquantile # TODO: account for uneven distribution of quantiles
    quantiles_climate = quantiles_climate[1:-1]
    quantiles_ensemble = quantiles_ensemble[:-1]
    cbins = cbins[1:-1]
    inds = np.searchsorted(quantiles_ensemble, quantiles_climate)
    res[0] = 2/np.pi*((cbins - ebins[inds])/np.sqrt(cbins*(1-cbins))*dq).sum()

def generate_efi(data_path, target_var, year):
    with ProgressBar():
        dsc = xr.open_dataarray(f"{data_path}/climate_{target_var}_{year}.nc").isel(day=slice(0, -1))#chunks={"longitude":50, "latitude":50}
        dse = xr.open_dataarray(f"{data_path}/ensemble_{target_var}_{year}.nc").isel(day=slice(0, -1))#chunks={"longitude":50, "latitude":50}
        dsc = dsc.sel(day=dse.day)
        cbins = dsc["quantile"].astype(np.float32)
        ebins = dse["quantile"].astype(np.float32)
        res = xr.apply_ufunc(efi_func, dsc.astype(np.float32), dse.astype(np.float32), cbins, ebins, input_core_dims=[["quantile"], ["quantile"], ["quantile"], ["quantile"]], vectorize=False)
        res = res.to_dataset(name="efi").swap_dims(day="valid_time").rename(valid_time="time")
        res = res.assign_coords(valid_time=res.time)
        res.to_netcdf(f"{data_path}/efi_{target_var}_{year}.nc")

"""
Script for generating EFI data from the ENS10 dataset
To generate EFI for t2m in 2016 for example, execute:
python calaulate_efi.py --data_path=... --ens10_path=... --target_var=t2m --year=2016 --gen_climate --gen_ensemble --gen_efi
"""

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default=".", help="Path for storing EFI and intermediate data")
    parser.add_argument('--ens10_path', type=str, default=".", help="Path storing then ENS10 dataset")
    parser.add_argument('--target_var', type=str, choices=["t2m", "z500", "t850"])
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--gen_climate', action='store_true', help="generate CDFs for model climate")
    parser.add_argument('--gen_ensemble', action='store_true', help="generate CDFs for ensembles")
    parser.add_argument('--gen_efi', action='store_true', help="generate EFI based on CDFs for model climate and ensembles")
    parser.add_argument('--year', default=2016, type=int, choices=[2016, 2017])
    args = parser.parse_args()
    if args.preprocess:
        preprocess_data(args.ens10_path, args.data_path, args.target_var)
    if args.gen_climate:
        generate_climate_quantile(args.data_path, args.target_var, args.year)
    if args.gen_ensemble:
        generate_ensemble_quantile(args.data_path, args.target_var, args.year)
    if args.gen_efi:
        generate_efi(args.data_path, args.target_var, args.year)