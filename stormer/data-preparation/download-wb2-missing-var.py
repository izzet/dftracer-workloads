import argparse
import os

import xarray as xr
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--start-year", type=int, default=1979)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument("--variables", nargs="*", default=["lake_depth"])

    args = parser.parse_args()
    
    file = args.file
    save_dir = args.save_dir
    
    os.makedirs(save_dir, exist_ok=True)
    ds = xr.open_zarr('gs://weatherbench2/datasets/era5/' + file)
    
    years = list(range(args.start_year, args.end_year + 1))
    variables = args.variables

    for var in tqdm(variables, desc="variables", position=0):
        ds_var = ds[[var]]
        if len(ds_var.dims) < 3: # constant variables
            out_path = os.path.join(save_dir, f"{var}.nc")
            if os.path.exists(out_path):
                continue
            ds_var.to_netcdf(out_path)
        else:
            save_dir_var = os.path.join(save_dir, var)
            os.makedirs(save_dir_var, exist_ok=True)
            for year in tqdm(years, desc="years", position=1, leave=False):
                out_path = os.path.join(save_dir_var, f"{year}.nc")
                if os.path.exists(out_path):
                    continue
                ds_var_year = ds_var.sel(time=str(year))
                ds_var_year.to_netcdf(out_path)
            

if __name__ == "__main__":
    main()
