import argparse
import os

import xarray as xr
from tqdm import tqdm


DEFAULT_VARS = [
    "anisotropy_of_sub_gridscale_orography",
    "angle_of_sub_gridscale_orography",
    "geopotential_at_surface",
    "high_vegetation_cover",
    "lake_cover",
    "land_sea_mask",
    "low_vegetation_cover",
    "slope_of_sub_gridscale_orography",
    "soil_type",
    "standard_deviation_of_filtered_subgrid_orography",
    "standard_deviation_of_orography",
    "type_of_high_vegetation",
    "type_of_low_vegetation",
    "mean_surface_latent_heat_flux",
    "mean_surface_net_long_wave_radiation_flux",
    "mean_surface_net_short_wave_radiation_flux",
    "mean_surface_sensible_heat_flux",
    "mean_top_downward_short_wave_radiation_flux",
    "snow_depth",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "10m_wind_speed",
    "mean_sea_level_pressure",
    "sea_ice_cover",
    "sea_surface_temperature",
    "surface_pressure",
    "total_cloud_cover",
    "total_precipitation_6hr",
    "total_precipitation_12hr",
    "total_precipitation_24hr",
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "wind_speed",
]

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--start-year", type=int, default=1979)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument("--all-variables", action="store_true")
    parser.add_argument("--variables", nargs="*", default=None)

    args = parser.parse_args()
    
    file = args.file
    save_dir = args.save_dir
    
    os.makedirs(save_dir, exist_ok=True)
    ds = xr.open_zarr('gs://weatherbench2/datasets/era5/' + file)
    
    years = list(range(args.start_year, args.end_year + 1))
    if args.all_variables:
        variables = list(ds.keys())
    elif args.variables is not None and len(args.variables) > 0:
        variables = args.variables
    else:
        variables = DEFAULT_VARS

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
