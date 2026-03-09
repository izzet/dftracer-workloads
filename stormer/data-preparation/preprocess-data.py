# https://github.com/tung-nd/stormer/blob/main/stormer/data_preprocessing/process_one_step_data.py

import os
import argparse
import numpy as np
import xarray as xr
import h5py
from tqdm import tqdm


# taken from https://github.com/tung-nd/stormer/blob/main/stormer/utils/data_utils.py
CONSTANTS = [
    "anisotropy_of_sub_gridscale_orography",
    "orography",
    "land_sea_mask",
    "slt",
    "lattitude",
    "longitude",
    "angle_of_sub_gridscale_orography",
    "geopotential_at_surface",
    "high_vegetation_cover",
    "lake_cover",
    "lake_depth",
    "low_vegetation_cover",
    "slope_of_sub_gridscale_orography",
    "soil_type",
    "standard_deviation_of_filtered_subgrid_orography",
    "standard_deviation_of_orography",
    "type_of_high_vegetation",
    "type_of_low_vegetation",
]

SINGLE_LEVEL_VARS = [
    "mean_surface_latent_heat_flux",
    "mean_surface_net_long_wave_radiation_flux",
    "mean_surface_net_short_wave_radiation_flux",
    "mean_surface_sensible_heat_flux",
    "mean_top_downward_short_wave_radiation_flux",
    "mean_top_net_long_wave_radiation_flux",
    "mean_top_net_short_wave_radiation_flux",
    "skin_temperature",
    "snow_depth",
    
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "10m_wind_speed",
    "surface_pressure",
    'sea_ice_cover',
    'sea_surface_temperature',
    "toa_incident_solar_radiation",
    "toa_incident_solar_radiation_6hr",
    "toa_incident_solar_radiation_12hr",
    "toa_incident_solar_radiation_24hr",
    'total_precipitation_6hr',
    'total_column_water_vapour',
    "total_precipitation_6hr",
    "total_precipitation_12hr",
    "total_precipitation_24hr",
    "total_cloud_cover",
    "land_sea_mask",
    "orography",
    "lattitude",
]

PRESSURE_LEVEL_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "wind_speed",
    "temperature",
    "relative_humidity",
    "specific_humidity",
    "vorticity",
    "potential_vorticity",
]

DEFAULT_PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# This is for 0.25deg
# VARS = [
#     "angle_of_sub_gridscale_orography", # x
#     "geopotential_at_surface", # x
#     "high_vegetation_cover", # x
#     "lake_cover", #x 
#     "lake_depth", # x
#     "land_sea_mask", # x
#     "low_vegetation_cover", # x
#     "slope_of_sub_gridscale_orography", # x
#     "soil_type", # x
#     "standard_deviation_of_filtered_subgrid_orography",  # x
#     "standard_deviation_of_orography", # x
#     "type_of_high_vegetation", # x
#     "type_of_low_vegetation", # x
    
#     "mean_surface_latent_heat_flux", # x
#     "mean_surface_net_long_wave_radiation_flux", # x
#     "mean_surface_net_short_wave_radiation_flux", # x
#     "mean_surface_sensible_heat_flux", # x
#     "mean_top_downward_short_wave_radiation_flux", # x
#     "mean_top_net_long_wave_radiation_flux", # x
#     "mean_top_net_short_wave_radiation_flux", # x
#     "skin_temperature", # x
#     "snow_depth", # x
    
#     "2m_temperature", # x
#     "10m_u_component_of_wind", # x
#     "10m_v_component_of_wind", # x
#     "10m_wind_speed", # x
#     "mean_sea_level_pressure", # x
    
#     "sea_ice_cover", # x
#     "sea_surface_temperature", # x
#     "surface_pressure", # x
#     "toa_incident_solar_radiation", # x
#     "toa_incident_solar_radiation_6hr", # x
#     "toa_incident_solar_radiation_12hr", # x
#     "toa_incident_solar_radiation_24hr", # x
#     "total_cloud_cover", # x
#     "total_precipitation_6hr", # x
#     "total_precipitation_12hr", # x
#     "total_precipitation_24hr", # x
#     "total_column_water_vapour", # x
    
#     "geopotential", # x
#     "specific_humidity", # x
#     "temperature", # x
#     "u_component_of_wind", # x
#     "v_component_of_wind", # x
#     "vertical_velocity", # x
#     "wind_speed" # x
# ]

# this is for 1.4deg
VARS = [
    "angle_of_sub_gridscale_orography",
    "geopotential_at_surface",
    "high_vegetation_cover",
    "lake_cover",
    "lake_depth",
    "land_sea_mask",
    "low_vegetation_cover",
    "slope_of_sub_gridscale_orography",
    "soil_type",
    "standard_deviation_of_filtered_subgrid_orography",
    "standard_deviation_of_orography",
    "type_of_high_vegetation",
    "type_of_low_vegetation",

    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "10m_wind_speed",
    "2m_temperature",
    "mean_sea_level_pressure",

    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity"
]

XR_OPEN_KWARGS = {
    "cache": False,
}


def open_netcdf(path):
    errors = []
    for engine in ("h5netcdf", "netcdf4"):
        try:
            return xr.open_dataset(path, engine=engine, **XR_OPEN_KWARGS)
        except Exception as exc:
            errors.append(f"{engine}: {exc}")
    raise OSError(f"failed to open {path} with supported backends: {'; '.join(errors)}")



def create_one_step_dataset(root_dir, save_dir, split, years, list_vars, chunk_size=None):
    save_dir_split = os.path.join(save_dir, split)
    os.makedirs(save_dir_split, exist_ok=True)
    
    list_constant_vars = [v for v in list_vars if v in CONSTANTS]
    list_single_vars = [v for v in list_vars if v in SINGLE_LEVEL_VARS and v not in CONSTANTS]
    list_pressure_vars = [v for v in list_vars if v in PRESSURE_LEVEL_VARS]
    
    # Load coordinates and constant fields once. Reopening constants inside the
    # per-sample loop is unnecessarily expensive and stresses Lustre metadata.
    with open_netcdf(os.path.join(root_dir, f"{list_constant_vars[0]}.nc")) as ds_constant:
        lat = ds_constant.latitude.to_numpy()
        lat.sort()
        lon = ds_constant.longitude.to_numpy()
        lon.sort()

    constant_fields = {}
    for var in list_constant_vars:
        constant_path = os.path.join(root_dir, f"{var}.nc")
        with open_netcdf(constant_path) as ds_constant_var:
            constant_field = ds_constant_var[var].to_numpy()
            constant_fields[var] = constant_field.reshape(constant_field.shape[-2:])

    np.save(os.path.join(save_dir, 'lat.npy'), lat)
    np.save(os.path.join(save_dir, 'lon.npy'), lon)
    
    for year in tqdm(years, desc='years', position=0):
        with open_netcdf(os.path.join(root_dir, list_single_vars[0], f"{year}.nc")) as ds_sample:
            if chunk_size is not None:
                n_chunks = len(ds_sample.time) // chunk_size + 1
            else:
                n_chunks = 1
                chunk_size = len(ds_sample.time)
        
        idx_in_year = 0
        
        ds_dict = {}
        try:
            for var in (list_single_vars + list_pressure_vars):
                ds_dict[var] = open_netcdf(os.path.join(root_dir, var, f"{year}.nc"))

            for chunk_id in tqdm(range(n_chunks), desc='chunks', position=1, leave=False):
                dict_np = {}
                list_time_stamps = None
                ### convert ds to numpy
                for var in (list_single_vars + list_pressure_vars):
                    ds = ds_dict[var].isel(time=slice(chunk_id * chunk_size, (chunk_id + 1) * chunk_size))
                    if list_time_stamps is None:
                        list_time_stamps = ds.time.values
                    if var in list_single_vars:
                        dict_np[var] = ds[var].values
                    else:
                        available_levels = ds.level.values
                        ds_np = ds[var].values
                        for i, level in enumerate(available_levels):
                            if level in DEFAULT_PRESSURE_LEVELS:
                                dict_np[f'{var}_{level}'] = ds_np[:, i]
                        
                for i in tqdm(range(len(list_time_stamps)), desc='time stamps', position=2, leave=False):
                    data_dict = {
                        'input': {'time': str(list_time_stamps[i])}
                    }
                    for var, values in dict_np.items():
                        data_dict['input'][var] = values[i]
                    for var, constant_field in constant_fields.items():
                        data_dict['input'][var] = constant_field
                        
                    with h5py.File(os.path.join(save_dir_split, f'{year}_{idx_in_year:04}.h5'), 'w', libver='latest') as f:
                        for main_key, sub_dict in data_dict.items():
                            # Create a group for the main key (e.g., 'input' or 'output')
                            group = f.create_group(main_key)
                            
                            # Now, save each array in the sub-dictionary to this group
                            for sub_key, array in sub_dict.items():
                                if sub_key != 'time':
                                    group.create_dataset(sub_key, data=array, compression=None, dtype=np.float32)
                                else:
                                    group.create_dataset(sub_key, data=array, compression=None)

                    idx_in_year += 1
        finally:
            for ds in ds_dict.values():
                ds.close()


def parse_args():
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--root-dir', type=str, required=True, help='Root directory containing input data.')
    parser.add_argument('--save-dir', type=str, required=True, help='Directory to save regridded files.')
    parser.add_argument('--start-year', type=int, default=1979, help='Start year for the data range.')
    parser.add_argument('--end-year', type=int, default=2019, help='End year for the data range.')
    parser.add_argument("--split", type=str, default="train", help="Split of the dataset (train, val, test).")
    parser.add_argument("--chunk-size", type=int, default=10, help="Chunk size for reading datasets (default=10).")
    
    return parser.parse_args()


def main():
    args = parse_args()

    create_one_step_dataset(
        root_dir=args.root_dir,
        save_dir=args.save_dir,
        split=args.split,
        years=list(range(args.start_year, args.end_year + 1)),
        list_vars=VARS,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()
