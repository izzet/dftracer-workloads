import numpy as np

NAME_TO_CMIP = {
    "geopotential": "zg",
    "u_component_of_wind": "ua",
    "v_component_of_wind": "va",
    "temperature": "ta",
    "relative_humidity": "r",
    "specific_humidity": "hus",
}

NAME_TO_VAR = {
    "mean_surface_latent_heat_flux": "mslhf",
    "mean_surface_net_long_wave_radiation_flux": "msnlwrf",
    "mean_surface_net_short_wave_radiation_flux": "msnswrf",
    "mean_surface_sensible_heat_flux": "msshf",
    "mean_top_downward_short_wave_radiation_flux": "mtdnswrf",
    "mean_top_net_long_wave_radiation_flux": "mtnlwrf",
    "mean_top_net_short_wave_radiation_flux": "mtnswrf",
    "skin_temperature": "skt",
    "snow_depth": "snd",
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "10m_wind_speed": "w10",
    "surface_pressure": "sp",
    "toa_incident_solar_radiation": "tisr",
    "toa_incident_solar_radiation_6hr": "tisr_6hr",
    "toa_incident_solar_radiation_12hr": "tisr_12hr",
    "toa_incident_solar_radiation_24hr": "tisr_24hr",
    "total_precipitation": "tp",
    "total_precipitation_6hr": "tp_6hr",
    "total_precipitation_12hr": "tp_12hr",
    "total_precipitation_24hr": "tp_24hr",
    "land_sea_mask": "lsm",
    "orography": "orography",
    "slt": "slt",
    "lattitude": "lat2d",
    "longitude": "lon2d",
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "vertical_velocity": "vel",
    "temperature": "t",
    "relative_humidity": "r",
    "specific_humidity": "q",
    "vorticity": "vo",
    "potential_vorticity": "pv",
    "total_cloud_cover": "tcc",
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

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
    "sea_ice_cover",
    "sea_surface_temperature",
    "toa_incident_solar_radiation",
    "toa_incident_solar_radiation_6hr",
    "toa_incident_solar_radiation_12hr",
    "toa_incident_solar_radiation_24hr",
    "total_precipitation_6hr",
    "total_column_water_vapour",
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

single_level_weight_dict = {
    "2m_temperature": 1.0,
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "mean_sea_level_pressure": 0.1,
}

pressure_weights = [l / sum(DEFAULT_PRESSURE_LEVELS) for l in DEFAULT_PRESSURE_LEVELS]  # noqa: E741
pressure_level_weight_dict = {}
for var in PRESSURE_LEVEL_VARS:
    for l, w in zip(DEFAULT_PRESSURE_LEVELS, pressure_weights):  # noqa: E741
        pressure_level_weight_dict[var + "_" + str(l)] = w

WEIGHT_DICT = {**single_level_weight_dict, **pressure_level_weight_dict}

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

NAME_LEVEL_TO_VAR_LEVEL = {}

for var in SINGLE_LEVEL_VARS:
    if var in NAME_TO_VAR:
        NAME_LEVEL_TO_VAR_LEVEL[var] = NAME_TO_VAR[var]

for var in PRESSURE_LEVEL_VARS:
    if var in NAME_TO_VAR:
        for l in DEFAULT_PRESSURE_LEVELS:  # noqa: E741
            NAME_LEVEL_TO_VAR_LEVEL[var + "_" + str(l)] = NAME_TO_VAR[var] + "_" + str(l)

VAR_LEVEL_TO_NAME_LEVEL = {v: k for k, v in NAME_LEVEL_TO_VAR_LEVEL.items()}

BOUNDARIES = {
    "NorthAmerica": {"lat_range": (15, 65), "lon_range": (220, 300)},  # 8x14
    "SouthAmerica": {"lat_range": (-55, 20), "lon_range": (270, 330)},  # 14x10
    "Europe": {"lat_range": (30, 65), "lon_range": (0, 40)},  # 6x8
    "SouthAsia": {"lat_range": (-15, 45), "lon_range": (25, 110)},  # 10, 14
    "EastAsia": {"lat_range": (5, 65), "lon_range": (70, 150)},  # 10, 12
    "Australia": {"lat_range": (-50, 10), "lon_range": (100, 180)},  # 10x14
    "Global": {"lat_range": (-90, 90), "lon_range": (0, 360)},  # 32, 64
}


def get_region_info(region, lat, lon, patch_size):
    region = BOUNDARIES[region]
    lat_range = region["lat_range"]
    lon_range = region["lon_range"]

    h, w = len(lat), len(lon)
    lat_matrix = np.expand_dims(lat, axis=1).repeat(w, axis=1)
    lon_matrix = np.expand_dims(lon, axis=0).repeat(h, axis=0)
    valid_cells = (
        (lat_matrix >= lat_range[0])
        & (lat_matrix <= lat_range[1])
        & (lon_matrix >= lon_range[0])
        & (lon_matrix <= lon_range[1])
    )
    h_ids, w_ids = np.nonzero(valid_cells)
    h_from, h_to = h_ids[0], h_ids[-1]
    w_from, w_to = w_ids[0], w_ids[-1]
    patch_idx = -1
    p = patch_size
    valid_patch_ids = []
    min_h, max_h = 1e5, -1e5
    min_w, max_w = 1e5, -1e5
    for i in range(0, h, p):
        for j in range(0, w, p):
            patch_idx += 1
            if (i >= h_from) & (i + p - 1 <= h_to) & (j >= w_from) & (j + p - 1 <= w_to):
                valid_patch_ids.append(patch_idx)
                min_h = min(min_h, i)
                max_h = max(max_h, i + p - 1)
                min_w = min(min_w, j)
                max_w = max(max_w, j + p - 1)
    return {
        "x_patch_ids": valid_patch_ids,
        "pos_emb_patch_ids": valid_patch_ids,
        "min_h": min_h,
        "max_h": max_h,
        "min_w": min_w,
        "max_w": max_w,
    }
