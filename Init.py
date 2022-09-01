from osgeo import gdal
from osgeo import osr
import numpy as np
import pandas as pd
from datetime import datetime
import os
import netCDF4 as nc
from tqdm import tqdm
import math


def read_tif(tif_path, band=1):
    if tif_path.endswith('.tif') or tif_path.endswith('TIF'):
        dataset = gdal.Open(tif_path)
        pcs = osr.SpatialReference()
        pcs.ImportFromWkt(dataset.GetProjection())
        gcs = pcs.CloneGeogCS()
        extend = dataset.GetGeoTransform()
        shape = (dataset.RasterYSize, dataset.RasterXSize)
    else:
        raise "Unsupported file format!"

    img = dataset.GetRasterBand(band).ReadAsArray()
    dataset = None
    return img, gcs, pcs, extend, shape


# def read_salinity(nc_path, lon, lat):
#     dataset = nc.Dataset(nc_path)
#     data = dataset.variables['salinity'][:]
#     if lon < 0:
#         return float(data[90-lat, 360+lon, 0])
#     else:
#         return float(data[90-lat, lon, 0])


# def read_chl(nc_path, lon, lat):
#     lon1 = lon - 0.25
#     lon2 = lon + 0.25
#     lat1 = lat - 0.25
#     lat2 = lat + 0.25
#     dataset = nc.Dataset(nc_path)
#     data = dataset.variables['CHL1_mean'][:]
#     return data[int((90-lat1)*4), int((lon1+180)*4)], data[int((90-lat1)*4), int((lon2+180)*4)], data[int((90-lat2)*4), int((lon1+180)*4)], data[int((90-lat2)*4), int((lon2+180)*4)]


def lonlat_to_rowcol(gcs, pcs, extend, lon, lat):
    ct = osr.CoordinateTransformation(gcs, pcs)
    coord = ct.TransformPoint(lon, lat)
    #print("coord:", coord)
    tx = coord[0]
    ty = coord[1]
    if tx < 0:
        tx += 360
    col = int((tx-extend[0])/extend[1])
    row = int((ty-extend[3])/extend[5])
    return row, col


# def isnum(x):
#     if x > -9999999 and x < 99999999:
#         return True
#     else:
#         # print(type(x))
#         return False


socat = pd.read_csv("../CO2.csv")
active_year = 0
active_month = 0
result = pd.DataFrame(
    columns=["date", "lon", "lat", "fCO2", "Chl", "Temp", "Salt"])
for i in tqdm(range(0, socat.shape[0])):
    try:
        date = datetime.strptime(socat.loc[i]["DATE"], "%Y/%m/%d")
        lon = socat.loc[i]["LON"]
        lat = socat.loc[i]["LAT"]
        if not active_year == date.year or not active_month == date.month:
            path = "../merge/"+str(date.year)+str(date.month).zfill(2)+".tif"
            chl_img, chl_gcs, chl_pcs, chl_extend, chl_shape = read_tif(
                path, 1)
            temp_img, temp_gcs, temp_pcs, temp_extend, temp_shape = read_tif(
                path, 2)
            salt_img, salt_gcs, salt_pcs, salt_extend, salt_shape = read_tif(
                path, 3)
            active_year = date.year
            active_month = date.month

        chl_row, chl_col = lonlat_to_rowcol(
            chl_gcs, chl_pcs, chl_extend, lon-0.25, lat-0.25)
        temp_row, temp_col = lonlat_to_rowcol(
            temp_gcs, temp_pcs, temp_extend, lon-0.25, lat-0.25)
        salt_row, salt_col = lonlat_to_rowcol(
            salt_gcs, salt_pcs, salt_extend, lon-0.25, lat-0.25)
        rec = [socat.loc[i]["DATE"], lon-0.25, lat-0.25, socat.loc[i]["FCO2_AVE_WEIGHTED_YEAR"],
               chl_img[chl_row, chl_col], temp_img[temp_row, temp_col], salt_img[salt_row, salt_col]]
        result.loc[i*4] = rec

        chl_row, chl_col = lonlat_to_rowcol(
            chl_gcs, chl_pcs, chl_extend, lon+0.25, lat-0.25)
        temp_row, temp_col = lonlat_to_rowcol(
            temp_gcs, temp_pcs, temp_extend, lon+0.25, lat-0.25)
        salt_row, salt_col = lonlat_to_rowcol(
            salt_gcs, salt_pcs, salt_extend, lon+0.25, lat-0.25)
        rec = [socat.loc[i]["DATE"], lon+0.25, lat-0.25, socat.loc[i]["FCO2_AVE_WEIGHTED_YEAR"],
               chl_img[chl_row, chl_col], temp_img[temp_row, temp_col], salt_img[salt_row, salt_col]]
        result.loc[i*4+1] = rec

        chl_row, chl_col = lonlat_to_rowcol(
            chl_gcs, chl_pcs, chl_extend, lon-0.25, lat+0.25)
        temp_row, temp_col = lonlat_to_rowcol(
            temp_gcs, temp_pcs, temp_extend, lon-0.25, lat+0.25)
        salt_row, salt_col = lonlat_to_rowcol(
            salt_gcs, salt_pcs, salt_extend, lon-0.25, lat+0.25)
        rec = [socat.loc[i]["DATE"], lon-0.25, lat+0.25, socat.loc[i]["FCO2_AVE_WEIGHTED_YEAR"],
               chl_img[chl_row, chl_col], temp_img[temp_row, temp_col], salt_img[salt_row, salt_col]]
        result.loc[i*4+2] = rec

        chl_row, chl_col = lonlat_to_rowcol(
            chl_gcs, chl_pcs, chl_extend, lon+0.25, lat+0.25)
        temp_row, temp_col = lonlat_to_rowcol(
            temp_gcs, temp_pcs, temp_extend, lon+0.25, lat+0.25)
        salt_row, salt_col = lonlat_to_rowcol(
            salt_gcs, salt_pcs, salt_extend, lon+0.25, lat+0.25)
        rec = [socat.loc[i]["DATE"], lon+0.25, lat+0.25, socat.loc[i]["FCO2_AVE_WEIGHTED_YEAR"],
               chl_img[chl_row, chl_col], temp_img[temp_row, temp_col], salt_img[salt_row, salt_col]]
        result.loc[i*4+3] = rec
        tempimg = None
    except Exception as e:
        print("error:", e)

errors = []
for i in tqdm(range(0, result.shape[0])):
    try:
        if math.isnan(result.loc[i]["Chl"]) or math.isnan(result.loc[i]["Temp"]) or math.isnan(result.loc[i]["Salt"]):
            errors.append(i)
    except Exception as e:
        print("error:", e)

result.drop(labels=errors, inplace=True)
# print(result)
result.to_csv("../data.csv", index=False)
