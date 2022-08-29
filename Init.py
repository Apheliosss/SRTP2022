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
        shape = (dataset.RasterXSize, dataset.RasterYSize)
    else:
        raise "Unsupported file format!"

    img = dataset.GetRasterBand(band).ReadAsArray()
    dataset = None
    return img, gcs, pcs, extend, shape


def read_salinity(nc_path, lon, lat):
    dataset = nc.Dataset(nc_path)
    data = dataset.variables['salinity'][:]
    if lon < 0:
        return float(data[90-lat, 360+lon, 0])
    else:
        return float(data[90-lat, lon, 0])


def read_chl(nc_path, lon, lat):
    lon1 = lon - 0.25
    lon2 = lon + 0.25
    lat1 = lat - 0.25
    lat2 = lat + 0.25
    dataset = nc.Dataset(nc_path)
    data = dataset.variables['CHL1_mean'][:]
    return data[int((90-lat1)*4), int((lon1+180)*4)], data[int((90-lat1)*4), int((lon2+180)*4)], data[int((90-lat2)*4), int((lon1+180)*4)], data[int((90-lat2)*4), int((lon2+180)*4)]


def lonlat_to_rowcol(gcs, pcs, extend, lon, lat):
    ct = osr.CoordinateTransformation(gcs, pcs)
    coord = ct.TransformPoint(lon, lat)
    #print("coord:", coord)
    tx = coord[0]
    ty = coord[1]
    if tx < 0:
        tx += 360
    x = int((tx-extend[0])/extend[1])
    y = int((ty-extend[3])/extend[5])
    return x, y


def isnum(x):
    if x > -9999999 and x < 99999999:
        return True
    else:
        # print(type(x))
        return False


socat = pd.read_csv("../CO2.csv")
result = pd.DataFrame(
    columns=["date", "lon", "lat", "fCO2", "Chl", "Temp", "Salt"])
for i in tqdm(range(0, socat.shape[0])):
    try:
        date = datetime.strptime(socat.loc[i]["DATE"], "%Y/%m/%d")
        chlpath = "../chl_nc/"+date.strftime("%Y/%m/")
        chlfile_list = os.listdir(chlpath)
        chlpath += chlfile_list[0]
        chl1, chl2, chl3, chl4 = read_chl(
            chlpath, socat.loc[i]["LON"], socat.loc[i]["LAT"])
        saltpath = "../salt/"+date.strftime("%Y")
        saltfile_list = os.listdir(saltpath)
        saltpath += "/"+saltfile_list[date.month-1]
        salt = read_salinity(saltpath, int(
            socat.loc[i]["LON"]), int(socat.loc[i]["LAT"]))
        #print("salt", salt)
        temppath = "../temp/"+date.strftime("%Y")+".tif"
        tempimg, tempgcs, temppcs, tempextend, tempshape = read_tif(
            temppath, date.month)
        col, row = lonlat_to_rowcol(
            tempgcs, temppcs, tempextend, socat.loc[i]["LON"]-0.25, socat.loc[i]["LAT"]-0.25)
        # print(chlextend, socat.loc[i]["LON"]-0.25, socat.loc[i]
        #       ["LAT"]-0.25, date.year, date.month, row, col)
        # print(chlimg[row, col])
        # print(tempimg[row, col])
        rec = [socat.loc[i]["DATE"], socat.loc[i]["LON"]-0.25, socat.loc[i]
               ["LAT"]-0.25, socat.loc[i]["FCO2_AVE_WEIGHTED_YEAR"], chl1, tempimg[row, col], salt]
        result.loc[i*4] = rec

        col, row = lonlat_to_rowcol(tempgcs, temppcs, tempextend, socat.loc[i]["LON"]+0.25, socat.loc[i]
                                    ["LAT"]-0.25)
        rec = [socat.loc[i]["DATE"], socat.loc[i]["LON"]+0.25, socat.loc[i]
               ["LAT"]-0.25, socat.loc[i]["FCO2_AVE_WEIGHTED_YEAR"], chl2, tempimg[row, col], salt]
        result.loc[i*4+1] = rec

        col, row = lonlat_to_rowcol(tempgcs, temppcs, tempextend, socat.loc[i]["LON"]-0.25, socat.loc[i]
                                    ["LAT"]+0.25)
        rec = [socat.loc[i]["DATE"], socat.loc[i]["LON"]-0.25, socat.loc[i]
               ["LAT"]+0.25, socat.loc[i]["FCO2_AVE_WEIGHTED_YEAR"], chl3, tempimg[row, col], salt]
        result.loc[i*4+2] = rec

        col, row = lonlat_to_rowcol(tempgcs, temppcs, tempextend, socat.loc[i]["LON"]+0.25, socat.loc[i]
                                    ["LAT"]+0.25)
        rec = [socat.loc[i]["DATE"], socat.loc[i]["LON"]+0.25, socat.loc[i]
               ["LAT"]+0.25, socat.loc[i]["FCO2_AVE_WEIGHTED_YEAR"], chl4, tempimg[row, col], salt]
        result.loc[i*4+3] = rec
        tempimg = None
    except Exception as e:
        print("error:", e)

errors = []
for i in tqdm(range(0, result.shape[0])):
    try:
        if math.isnan(result.loc[i]["Chl"]) or math.isnan(result.loc[i]["Temp"]) or isnum(result.loc[i]["Salt"]) == False:
            errors.append(i)
    except Exception as e:
        print("error:", e)

result.drop(labels=errors, inplace=True)
# print(result)
result.to_csv("../data.csv", index=False)
