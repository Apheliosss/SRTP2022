import numpy as np
from osgeo import gdal
from osgeo import osr
import os
import netCDF4 as nc
from tqdm import tqdm
import math
import warnings


def read_tif(tif_path, band=1):
    if tif_path.endswith('.tif') or tif_path.endswith('TIF'):
        dataset = gdal.Open(tif_path)
        pcs = osr.SpatialReference()
        pcs.ImportFromWkt(dataset.GetProjection())
        gcs = pcs.CloneGeogCS()
        extend = dataset.GetGeoTransform()
        prj = dataset.GetProjection()
        shape = (dataset.RasterYSize, dataset.RasterXSize)
    else:
        raise "Unsupported file format!"

    img = dataset.GetRasterBand(band).ReadAsArray()
    dataset = None
    return img, gcs, pcs, extend, prj, shape


warnings.filterwarnings('ignore')
for i in range(1997, 1998):
    print(i)
    for j in tqdm(range(9, 13)):
        chlpath = "../chl_nc/"+str(i)+'/'+str(j).zfill(2)+'/'
        chlfile_list = os.listdir(chlpath)
        chlpath += chlfile_list[0]
        saltpath = "../salt/"+str(i)
        saltfile_list = os.listdir(saltpath)
        saltpath += "/"+saltfile_list[j-1]
        temppath = "../temp/"+str(i)+".tif"
        tempimg, gcs, pcs, extend, prj, shape = read_tif(temppath, j)
        salt_dataset = nc.Dataset(saltpath)
        salt_data = salt_dataset.variables["salinity"][:]
        chl_dataset = nc.Dataset(chlpath)
        try:
            chl_data = chl_dataset.variables['CHL1_mean'][:]
        except:
            chl_data = chl_dataset.variables['CHL2_mean'][:]
        chl = np.zeros(shape, dtype=float)
        temp = np.zeros(shape, dtype=float)
        salt = np.zeros(shape, dtype=float)

        driver = gdal.GetDriverByName("GTiff")
        dst_path = "../merge/"+str(i)+str(j).zfill(2)+".tif"
        dst_dataset = driver.Create(
            dst_path, shape[1], shape[0], 3, gdal.GDT_Float64)
        dst_dataset.SetGeoTransform(extend)
        dst_dataset.SetProjection(prj)
        for row in range(0, shape[0]):
            for col in range(0, shape[1]):
                try:
                    x = extend[0] + col * extend[1] + row * extend[2]
                    y = extend[3] + col * extend[4] + row * extend[5]
                    ct = osr.CoordinateTransformation(gcs, pcs)
                    lon, lat, _ = ct.TransformPoint(x, y)
                    if lon >= 180:
                        lon = lon-360
                    #print(lon, lat)
                    t_chl = float(chl_data[int((90-lat)*4), int((lon+180)*4)])
                    if lon < 0:
                        t_salt = float(salt_data[90+int(lat), 360+int(lon), 0])
                    else:
                        t_salt = float(salt_data[90+int(lat), int(lon), 0])
                    t_temp = float(tempimg[row, col])
                    if not math.isnan(t_chl):
                        chl[row, col] = t_chl
                    if not math.isnan(t_temp):
                        temp[row, col] = t_temp
                    if not math.isnan(t_salt):
                        salt[row, col] = t_salt
                    #print(t_chl, t_salt, t_temp)
                except Exception as e:
                    print(row, col)
                    print(e)
        dst_dataset.GetRasterBand(1).WriteArray(chl)
        dst_dataset.GetRasterBand(2).WriteArray(temp)
        dst_dataset.GetRasterBand(3).WriteArray(salt)
        dst_dataset = None
