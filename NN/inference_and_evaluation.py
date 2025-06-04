import torch
from NN.model.BHFPModel_f import *
import os
import tifffile as tif
from osgeo import gdal
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
from NN.util.evaluationUtils import *


if __name__ == '__main__':
    # 设置device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 设置模型
    model = BHFPModel_f()
    resume = 'model_save/92_1_model_epoch_69.pth'
    # 加载模型参数
    model.load_state_dict(torch.load(resume)['state_dict'])
    model.to(device)

    # 设置数据路径
    dataPath="testData/sample"
    dataSavePath = 'testData/output'       # 数据保存路径

    datalist=[]
    # 遍历所有文件名
    file_names = os.listdir(dataPath+"/s1")
    for file_name in file_names:
        # 检查文件名是否以 .tif 结尾
        if file_name.endswith('.tif'):
            # 按 _ 分割文件名
            parts = file_name.split('_')
            # 取前两个部分和最后的 .tif 组成新的文件名
            new_name = '_'.join(parts[1:])
            # 将新文件名添加到过滤后的列表中
            datalist.append(new_name)

    iou_values = []
    f1_values = []
    acc_values = []
    rec_values = []
    prec_values = []
    rmse_values = []
    mae_values = []
    r2_values = []
    mape_values = []

    for data_suffix in datalist:
        s1_path = os.path.join(dataPath, 's1', 's1_' + data_suffix )
        s2_path = os.path.join(dataPath, 's2', 's2_' + data_suffix )
        palsar_path = os.path.join(dataPath, 'palsar', 'palsar_' + data_suffix )
        bh_path = os.path.join(dataPath, 'bh', 'bh_' + data_suffix )
        fp_path = os.path.join(dataPath, 'fp', 'fp_' + data_suffix)
        strm_path = os.path.join(dataPath, 'strm', 'strm_' + data_suffix)


        # 读取数据
        s1_datasat = gdal.Open(s1_path)
        s1_data = s1_datasat.ReadAsArray()
        im_height = s1_datasat.RasterYSize
        im_width = s1_datasat.RasterXSize
        Type = s1_datasat.GetRasterBand(1).DataType
        Transform = s1_datasat.GetGeoTransform()
        Projection = s1_datasat.GetProjection()
        im_band = 1

        # s2
        s2_data = tif.imread(s2_path)
        s2_data = np.transpose(s2_data, (2, 0, 1))



        # palsar
        palsar_data = tif.imread(palsar_path)
        palsar_data = np.transpose(palsar_data, (2, 0, 1))
        # strm
        strm = tif.imread(strm_path)
        strm = strm.reshape(1, 128, 128)

        s1 = torch.tensor(s1_data, dtype=torch.float32)
        s2 = torch.tensor(s2_data, dtype=torch.float32)
        palsar = torch.tensor(palsar_data.astype(np.float32), dtype=torch.float32)
        strm = torch.tensor(strm, dtype=torch.float32)

        # 直接先归一化吧
        data_mean_s2 = [8.8033e-02, 1.0361e-01, 1.0447e-01, 1.7948e-01]
        data_std_s2 = [3.0823e-02, 3.2570e-02, 3.9694e-02, 5.5232e-02]
        data_normalize_s2 = transforms.Normalize(mean=data_mean_s2, std=data_std_s2)
        s2 = data_normalize_s2(s2)

        data_mean_s1 = [-8.0055e+00, -1.5418e+01]
        data_std_s1 = [3.8221e+00, 3.4108e+00]
        data_normalize_s1 = transforms.Normalize(mean=data_mean_s1, std=data_std_s1)
        s1 = data_normalize_s1(s1)

        data_mean_palsar = [7.8584e+03, 2.8027e+03]
        data_std_palsar = [4.6073e+03, 1.4685e+03]
        data_normalize_palsar = transforms.Normalize(mean=data_mean_palsar, std=data_std_palsar)
        palsar = data_normalize_palsar(palsar)

        data_mean_strm = [1.1333e+02]
        data_std_strm = [1.0113e+01]
        data_normalize_strm = transforms.Normalize(mean=data_mean_strm, std=data_std_strm)
        strm = data_normalize_strm(strm)


        # Merge bands
        feature_bands = torch.cat((s2, s1, palsar, strm), dim=0)
        feature_bands = feature_bands.reshape(1, 9, 128, 128)
        feature_bands = feature_bands.to(device, non_blocking=True)

        # 开始预测
        model.eval()
        with torch.no_grad():
            bh, fp = model(feature_bands)

        bh = torch.reshape(bh, (1, 128, 128))
        fp = torch.reshape(fp, (1, 128, 128))
        one = torch.ones_like(fp)
        zero = torch.zeros_like(fp)
        fp = torch.where(fp > 0.5, one, zero)


        bh_true = tif.imread(bh_path)
        bh_true = bh_true.reshape(1,128,128)
        # bh_true=torch.tensor(bh_true, dtype=torch.float32, device=device)

        fp_true = tif.imread(fp_path)
        fp_true = fp_true.reshape(1, 128, 128)
        # fp_true = torch.tensor(fp_true, dtype=torch.float32, device=device)

        bh=bh.cpu().detach().numpy()
        fp = fp.cpu().detach().numpy()


        iou_val, f1, acc, rec, prec, rmse_val, mae_val, r2, mape_val = evaluate_all(bh_true,bh,fp_true,fp)
        iou_values.append(iou_val)
        f1_values.append(f1)
        acc_values.append(acc)
        rec_values.append(rec)
        prec_values.append(prec)
        rmse_values.append(rmse_val)
        mae_values.append(mae_val)
        r2_values.append(r2)
        mape_values.append(mape_val)


        print(data_suffix, iou_val, f1, acc, rec, prec, rmse_val, mae_val, r2, mape_val)

        ##################################################

        # 保存bh
        output_path = os.path.join(dataSavePath,"bh","bh_"+data_suffix )
        driver = gdal.GetDriverByName("GTiff")
        sr = driver.Create(output_path, im_width, im_height, im_band, Type)
        sr.GetRasterBand(1).WriteArray(bh[0])
        sr.SetGeoTransform(Transform)
        sr.SetProjection(Projection)
        sr.FlushCache()
        del sr

        # 保存fp
        output_path = os.path.join(dataSavePath,"fp","fp_"+data_suffix)
        driver = gdal.GetDriverByName("GTiff")
        sr = driver.Create(output_path, im_width, im_height, im_band, Type)
        sr.GetRasterBand(1).WriteArray(fp[0])
        sr.SetGeoTransform(Transform)
        sr.SetProjection(Projection)
        sr.FlushCache()
        del sr

    print("数量：" + str(len(iou_values)))
    print("iou：" , np.mean(iou_values))
    print("f1：" , np.mean(f1_values))
    print("acc：" , np.mean(acc_values))
    print("rec：" , np.mean(rec_values))
    print("prec：" , np.mean(prec_values))
    print("rmse：" , np.mean(rmse_values))
    print("mae：" , np.mean(mae_values))
    print("r2：" , np.mean(r2_values))
    print("mape：" , np.mean(mape_values))

