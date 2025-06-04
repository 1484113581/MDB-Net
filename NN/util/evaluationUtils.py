# -*- coding:utf-8 -*-
# @Time      :2025/2/20 22:07
# @Author    :Chen

"""
确定一些评估指标
"""
import os
import tifffile as tif
import numpy as np
from sklearn.metrics import mean_squared_error


##################### 建筑物足迹相关指标 #############################################
def iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-7)  # 防止除零

def calculate_iou(ground_truth,predicted):
    # 计算交并比（IoU）
    intersection = np.sum(np.logical_and(predicted == 1, ground_truth == 1))
    union = np.sum(np.logical_or(predicted == 1, ground_truth == 1))
    iou = intersection / union if union > 0 else 0
    return iou


# 准确率
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / y_true.size


# 召回率
def recall(y_true, y_pred):
    tp = np.sum(y_true * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    return tp / (tp + fn + 1e-7)


# 精确率
def precision(y_true, y_pred):
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    return tp / (tp + fp + 1e-7)


def f1score(y_true, y_pred):
    ppv = precision(y_true, y_pred)
    tpr = recall(y_true, y_pred)
    return (2*ppv*tpr)/(ppv+tpr+1e-7)

##################### 建筑物高度相关指标 #############################################
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# R方（决定系数）
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / (ss_tot + 1e-7))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-7))) * 100


#########################################
# 足迹提取评估函数
def evaluate_segmentation(y_true, y_pred):
    # y_true和y_pred为二值化矩阵（0或1）
    iou_val = iou(y_true, y_pred)
    f1 = f1score(y_true, y_pred)
    acc = accuracy(y_true, y_pred)
    rec = recall(y_true, y_pred)
    prec = precision(y_true, y_pred)

    # # 检查是否为全负样本
    # # if np.sum(y_true) == 0 and np.sum(y_pred) == 0:
    # if np.sum(y_pred) == 0:
    #     return 1,1,acc,1,1
    # 检查是否为全负样本
    if np.sum(y_true) == 0 and np.sum(y_pred) == 0:
        return 1, 1, acc, 1, 1

    return iou_val,f1,acc,rec,prec
    # return {'IoU': iou_val, 'f1score': f1, 'Accuracy': acc, 'Recall': rec, 'Precision': prec}

# 高度估算评估函数
def evaluate_regression(y_true, y_pred):
    rmse_val = rmse(y_true, y_pred)
    mae_val = mae(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    return rmse_val,mae_val,r2,mape_val
    # return {'RMSE': rmse_val, 'MAE': mae_val, 'R²': r2, 'MAPE (%)': mape_val}

def evaluate_all(bh_true,bh_pred,fp_true,fp_pred):
    iou_val, f1, acc, rec, prec = evaluate_segmentation(fp_true,fp_pred)
    rmse_val, mae_val, r2, mape_val = evaluate_regression(bh_true,bh_pred)
    return iou_val, f1, acc, rec, prec,rmse_val, mae_val, r2, mape_val

def calculate_rmse(ground_truth_heights,predicted_heights):
    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mean_squared_error(ground_truth_heights, predicted_heights))
    return rmse

if __name__ == "__main__":
    datasetPath = r"D:\Test\Pycharm\sentinle12\NN\dataTest"
    base_files = [f for f in os.listdir(os.path.join(datasetPath, "bh")) if f.endswith(".tif")]
    base_names = ['_'.join(f.split('_')[1:])[:-4] for f in base_files]  # 提取"baoding_1"部分
    for base in base_names:
        bh = tif.imread(os.path.join(datasetPath, 'bh', 'bh_'+base+'.tif'))
        bh_pred = tif.imread(os.path.join(datasetPath, 'bh_pred', 'bh_'+base+'.tif'))
        fp = tif.imread(os.path.join(datasetPath, 'fp', 'fp_' + base + '.tif'))
        fp_pred = tif.imread(os.path.join(datasetPath, 'fp_pred', 'fp_' + base + '.tif'))

        iou_val, f1, acc, rec, prec = evaluate_segmentation(fp,fp_pred)
        rmse_val, mae_val, r2, mape_val = evaluate_regression(bh,bh_pred)


        print(base,iou_val, f1, acc, rec, prec,rmse_val, mae_val, r2, mape_val)




