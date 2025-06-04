import torch
import time
import numpy as np
import random
from util.data_loader import *
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from util.loss_function import *
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from model.BHFPModel_f import *




def calculate_iou(pred_mask, true_mask):
    one = torch.ones_like(pred_mask)
    zero = torch.zeros_like(pred_mask)
    pred_mask = torch.where(pred_mask > 0.5, one, zero)

    intersection = torch.logical_and(pred_mask, true_mask).sum(dim=(2, 3))  # 计算交集
    union = torch.logical_or(pred_mask, true_mask).sum(dim=(2, 3))  # 计算并集
    iou = intersection.float() / (union.float() + 1e-10)  # 避免除以零，加上一个很小的数
    iou = torch.mean(iou)
    return iou


# -----训练函数
def train(net, train_iter, test_iter, num_epochs, lr, device, num_trian, num_test, model_save_path, writer):
    print('training on', device)
    num_batches = len(train_iter)
    # 模型、移动到相应的设备上 cpu or gpu
    net.to(device)
    # 选择优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 定义学习率衰减策略
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=lr / 1000)

    train_batch_step = 0  # 每训练一个batch就加1
    # 开始训练
    for epoch in range(num_epochs):
        start_time = time.time()
        train_rmse_value = 0
        train_loss_classification_list = []
        train_loss_regression_list = []
        train_loss_total_list = []
        trian_iou_value = 0
        net.train()  # 设置为训练模式
        print("开始第 {} 个epoch的训练".format(epoch))
        for i, sample in enumerate(train_iter):
            X, bh_true, pf_true,boundary = sample["feature"], sample["height"], sample["footprint"],sample["boundary"]

            X, bh_true, pf_true,boundary = X.to(device, non_blocking=True), bh_true.to(device, non_blocking=True), pf_true.to(
                device, non_blocking=True),boundary.to(device, non_blocking=True)
            bh_pred, pf_pred = net(X)

            train_loss_classification, train_loss_regression, train_loss_total = building_loss_multi(bh_true, pf_true,
                                                                                                 bh_pred, pf_pred,True,True,boundary)
            train_loss_classification_list.append(train_loss_classification.cpu().detach().numpy())
            train_loss_regression_list.append(train_loss_regression.cpu().detach().numpy())
            train_loss_total_list.append(train_loss_total.cpu().detach().numpy())

            with torch.no_grad():  # 方便精度评估，计算RMSE
                train_rmse_value += F.mse_loss(bh_pred, bh_true, reduction='mean').cpu().detach().numpy() * \
                                    bh_pred.shape[0]
                trian_iou_value += calculate_iou(pf_pred, pf_true).cpu().detach().numpy() * bh_pred.shape[0]

            # 优化器优化模型
            optimizer.zero_grad()
            train_loss_total.backward()
            optimizer.step()

            train_batch_step += 1
            writer.add_scalar("train_loss_total_batch", train_loss_total, train_batch_step)
        # 一个epoch结束后，计算所有样本的RMSE
        train_rmse_total_epoch = np.sqrt(train_rmse_value / num_trian)
        train_loss_total_epoch = np.mean(train_loss_total_list)
        train_loss_classification_epoch = np.mean(train_loss_classification_list)
        train_loss_regression_epoch = np.mean(train_loss_regression_list)
        train_iou_epoch = trian_iou_value / num_trian

        # ************** test ********************
        net.eval()
        with torch.no_grad():
            test_rmse_value = 0
            test_loss_classification_list = []
            test_loss_regression_list = []
            test_loss_total_list = []
            test_iou_value = 0
            for sample in test_iter:
                X, bh_true, pf_true, boundary = sample["feature"], sample["height"], sample["footprint"], sample["boundary"]
                X, bh_true, pf_true,boundary = X.to(device, non_blocking=True), bh_true.to(device,non_blocking=True), pf_true.to(device,
                                        non_blocking=True),boundary.to(device, non_blocking=True)
                bh_pred, pf_pred = net(X)
                test_loss_classification, test_loss_regression, test_loss_total = building_loss_multi(bh_true, pf_true,
                                                                                                 bh_pred, pf_pred,True,True,boundary)
                test_loss_classification_list.append(test_loss_classification.cpu().detach().numpy())
                test_loss_regression_list.append(test_loss_regression.cpu().detach().numpy())
                test_loss_total_list.append(test_loss_total.cpu().detach().numpy())

                test_rmse_value += F.mse_loss(bh_pred, bh_true, reduction='mean').cpu().detach().numpy() * \
                                   bh_pred.shape[0]
                test_iou_value += calculate_iou(pf_pred, pf_true).cpu().detach().numpy() * bh_pred.shape[0]

            test_rmse_total_epoch = np.sqrt(test_rmse_value / num_test)
            test_loss_total_epoch = np.mean(test_loss_total_list)
            test_loss_classification_epoch = np.mean(test_loss_classification_list)
            test_loss_regression_epoch = np.mean(test_loss_regression_list)
            test_iou_epoch = test_iou_value / num_test

        end_time = time.time()
        print("训练第{}个epoch所需时间:{}秒".format(epoch, end_time - start_time))
        print("epoch:{}\t|Train RMSE:{}\t|Train Loss:{}\t|Test RMSE:{}\tTest Loss:{}".format(epoch,
                                                                                             train_rmse_total_epoch,
                                                                                             train_loss_total_epoch,
                                                                                             test_rmse_total_epoch,
                                                                                             test_loss_total_epoch))
        print("*" * 50)

        writer.add_scalar("train_loss_total_epoch", train_loss_total_epoch, epoch)
        writer.add_scalar("train_loss_classification_epoch", train_loss_classification_epoch, epoch)
        writer.add_scalar("train_loss_regression_epoch", train_loss_regression_epoch, epoch)
        writer.add_scalar("train_rmse_total_epoch", train_rmse_total_epoch, epoch)
        writer.add_scalar("train_iou_epoch", train_iou_epoch, epoch)

        writer.add_scalar("test_loss_total_epoch", test_loss_total_epoch, epoch)
        writer.add_scalar("test_loss_classification_epoch", test_loss_classification_epoch, epoch)
        writer.add_scalar("test_loss_regression_epoch", test_loss_regression_epoch, epoch)
        writer.add_scalar("test_rmse_total_epoch", test_rmse_total_epoch, epoch)
        writer.add_scalar("test_iou_epoch", test_iou_epoch, epoch)

        # 记录一下学习率的变化
        lr_scheduler.step()
        writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)

        # ************** save model ********************
        if epoch % 50 == 0:
            save_file_name = os.path.join(model_save_path, 'model_epoch_' + str(epoch) + ".pth")
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_total_epoch,
                'train_rmse': train_rmse_total_epoch,
                'test_loss': test_loss_total_epoch,
                'test_rmse': test_rmse_total_epoch
            }, save_file_name)


if __name__ == '__main__':
    # 设置随机数种子，以便在训练过程中获得可重复的结果
    torch.manual_seed(86)
    torch.cuda.manual_seed(86)
    np.random.seed(86)
    random.seed(86)

    # 设置device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 设置一些参数
    batchsize = 8
    learning_rate = 1e-3
    num_epochs = 201

    # 数据路径
    data_path = "../datasetSample_final"
    model_save_path = "model_save"
    logs_path = "logs"
    # 导入数据
    train_iter, test_iter, num_trian, num_test = load_data_new(data_path, batchsize=batchsize)
    print("训练集样本数量：{}，batch数量：{}".format(num_trian, len(train_iter)))
    print("测试集样本数据：{}，batch数量：{}".format(num_test, len(test_iter)))

    # 模型
    model = BHFPModel_f()

    # 训练
    writer = SummaryWriter(log_dir=logs_path)
    train(model, train_iter, test_iter, num_epochs, learning_rate, device, num_trian, num_test, model_save_path, writer)
    writer.close()
