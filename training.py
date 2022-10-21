#!/usr/bin/env Python
# coding=utf-8

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import *
from torchvision import transforms as tr
import torchvision
import torch.nn.functional as F
from Model.model import *
from torch.cuda.amp import GradScaler, autocast

# from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import cv2, os, json, random, time, json, shutil
from tqdm import tqdm

from Tools.Config_Dataset import *
from Tools.Dataset_ImageNet import get_split_loader as get_split_loader_ImageNet
from Tools.Dataset_ALL_DataLoader import get_split_loader
from Tools.Metrics import *
from Tools.Triplet_Seg_Tool import *
from Tools.Boundary_Alignment_Tool import *
from Tools.Generate_Mask import *
from Tools.Generate_Feature_Heatmap import *
from Tools.Utils import *
from Model.model import *
import warnings

warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x
use_cuda = True


class trainer(object):
    def __init__(self):
        self.backbone = ['3DCNN']  # ['ResNet18', 'ShuffleNetV2', 'MobileNetV2','MedNet_L', 'MedNet_S']
        self.times = current_time()
        self.project_root = project_root
        self.keyword = ['Finetune_Cognition_WMGM']  # ['Finetune_CIBM_CAT_WB_Cognitive_Random', 'Finetune_CIBM_CAT_WB_Cognitive_FullImageNet']
        self.mode = ['SZonly', 'SZwCognitive_CLS', 'SZwCognitive_REG', 'SZwCognitive_CLSREG']  # ['SZonly', 'SZwCognitive_CLS', 'SZwCognitive_REG', 'SZwCognitive_CLSREG']
        self.cv_n = 5
        self.fold_n = range(self.cv_n)
        self.save_log = True
        self.save_tb_log = False
        self.save_weights = False
        self.cls_num = [5]
        self.img_size = 'min'
        self.correction_flag = False

    def __del__(self):
        if self.save_log:
            if self.log_file_iter_train_buddle is not None:
                self.log_file_iter_train_buddle.close()
            if self.log_file_iter_valid_buddle is not None:
                self.log_file_iter_valid_buddle.close()

    def set_environment_seed(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def log(self, info_dict):
        self.log_file_buddle.write(json.dumps(info_dict) + '\n')
        self.log_file_buddle.flush()

    def log_iteration(self, log_file_buddle, info_dict):
        log_file_buddle.write(json.dumps(info_dict) + '\n')
        log_file_buddle.flush()

    def config(self, **kwargs):
        self.epochs = 400
        self.batch_size = 11
        self.lr = 6e-4 if self.correction_flag else 5e-4
        self.reproduce_seed = 400

        self.cur_backbone = kwargs['backbone']
        self.cur_mode = kwargs['mode']
        self.cur_keyword = kwargs['keyword']
        self.cur_fold = kwargs['fold']
        self.cur_num_class = kwargs['num_class']


        self.log_root = self.project_root + '/Results/Logs_DL/Experiment_2'  # Save info after each iteration
        self.auc_root = project_root + '/Results/Logs_DL_AUC_Probs/Experiment_2'

        self.tb_root = self.project_root + '/Results/TensorBoard_logs/Experiment_2/{}'.format(
            self.keyword)  # Tensorboard root
        self.weight_root = self.project_root + '/Results/CheckPoints/Experiment_2/{}'.format(
            self.keyword)  # Weight file root

    def start_training(self, **kwargs):
        self.param = kwargs
        print(kwargs)

        self.config(**kwargs)

        if self.save_log:
            if not os.path.exists(self.log_root):
                os.makedirs(self.log_root)
            # log_path_ = self.log_root + '/{}_{}_{}.txt'.format(
            #     self.cur_keyword,
            #     self.cur_backbone,
            #     self.cur_mode
            # )
            # self.log_file_buddle = open(log_path_, 'w')

            log_path_valid = self.log_root + '/{}_{}_mode_{}_Class_{}_seed_{}_cv_{}_fold_{}_iter_valid.txt'.format(
                self.cur_keyword,
                self.cur_backbone,
                self.cur_mode,
                self.cur_num_class,
                self.reproduce_seed,
                self.cv_n,
                self.cur_fold
            )
            log_path_train = self.log_root + '/{}_{}_mode_{}_Class_{}_seed_{}_cv_{}_fold_{}_iter_train.txt'.format(
                self.cur_keyword,
                self.cur_backbone,
                self.cur_mode,
                self.cur_num_class,
                self.reproduce_seed,
                self.cv_n,
                self.cur_fold
            )
            self.log_file_iter_train_buddle = open(log_path_train, 'w')
            self.log_file_iter_valid_buddle = open(log_path_valid, 'w')

            if not os.path.exists(self.auc_root):
                os.makedirs(self.auc_root)
            auc_path = self.auc_root + '/{}_{}_mode_{}_Class_{}_seed_{}_cv_{}_fold_{}_iter_auc.txt'.format(
                self.cur_keyword,
                self.cur_backbone,
                self.cur_mode,
                self.cur_num_class,
                self.reproduce_seed,
                self.cv_n,
                self.cur_fold
            )
            self.log_auc_file_buddle = open(auc_path, 'w')


        # if self.save_tb_log:
        #     tb_path = self.tb_root + '/{}_{}'.format(self.cur_keyword, self.cur_backbone)
        #     if not os.path.exists(tb_path):
        #         os.makedirs(tb_path)
        #     self.tb_writer = SummaryWriter(log_dir=tb_path)
        if self.save_weights:
            if not os.path.exists(self.weight_root):
                os.makedirs(self.weight_root)
            self.weight_path = self.weight_root + '/{}_{}.pth'.format(self.cur_keyword, self.cur_backbone)

        self.set_environment_seed(self.reproduce_seed)
        self.train()

    def iteration(self):
        for backbone in self.backbone:
            for keyword in self.keyword:
                for mode in self.mode:
                    for num_class in self.cls_num:
                        for fold in self.fold_n:
                            if self.cv_n == 10 and fold == 9:
                                continue
                            if backbone == '3DCNN' and mode == 'ImageNet':
                                continue
                            # if fold in [0, 1, 2, 4]:
                            #     continue
                            # else:
                            #     self.correction_flag = True
                            self.start_training(
                                backbone=backbone,
                                mode=mode,
                                keyword=keyword,
                                fold=fold,
                                num_class=num_class
                            )

    def train(self):
        epochs = self.epochs
        batch_size = self.batch_size
        lr = self.lr

        image_transform = tr.Compose([
            tr.Resize(128, 128),
            tr.ToTensor(),
            tr.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
        ])

        train_dataloader, test_dataloader, val_dataloader, all_dataloader = get_split_loader(
            image_transform=image_transform,
            mask_transform=None,
            batch_size=self.batch_size,
            num_workers=10,
            dataset_name='CIBM_sMRI_7T_CAT_MNI_ALL_Cognition_SubjectLevel_Array',
            path_only=True,
            img_size='min',
            seed=self.reproduce_seed,
            cv_n=self.cv_n,
            fold_n=self.cur_fold,
            label_mode='all',
            cls_num=self.cur_num_class,
        )

        if self.cur_backbone in ['ResNet18', 'ShuffleNetV2', 'MobileNetV2', 'MNasNet']:
            net = CIBM_Cognitive_2DCNN_CLSREG(backbone=self.cur_backbone, num_cls=self.cur_num_class, pretrained=True if self.cur_mode == 'ImageNet' else False)
        else:
            net = CIBM_Cognitive_3DCNN_CLSREG(backbone=self.cur_backbone, num_cls=self.cur_num_class, pretrained=False)

        if use_cuda:
            cuda(net)

        loss_func_ce = torch.nn.CrossEntropyLoss()
        loss_func_be = torch.nn.BCEWithLogitsLoss()
        loss_func_mse = torch.nn.MSELoss()
        loss_func_dice = Dice_Loss()
        loss_func_triplet = torch.nn.TripletMarginLoss(margin=1.0)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, amsgrad=True, weight_decay=0.01 if self.correction_flag else 0.01)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.7)
        scaler = GradScaler()

        def train(net, tq, epoch):
            ACC_score = 0
            F1_score = 0
            Precision_score = 0
            Recall_score = 0
            Train_loss = 0

            tank_train_label = None
            tank_train_pred = None
            tank_train_reg_label = None
            tank_train_reg_pred = None
            tank_train_label_SZ = None
            tank_train_pred_SZ = None

            log_loss_train_SZ = []
            log_loss_train_Cognitive = []
            log_loss_train_Cognitive_cls1 = []
            log_loss_train_Cognitive_cls2 = []
            log_loss_train_Cognitive_cls3 = []
            log_loss_train_Cognitive_cls4 = []
            log_loss_train_Cognitive_cls5 = []
            log_loss_train_Cognitive_cls6 = []


            for index, (img, label_) in enumerate(train_dataloader):
                optimizer.zero_grad()

                if use_cuda:
                    img = cuda(img)
                    label_ = cuda(label_)
                    label_CLS = torch.cat([label_[:, :6], label_[:, -1].unsqueeze(-1)], -1).long()
                    label_REG = label_[:, 6:12].float()

                with autocast():
                    logits_cognitive_CLS, logits_cognitive_REG, logits_SZ = net(img)  # Original Images
                    # probs = F.softmax(logits, 1)

                    cur_result = torch.stack([
                        torch.argmax(F.softmax(logits_cognitive_CLS[:, :self.cur_num_class * 1], 1), dim=1),
                        torch.argmax(F.softmax(logits_cognitive_CLS[:, self.cur_num_class * 1: self.cur_num_class * 2], 1), dim=1),
                        torch.argmax(F.softmax(logits_cognitive_CLS[:, self.cur_num_class * 2: self.cur_num_class * 3], 1), dim=1),
                        torch.argmax(F.softmax(logits_cognitive_CLS[:, self.cur_num_class * 3: self.cur_num_class * 4], 1), dim=1),
                        torch.argmax(F.softmax(logits_cognitive_CLS[:, self.cur_num_class * 4: self.cur_num_class * 5], 1), dim=1),
                        torch.argmax(F.softmax(logits_cognitive_CLS[:, self.cur_num_class * 5: self.cur_num_class * 6], 1), dim=1),
                    ], 1)
                    cur_label = torch.stack([
                        label_CLS[:, 0],
                        label_CLS[:, 1],
                        label_CLS[:, 2],
                        label_CLS[:, 3],
                        label_CLS[:, 4],
                        label_CLS[:, 5],
                    ], 1)
                    cur_result_reg = logits_cognitive_REG
                    cur_label_reg = label_REG
                    cur_result_SZ = torch.sigmoid(logits_SZ)
                    cur_label_SZ = label_CLS[:, -1]
                    if tank_train_pred is None:
                        tank_train_pred = cur_result
                        tank_train_label = cur_label
                        tank_train_reg_pred = cur_result_reg
                        tank_train_reg_label = cur_label_reg
                        tank_train_pred_SZ = cur_result_SZ
                        tank_train_label_SZ = cur_label_SZ
                    else:
                        tank_train_pred = torch.cat([
                            tank_train_pred,
                            cur_result
                        ], 0)
                        tank_train_label = torch.cat([
                            tank_train_label,
                            cur_label
                        ], 0)

                        tank_train_reg_pred = torch.cat([
                            tank_train_reg_pred,
                            cur_result_reg
                        ], 0)
                        tank_train_reg_label = torch.cat([
                            tank_train_reg_label,
                            cur_label_reg
                        ], 0)

                        tank_train_pred_SZ = torch.cat([
                            tank_train_pred_SZ,
                            cur_result_SZ
                        ], 0)
                        tank_train_label_SZ = torch.cat([
                            tank_train_label_SZ,
                            cur_label_SZ
                        ], 0)

                    batch_loss_train_cls1 = loss_func_ce(logits_cognitive_CLS[:, :self.cur_num_class * 1], label_CLS[:, 0])
                    batch_loss_train_cls2 = loss_func_ce(logits_cognitive_CLS[:, self.cur_num_class * 1: self.cur_num_class * 2], label_CLS[:, 1])
                    batch_loss_train_cls3 = loss_func_ce(logits_cognitive_CLS[:, self.cur_num_class * 2: self.cur_num_class * 3], label_CLS[:, 2])
                    batch_loss_train_cls4 = loss_func_ce(logits_cognitive_CLS[:, self.cur_num_class * 3: self.cur_num_class * 4], label_CLS[:, 3])
                    batch_loss_train_cls5 = loss_func_ce(logits_cognitive_CLS[:, self.cur_num_class * 4: self.cur_num_class * 5], label_CLS[:, 4])
                    batch_loss_train_cls6 = loss_func_ce(logits_cognitive_CLS[:, self.cur_num_class * 5: self.cur_num_class * 6], label_CLS[:, 5])
                    batch_loss_train_cognitive_CLS = (batch_loss_train_cls1 + \
                                                 batch_loss_train_cls2 + \
                                                 batch_loss_train_cls3 + \
                                                 batch_loss_train_cls4 + \
                                                 batch_loss_train_cls5 + \
                                                 batch_loss_train_cls6) / 6

                    batch_loss_train_cognitive_REG = loss_func_mse(logits_cognitive_REG, label_REG)

                    batch_loss_train_SZ = loss_func_be(logits_SZ.squeeze(-1), label_CLS[:, -1].float())


                if self.cur_mode == 'SZwCognitive_CLSREG':
                    batch_loss = batch_loss_train_cognitive_CLS + 0.001 * batch_loss_train_cognitive_REG + 2 * batch_loss_train_SZ
                elif self.cur_mode == 'SZwCognitive_CLS':
                    batch_loss = batch_loss_train_cognitive_CLS + 2 * batch_loss_train_SZ
                elif self.cur_mode == 'SZwCognitive_REG':
                    batch_loss = 0.001 * batch_loss_train_cognitive_REG + 2 * batch_loss_train_SZ
                elif self.cur_mode == 'SZonly':
                    batch_loss = batch_loss_train_SZ

                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # batch_loss.backward()
                # optimizer.step()
                # scheduler.step()

                F1_cls1, Precision_cls1, Specificity_cls1, Sensitivity_cls1, Accuracy_cls1 = F1_Multi_Class_ConfuMatrix(ground_truth=tank_train_label[:, 0], prediction=tank_train_pred[:, 0], num_classes=self.cur_num_class)
                F1_cls2, Precision_cls2, Specificity_cls2, Sensitivity_cls2, Accuracy_cls2 = F1_Multi_Class_ConfuMatrix(ground_truth=tank_train_label[:, 1], prediction=tank_train_pred[:, 1], num_classes=self.cur_num_class)
                F1_cls3, Precision_cls3, Specificity_cls3, Sensitivity_cls3, Accuracy_cls3 = F1_Multi_Class_ConfuMatrix(ground_truth=tank_train_label[:, 2], prediction=tank_train_pred[:, 2], num_classes=self.cur_num_class)
                F1_cls4, Precision_cls4, Specificity_cls4, Sensitivity_cls4, Accuracy_cls4 = F1_Multi_Class_ConfuMatrix(ground_truth=tank_train_label[:, 3], prediction=tank_train_pred[:, 3], num_classes=self.cur_num_class)
                F1_cls5, Precision_cls5, Specificity_cls5, Sensitivity_cls5, Accuracy_cls5 = F1_Multi_Class_ConfuMatrix(ground_truth=tank_train_label[:, 4], prediction=tank_train_pred[:, 4], num_classes=self.cur_num_class)
                F1_cls6, Precision_cls6, Specificity_cls6, Sensitivity_cls6, Accuracy_cls6 = F1_Multi_Class_ConfuMatrix(ground_truth=tank_train_label[:, 5], prediction=tank_train_pred[:, 5], num_classes=self.cur_num_class)
                F1_SZ, Precision_SZ, Sensitivity_SZ, Specificity_SZ, Accuracy_SZ = F1_Binary_CLS(y_true=tank_train_label_SZ, y_pred_probs=tank_train_pred_SZ[:, 0])

                MAE_cls1 = MAE(y_true=tank_train_reg_label[:, 0], y_pred=tank_train_reg_pred[:, 0]).item()
                MAE_cls2 = MAE(y_true=tank_train_reg_label[:, 1], y_pred=tank_train_reg_pred[:, 1]).item()
                MAE_cls3 = MAE(y_true=tank_train_reg_label[:, 2], y_pred=tank_train_reg_pred[:, 2]).item()
                MAE_cls4 = MAE(y_true=tank_train_reg_label[:, 3], y_pred=tank_train_reg_pred[:, 3]).item()
                MAE_cls5 = MAE(y_true=tank_train_reg_label[:, 4], y_pred=tank_train_reg_pred[:, 4]).item()
                MAE_cls6 = MAE(y_true=tank_train_reg_label[:, 5], y_pred=tank_train_reg_pred[:, 5]).item()

                # Logging
                log_loss_train_SZ.append(batch_loss_train_SZ.item())
                log_loss_train_Cognitive.append(batch_loss_train_cognitive_CLS.item())
                log_loss_train_Cognitive_cls1.append(batch_loss_train_cls1.item())
                log_loss_train_Cognitive_cls2.append(batch_loss_train_cls2.item())
                log_loss_train_Cognitive_cls3.append(batch_loss_train_cls3.item())
                log_loss_train_Cognitive_cls4.append(batch_loss_train_cls4.item())
                log_loss_train_Cognitive_cls5.append(batch_loss_train_cls5.item())
                log_loss_train_Cognitive_cls6.append(batch_loss_train_cls6.item())

                tq.update(batch_size)
                tq.set_postfix(
                    train_loss_SZ='{:.3f}'.format(np.average(log_loss_train_SZ)),
                    train_loss_Cog='{:.3f}'.format(np.average(log_loss_train_Cognitive)),
                    F1_c1='{:.3f}'.format(F1_cls1),
                    F1_c2='{:.3f}'.format(F1_cls2),
                    F1_c3='{:.3f}'.format(F1_cls3),
                    F1_c4='{:.3f}'.format(F1_cls4),
                    F1_c5='{:.3f}'.format(F1_cls5),
                    F1_c6='{:.3f}'.format(F1_cls6),
                    F1_SZ='{:.3f}'.format(F1_SZ),
                )

                # Logging Into File
                self.info_dict_train = {}
                self.info_dict_train['iteration'] = epoch * len(train_dataloader) + index
                self.info_dict_train['train_loss_SZ'] = np.average(log_loss_train_SZ)
                self.info_dict_train['train_loss_Cognitive_cls1'] = np.average(log_loss_train_Cognitive_cls1)
                self.info_dict_train['train_loss_Cognitive_cls2'] = np.average(log_loss_train_Cognitive_cls2)
                self.info_dict_train['train_loss_Cognitive_cls3'] = np.average(log_loss_train_Cognitive_cls3)
                self.info_dict_train['train_loss_Cognitive_cls4'] = np.average(log_loss_train_Cognitive_cls4)
                self.info_dict_train['train_loss_Cognitive_cls5'] = np.average(log_loss_train_Cognitive_cls5)
                self.info_dict_train['train_loss_Cognitive_cls6'] = np.average(log_loss_train_Cognitive_cls6)

                self.info_dict_train['F1_SZ'] = F1_SZ
                self.info_dict_train['Pre_SZ'] = Precision_SZ
                self.info_dict_train['Sen_SZ'] = Sensitivity_SZ
                self.info_dict_train['Spe_SZ'] = Specificity_SZ
                self.info_dict_train['Acc_SZ'] = Accuracy_SZ

                self.info_dict_train['F1_cls1'] = F1_cls1
                self.info_dict_train['Pre_cls1'] = Precision_cls1
                self.info_dict_train['Sen_cls1'] = Sensitivity_cls1
                self.info_dict_train['Spe_cls1'] = Specificity_cls1
                self.info_dict_train['Acc_cls1'] = Accuracy_cls1
                self.info_dict_train['F1_cls2'] = F1_cls2
                self.info_dict_train['Pre_cls2'] = Precision_cls2
                self.info_dict_train['Sen_cls2'] = Sensitivity_cls2
                self.info_dict_train['Spe_cls2'] = Specificity_cls2
                self.info_dict_train['Acc_cls2'] = Accuracy_cls2
                self.info_dict_train['F1_cls3'] = F1_cls3
                self.info_dict_train['Pre_cls3'] = Precision_cls3
                self.info_dict_train['Sen_cls3'] = Sensitivity_cls3
                self.info_dict_train['Spe_cls3'] = Specificity_cls3
                self.info_dict_train['Acc_cls3'] = Accuracy_cls3
                self.info_dict_train['F1_cls4'] = F1_cls4
                self.info_dict_train['Pre_cls4'] = Precision_cls4
                self.info_dict_train['Sen_cls4'] = Sensitivity_cls4
                self.info_dict_train['Spe_cls4'] = Specificity_cls4
                self.info_dict_train['Acc_cls4'] = Accuracy_cls4
                self.info_dict_train['F1_cls5'] = F1_cls5
                self.info_dict_train['Pre_cls5'] = Precision_cls5
                self.info_dict_train['Sen_cls5'] = Sensitivity_cls5
                self.info_dict_train['Spe_cls5'] = Specificity_cls5
                self.info_dict_train['Acc_cls5'] = Accuracy_cls5
                self.info_dict_train['F1_cls6'] = F1_cls6
                self.info_dict_train['Pre_cls6'] = Precision_cls6
                self.info_dict_train['Sen_cls6'] = Sensitivity_cls6
                self.info_dict_train['Spe_cls6'] = Specificity_cls6
                self.info_dict_train['Acc_cls6'] = Accuracy_cls6

                self.info_dict_train['MAE_reg1'] = MAE_cls1
                self.info_dict_train['MAE_reg2'] = MAE_cls2
                self.info_dict_train['MAE_reg3'] = MAE_cls3
                self.info_dict_train['MAE_reg4'] = MAE_cls4
                self.info_dict_train['MAE_reg5'] = MAE_cls5
                self.info_dict_train['MAE_reg6'] = MAE_cls6

                if self.save_log and index in [0, 2, 4]:
                    self.log_iteration(log_file_buddle=self.log_file_iter_train_buddle, info_dict=self.info_dict_train)

                    # Validation Procedure
                    self.info_dict_val = {}
                    val(net, iteration=epoch * len(train_dataloader) + index)
                    if self.save_log:
                        self.log_iteration(log_file_buddle=self.log_file_iter_valid_buddle, info_dict=self.info_dict_val)
                    net.train()

        def val(net, iteration):
            with torch.no_grad():
                tank_test_label = None
                tank_test_pred = None
                tank_test_reg_label = None
                tank_test_reg_pred = None
                tank_test_label_SZ = None
                tank_test_pred_SZ = None

                log_loss_test_SZ = []
                log_loss_test_Cognitive_cls1 = []
                log_loss_test_Cognitive_cls2 = []
                log_loss_test_Cognitive_cls3 = []
                log_loss_test_Cognitive_cls4 = []
                log_loss_test_Cognitive_cls5 = []
                log_loss_test_Cognitive_cls6 = []

                for index, (img, label_) in enumerate(test_dataloader):
                    optimizer.zero_grad()

                    if use_cuda:
                        img = cuda(img)
                        label_ = cuda(label_)
                        label_CLS = torch.cat([label_[:, :6], label_[:, -1].unsqueeze(-1)], -1).long()
                        label_REG = label_[:, 6:12].float()

                    with autocast():
                        logits_cognitive_CLS, logits_cognitive_REG, logits_SZ = net(img)  # Original Images

                        cur_result = torch.stack([
                            torch.argmax(F.softmax(logits_cognitive_CLS[:, :self.cur_num_class * 1], 1), dim=1),
                            torch.argmax(F.softmax(logits_cognitive_CLS[:, self.cur_num_class * 1: self.cur_num_class * 2], 1), dim=1),
                            torch.argmax(F.softmax(logits_cognitive_CLS[:, self.cur_num_class * 2: self.cur_num_class * 3], 1), dim=1),
                            torch.argmax(F.softmax(logits_cognitive_CLS[:, self.cur_num_class * 3: self.cur_num_class * 4], 1), dim=1),
                            torch.argmax(F.softmax(logits_cognitive_CLS[:, self.cur_num_class * 4: self.cur_num_class * 5], 1), dim=1),
                            torch.argmax(F.softmax(logits_cognitive_CLS[:, self.cur_num_class * 5: self.cur_num_class * 6], 1), dim=1),
                        ], 1)
                        cur_label = torch.stack([
                            label_CLS[:, 0],
                            label_CLS[:, 1],
                            label_CLS[:, 2],
                            label_CLS[:, 3],
                            label_CLS[:, 4],
                            label_CLS[:, 5],
                        ], 1)
                        cur_result_SZ = torch.sigmoid(logits_SZ)
                        cur_label_SZ = label_CLS[:, -1]
                        cur_result_reg = logits_cognitive_REG
                        cur_label_reg = label_REG
                        if tank_test_pred is None:
                            tank_test_pred = cur_result
                            tank_test_label = cur_label
                            tank_test_reg_pred = cur_result_reg
                            tank_test_reg_label = cur_label_reg
                            tank_test_pred_SZ = cur_result_SZ
                            tank_test_label_SZ = cur_label_SZ
                        else:
                            tank_test_pred = torch.cat([
                                tank_test_pred,
                                cur_result
                            ], 0)
                            tank_test_label = torch.cat([
                                tank_test_label,
                                cur_label
                            ], 0)

                            tank_test_reg_pred = torch.cat([
                                tank_test_reg_pred,
                                cur_result_reg
                            ], 0)
                            tank_test_reg_label = torch.cat([
                                tank_test_reg_label,
                                cur_label_reg
                            ], 0)

                            tank_test_pred_SZ = torch.cat([
                                tank_test_pred_SZ,
                                cur_result_SZ
                            ], 0)
                            tank_test_label_SZ = torch.cat([
                                tank_test_label_SZ,
                                cur_label_SZ
                            ], 0)

                        batch_loss_test_cls1 = loss_func_ce(logits_cognitive_CLS[:, :self.cur_num_class * 1], label_CLS[:, 0])
                        batch_loss_test_cls2 = loss_func_ce(logits_cognitive_CLS[:, self.cur_num_class * 1: self.cur_num_class * 2], label_CLS[:, 1])
                        batch_loss_test_cls3 = loss_func_ce(logits_cognitive_CLS[:, self.cur_num_class * 2: self.cur_num_class * 3], label_CLS[:, 2])
                        batch_loss_test_cls4 = loss_func_ce(logits_cognitive_CLS[:, self.cur_num_class * 3: self.cur_num_class * 4], label_CLS[:, 3])
                        batch_loss_test_cls5 = loss_func_ce(logits_cognitive_CLS[:, self.cur_num_class * 4: self.cur_num_class * 5], label_CLS[:, 4])
                        batch_loss_test_cls6 = loss_func_ce(logits_cognitive_CLS[:, self.cur_num_class * 5: self.cur_num_class * 6], label_CLS[:, 5])
                        batch_loss_test_cognitive = ( batch_loss_test_cls1 + \
                                                      batch_loss_test_cls2 + \
                                                      batch_loss_test_cls3 + \
                                                      batch_loss_test_cls4 + \
                                                      batch_loss_test_cls5 + \
                                                      batch_loss_test_cls6) / 6

                        batch_loss_test_SZ = loss_func_be(logits_SZ.squeeze(-1), label_CLS[:, -1].float())

                        # Logging
                        log_loss_test_SZ.append(batch_loss_test_SZ.item())
                        log_loss_test_Cognitive_cls1.append(batch_loss_test_cls1.item())
                        log_loss_test_Cognitive_cls2.append(batch_loss_test_cls2.item())
                        log_loss_test_Cognitive_cls3.append(batch_loss_test_cls3.item())
                        log_loss_test_Cognitive_cls4.append(batch_loss_test_cls4.item())
                        log_loss_test_Cognitive_cls5.append(batch_loss_test_cls5.item())
                        log_loss_test_Cognitive_cls6.append(batch_loss_test_cls6.item())


                F1_cls1, Precision_cls1, Specificity_cls1, Sensitivity_cls1, Accuracy_cls1 = F1_Multi_Class_ConfuMatrix(ground_truth=tank_test_label[:, 0], prediction=tank_test_pred[:, 0], num_classes=self.cur_num_class)
                F1_cls2, Precision_cls2, Specificity_cls2, Sensitivity_cls2, Accuracy_cls2 = F1_Multi_Class_ConfuMatrix(ground_truth=tank_test_label[:, 1], prediction=tank_test_pred[:, 1], num_classes=self.cur_num_class)
                F1_cls3, Precision_cls3, Specificity_cls3, Sensitivity_cls3, Accuracy_cls3 = F1_Multi_Class_ConfuMatrix(ground_truth=tank_test_label[:, 2], prediction=tank_test_pred[:, 2], num_classes=self.cur_num_class)
                F1_cls4, Precision_cls4, Specificity_cls4, Sensitivity_cls4, Accuracy_cls4 = F1_Multi_Class_ConfuMatrix(ground_truth=tank_test_label[:, 3], prediction=tank_test_pred[:, 3], num_classes=self.cur_num_class)
                F1_cls5, Precision_cls5, Specificity_cls5, Sensitivity_cls5, Accuracy_cls5 = F1_Multi_Class_ConfuMatrix(ground_truth=tank_test_label[:, 4], prediction=tank_test_pred[:, 4], num_classes=self.cur_num_class)
                F1_cls6, Precision_cls6, Specificity_cls6, Sensitivity_cls6, Accuracy_cls6 = F1_Multi_Class_ConfuMatrix(ground_truth=tank_test_label[:, 5], prediction=tank_test_pred[:, 5], num_classes=self.cur_num_class)
                F1_SZ, Precision_SZ, Sensitivity_SZ, Specificity_SZ, Accuracy_SZ = F1_Binary_CLS(y_true=tank_test_label_SZ, y_pred_probs=tank_test_pred_SZ[:, 0])

                MAE_cls1 = MAE(y_true=tank_test_reg_label[:, 0], y_pred=tank_test_reg_pred[:, 0]).item()
                MAE_cls2 = MAE(y_true=tank_test_reg_label[:, 1], y_pred=tank_test_reg_pred[:, 1]).item()
                MAE_cls3 = MAE(y_true=tank_test_reg_label[:, 2], y_pred=tank_test_reg_pred[:, 2]).item()
                MAE_cls4 = MAE(y_true=tank_test_reg_label[:, 3], y_pred=tank_test_reg_pred[:, 3]).item()
                MAE_cls5 = MAE(y_true=tank_test_reg_label[:, 4], y_pred=tank_test_reg_pred[:, 4]).item()
                MAE_cls6 = MAE(y_true=tank_test_reg_label[:, 5], y_pred=tank_test_reg_pred[:, 5]).item()

            self.info_dict_val['iteration'] = iteration
            self.info_dict_val['test_loss_SZ'] = np.average(log_loss_test_SZ)
            self.info_dict_val['test_loss_Cognitive_cls1'] = np.average(log_loss_test_Cognitive_cls1)
            self.info_dict_val['test_loss_Cognitive_cls2'] = np.average(log_loss_test_Cognitive_cls2)
            self.info_dict_val['test_loss_Cognitive_cls3'] = np.average(log_loss_test_Cognitive_cls3)
            self.info_dict_val['test_loss_Cognitive_cls4'] = np.average(log_loss_test_Cognitive_cls4)
            self.info_dict_val['test_loss_Cognitive_cls5'] = np.average(log_loss_test_Cognitive_cls5)
            self.info_dict_val['test_loss_Cognitive_cls6'] = np.average(log_loss_test_Cognitive_cls6)

            self.info_dict_val['F1_SZ'] = F1_SZ
            self.info_dict_val['Pre_SZ'] = Precision_SZ
            self.info_dict_val['Sen_SZ'] = Sensitivity_SZ
            self.info_dict_val['Spe_SZ'] = Specificity_SZ
            self.info_dict_val['Acc_SZ'] = Accuracy_SZ

            self.info_dict_val['F1_cls1'] = F1_cls1
            self.info_dict_val['Pre_cls1'] = Precision_cls1
            self.info_dict_val['Sen_cls1'] = Sensitivity_cls1
            self.info_dict_val['Spe_cls1'] = Specificity_cls1
            self.info_dict_val['Acc_cls1'] = Accuracy_cls1
            self.info_dict_val['F1_cls2'] = F1_cls2
            self.info_dict_val['Pre_cls2'] = Precision_cls2
            self.info_dict_val['Sen_cls2'] = Sensitivity_cls2
            self.info_dict_val['Spe_cls2'] = Specificity_cls2
            self.info_dict_val['Acc_cls2'] = Accuracy_cls2
            self.info_dict_val['F1_cls3'] = F1_cls3
            self.info_dict_val['Pre_cls3'] = Precision_cls3
            self.info_dict_val['Sen_cls3'] = Sensitivity_cls3
            self.info_dict_val['Spe_cls3'] = Specificity_cls3
            self.info_dict_val['Acc_cls3'] = Accuracy_cls3
            self.info_dict_val['F1_cls4'] = F1_cls4
            self.info_dict_val['Pre_cls4'] = Precision_cls4
            self.info_dict_val['Sen_cls4'] = Sensitivity_cls4
            self.info_dict_val['Spe_cls4'] = Specificity_cls4
            self.info_dict_val['Acc_cls4'] = Accuracy_cls4
            self.info_dict_val['F1_cls5'] = F1_cls5
            self.info_dict_val['Pre_cls5'] = Precision_cls5
            self.info_dict_val['Sen_cls5'] = Sensitivity_cls5
            self.info_dict_val['Spe_cls5'] = Specificity_cls5
            self.info_dict_val['Acc_cls5'] = Accuracy_cls5
            self.info_dict_val['F1_cls6'] = F1_cls6
            self.info_dict_val['Pre_cls6'] = Precision_cls6
            self.info_dict_val['Sen_cls6'] = Sensitivity_cls6
            self.info_dict_val['Spe_cls6'] = Specificity_cls6
            self.info_dict_val['Acc_cls6'] = Accuracy_cls6

            self.info_dict_val['MAE_reg1'] = MAE_cls1
            self.info_dict_val['MAE_reg2'] = MAE_cls2
            self.info_dict_val['MAE_reg3'] = MAE_cls3
            self.info_dict_val['MAE_reg4'] = MAE_cls4
            self.info_dict_val['MAE_reg5'] = MAE_cls5
            self.info_dict_val['MAE_reg6'] = MAE_cls6


            y_test = tank_test_label_SZ.detach().cpu().numpy()
            y_pred_prob = tank_test_pred_SZ[:, 0].detach().cpu().numpy()
            self.auc_dict = {}
            self.auc_dict['iteration'] = iteration
            self.auc_dict['y_true'] = '_'.join([str(y) for y in y_test])
            self.auc_dict['y_pred'] = '_'.join([str(y) for y in y_pred_prob])
            self.log_iteration(log_file_buddle=self.log_auc_file_buddle, info_dict=self.auc_dict)



        def eval(net, tq_, epoch, k=2):
            with torch.no_grad():
                total = 0
                top1 = 0
                topk = 0
                AUC_score = []
                for index, (img, mask, label) in enumerate(test_dataloader):
                    if use_cuda:
                        img = cuda(img)
                        mask = cuda(mask.long())

                    logits, _ = net(img)  # Original Images
                    probs = F.softmax(logits, 1)

                    mIoU = batch_intersection_union(predict=probs.detach(), target=mask.detach(), nclass=21)
                    AUC_score.extend([mIoU])

                    batch_loss = loss_func(logits, mask)
                    log_loss_test.append(batch_loss.item())

                    # _, maxk = torch.topk(probs, k, dim=-1)
                    test_labels = label
                    total += test_labels.size(0)
                    test_labels = test_labels.view(-1, 1)  # reshape labels from [n] to [n,1] to compare [n,k]

                    # top1 += (test_labels == maxk[:, 0:1]).sum().item()
                    # topk += (test_labels == maxk).sum().item()

                    tq_.update(batch_size)
                    tq_.set_postfix(
                        test_loss='{:.3f}'.format(np.average(log_loss_test)),
                        AUC='{:.3f}'.format(np.nanmean(AUC_score))
                    )

                print(
                    'Accuracy of the network on total {} test images: @top1={:.3f}%; @top{}={:.3f}%; @AUC={:.3f}'.format(
                        total,
                        100 * top1 / total,
                        k,
                        100 * topk / total,
                        np.nanmean(AUC_score))
                )

                if self.save_tb_log:
                    self.tb_writer.add_scalar('test/Top1_acc', 100 * top1 / total, epoch)
                    self.tb_writer.add_scalar('test/Top5_acc', 100 * topk / total, epoch)
                    self.tb_writer.add_scalar('test/loss', np.average(log_loss_test), epoch)

                self.cur_test_Top5ACC_ = 100 * topk / total
                self.info_dict['epoch'] = epoch
                self.info_dict['test_top1_acc'] = 100 * top1 / total
                self.info_dict['test_top5_acc'] = 100 * topk / total
                self.info_dict['test_AUC'] = np.nanmean(AUC_score)
                self.info_dict['test_loss'] = np.average(log_loss_test)
                return

        self.best_test_Top5ACC_ = 0.0
        self.cur_test_Top5ACC_ = None
        for epoch_index in range(epochs + 1):
            self.info_dict = {}


            tq = tqdm(total=(len(train_dataloader) * batch_size))
            tq.set_description('Epoch {}'.format(epoch_index))
            train(net, tq, epoch=epoch_index)
            lr_scheduler.step()
            tq.close()

            # tq_test = tqdm(total=(len(test_dataloader) * batch_size))
            # tq_test.set_description('Epoch {}'.format(epoch_index))
            # eval(net, tq_test, k=1, epoch=epoch_index)
            # tq_test.close()

            # if self.best_test_Top5ACC_ < self.cur_test_Top5ACC_:
            #     self.best_test_Top5ACC_ = self.cur_test_Top5ACC_
            #     if self.save_weights:
            #         torch.save(net, self.weight_path)
            #         print(' + Save CheckPoint !')
            # if self.save_log:
            #     self.log(self.info_dict)


if __name__ == '__main__':
    t = trainer()
    t.iteration()
