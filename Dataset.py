from torch.utils.data import Dataset
from pyquaternion import Quaternion
import pandas as pd
import os
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import shutil  
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import logging
logging.basicConfig(level='INFO')
import coloredlogs
coloredlogs.install(level='INFO')

from data_classes import PointCloud, Box
from dataset_utils import *


def plot_and_show(data, category_name):
    # 设置中文字体和负号正常显示
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    label_list = ['50', '100', '150', '200', '250', '300', '350', '400', '450', '500', '>500']    # 横坐标刻度显示值
    num_list1 = data.tolist()     # 纵坐标值1
    x = range(len(num_list1))
    """
    绘制条形图
    left:长条形中点横坐标
    height:长条形高度
    width:长条形宽度，默认值0.8
    label:为后面设置legend准备
    """
    rects1 = plt.bar(left=[i + 0.2 for i in x], height=num_list1, width=0.4, alpha=0.8, color='red')
    max_ = int(np.max(data) * 1.2)
    plt.ylim(0, max_)     # y轴取值范围
    plt.ylabel("Number of frames")
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """
    plt.xticks([index + 0.2 for index in x], label_list)
    plt.xlabel("Number of the points on KITTI’s " + category_name)
    plt.title(category_name)
    plt.legend()     # 设置题注
    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
    plt.show()


class kittiDataset():
    def __init__(self, path):
        self.KITTI_Folder = path
        self.KITTI_velo = os.path.join(self.KITTI_Folder, "velodyne")
        self.KITTI_label = os.path.join(self.KITTI_Folder, "label_02")
        logging.debug("KITTI_velo_path:\n%s\n", self.KITTI_velo)
        logging.debug("KITTI_label_path:\n%s\n", self.KITTI_label)   

    def make_sure_path_valid(self, dirs):   
        if not os.path.exists(dirs):
            os.makedirs(dirs)
    
    def generate_pcd_and_txt(self, sceneID, save_path, category_name="Car", Replace=False):
        pcd_path = os.path.join(save_path, category_name, 'lidar') 
        label_path = os.path.join(save_path, category_name, 'label')

        self.make_sure_path_valid(pcd_path)
        self.make_sure_path_valid(label_path)

        if Replace is True:
            shutil.rmtree(pcd_path)
            shutil.rmtree(label_path)
            os.mkdir(pcd_path)  
            os.mkdir(label_path)

        list_of_scene = [
            path for path in os.listdir(self.KITTI_velo)
            if os.path.isdir(os.path.join(self.KITTI_velo, path)) and
            int(path) in sceneID
        ]
        logging.debug("list_of_scene:\n%s\n", list_of_scene)
        pts_cnt = np.zeros([11,])
        # 遍历每一个序列列表中的序列
        for scene in list_of_scene:
            logging.info("scene: %s", scene)
            #标签路径
            label_file = os.path.join(self.KITTI_label, scene + ".txt")
            #读取标签txt文件
            df = pd.read_csv(
                label_file,
                sep=' ',
                names=[
                    "frame", "track_id", "type", "truncated", "occluded",
                    "alpha", "bbox_left", "bbox_top", "bbox_right",
                    "bbox_bottom", "height", "width", "length", "x", "y", "z",
                    "rotation_y"
                ])
            df = df[df["type"] == category_name] #筛选出类别是car的标签
            df.insert(loc=0, column="scene", value=scene) #在标签中插入一列表示这是哪个场景
            # 还原索引，将df中的数据的每一行的索引变成默认排序的形式
            df = df.reset_index(drop=True)
            length = df.shape[0]

            for label_row in tqdm(range(length)):
                this_anno = df.loc[label_row]
                logging.debug("this_anno\n%s\n", this_anno)
                this_pc, this_box, state = self.getBBandPC(this_anno)  # this_pc's shape is (3, N)
                if state is True:
                    pc_in_box = cropPC(this_pc, this_box)
                    points = pc_in_box.points.transpose()

                    file_name = get_name_by_read_dir(pcd_path)
                    if points.shape[0] < 50:
                        pts_cnt[0] += 1
                    elif points.shape[0] < 100 and points.shape[0] >= 50:
                        pts_cnt[1] += 1
                    elif points.shape[0] < 150 and points.shape[0] >= 100:
                        pts_cnt[2] += 1           
                    elif points.shape[0] < 200 and points.shape[0] >= 150:
                        pts_cnt[3] += 1
                    elif points.shape[0] < 250 and points.shape[0] >= 200:
                        pts_cnt[4] += 1
                    elif points.shape[0] < 300 and points.shape[0] >= 250:
                        pts_cnt[5] += 1
                    elif points.shape[0] < 350 and points.shape[0] >= 300:
                        pts_cnt[6] += 1           
                    elif points.shape[0] < 400 and points.shape[0] >= 350:
                        pts_cnt[7] += 1
                    elif points.shape[0] < 450 and points.shape[0] >= 400:
                        pts_cnt[8] += 1
                    elif points.shape[0] < 500 and points.shape[0] >= 450:
                        pts_cnt[9] += 1
                    else:
                        pts_cnt[10] += 1
                    pc_save_pcd(points, pcd_path, file_name + '.pcd')
                    bbox_label_save_txt(this_anno[1:], label_path, file_name + '.txt')
        plot_and_show(pts_cnt.astype(np.int), category_name=category_name)

    # 获取包含序列的列表
    def getSceneList(self, split):
        if "TRAIN" in split.upper():  # Training SET
            sceneID = list(range(0, 17))
        elif "VALID" in split.upper():  # Validation Set
            sceneID = list(range(17, 19))
        elif "TEST" in split.upper():  # Testing Set
            sceneID = list(range(19, 21))
        else:  # Full Dataset
            sceneID = list(range(21))
        # logging.info("sceneID_path:\n%s\n", sceneID)   
        return sceneID

    def getBBandPC(self, anno):
        calib_path = os.path.join(self.KITTI_Folder, 'calib',
                                  anno['scene'] + ".txt")
        calib = self.read_calib_file(calib_path)
        #在矩阵最下面叠加一行(0,0,0,1)
        transf_mat = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))
        PC, box, state= self.getPCandBBfromPandas(anno, transf_mat)
        return PC, box, state

    def getPCandBBfromPandas(self, box, calib):
        #求出车辆的中心点 从此处的中心点是根据KITTI中相机坐标系下的中心点
        # 减去一半的高度移到地面上
        center = [box["x"], box["y"] - box["height"] / 2, box["z"]]
        size = [box["width"], box["length"], box["height"]]
        #下面这个函数没有完全看明白　　应该是将roy角转换成四元数吧
        orientation = Quaternion(
            axis=[0, 1, 0], radians=box["rotation_y"]) * Quaternion(
                axis=[1, 0, 0], radians=np.pi / 2)
        BB = Box(center, size, orientation) #用中心点坐标和w,h,l以及旋转角来初始化BOX这个类
        State = True
        try:
            # VELODYNE PointCloud
            velodyne_path = os.path.join(self.KITTI_velo, box["scene"],
                                         '%06d.bin'%(box["frame"])) #f'{box["frame"]:06}.bin')
            #从点云的.bin文件中读取点云数据并且转换为4*x的矩阵，且去掉最后的一行的点云的密度表示数据
            PC = PointCloud(
                np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4).T)
            #将点云转换到相机坐标系下　因为label中的坐标和h,w,l在相机坐标系下的
            PC.transform(calib)
        except FileNotFoundError:
            # logging.error("No such file found\n%s\n", velodyne_path)
            PC = PointCloud(np.array([[0, 0, 0]]).T)
            State = False

        return PC, BB, State

    def read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                values = line.split()
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[values[0]] = np.array(
                        [float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    pass
        #返回一个字典　字典中有6个键对　每个键对应的是calib文件中的一行，
        # key是'P0'，value是后面的对应的表示数值转换的一个3*4的numpy矩阵
        return data


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Generate the pcd and label.txt file of fixed sequence in KITTI',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--category', required=False, type=str,
        default='Cyclist', help='category_name')

    parser.add_argument('--dataset_path', required=False, type=str,
        default='/media/echo/仓库卷/DataSet/Autonomous_Driving/KITTI/tracking/origin_dataset/training',
        help='dataset Path')

    parser.add_argument('--save_path', required=False, type=str,
        default='/media/echo/仓库卷/DataSet/Autonomous_Driving/KITTI/tracking/Specific_class',
        help='save Path')

    parser.add_argument('--replace', required=False, type=bool,
        default=True, help='whether delete the all files and generate again or not')

    args = parser.parse_args()
    kitti = kittiDataset(args.dataset_path)
    scene_list = kitti.getSceneList('TRAIN')
    kitti.generate_pcd_and_txt(scene_list, save_path=args.save_path, category_name=args.category, Replace=args.replace)
