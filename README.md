### 代码说明

本代码包使用python对kitti的原始数据集做处理，生成某一个序列中，对应 `category` 的数据文件，包括点云`.bin`文件以及label的`.txt`文件，二者互相对应，**默认在序列 0000-0016 中进行筛选**

- 传入参数：
  
- `category`
  
  - 要创建数据集的类别，KITTI中包含的类别有
    
    ```python
      'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'
    ```
    
  - `dataset_path`
  
    - KITTI原始数据集的路径，此路径下应包含
  
      ```python
      ├── calib
      ├── label_02
      └── velodyne
      ```
  
  - `save_path`
  
    - 输出对应类别的数据集的路径，代码输出完成后，会在此目录下建立对应的类别文件夹，如下：
  
      ```python
      ├── your_category1
      │   ├── label
      │   ├── lidar
      ```
  
  - `replace`
  
    - 是否清空 `save_path`  下的文件重新生成？若选择否，则会计算原有目录下的文件数量，接着最后一个文件名的序号生成文件
  
- 环境要求：

  ```python
  torch
  pyquaternion
  pandas
  os
  tqdm
  argparse
  shutil  
  numpy
  matplotlib
  logging
  coloredlogs
  python=3.5(系统python)
  ```

- 运行

  ```python
  python Dataset.py --category='Car' --dataset_path=<your_path> --save_path=<your_path> --replace=True
  ```

- 输出截图

  ```python
  2020-07-22 20:32:33 OMEN root[25001] INFO scene: 0000
  100%|█████████████████████████████████████████| 154/154 [00:02<00:00, 57.62it/s]
  ...
  ```

  算法运行完毕会输出对应类别各个点云数量的条形统计图，便于统计



