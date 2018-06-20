# 未来杯AI挑战赛区域赛 AI专业组-图像场景分类

# 东北大区 Deep Neuron 100126

## 程序简介

本程序一共包含五个文件和两个文件夹。   
inception_v3_ft_futurelab.py为第一个模型的训练文件；  
inception_resnet_v2_ft_futurelab.py为第二个模型训练文件；  
ensemble_model_prediction.py为集成模型预测文件；  
testb_predict_results.csv为testb预测结果；  
checkpoint文件夹为训练模型存放文件夹，默认为空；  
checkpoint_prediction为预测使用模型文件夹(已有训练完成的两个模型)；  
README.md为本文件。

## 系统需求
* Python 2.7
* Tensorflow-gpu 1.7.0（CUDA 9.0 cuDNN 7.0）
* Pillow 5.1.0
* Keras 2.1.5
* scikit-image 0.13.1
* scikit-learn 0.19.1
* pandas 0.22.0 

## 运行程序

### 训练模型

#### 训练inception-v3

使用命令行执行如下操作：

```
python inception_v3_ft_futurelab.py --dataset_dir=<dir> --checkpoint_dir=<checkpoint_dir>
```

参数说明

* --dataset_dir 训练数据集所在目录, 例如：/home/ubuntu/image_scene_training_v1/
* --checkpoint_dir checkpoint目录，训练后的模型参数。默认为：./checkpoint/
* 模型默认训练150个epoch，batch_size默认为32，如需修改，请更改代码第23和24行


#### 训练inception-resnet-v2

使用命令行执行如下操作：

```
python inception_resnet_v2_ft_futurelab.py --dataset_dir=<dir> --checkpoint_dir=<checkpoint_dir>
```

参数说明

* --dataset_dir 训练数据集所在目录, 例如：/home/ubuntu/image_scene_training_v1/
* --checkpoint_dir checkpoint目录，训练后的模型参数。默认为：./checkpoint/
* 模型默认训练150个epoch，batch_size默认为32，如需修改，请更改代码第23和24行


### 执行测试

使用命令行执行如下操作：

```
python ensemble_model_prediction.py --dataset_dir=<testset_dir> --checkpoint_dir=<checkpoint_dir> --target_file=<target_file>
```

参数说明

* --dataset_dir 测试数据集所在目录, 例如：/home/ubuntu/image_scene_test_b_0515
* --checkpoint_dir checkpoint目录，训练后的模型参数。默认加载已经训练好的模型，路径为：./checkpoint_prediction/
* --target_file 结果文件存放路径，应该指向一个csv文件地址。默认存放在根目录下，文件名为：testb_predict_results.csv
