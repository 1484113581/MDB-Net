# MDB-Net
一个多模态双分支建筑信息联合提取模型（Multi-modal Dual-branch Building Information Joint Extraction Model, MDB-Net）

### 文件夹说明
- `NN/logs`
	- 训练时存放日志文件的文件夹
- `NN/model_save`
	- 存放着训练后得到的参数文件，在推理时使用该参数文件
- `NN/model`
	- 存放着模型文件`BHFPModel_f.py`
- `NN/util`
	- 存放一些工具代码文件
- `NN/testData`
	- 可以使用该文件夹中的样本来推理，验证模型性能
	- 其中`sampel`中是输入数据，`output`将存放推理得到的结果
- `NN/train.py`
	- 训练文件，使用该文件来训练
- `NN/inference_and_evaluation.py`
	- 推理文件，使用该文件来推理和评估结果
- `datasetSample_final`
	- 存放着数据集，包括训练集(train)、测试集(test)、验证集(val)
	- 其中`train_augmentation`中存放着增广后的训练集，直接使用该文件夹中样本训练
        - 请联系 1484113581@qq.com 获取数据

### 训练说明
- 运行`NN/train.py`文件
	- 训练过程中会生成日志文件和模型参数文件
		- 日志文件存放在`NN/logs`中
		- 默认每50个epoch生成一个参数文件，放在`NN/model_save`中，可根据需求修改代码


### 推理说明
- 运行`NN/inference_and_evaluation.py`文件
	- 目前默认会对`NN/testData/sample`文件夹中的样本进行推理，生成的结果会存放在`NN/testData/output/bh`(建筑物高度)和`NN/testData/output/fp`(建筑物足迹)中。
