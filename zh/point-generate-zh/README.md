# point-generate-zh

## 指针网络在中文数据下 news2016zh_train.json 使用jieba分词后的实现。


|-data_util-|-data.py 读取样本文件，生成二进制文件存储
|           |-vocab.py 读取词表文件，生成词表
|
|-model-    |model.py  模型相关代码文件
|
|-decoder.py  模型进行decoder相关代码文件
|
|-zh_config.py  中文数据集的相关配置参数
|
|-main.py   这个不用多说，入口函数文件
|
|-train_util.py  模型进行 train 和 evaluate的文件




如果有什么问题，请加我QQ:997251095或者QQ邮箱交流