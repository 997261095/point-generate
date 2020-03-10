# zh-single-word
中文数据集单个分词下，指针生成网络模型部分更改后的实现

### news-tokenizer
news2016zh_train.json 使用按照单个字分词，生成相应的train，test，dev和vocab.json


### point-generate-zh-single
指针生成网络在news2016zh_train数据集下的应用，这部分代码的运行，依赖news2016zh_train使用news-tokenizer单个分词后的数据。



