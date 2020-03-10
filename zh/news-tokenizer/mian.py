import argparse
import news_config as config
import os
import json
import random
import jieba
import collections


class Tokenizer(object):
    def __init__(self,original_data_dir,sub_dir,tokenized_output_dir,train_num,test_num,dev_num):

        sub_dir = os.path.join(original_data_dir,sub_dir)
        assert os.path.exists(sub_dir)
        assert 0 != len(os.listdir(sub_dir))

        tokenized_output_dir = os.path.join(tokenized_output_dir)
        if not os.path.exists(tokenized_output_dir):
            os.makedirs(tokenized_output_dir)
        self.train_dir = os.path.join(tokenized_output_dir,"train")
        self.test_dir = os.path.join(tokenized_output_dir,"test")
        self.dev_dir = os.path.join(tokenized_output_dir,"dev")

        if not os.path.exists(self.train_dir):
            os.mkdir(self.train_dir)
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        if not os.path.exists(self.dev_dir):
            os.mkdir(self.dev_dir)

        self.original_data_dir = original_data_dir
        self.sub_dir = sub_dir
        self.tokenized_output_dir = tokenized_output_dir
        self.train_num = train_num
        self.test_num = test_num
        self.dev_num = dev_num




    def tokenize(self,sub_file_sample_num):




        total_sample_num = self.train_num + self.test_num + self.dev_num
        train_terminate = self.train_num
        test_terminate = self.train_num + self.test_num
        dev_terminate = self.train_num + self.test_num + self.dev_num


        file_list = os.listdir(self.sub_dir)
        all_sample_list = []
        read_sample_num = 0
        train_file_no = 0
        test_file_no = 0
        dev_file_no = 0
        for file in file_list:
            path_file = os.path.join(self.sub_dir,file)
            with open(path_file,"r",encoding='utf-8') as f:
                data = json.load(f)["data"]
                f.close()
            for sample in data:
                if read_sample_num >= total_sample_num:
                    break
                read_sample_num += 1

                sample_list = []
                news_id = sample['news_id']
                title = sample['title']
                content = sample['content']

                sample_list.append(int(news_id))
                sample_list.append(list(jieba.cut(title)))
                sample_list.append(list(jieba.cut(content)))

                all_sample_list.append(sample_list)

                if 1 == len(all_sample_list) or len(all_sample_list) % 10000 == 0:
                    print(len(all_sample_list))


                if (read_sample_num == train_terminate) or \
                        (read_sample_num < train_terminate and len(all_sample_list) == sub_file_sample_num):

                    train_file_no += 1
                    filename = "train_{:0>2d}.json".format(train_file_no)
                    filename = os.path.join(self.train_dir, filename)
                    print("save file {}".format(filename))
                    save_dict = {}
                    save_dict["data"] = all_sample_list
                    with open(filename, 'w', encoding='utf-8') as save_f:
                        json.dump(save_dict, save_f)
                        save_f.close()
                    all_sample_list.clear()
                    save_dict.clear()

                elif (read_sample_num == test_terminate) or \
                        (read_sample_num < test_terminate and len(all_sample_list) == sub_file_sample_num):
                    test_file_no += 1
                    filename = "test_{:0>2d}.json".format(test_file_no)
                    filename = os.path.join(self.test_dir,filename)
                    print("save file {}".format(filename))
                    save_dict = {}
                    save_dict["data"] = all_sample_list

                    with open(filename, 'w', encoding='utf-8') as save_f:
                        json.dump(save_dict, save_f)
                        save_f.close()
                    all_sample_list.clear()
                    save_dict.clear()

                elif (read_sample_num == dev_terminate) or \
                        (read_sample_num < dev_terminate and len(all_sample_list) == sub_file_sample_num):
                    dev_file_no += 1
                    filename = "dev_{:0>2d}.json".format(dev_file_no)
                    filename = os.path.join(self.dev_dir, filename)
                    print("save file {}".format(filename))
                    save_dict = {}
                    save_dict["data"] = all_sample_list
                    with open(filename, 'w', encoding='utf-8') as save_f:
                        json.dump(save_dict, save_f)
                        save_f.close()
                    all_sample_list.clear()
                    save_dict.clear()
                else:
                    pass

            if read_sample_num >= total_sample_num:
                break



    def gene_word_freq(self,vocab_file):
        assert 0 != len(os.listdir(self.train_dir))
        file_list = os.listdir(self.train_dir)
        word_freq = collections.Counter()

        for file in file_list:
            file = os.path.join(self.train_dir,file)
            with open(file,'r',encoding='utf-8') as f:
                data = json.load(f)['data']
                f.close()
            for sample in data:
                title = sample[1]
                content = sample[2]

                word_freq.update(title)
                word_freq.update(content)

        word_freq = word_freq.most_common(len(word_freq))
        word_freq = dict(word_freq)

        vocab_file = os.path.join(self.tokenized_output_dir,vocab_file)
        print("save word freq file {}".format(vocab_file))
        with open(vocab_file,"w",encoding='utf-8') as f:
            json.dump(word_freq,f)
            f.close()









def set_seed(seed):
    random.seed(seed)



def split_original_json(original_json_path,sub_dir,sub_file_sample_num):
    sub_dir = os.path.join(original_json_path,sub_dir)
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)

    real_sub_file_num = len(os.listdir(sub_dir))
    except_sub_file_num = config.except_sample_num // sub_file_sample_num
    if 0 != config.except_sample_num % sub_file_sample_num:
        except_sub_file_num += 1
    if real_sub_file_num == except_sub_file_num:
        return

    if 0 != real_sub_file_num:
        raise ValueError("子文件夹[{}]不为空，且真实数量与预期不同".format(sub_dir))

    original_json_path = os.path.join(original_json_path, config.file_name)

    sub_file_prefix = "sub"
    sample_num = 0
    file_num = 0
    sub_file_list = []


    f = open(original_json_path,"r",encoding='utf-8')
    line = "init_str"
    while line:
        line = f.readline().strip()
        if "" == line:
            break
        sample_num += 1
        sub_file_list.append(eval(line))
        if sample_num % sub_file_sample_num == 0:
            file_num += 1
            current_dict = {}
            assert sub_file_sample_num == len(sub_file_list)
            current_dict["data"] = sub_file_list
            file_name = "{}_{:0>2d}.json".format(sub_file_prefix, file_num)
            file_path_name = os.path.join(sub_dir, file_name)
            print("第{}-{}个样本，存储于文件{}".format(int(sample_num - sub_file_sample_num + 1), sample_num, file_name))
            with open(file_path_name, "w", encoding='utf-8') as json_file:
                json.dump(current_dict, json_file)
                json_file.close()
            sub_file_list.clear()

    if 0 != len(sub_file_list):
        file_num += 1
        current_dict = {}
        current_dict["data"] = sub_file_list
        file_name = "{}_{:0>2d}.json".format(sub_file_prefix, file_num)
        file_path_name = os.path.join(sub_dir, file_name)
        print("第{}-{}个样本，存储于文件{}".format(int(sample_num - len(sub_file_list) + 1), sample_num, file_name))
        with open(file_path_name, "w", encoding='utf-8') as json_file:
            json.dump(current_dict, json_file)
            json_file.close()
        sub_file_list.clear()
    f.close()




def main():

    parser = argparse.ArgumentParser()

    # F:\data\zh\news
    parser.add_argument("--original_data_dir",default=None,type = str,required=True,
                        help="包含文件news2016zh_train.json的文件夹路径")
    # F:\data\zh\tokenized
    parser.add_argument("--tokenized_dir", default=None, type=str, required=True,
                        help="分词后文件所存储的文件夹")

    parser.add_argument("--split_data_dir",default = "sub",type=str,
                        help="切分后的文件存储的文件夹(源json文件太大了)")

    parser.add_argument("--sub_file_sample_num",default=1e+5,type=int,
                        help="每个子文件存储的样本数量")
    parser.add_argument("--seed",default=1234,type=int,
                        help="随机种子")


    parser.add_argument("--word_freq",default="vocab.json",type = str,
                        help="词表文件")
    parser.add_argument("--train_sample_num",default=3e+5,type=int,
                        help="训练集样本的数量")
    parser.add_argument("--dev_sample_num",default=1.2e+4,type=int,
                        help="验证集样本的数量")
    parser.add_argument("--test_sample_num",default=1.2e+4,type=int,
                        help="测试集样本的数量")

    args = parser.parse_args()

    set_seed(args.seed)

    split_original_json(original_json_path = args.original_data_dir,
                        sub_dir = args.split_data_dir,
                        sub_file_sample_num = args.sub_file_sample_num)

    tokenizer = Tokenizer(original_data_dir = args.original_data_dir,
                          sub_dir = args.split_data_dir,
                          tokenized_output_dir = args.tokenized_dir,
                          train_num = args.train_sample_num,
                          test_num = args.test_sample_num,
                          dev_num = args.test_sample_num,)

    tokenizer.tokenize(sub_file_sample_num=args.sub_file_sample_num)

    tokenizer.gene_word_freq(args.word_freq)


if __name__ == "__main__":
    main()






