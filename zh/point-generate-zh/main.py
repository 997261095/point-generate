import random
import numpy as np
import torch
import argparse
import os
from data_util.vocab import Vocab
import zh_config as config
from data_util.data import get_features
from model.model import PointerGeneratorNetworks
from torch.optim import Adagrad,Adam
from train_util import train
from decoder import decoder



def check(args,vocab):

    train_token_dir = os.path.join(args.token_data,"train")
    test_token_dir = os.path.join(args.token_data,"test")
    val_token_dir = os.path.join(args.token_data,"dev")

    assert os.path.exists(train_token_dir)
    assert os.path.exists(test_token_dir)
    assert os.path.exists(val_token_dir)



    # features_50000_400_100
    feature_dir = "{}_{}_{}_{}".format(args.feature_dir_prefix,args.vocab_num,args.content_len,args.title_len)
    args.feature_dir = os.path.join(".",feature_dir)

    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)
        print("创建的特征目录:{}".format(feature_dir))

    train_feature_dir = os.path.join(".",feature_dir,"train")
    test_feature_dir = os.path.join(".", feature_dir, "test")
    val_feature_dir = os.path.join(".", feature_dir, "val")
    if not os.path.exists(train_feature_dir):
        os.mkdir(train_feature_dir)
    if not os.path.exists(test_feature_dir):
        os.mkdir(test_feature_dir)
    if not os.path.exists(val_feature_dir):
        os.mkdir(val_feature_dir)


    expect_train_feature_file_num = config.expect_train_feature_file_num
    real_train_feature_file_num = len(os.listdir(train_feature_dir))
    if real_train_feature_file_num == 0:
        get_features(token_dir=train_token_dir, feature_dir=train_feature_dir, vocab=vocab, args=args, data_set="train")
    elif real_train_feature_file_num != expect_train_feature_file_num:
        raise ValueError("train feature dir {} not empty".format(train_feature_dir))

    expect_test_feature_file_num = config.expect_test_feature_file_num
    real_test_feature_file_num = len(os.listdir(test_feature_dir))
    if real_test_feature_file_num == 0:
        get_features(token_dir=test_token_dir, feature_dir=test_feature_dir, vocab=vocab, args=args, data_set="test")
    elif real_test_feature_file_num != expect_test_feature_file_num:
        raise ValueError("test feature dir {} not empty".format(test_feature_dir))

    expect_dev_feature_file_num = config.expect_dev_feature_file_num
    real_val_feature_file_num = len(os.listdir(val_feature_dir))
    if real_val_feature_file_num == 0:
        get_features(token_dir=val_token_dir, feature_dir=val_feature_dir, vocab=vocab, args=args, data_set="dev")
    elif real_val_feature_file_num != expect_dev_feature_file_num:
        raise ValueError("val feature dir {} not empty".format(val_feature_dir))





def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--token_data",default=None,type = str,required=True,
                       help="包含 train，test，dev 和 vocab.json的文件夹")

    parser.add_argument("--feature_dir_prefix",default="features",
                        help="train，test，evl从样本转化成特征所存储的文件夹前缀置")

    parser.add_argument("--do_train",action='store_true',
                        help="是否进行训练")
    parser.add_argument("--do_decode", action='store_true',
                        help="是否对测试集进行测试")

    parser.add_argument("--example_num", default= 1024 * 8,type = int,
                        help="每一个特征文件所包含的样本数量")

    parser.add_argument("--content_len",default=500,type=int,
                       help="文章的所允许的最大长度")

    parser.add_argument("--title_len", default=50, type=int,
                        help="训练和生成时摘要所允许的最大长度")

    parser.add_argument("--min_decoder_len",default = 15,type=int,
                        help="生成摘要时，生成最短的摘要长度")

    parser.add_argument("--vocab_num",default=50000,type = int,
                        help="词表所允许的最大长度")

    parser.add_argument("--pointer_gen",action='store_true',
                        help="是否使用指针机制")

    parser.add_argument("--use_coverage",action="store_true",
                        help="是否使用汇聚机制")

    parser.add_argument("--no_cuda", action='store_true',
                        help="当GPU可用时，选择不用GPU")

    parser.add_argument("--epoch_num", default=10,type = int,
                        help="epoch")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="train batch size")

    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="evaluate batch size")

    parser.add_argument("--hidden_dim", default=256,type=int,
                        help="hidden dimension")
    parser.add_argument("--embedding_dim",default=128,type=int,
                        help="embedding dimension")
    parser.add_argument("--coverage_loss_weight",default=1.0,type=float,
                        help="coverage loss weight ")
    parser.add_argument("--eps",default=1e-12,type = float,
                        help="log(v + eps) Avoid  v == 0,")
    parser.add_argument("--dropout",default= 0.5,type =float,
                        help="dropout")

    parser.add_argument("--lr",default=1e-3,type=float,
                        help="learning rate")
    parser.add_argument("--max_grad_norm",default=1.0,type=float,
                        help="Max gradient norm.")

    parser.add_argument("--adagrad_init_acc", default=0.1, type=float,
                        help="learning rate")

    parser.add_argument("--adam_epsilon",default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--gradient_accumulation_steps",default=1,type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--output_dir",default="output",type=str,
                        help="Folder to store models and results")


    parser.add_argument("--evaluation_steps",default = 500,type=int,
                        help="Evaluation every N steps of training")
    parser.add_argument("--seed",default=4321,type=int,
                        help="Random seed")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    set_seed(args.seed)

    vocab_file = os.path.join(args.token_data, 'vocab.json')
    assert os.path.exists(vocab_file)

    vocab = Vocab(vocab_file=vocab_file, vob_num=args.vocab_num)



    check(args, vocab=vocab)

    model = PointerGeneratorNetworks(vob_size=args.vocab_num, embed_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
                                     pad_idx=vocab.pad_idx, dropout=args.dropout, pointer_gen=args.pointer_gen,
                                     use_coverage=args.use_coverage,min_decoder_len = args.min_decoder_len)

    model = model.to(args.device)
    model = model.to(args.device)
    if args.do_train:
        optimizer = Adam(model.parameters(),lr = args.lr)
        train(args = args,model=model,optimizer = optimizer,with_eval = True)
    if args.do_decode:
        decoder(args,model,vocab=vocab)





if __name__ == "__main__":
    main()