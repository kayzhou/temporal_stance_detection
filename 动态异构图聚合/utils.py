import os
from torch.utils.data import Dataset, DataLoader
import _pickle as cPickle
import dgl
import torch
import numpy as np
import pandas as pd



def pickle_loader(path):
    a = cPickle.load(open(path, 'rb'))
    return a

def user_neg(data, item_num):  #data = stance_train.csv,item_num=len(item_id)
    item = range(item_num)   
    def select(data_u, item):  #data_u = data中每个user下的item_id
        return np.random.choice(np.setdiff1d(item, data_u),20) #这个函数用以比较两个数组  返回再item中但不在data_u的一排序的唯一值  随机抽取10个作为负样本
    return data.groupby('user_id')['item_id'].apply(lambda x: select(x, item))


# def user_neg(data, item_num):  #data = stance_train.csv,item_num=len(item_id)
#     item = range(item_num)   
#     def select(data_u, item):  #data_u = data中每个user下的item_id
#         return np.setdiff1d(item, data_u) #这个函数用以比较两个数组  返回再item中但不在data_u的一排序的唯一值
#     return data.groupby('user_id')['item_id'].apply(lambda x: select(x, item))  #返回目标user（就是根据user排序）下没有的 但其他user组中出现的item

def neg_generate(user, data_neg, neg_num=100):
    neg = np.zeros((len(user), neg_num), np.int32)
    for i, u in enumerate(user):
        neg[i] = np.random.choice(data_neg[u], neg_num, replace=False)  #指定user 从user的负样本中抽取
    return neg


class myFloder(Dataset):
    def __init__(self, root_dir, loader,jioaohu):
        self.root = root_dir
        self.loader = loader
        self.dir_list = load_data(root_dir,jioaohu)  #bin文件列表
        self.size = len(self.dir_list)

    def __getitem__(self, index):
        dir_ = self.dir_list[index]
        data = self.loader(dir_)
        return data

    def __len__(self):
        return self.size


def collate(data):
    user = []
    graph = []
    last_item = []
    label = []
    for da in data:
        user.append(da[0])
        graph.append(da[1])
        last_item.append(da[2])
        label.append(da[3])
    return torch.Tensor(user).long(), dgl.batch_hetero(graph), torch.Tensor(last_item).long(), torch.Tensor(label).long()



def load_data(data_path,jioaohu):# ...../train/ 
    data_dir = []
    dir_list = os.listdir(data_path)  #返回指定路径下的文件和文件夹列表
    dir_list.sort()
    for filename in dir_list: #[trian,val,test] filename = 用户id
        subfolder = os.listdir(os.path.join(data_path, filename))
        if len(subfolder) >= jioaohu:
            for fil in subfolder:  #返回路径下的文件列表
                data_dir.append(os.path.join(os.path.join(data_path, filename), fil))
    return data_dir  #返回bin文件路径  并以列表形式展示



def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def eval_metric(all_top, all_label, all_length, random_rank=True):
    recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = [], [], [], [], [], []
    data_l = np.zeros((100, 7))
    for index in range(len(all_top)):
        per_length = all_length[index]
        if random_rank:
            prediction = (-all_top[index]).argsort(1).argsort(1)
            predictions = prediction[:, 0]
            for i, rank in enumerate(predictions):
                # data_l[per_length[i], 6] += 1
                if rank < 20:
                    ndgg20.append(1 / np.log2(rank + 2))
                    recall20.append(1)
                    # if per_length[i]-1 < 100:
                    #     data_l[per_length[i], 5] += 1 / np.log2(rank + 2)
                    #     data_l[per_length[i], 2] += 1
                    # else:
                    #     data_l[99, 5] += 1 / np.log2(rank + 2)
                    #     data_l[99, 2] += 1
                else:
                    ndgg20.append(0)
                    recall20.append(0)
                if rank < 10:
                    ndgg10.append(1 / np.log2(rank + 2))
                    recall10.append(1)
                    # if per_length[i]-1 < 100:
                    #     data_l[per_length[i], 4] += 1 / np.log2(rank + 2)
                    #     data_l[per_length[i], 1] += 1
                    # else:
                    #     data_l[99, 4] += 1 / np.log2(rank + 2)
                    #     data_l[99, 1] += 1
                else:
                    ndgg10.append(0)
                    recall10.append(0)
                if rank < 5:
                    ndgg5.append(1 / np.log2(rank + 2))
                    recall5.append(1)
                    # if per_length[i]-1 < 100:
                    #     data_l[per_length[i], 3] += 1 / np.log2(rank + 2)
                    #     data_l[per_length[i], 0] += 1
                    # else:
                    #     data_l[99, 3] += 1 / np.log2(rank + 2)
                    #     data_l[99, 0] += 1
                else:
                    ndgg5.append(0)
                    recall5.append(0)

        else:
            for top_, target in zip(all_top[index], all_label[index]):
                recall20.append(np.isin(target, top_))
                recall10.append(np.isin(target, top_[0:10]))
                recall5.append(np.isin(target, top_[0:5]))
                if len(np.where(top_ == target)[0]) == 0:
                    ndgg20.append(0)
                else:
                    ndgg20.append(1 / np.log2(np.where(top_ == target)[0][0] + 2))
                if len(np.where(top_ == target)[0]) == 0:
                    ndgg10.append(0)
                else:
                    ndgg10.append(1 / np.log2(np.where(top_ == target)[0][0] + 2))
                if len(np.where(top_ == target)[0]) == 0:
                    ndgg5.append(0)
                else:
                    ndgg5.append(1 / np.log2(np.where(top_ == target)[0][0] + 2))
    #pd.DataFrame(data_l, columns=['r5','r10','r20','n5','n10','n10','number']).to_csv(name+'.csv')
    return np.mean(recall5), np.mean(recall10), np.mean(recall20), np.mean(ndgg5), np.mean(ndgg10), np.mean(ndgg20), \
           pd.DataFrame(data_l, columns=['r5','r10','r20','n5','n10','n20','number'])



def format_arg_str(args, exclude_lst, max_len=20):
    linesep = os.linesep
    arg_dict = vars(args)
    keys = [k for k in arg_dict.keys() if k not in exclude_lst]
    values = [arg_dict[k] for k in keys]
    key_title, value_title = 'Arguments', 'Values'
    key_max_len = max(map(lambda x: len(str(x)), keys))
    value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
    key_max_len, value_max_len = max([len(key_title), key_max_len]), max([len(value_title), value_max_len])
    horizon_len = key_max_len + value_max_len + 5
    res_str = linesep + '=' * horizon_len + linesep
    res_str += ' ' + key_title + ' ' * (key_max_len - len(key_title)) + ' | ' \
               + value_title + ' ' * (value_max_len - len(value_title)) + ' ' + linesep + '=' * horizon_len + linesep
    for key in sorted(keys):
        value = arg_dict[key]
        if value is not None:
            key, value = str(key), str(value).replace('\t', '\\t')
            value = value[:max_len-3] + '...' if len(value) > max_len else value
            res_str += ' ' + key + ' ' * (key_max_len - len(key)) + ' | ' \
                       + value + ' ' * (value_max_len - len(value)) + linesep
    res_str += '=' * horizon_len
    return res_str