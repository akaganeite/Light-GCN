"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import os


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.config=config
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")

    '''written by zxb'''
    def computer(self):#获取embedding
        return self.embedding_user.weight,self.embedding_item.weight

    def attack_neg(self,user,pos,neg,ran):
        user = user.long().to(world.device)
        pos_items = torch.tensor(pos).long().to(world.device)
        neg_items = torch.tensor(neg).long().to(world.device)

        users_emb = self.embedding_user(user)
        pos_emb = self.embedding_item(pos_items)
        neg_emb = self.embedding_item(neg_items)

        if ran == True:
            delta_n = torch.rand(neg_emb.size())
            delta_n = nn.functional.normalize(delta_n, p=2, dim=1)
            return delta_n.to(world.device)

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(1)
        neg_emb.retain_grad()
        weight_decay = self.config['decay']
        total_loss=loss+reg_loss*weight_decay
        total_loss.backward()
        grad_n = neg_emb.grad
        delta_n = nn.functional.normalize(grad_n, p=2, dim=1)
        return delta_n

    def attack_user_pos(self,users,pos,neg,ran):
        # users只有一个用户
        users = users.long()
        pos_items = torch.tensor(pos).long()
        neg_items = torch.tensor(neg).long()

        users = users.to(world.device)
        pos_items = pos_items.to(world.device)
        neg_items = neg_items.to(world.device)
        users_emb = self.embedding_user(users)
        pos_emb = self.embedding_item(pos_items)
        neg_emb = self.embedding_item(neg_items)

        if ran == True:
            delta_u = torch.rand(users_emb.size())
            delta_u = nn.functional.normalize(delta_u, p=2, dim=0)
            delta_p = torch.rand(pos_emb.size())
            delta_p = nn.functional.normalize(delta_p, p=2, dim=1)
            return delta_u.to(world.device), delta_p.to(world.device)



        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)

        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(1)
        users_emb.retain_grad()
        pos_emb.retain_grad()

        weight_decay = self.config['decay']
        total_loss = loss + reg_loss * weight_decay
        total_loss.backward()
        grad_u = users_emb.grad
        grad_p = pos_emb.grad
        delta_u = nn.functional.normalize(grad_u, p=2, dim=0)
        delta_p = nn.functional.normalize(grad_p, p=2, dim=1)

        return delta_u, delta_p

    '''end'''
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']#the batch size for bpr loss training procedure，default=0.6
        self.A_split = self.config['A_split']#默认false
        self.embedding_user = torch.nn.Embedding(#weight存储了一个num_users*laten_dim大小的矩阵，代表每个user的embedding
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)#随机初始化embedding
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()#拿到的图是已经进行归一化的，可以直接进行运算
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight#取得所有用户和物品的embedding矩阵
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])#(num_items+num_users)*laten_dim
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]#由三个E组成，每个E都是一个二维tensor，代表GCN每一层输出的embedding

        '''
        dropout设置，默认不dropout，g_droped就是graph
        '''
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    

        '''
         GCN的部分
        '''
        for layer in range(self.n_layers):
            if self.A_split:#A_split默认false
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:#和论文一致，直接做积
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)# 把列表转化为一个tensor，将列表中每一项拼接embs:[(num_items+num_users),3,laten_dim]
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)#每个embedding都有三层，对这三层取平均值，light_out:[(num_items+num_users),laten_dim]
        users, items = torch.split(light_out, [self.num_users, self.num_items])#分割用户和物品的embedding
        return users, items
    
    def getUsersRating(self, users):#test的时候用来打分
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    '''
    written by zxb
    '''

    def getUsersRating_adv(self,user_emb,all_item):
        rating=self.f(torch.matmul(user_emb, all_item.t()))
        return rating


    def attack_neg(self,user,pos,neg,ran):
        if ran==True:
            neg_len=len(neg)
            emb_size=self.config['latent_dim_rec']
            delta_n = torch.rand(neg_len,emb_size)
            delta_n = nn.functional.normalize(delta_n, p=2, dim=1)
            return delta_n.to(world.device)
        user = user.long().to(world.device)
        pos_items = torch.tensor(pos).long().to(world.device)
        neg_items = torch.tensor(neg).long().to(world.device)

        all_users, all_items = self.computer()  # 获得所有user，item经过GCN后，平均过的embedding
        user_emb = all_users[user]  # 筛选出本次batch对应的embedding
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        if ran==True:
            delta_n=torch.rand(neg_emb.size())
            delta_n = nn.functional.normalize(delta_n, p=2, dim=1)
            return delta_n.to(world.device)
        neg_emb.retain_grad()

        userEmb0 = self.embedding_user(user)  # 获得没有经过GCN的embedding
        posEmb0 = self.embedding_item(pos_items)
        negEmb0 = self.embedding_item(neg_items)
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(
            1)  # 正则化，tensor(0.9578, device='cuda:0', grad_fn=<DivBackward0>)
        pos_scores = torch.mul(user_emb, pos_emb)  # 二维，点积后的矩阵torch.Size([2048, 64])
        pos_scores = torch.sum(pos_scores, dim=1)  # 一维，用户对物品评分表 torch.Size([2048])
        neg_scores = torch.mul(user_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        weight_decay = self.config['decay']
        total_loss = loss + reg_loss * weight_decay
        total_loss.backward()

        grad_n = neg_emb.grad
        delta_n = nn.functional.normalize(grad_n, p=2, dim=1)
        return delta_n


    def attack_user_pos(self,users,pos,neg,ran):
        #users只有一个用户
        if ran==True:
            emb_size = self.config['latent_dim_rec']
            #user_len=len(users)
            user_len=1
            pos_len=len(pos)
            delta_u = torch.rand(emb_size)
            delta_u = nn.functional.normalize(delta_u, p=2, dim=0)
            delta_p = torch.rand(pos_len,emb_size)
            delta_p = nn.functional.normalize(delta_p, p=2, dim=1)
            return delta_u.to(world.device), delta_p.to(world.device)


        users=users.long()
        pos_items=torch.tensor(pos).long()
        neg_items=torch.tensor(neg).long()

        users=users.to(world.device)
        pos_items=pos_items.to(world.device)
        neg_items=neg_items.to(world.device)

        all_users, all_items = self.computer()  # 获得所有user，item经过GCN后，平均过的embedding
        users_emb = all_users[users]  # 筛选出本次batch对应的embedding
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        if ran == True:
            delta_u = torch.rand(users_emb.size())
            delta_u = nn.functional.normalize(delta_u, p=2, dim=0)
            delta_p=torch.rand(pos_emb.size())
            delta_p=nn.functional.normalize(delta_p,p=2,dim=1)
            return delta_u.to(world.device),delta_p.to(world.device)

        users_emb.retain_grad()
        pos_emb.retain_grad()

        userEmb0 = self.embedding_user(users)  # 获得没有经过GCN的embedding
        posEmb0 = self.embedding_item(pos_items)
        negEmb0 = self.embedding_item(neg_items)
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(1)  # 正则化，tensor(0.9578, device='cuda:0', grad_fn=<DivBackward0>)
        pos_scores = torch.mul(users_emb, pos_emb)  # 二维，点积后的矩阵torch.Size([2048, 64])
        pos_scores = torch.sum(pos_scores, dim=1)  # 一维，用户对物品评分表 torch.Size([2048])
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)


        # 等价于ln(sigmoid())，mean将所有的loss(一对样本就是一个loss)求平均
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        weight_decay = self.config['decay']
        total_loss=loss+reg_loss*weight_decay
        total_loss.backward()
        grad_u=users_emb.grad
        grad_p=pos_emb.grad
        delta_u = nn.functional.normalize(grad_u, p=2, dim=0)
        delta_p = nn.functional.normalize(grad_p, p=2, dim=1)

        return delta_u,delta_p
    '''
    end
    '''


    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()#获得所有user，item经过GCN后，平均过的embedding
        users_emb = all_users[users]#筛选出本次batch对应的embedding
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)#获得没有经过GCN的embedding
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        #所有的embedding大小:torch.Size([2048, 64])
        #
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))#正则化，tensor(0.9578, device='cuda:0', grad_fn=<DivBackward0>)
        pos_scores = torch.mul(users_emb, pos_emb)#二维，点积后的矩阵torch.Size([2048, 64])
        pos_scores = torch.sum(pos_scores, dim=1)#一维，用户对物品评分表 torch.Size([2048])
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        #等价于ln(sigmoid())，mean将所有的loss(一对样本就是一个loss)求平均
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
