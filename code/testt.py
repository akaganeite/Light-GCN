import world
import torch
import dataloader
import Procedure
import model
import utils
import numpy as np
import os
import time
import random
from tensorboardX import SummaryWriter
'''
def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])
    print(kwargs)
    print(tensors)
    if len(tensors) == 1:
        print(1)
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

users=[0,1,2,3]
posItems=[10,15,20,3]
negItems=[11,4,2]
for (batch_i,(batch_users,batch_pos,batch_neg)) in enumerate(minibatch(users,posItems,negItems,batch_size=world.config['bpr_batch_size'])):
    print('batch_i:',batch_i,' batch_users,batch_pos,batch_neg:',batch_users,batch_pos,batch_neg)
'''
'''
dataset=dataloader.LastFM()
recmodel=model.LightGCN(world.config, dataset)
recmodel.load_state_dict(torch.load('./checkpoints/lastfm-3-64-base.pth.tar'))
recmodel = recmodel.to(world.device)
Procedure.Test(dataset,recmodel,0)

'''
'''
dataset=dataloader.LastFM()
testDict=dataset.testDict
users = list(testDict.keys())
#groundTrue = [testDict[u] for u in batch_users]
with open('lastfm_neg.txt','w') as f:
    for u in users:
        data=str(u)+' '+str(testDict[u]).replace('\n','')
        wr=data+'\n'
        f.write(wr) 
'''
'''
    def attack_user_pos(self,users,pos,neg):
        self.eps=1
        #users只有一个用户
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

        users_emb.retain_grad()
        pos_emb.retain_grad()
        neg_emb.retain_grad()

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
        grad_p=pos_emb.grad.cpu()
        grad_n=neg_emb.grad.cpu()
        delta_u = nn.functional.normalize(grad_u, p=2, dim=0)
        delta_p = nn.functional.normalize(grad_p, p=2, dim=1).cpu()
        delta_n = nn.functional.normalize(grad_n, p=2, dim=1).cpu()

        users_emb_adv=users_emb+grad_u
        all_items_adv=all_items.cpu().detach()
        neg_emb=neg_emb.cpu()
        all_items_adv[neg_items]=neg_emb+grad_n
        pos_emb=pos_emb.cpu()
        all_items_adv[pos_items]=pos_emb+grad_p
        all_items_adv=all_items_adv.to(world.device)
        rating_ori = self.f(torch.matmul(users_emb, all_items.t()))
        rating_adv=self.f(torch.matmul(users_emb_adv,all_items_adv.t()))
        return rating_ori,rating_adv
'''



class atk_Train():

    def __init__(self,recmodel,dataset,model_path,eps,ran,w=None,users=None,mode='train'):
        if dataset=='lastfm':
            self.dataset=dataloader.LastFM()
        else:
            self.dataset=dataloader.Loader()

        if recmodel=='mf':
            self.recmodel=model.PureMF(world.config,self.dataset)
        elif recmodel=='lgn':
            self.recmodel=model.LightGCN(world.config,self.dataset)
        self.recmodel.load_state_dict(torch.load(model_path))
        self.recmodel = self.recmodel.to(world.device)

        self.mode=mode
        self.ran=ran
        self.w=w
        self.eps=eps
        if users ==None and mode=='train':
            self.users=self.dataset.trainUniqueUsers
        elif users ==None and mode=='test':
            self.users=self.dataset.testUniqueUsers
        else:
            self.users=users

        self.results={
                'precision': [],
               'recall': [],
               'ndcg': []}


    def atk_user(self,user):
        recmodel: model.LightGCN=self.recmodel
        dataset:dataloader.BasicDataset=self.dataset
        neg = dataset.allNeg[user].tolist()
        np.random.shuffle(neg)
        pos=dataset.allPos[user].tolist()
        np.random.shuffle(pos)
        rating_adv=[]

        all_users, all_items=recmodel.computer()
        #all_items=all_items.cpu()
        #all_users=all_users.cpu()
        all_users=all_users[user]

        all_items_delta=torch.zeros_like(all_items)
        '''attack neg'''
        neg_list=[]
        step=len(pos)
        if step==0:
            for i in self.eps:
                rating_adv.append(self.recmodel.getUsersRating(user))
            return rating_adv

        for i in range(0,len(neg),step):
            b=neg[i:i+step]
            neg_list.append(b)
        for data in neg_list:
            if len(data)==step:
                delta_n=recmodel.attack_neg(user,pos,data,self.ran)
                #delta_n=delta_n.to(world.device)
                all_items_delta[data]=delta_n
            else:
                p=random.sample(pos,len(data))
                delta_n=recmodel.attack_neg(user,p,data,self.ran)
                #delta_n = delta_n.to(world.device)
                all_items_delta[data] = delta_n
        '''atk_user'''
        neg_items = random.sample(neg, len(dataset.allPos[user]))
        delta_u,delta_p=recmodel.attack_user_pos(user,pos,neg_items,self.ran)
        #delta_u=delta_u.cpu()
        #delta_p=delta_p.cpu()
        all_items_delta[pos]=delta_p
        all_items_delta=all_items_delta.to(world.device)
        f = torch.nn.Sigmoid()
        for i in self.eps:
            all_items=all_items_delta*i+all_items
            all_users=delta_u*i+all_users
            all_items=all_items.to(world.device)
            all_users=all_users.to(world.device)
            rating_adv.append(f(torch.matmul(all_users, all_items.t())))
        return rating_adv

    def atk_train(self):
        t_users=torch.Tensor(self.users).long()
        rating_list_adv=[]
        ground_true=[]
        count=1
        max_K = max(world.topks)
        for u in t_users:
            rating_adv=self.atk_user(u)
            temp=[]
            for data in rating_adv:
                _, rating_K_adv = torch.topk(data, k=max_K)
                #print(rating_K_adv)
                rating = data.cpu().detach().numpy()
                del rating
                temp.append(rating_K_adv)
            temp=torch.stack(temp,dim=0)
            rating_list_adv.append(temp)
            ground_true.append(self.dataset.allPos[u])
            #print(ground_true)
            print(f"{count}/{len(self.users)}")
            count+=1
        rating_list_adv=torch.stack(rating_list_adv,dim=1).cpu()
        #print(rating_list_adv)
        #print(ground_true)
        eps=self.eps
        for j,data in enumerate(rating_list_adv):
            print(f'eps:{eps[j]}')
            temp=self.Test_one_list(data, ground_true, self.users)
            key = self.results.keys()
            for i in key:
                self.results[i].append(temp[i])
        self.get_results()


    def atk_test(self):
        u_batch_size = 1  # 测试batchsize：100
        testDict: dict = self.dataset.testDict
        max_K = max(world.topks)  # 20
        users = list(testDict.keys())  # 测试数据中的所有用户
        #users=users[104:105]
        # print('users:',len(users))
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        count=1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = self.dataset.getUserPosItems(batch_users)
            groundTrue = testDict[batch_users[0]]  # 测试数据中user的positem[[],...,[]]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating_adv = self.atk_user(batch_users_gpu)  # users对所有item的分数(test_batchsize,4489)
            # print(f'batch_users:{batch_users}\n{len(batch_users)}\ngroundTrue:{groundTrue}\nrating:{rating}\n{rating.shape}')
            # rating = rating.cpu()
            temp=[]
            for rating in rating_adv:
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):  # 去掉pos，只预测neg？
                    # print(range_i,items) 下标和positem的列表
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1 << 10)  # -1024，从user对所有item的评分中去除user对应的positem
                _, rating_K = torch.topk(rating, k=max_K)  # 前20的下标，不是分数
                rating = rating.cpu().detach().numpy()
                del rating
                rating_K=rating_K.flatten()
                temp.append(rating_K.cpu())
            temp=torch.stack(temp,dim=0)
            rating_list.append(temp)
            groundTrue_list.append(groundTrue)
            print(f"{count}/{len(self.users)}")
            count += 1
        rating_list = torch.stack(rating_list, dim=1).cpu()
        #print(groundTrue_list)
        # print(rating_list_adv)
        eps = self.eps
        for j, data in enumerate(rating_list):
            print(f'eps:{eps[j]}')
            temp = self.Test_one_list(data, groundTrue_list, self.users)
            key = self.results.keys()
            for i in key:
                self.results[i].append(temp[i])
        self.get_results()

    def Test_one_list(self,rating_list,ground_true,users):
        results = {'precision': np.zeros(1),
                   'recall': np.zeros(1),
                   'ndcg': np.zeros(1)}
       # print(rating_list)
        X = (torch.tensor(np.array(rating_list)), ground_true)
        pre_results=[]
        pre_results.append(Procedure.test_one_batch(X))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results['recall']=round(results['recall'].item(),6)
        results['precision']=round(results['precision'].item(),6)
        results['ndcg']=round(results['ndcg'].item(),6)
        print(results)
        return results

    def get_results(self):
        print(self.results)
        with open('results.txt', 'a') as f:
            f.write(time.strftime("-%m-%d-%Hh%Mm%Ss")+'\n')
            if self.w!=None:
                f.write(self.w+'\n')
            f.write('eps:'+str(self.eps).replace('\n','')+'\n')
            key=self.results.keys()
            for i in key:
                f.write(i+':'+str(self.results[i]).replace('\n','')+'\n')



    def attack(self):
        if self.mode=='train':
            self.atk_train()
        elif self.mode=='test':
            self.atk_test()



#w=SummaryWriter(logdir='./runs/mfatk_shuffle'+time.strftime("-%m-%d-%Hh%Mm%Ss"))
Rec=['lgn','mf']
dataset=['lastfm', 'gowalla', 'yelp2018', 'amazon-book']
model_path='./checkpoints/mf-lastfm-64.pth.tar'#mf-lastfm-64.pth.tar lastfm-3-64-base.pth.tar
ran=True
eps=[0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
#eps=np.linspace(0,1,num = 101)
#eps=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0]
users=[0,1]
w='ran_pureMF_test_100'
test=atk_Train(Rec[1],dataset[0],model_path,eps,ran,w=w,users=None,mode='test')
test.attack()
#testetsetstetse




'''
dataset=dataloader.LastFM()
recmodel=model.PureMF(world.config,dataset)
recmodel.load_state_dict(torch.load('./checkpoints/mf-lastfm-64.pth.tar'))
recmodel = recmodel.to(world.device)
Procedure.Test(dataset,recmodel,0)
'''
'''
dataset=dataloader.LastFM()
recmodel=model.LightGCN(world.config, dataset)
recmodel.load_state_dict(torch.load('./checkpoints/lastfm-3-64-base.pth.tar'))
recmodel = recmodel.to(world.device)
Procedure.Test(dataset,recmodel,0)
'''
'''
eps=[0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
predict={}#    precision   recall    ndcg
predict['precision']=[0.73083067,0.71730564,0.68711395,0.63953674,
0.56778488,0.4701278,0.35039936,0.22326944,0.11664004,0.04880192,0.01605431]
predict['recall']=[0.69086117,0.67924705,0.65275653,0.61098851,
0.56778488,0.46049743,0.35183651,0.23460921,0.13297472,0.06520187,0.02966257]
predict['ndcg']=[0.81771278,0.80335733,0.7707755,0.71733174,
0.63609673,0.5240083,0.38528285,0.24311592,0.12826632,0.05704324,0.02255579]

viz=visdom.Visdom(env='Test')
viz.line(
    X=eps,
    Y=predict['precision'],
    name='precision',
    opts={
        'xlabel':'eps',
        'ylabel':'precision',
    }
)
'''


