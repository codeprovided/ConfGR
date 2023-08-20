import scipy.sparse as sp
from scipy.sparse import vstack
import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

class UserDataset(object):

    def __init__(self, args, path ,dataset , num_negatives):
        '''
        Constructor
        '''
        self.args=args
        self.num_negatives = num_negatives
        
        self.group_member_txt=path+dataset+'/groupMember.txt'
        self.user_rating_train_txt=path+dataset+'/userRatingTrain.txt'
        self.userFollow_txt=path+dataset+'/userFollow.txt'
        self.userRatingTest_txt=path+dataset+'/userRatingTest.txt'
        self.userRatingNegative_txt=path+dataset+'/userRatingNegative.txt'
        # user data
        # self.user_trainMatrix = self.load_rating_file_as_matrix(user_path + "Train.txt")
        # self.user_testRatings = self.load_rating_file_as_list(user_path + "Test.txt")
        # self.user_testNegatives = self.load_negative_file(user_path + "Negative.txt")
        # self.num_users, self.num_items = self.user_trainMatrix.shape
        # group data
        # self.group_trainMatrix = self.load_rating_file_as_matrix(group_path + "Train.txt")
        # self.group_testRatings = self.load_rating_file_as_list(group_path + "Test.txt")
        # self.group_testNegatives = self.load_negative_file(group_path + "Negative.txt")
        
        # user data
        

    # 成员级社会选择+成员级社会影响
    def get_u_i_group(self):
        ui_data=pd.read_csv(self.user_rating_train_txt,sep=' ',header=None)
        user_ui=ui_data[0]
        item_ui=ui_data[1]
        self.user_num=ui_data[0].max()+1
        self.item_num=ui_data[1].max()+1
        ui_matrix=sp.csr_matrix((np.ones_like(user_ui),(user_ui,item_ui)),
                                dtype=np.float32,shape=(self.user_num,self.item_num))
        iu_matrix=sp.csr_matrix((np.ones_like(user_ui),(item_ui,user_ui)),
                                dtype=np.float32,shape=(self.item_num,self.user_num))
        # group,list user
        u_g_data=pd.read_csv(self.group_member_txt,sep=' ',header=None)
        self.group_num=len(u_g_data)
        user_index=[]
        group_index=[]
        for i in range(self.group_num): 
            group_id,listuserid=u_g_data.iloc[i][0],u_g_data.iloc[i][1]
            for userid in listuserid.split(','):
                group_index.append(int(group_id))
                user_index.append(int(userid))
                
        ug_matrix=sp.csr_matrix((np.ones_like(user_index),(np.array(user_index),np.array(group_index))), 
                                dtype=np.float32,shape=(self.user_num, self.group_num))    
        ig=iu_matrix.dot(ug_matrix)
        ig_matrix=sp.csr_matrix((np.ones_like(ig.data),ig.indices,ig.indptr),
                                dtype=np.float32,shape=(self.item_num, self.group_num))
        
        ui_g_matrix=vstack((ug_matrix,ig_matrix))
        # # 成员级社会影响
        ufollow=pd.read_csv(self.userFollow_txt,header=None,sep=':')
        user_uf=[]
        follow_uf=[]
        for i in range(len(ufollow)):
            user_id,followerList=ufollow.iloc[i][0],ufollow.iloc[i][1]
            for follower_id in followerList.split(" "):
                if int(follower_id)<self.user_num:
                    user_uf.append(user_id)
                    follow_uf.append(int(follower_id))
        fu_matrix=sp.csr_matrix((np.ones_like(user_uf),(np.array(follow_uf),np.array(user_uf))),
                                dtype=np.float32,shape=(self.user_num, self.user_num))
        fg=fu_matrix.dot(ug_matrix)
        fg=fg+ug_matrix
        ug_affect_matrix=sp.csr_matrix((np.ones_like(fg.data),fg.indices,fg.indptr),
                                dtype=np.float32,shape=(self.user_num, self.group_num))
        
        return ui_g_matrix,ug_affect_matrix,ui_matrix
    
    def getUserDataloader(self,ui_matrix,neg_num):
        # train=pd.read_csv(userRatingTrain_txt,sep=' ',header=None)
        # train_len=len(train)
        ui_matrix=ui_matrix.todok()
        all_user=[]
        all_pos_item=[]
        all_neg_item=[]
        for (u,i) in ui_matrix.keys():
            for _ in range(neg_num):
                j = np.random.randint(self.item_num)
                while(u,j) in ui_matrix:
                    j = np.random.randint(self.item_num)
                all_user.append(u)
                all_pos_item.append(i)
                all_neg_item.append(j)
                
        traindata=TensorDataset(torch.LongTensor(all_user),torch.LongTensor(all_pos_item),torch.LongTensor(all_neg_item)) 
        trainloader=DataLoader(traindata,batch_size=self.args.batch_size,shuffle=True)          
        return trainloader
    
    
    
    def getUserTest(self,):
        ratingList = []
        with open(self.userRatingTest_txt, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def getUserNeg(self,):
        negativeList = []
        with open(self.userRatingNegative_txt, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    
    
    
    
    
    
    
    
    
    
    
    
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                if len(arr) > 2:
                    user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                    if (rating > 0):
                        mat[user, item] = 1.0
                else:
                    user, item = int(arr[0]), int(arr[1])
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

    def get_train_instances(self, train):
        user_input, pos_item_input, neg_item_input = [], [], []
        num_users = train.shape[0]
        num_items = train.shape[1]
        for (u, i) in train.keys():
            # positive instance
            for _ in range(self.num_negatives):
                pos_item_input.append(i)
            # negative instances
            for _ in range(self.num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                user_input.append(u)
                neg_item_input.append(j)
        pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]
        return user_input, pi_ni

    def get_user_dataloader(self, batch_size):
        user, positem_negitem_at_u = self.get_train_instances(self.user_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(user), torch.LongTensor(positem_negitem_at_u))
        user_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return user_train_loader

    def get_group_dataloader(self, batch_size):
        group, positem_negitem_at_g = self.get_train_instances(self.group_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(group), torch.LongTensor(positem_negitem_at_g))
        group_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return group_train_loader