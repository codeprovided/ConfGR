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
    
    