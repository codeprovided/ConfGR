import scipy.sparse as sp
from scipy.sparse import vstack
import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
class GroupDataset(object):

    def __init__(self, args, path ,dataset , num_negatives):
        '''
        Constructor
        '''
        self.args=args
        self.num_negatives = num_negatives
        self.group_member_txt=path+dataset+'/groupMember.txt'
        self.group_rating_train_txt=path+dataset+'/groupRatingTrain.txt'
        
        self.user_rating_train_txt=path+dataset+'/userRatingTrain.txt'
        
        self.groupRatingTest_txt=path+dataset+'/groupRatingTest.txt'
        self.groupRatingNegative_txt=path+dataset+'/groupRatingNegative.txt'
        # self.userFollow_txt=path+dataset+'/userFollow.txt'
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
        
    def get_gi_g_matrix(self,):
        
        ui_data=pd.read_csv(self.user_rating_train_txt,sep=' ',header=None)
        self.user_num=ui_data[0].max()+1
        self.item_num=ui_data[1].max()+1
        
        
        
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
                
        gu_matrix=sp.csr_matrix((np.ones_like(user_index),(np.array(group_index),np.array(user_index))), 
                                dtype=np.float32,shape=(self.group_num, self.user_num))  
        ug_matrix=sp.csr_matrix((np.ones_like(user_index),(np.array(user_index),np.array(group_index))), 
                                dtype=np.float32,shape=(self.user_num, self.group_num))  
        # 共同用户
        gg=gu_matrix.dot(ug_matrix) 
        gg_matrix=sp.csr_matrix((np.ones_like(gg.data),gg.indices,gg.indptr),
                                dtype=np.float32,shape=(self.group_num, self.group_num)) 
        
        g_u_dict=defaultdict(list)
        for i in range(self.group_num):
            group_id,listuserid=u_g_data.iloc[i][0],u_g_data.iloc[i][1]
            for userid in listuserid.split(','):
                g_u_dict[int(group_id)].append(int(userid))
                # user_index.append(int(userid))
                
        gi=pd.read_csv(self.group_rating_train_txt,header=None,sep=' ')    
        group_index_gi=gi[0]
        item_index_gi=gi[1]
        ig_matrix=sp.csr_matrix((np.ones_like(group_index_gi),(np.array(item_index_gi),np.array(group_index_gi))),
                         dtype=np.float32,shape=(self.item_num, self.group_num))
        gi_matrix=sp.csr_matrix((np.ones_like(item_index_gi),(np.array(group_index_gi),np.array(item_index_gi))),
                         dtype=np.float32,shape=(self.group_num, self.item_num))
        gi_g_matrix=vstack((gg_matrix,ig_matrix))
        
                        # 计算D^-1,  计算A用的
        return gi_g_matrix,gg_matrix,gg ,gi_matrix,gu_matrix,ug_matrix,g_u_dict
    
    
    
    def getGroupTrainLoader(self,gi_matrix,neg_num):
        gi_matrix=gi_matrix.todok()
        all_group=[]
        all_pos_item=[]
        all_neg_item=[]
        for (g,i) in gi_matrix.keys():
            for _ in range(neg_num):
                j = np.random.randint(self.item_num)
                while(g,j) in gi_matrix:
                    j = np.random.randint(self.item_num)
                all_group.append(g)
                all_pos_item.append(i)
                all_neg_item.append(j)
        
        traindata=TensorDataset(torch.LongTensor(all_group),torch.LongTensor(all_pos_item),torch.LongTensor(all_neg_item)) 
        trainloader=DataLoader(traindata,batch_size=self.args.batch_size,shuffle=True)          
        return trainloader
    
    
    def getGroupTest(self,):
        ratingList = []
        with open(self.groupRatingTest_txt, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def getGroupNeg(self,):
        negativeList = []
        with open(self.groupRatingNegative_txt, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList