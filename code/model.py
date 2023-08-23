import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
from torch_geometric.nn import global_mean_pool, global_add_pool,global_max_pool

class ModelName(nn.Module):
    def __init__(self, args,user_num,item_num,group_num,ui_g_H,ug_affect_matrix,ui_matrix,gu_matrix,gi_g_matrix,gg_matrix,gg ,ug_matrix,g_u_dict):
        super(ModelName, self).__init__()
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args=args
        self.dim=args.dim
        self.user_num=user_num
        self.item_num=item_num
        self.group_num=group_num
        
        self.userEmbedding=nn.Embedding(self.user_num,self.dim)
        self.itemEmbedding=nn.Embedding(self.item_num,self.dim)
        self.groupEmbedding=nn.Embedding(self.group_num,self.dim)
        
        self.predict=nn.Sequential(
            nn.Linear(3*self.dim,8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        
        self.att=nn.Sequential(
            nn.Linear(2*self.dim,16),
            nn.ReLU(),
            nn.Linear(16,1),
        )
        
        
        self.per_group_num=self.PerGroupNum(gu_matrix)
        self.ug_matrix=ug_matrix
        self.ug_affect_matrix=ug_affect_matrix
        self.gg_matrix=gg_matrix
        self.gu_matrix=gu_matrix
        self.g_u_dict=g_u_dict
        
        
    def gengerateGfromH(self,ug_matrix):
        ug_matrix=ug_matrix.toarray()
        H = np.array(ug_matrix)
        H=torch.from_numpy(H).float().to(self.device)
        DV = torch.sum(H, dim=1) + 1e-5
        DE = torch.sum(H, dim=0) + 1e-5
        invDE = torch.diag(torch.pow(DE, -1))
        DV2 = torch.diag(torch.pow(DV, -1))
        HT = H.T
        
        G = DV2@H@invDE@HT
        return G
    
    def userchooseneed(self,u_g_H):
        u_g_H=u_g_H.toarray()
        H = np.array(u_g_H)
        H=torch.from_numpy(H).float().to(self.device).to(torch.float32)
        DV = torch.sum(H, dim=1) + 1e-5
        DE = torch.sum(H, dim=0) + 1e-5
        invDE = torch.diag(torch.pow(DE, -1))
        DV2 = torch.diag(torch.pow(DV, -1))
        HT = H.T
        G = DV2@H@invDE@HT
        
        return G
    
    def useraffectneed(self,ug_affect_matrix):
        ug_affect_matrix=ug_affect_matrix.toarray()
        H = np.array(ug_affect_matrix)
        H=torch.from_numpy(H).float().to(self.device)
        DV = torch.sum(H, dim=1) + 1e-5
        DE = torch.sum(H, dim=0) + 1e-5
        invDE = torch.diag(torch.pow(DE, -1))
        DV2 = torch.diag(torch.pow(DV, -1))
        HT = H.T
        
        G = DV2@H@invDE@HT
        return G
    
    def predictUser(self,):
        useremb=self.userEmbedding(torch.arange(self.user_num).to(self.device))
        itememb=self.itemEmbedding(torch.arange(self.item_num).to(self.device))
        u_emb=useremb
        G_choose=self.userchooseneed(self.ug_matrix)
        choose_ans=[u_emb]
        all_choose_emb=u_emb
        for i in range(self.args.klayer):
            temp=G_choose@choose_ans[-1]
            choose_ans.append(temp)
            
        final_choose_emb=choose_ans[-1]
        
        final_user_emb_index=torch.arange(self.user_num).to(self.device)
              
        
        G_affect=self.useraffectneed(self.ug_affect_matrix)
        affect_ans=[u_emb]
        all_affect_emb=useremb
        for i in range(self.args.klayer):
            temp=G_affect@affect_ans[-1]
            affect_ans.append(temp)

        final_affect_emb=affect_ans[-1]
        
        
        user=(final_choose_emb+final_affect_emb)/2
        return user,itememb
        
    def modelUser(self,userid,itemid):    
        useremb,itememb=self.predictUser()
        user=useremb[userid]
        item=itememb[itemid]
        element_embeds=torch.mul(user, item)
        newemb=torch.cat((element_embeds,user,item),dim=1)
        y = torch.sigmoid(self.predict(newemb))
        return y
    
    def groupchooseneed(self,gg_matrix):
        gg_matrix=gg_matrix.toarray()
        H = np.array(gg_matrix)
        H=torch.from_numpy(H).float().to(self.device)
        DV = torch.sum(H, dim=1) + 1e-5
        DE = torch.sum(H, dim=0) + 1e-5
        invDE = torch.diag(torch.pow(DE, -1))
        DV2 = torch.diag(torch.pow(DV, -1))
        HT = H.T
                
        G = DV2@H@invDE@HT
        
        return G
    
    def PerGroupNum(self,gu_matrix):
        gu_matrix=gu_matrix.toarray()
        H = np.array(gu_matrix)
        
        H=torch.from_numpy(H).float()
        per_group_num = torch.sum(H, dim=1).int()
        return per_group_num.to(self.device)
    
    def modelGroup(self,groupid,itemid):
        group_emb=self.groupEmbedding(torch.arange(self.group_num).to(self.device))
        item_emb=self.itemEmbedding(torch.arange(self.item_num).to(self.device))
        emb=group_emb
        choose_ans=[emb]
        all_choose_emb=emb
        G_choose=self.groupchooseneed(self.gg_matrix)
        for i in range(self.args.klayer):
            temp=G_choose@choose_ans[-1]
            choose_ans.append(temp)
        
        final_choose_emb=choose_ans[-1]    
        
        choose_emb=final_choose_emb[groupid]
        
        user,item=self.predictUser()
        
        f=0
        alluser=0
        for gid in groupid:
            
            curgroupuseremb=self.g_u_dict[gid.item()]
            
            curgroupuseremb=user[curgroupuseremb]
            
            
            if f==0:
                alluser=curgroupuseremb
            else:
                alluser=torch.cat((alluser,curgroupuseremb),dim=0)
            f=f+1
        itememb=item[itemid] 
        group_num=self.per_group_num[groupid]
        
        itememb=torch.repeat_interleave(itememb,group_num,dim=0)
        
        att=torch.cat((alluser,itememb),dim=1)
        att=self.att(att)
        att=torch.exp(att)
        batch=torch.repeat_interleave(torch.arange(len(groupid)).to(self.device),group_num)
        fenmu=global_add_pool(att,batch)
        
        fenmu=torch.repeat_interleave(fenmu,group_num,dim=0)
        
        final_att=att/fenmu
        
        affect_group_emb=global_add_pool(torch.mul(final_att,alluser),batch)  
        
        group=affect_group_emb+choose_emb
        
        element_embeds=torch.mul(group,item[itemid])
        new_embeds = torch.cat((element_embeds, group, item[itemid]), dim=1)
        y = torch.sigmoid(self.predict(new_embeds))
        return y        
        