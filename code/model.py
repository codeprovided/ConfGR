import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
from torch_geometric.nn import global_mean_pool, global_add_pool,global_max_pool

class GroupDc(nn.Module):
    def __init__(self, args,user_num,item_num,group_num,ui_g_H,ug_affect_matrix,ui_matrix,gu_matrix,gi_g_matrix,gg_matrix,gg ):
        super(ModelName, self).__init__()
        
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args=args
        self.dim=args.dim
        self.user_num=user_num
        self.item_num=item_num
        self.group_num=group_num
        
        self.userEmb=nn.Embedding(user_num,self.dim)
        self.itemEmb=nn.Embedding(item_num,self.dim)
        self.groupEmb=nn.Embedding(group_num,self.dim)

        self.user_bias=nn.Embedding(user_num,1)
        self.item_bias=nn.Embedding(item_num,1)
        
        self.userMlp=nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(self.dim,self.dim),
            nn.Sigmoid(),
        )
        
        self.itemMlp=nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(self.dim,self.dim),
            nn.Sigmoid(),
        )
        
        
        self.W_user1=torch.nn.Parameter(torch.from_numpy(np.eye(self.group_num)*1/self.group_num).to(torch.float32))
        
        self.ui_g_H=ui_g_H
        
        self.W_user2=torch.nn.Parameter(torch.from_numpy(np.eye(self.group_num)*1/self.group_num).to(torch.float32))
       
        self.ug_affect_matrix=ug_affect_matrix
        
        
        self.W_group3=torch.nn.Parameter(torch.from_numpy(np.eye(self.group_num)*1/self.group_num).to(torch.float32))
        
        self.attenMlp=nn.Sequential(
            
            nn.Linear(2*self.dim,1),
        )
        
        
        self.predictUserMlp=nn.Sequential(
            nn.Linear(2*self.dim,1),
            nn.Sigmoid(),
        )
        
        self.predictGroupMlp=nn.Sequential(
            nn.Linear(2*self.dim,1),
            nn.Sigmoid(),
        )
        
        
        self.ui_matrix=ui_matrix
        
        
        self.gu_matrix=gu_matrix
        self.per_group_num=self.userchooseattneed(gu_matrix)
        self.gi_g_matrix=gi_g_matrix
        
        self.gg=gg
        self.gg_matrix=gg_matrix
        
        self.group_attA=nn.Sequential(
            nn.Linear(self.dim,1),
            nn.Sigmoid(),
        )
        self.group_attB=nn.Sequential(
            nn.Linear(self.dim,1),
            nn.Sigmoid(),
        )
        
        self.piW1=nn.Linear(self.dim,self.dim)
        self.piW2=nn.Linear(self.dim,self.dim,bias=False)
        self.pi=nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(self.dim,1)
        )
        self.miW=Parameter(torch.Tensor(self.dim, self.dim))
        self.guWeight=self.getAffectGroupEmb(gu_matrix)
        
        
    def getUserEmbedding(self,userid):
        useremb=self.userEmb(userid)
        return self.userMlp(useremb)
    
    def getItemEmbedding(self,itemid):
        itememb=self.itemEmb(itemid)
        return self.itemMlp(itememb)
    
    
    
    def userchooseneed(self,ui_g_H):
        ui_g_H=ui_g_H.toarray()
        H = np.array(ui_g_H)
        
        H=torch.from_numpy(H).float().to(self.device).to(torch.float32)
        
        DV = torch.sum(H, dim=1) + 1e-5
        
        DE = torch.sum(H, dim=0) + 1e-5
        
        invDE = torch.diag(torch.pow(DE, -1))
        
        DV2 = torch.diag(torch.pow(DV, -1))
        
        HT = H.T
        
        
        G = DV2@H@self.W_user1@invDE@HT
        
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
        G = DV2@H@self.W_user2@invDE@HT
        return G
    
    def userchooseattneed(self,gu_matrix):
        gu_matrix=gu_matrix.toarray()
        H = np.array(gu_matrix)
        H=torch.from_numpy(H).float()
        per_group_num = torch.sum(H, dim=1).int()
        return per_group_num.to(self.device)
    
    def predictUser(self,):
        useremb=self.getUserEmbedding(torch.arange(self.user_num).to(self.device))
        itememb=self.getItemEmbedding(torch.arange(self.item_num).to(self.device))
        ui_emb=torch.cat((useremb,itememb),dim=0)
        
        # choose
        G_choose=self.userchooseneed(self.ui_g_H)
        choose_ans=[ui_emb]
        all_choose_emb=ui_emb
        for i in range(self.args.klayer):
            temp=G_choose@choose_ans[-1]
            choose_ans.append(temp)
            all_choose_emb=all_choose_emb+temp
            
        final_choose_emb=all_choose_emb/(self.args.klayer+1)
        final_user_emb_index=torch.arange(self.user_num).to(self.device)
        final_item_emb_index=torch.arange(self.user_num,self.user_num+self.item_num).to(self.device)
        finalchoose_user_emb=final_choose_emb[final_user_emb_index]
        finalchoose_item_emb=final_choose_emb[final_item_emb_index]
        # affect
        
        G_affect=self.useraffectneed(self.ug_affect_matrix)
        affect_ans=[useremb]
        all_affect_emb=useremb
        for i in range(self.args.klayer):
            temp=G_affect@affect_ans[-1]
            affect_ans.append(temp)
            all_affect_emb=all_affect_emb+temp
        
        final_affect_emb=all_affect_emb/(self.args.klayer+1)
        return finalchoose_user_emb,finalchoose_item_emb,final_affect_emb
        
    def modelUser(self,userid,itemid):
        finalchoose_user_emb,finalchoose_item_emb,final_affect_emb=self.predictUser()
        u_emb=finalchoose_user_emb[userid]+final_affect_emb[userid]
        i_emb=finalchoose_item_emb[itemid]
        y_hat=self.predictUserMlp(torch.cat((u_emb,i_emb),dim=1)).reshape(-1)
        return y_hat    

        
    
    def groupchooseneed(self,gi_g_matrix):
        gi_g_matrix=gi_g_matrix.toarray()
        H = np.array(gi_g_matrix)
        H=torch.from_numpy(H).float().to(self.device)
        DV = torch.sum(H, dim=1) + 1e-5
        DE = torch.sum(H, dim=0) + 1e-5
        invDE = torch.diag(torch.pow(DE, -1))
        DV2 = torch.diag(torch.pow(DV, -1))
        HT = H.T         
        G = DV2@H@self.W_group3@invDE@HT
        
        return G
    
    def groupaffectneed(self,gg,gg_matrix):
        gg_matrix=gg_matrix.toarray()
        gg=gg.toarray()
        gg_matrix=np.array(gg_matrix)
        gg_matrix=torch.from_numpy(gg_matrix).float()
        DV=torch.sum(gg_matrix, dim=1) + 1e-5
        DV2 = torch.diag(torch.pow(DV, -1))
        gg=np.array(gg)
        gg=torch.from_numpy(gg).float()
        fenmu=DV.reshape(-1,1)+DV.reshape(1,-1)
        fenmu=fenmu-gg
        A=gg/fenmu
        G=DV2@A
        
        return G,A
    # 
    def getAffectGroupEmb(self,gu_matrix):
        gu_matrix=gu_matrix.toarray()
        gu_matrix = np.array(gu_matrix)
        gu_matrix=torch.from_numpy(gu_matrix).float().to(self.device)
        _,A=self.groupaffectneed(self.gg,self.gg_matrix)
        A=A.to(self.device)
        final_gu=[]
        for i in range(len(gu_matrix)):
            temp=gu_matrix[i]*gu_matrix
            temp=temp*(A[i].reshape(-1,1))
            temp=torch.sum(temp,dim=0)
            final_gu.append(temp)
        guWeight=torch.stack(final_gu,dim=0).to(self.device)
        return guWeight
    
    def predictGroup(self,):
        group_emb=self.groupEmb(torch.arange(self.group_num).to(self.device))
        item_emb=self.getItemEmbedding(torch.arange(self.item_num).to(self.device))
        
        emb=torch.cat((group_emb,item_emb),dim=0)
        choose_ans=[emb]
        all_choose_emb=emb
        G_choose=self.groupchooseneed(self.gi_g_matrix)
        for i in range(self.args.klayer):
            temp=G_choose@choose_ans[-1]
            choose_ans.append(temp)
            all_choose_emb=all_choose_emb+temp
            
        final_choose_emb=all_choose_emb/(1+self.args.klayer)  
        final_choose_group_emb_index=torch.arange(self.group_num).to(self.device)
        final_choose_item_emb_index=torch.arange(self.item_num).to(self.device)
        final_choose_group_emb=final_choose_emb[final_choose_group_emb_index] 
        final_choose_item_emb=final_choose_emb[ final_choose_item_emb_index]
        
        user_emb=self.getUserEmbedding(torch.arange(self.user_num).to(self.device))
        g_emb=self.guWeight@user_emb
        affect_ans=[g_emb]
        all_affect_emb=g_emb
        G_affect,_=self.groupaffectneed(self.gg,self.gg_matrix)
        G_affect=G_affect.to(self.device)
        for i in range(self.args.klayer):
            temp=G_affect@affect_ans[-1]
            affect_ans.append(temp)
            all_affect_emb=all_affect_emb+temp
        
        final_affect_emb=all_affect_emb/(1+self.args.klayer)
        
        return final_choose_group_emb,final_choose_item_emb,final_affect_emb
        
           
    
    def modelGroup(self,groupid,itemid):
        final_choose_group_emb,final_choose_item_emb,final_affect_emb=self.predictGroup()
        choose_group_emb=final_choose_group_emb[groupid]
        choose_item_emb=final_choose_item_emb[itemid]
        attA=self.group_attA(choose_group_emb)
        
        affect_group_emb=final_affect_emb[groupid]
        attB=self.group_attB(affect_group_emb)
        group=attA*choose_group_emb+attB*affect_group_emb 
        
        y_hat=self.predictGroupMlp(torch.cat((group,choose_item_emb),dim=1)).reshape(-1)
        return y_hat 
    
    def MI(self,U,G):
        return  torch.sigmoid(U@self.miW@G.T)  
    
    def modelssl(self,groupid,itemid,neg_num=1):
        U_choose_user_emb,U_choose_item_emb,U_affect_emb=self.predictUser()
        G_choose_group_emb,G_choose_item_emb,G_affect_emb=self.predictGroup()
        
        attA=self.group_attA(G_choose_group_emb)
        attB=self.group_attB(G_affect_emb)
        G_target_allgroup=attA*G_choose_group_emb+attB*G_affect_emb
        G_target_group=G_target_allgroup[groupid]
        
        
        user_index=torch.tensor(self.gu_matrix.indices).to(self.device).long()
        per_group_num=self.per_group_num
        batch=torch.repeat_interleave(torch.arange(self.group_num).to(self.device),self.per_group_num)
        
        group_negative_index=torch.cat((torch.arange(1,self.group_num),torch.tensor([0])),dim=0).to(self.device)
        
        size=len(groupid)
        grouplist=[]
        group_neg_list=[]
        for i in range(size):
            curitemid=itemid[i]
            
            att=self.attenMlp(torch.cat((U_choose_user_emb,U_choose_item_emb[curitemid].repeat(self.user_num,1)),dim=1))
            user_index_att=torch.exp(att[user_index])
            fenmu=global_add_pool(user_index_att,batch)
            fenmu=torch.repeat_interleave(fenmu,self.per_group_num).unsqueeze(1)
            softmax_att=user_index_att/fenmu
            
            group_emb=global_add_pool(softmax_att*U_choose_user_emb[user_index],batch)
            grouplist.append(group_emb[groupid[i]])
            group_neg_list.append(group_emb[group_negative_index[groupid[i]]])
            
            
        U_choose_group=torch.stack(grouplist,dim=0) 
        U_choose_neg_group=torch.stack(group_neg_list,dim=0)
        
        
        user_index_emb=U_choose_user_emb[user_index]
        
        graph_emb=global_add_pool(user_index_emb,batch)
        
        graph_del_u=torch.repeat_interleave(graph_emb,self.per_group_num,dim=0)-user_index_emb
        
        graph_del_u=graph_del_u/(torch.repeat_interleave(self.per_group_num,self.per_group_num).reshape(-1,1)-1)
        pi=self.piW1(user_index_emb)+self.piW2(graph_del_u)
        pi=self.pi(pi)
        
        fenzi=torch.exp(pi)
        fenmu=global_add_pool(fenzi,batch)
        fenmu=torch.repeat_interleave(fenmu,self.per_group_num).unsqueeze(1)
        
        final_pi=fenzi/fenmu
        
        U_affect_allgroup=global_add_pool(U_affect_emb[user_index]*final_pi,batch)
        
        
        U_affect_group=U_affect_allgroup[groupid]
        U_affect_neg_group=U_affect_allgroup[group_negative_index[groupid]]
        
        U_target_group=U_choose_group+U_affect_group
        U_neg_target_group=U_choose_neg_group+U_affect_neg_group
        
        G_neg_target_group=G_target_allgroup[group_negative_index[groupid]]
        
        
        
        l=torch.log(self.MI(U_target_group,G_target_group))+torch.log(1-self.MI(U_neg_target_group,U_target_group))+torch.log(1-self.MI(G_neg_target_group,G_target_group))
        l=l.reshape(-1)
        l=torch.sum(l)
        l=-l
        return l