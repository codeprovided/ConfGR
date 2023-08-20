from user_utils import UserDataset
from group_utils import GroupDataset
import argparse
import os
from model import GroupDc
from train import train,evaluate
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MaFengWo', help='which dataset to use')
parser.add_argument('--dim', type=int, default=32, help='dimension of entity and relation embeddings')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=2, help='the number of epochs')
parser.add_argument('--klayer', type=int, default=3, help='the number of layers')
parser.add_argument('--lam', type=float, default=0.1)
args = parser.parse_args()
data_user=UserDataset(args,'../',args.dataset,1)
ui_g_matrix,ug_affect_matrix,ui_matrix=data_user.get_u_i_group()
user_train_loader=data_user.getUserDataloader(ui_matrix,1)

userTest=data_user.getUserTest()
userNeg=data_user.getUserNeg()

user_num=data_user.user_num
item_num=data_user.item_num
group_num=data_user.group_num



data_group=GroupDataset(args,'../',args.dataset,1)
gi_g_matrix,gg_matrix,gg,gi_matrix,gu_matrix=data_group.get_gi_g_matrix()
group_train_loader=data_group.getGroupTrainLoader(gi_matrix,1)

groupTest=data_group.getGroupTest()
groupNeg=data_group.getGroupNeg()


model=GroupDc(args,user_num,item_num,group_num,ui_g_matrix,ug_affect_matrix,ui_matrix,gu_matrix,gi_g_matrix,gg_matrix,gg )
for i in range(args.epochs):
    train(args,user_train_loader,group_train_loader,model)
    evaluate(model,userTest,userNeg,5,'user')
    evaluate(model,groupTest,groupNeg,5,'group')