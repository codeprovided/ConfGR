from user_utils import UserDataset
from group_utils import GroupDataset
import argparse
import os
from model import ModelName
from train import log, train,evaluate
# import user_utils
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MaFengWo', help='which dataset to use')
parser.add_argument('--dim', type=int, default=64, help='dimension of entity and relation embeddings')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr2', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=40, help='the number of epochs')
parser.add_argument('--klayer', type=int, default=2, help='the number of layers')
parser.add_argument('--userneg', type=int, default=2, help='')
parser.add_argument('--groupneg', type=int, default=15, help='')
args = parser.parse_args()
data_user=UserDataset(args,'../',args.dataset,args.userneg)
ui_g_matrix,ug_affect_matrix,ui_matrix=data_user.get_u_i_group()
user_train_loader=data_user.getUserDataloader(ui_matrix,args.userneg)

userTest=data_user.getUserTest()
userNeg=data_user.getUserNeg()

user_num=data_user.user_num
item_num=data_user.item_num
group_num=data_user.group_num



data_group=GroupDataset(args,'../',args.dataset,args.groupneg)
gi_g_matrix,gg_matrix,gg,gi_matrix,gu_matrix,ug_matrix,g_u_dict=data_group.get_gi_g_matrix()
group_train_loader=data_group.getGroupTrainLoader(gi_matrix,args.groupneg)

groupTest=data_group.getGroupTest()
groupNeg=data_group.getGroupNeg()

log('dim:{},batch_size:{},klayer:{},userneg:{},groupneg:{}'.format(args.dim,args.batch_size,args.klayer,args.userneg,args.groupneg))
model=ModelName(args,user_num,item_num,group_num,ui_g_matrix,ug_affect_matrix,ui_matrix,gu_matrix,gi_g_matrix,gg_matrix,gg,ug_matrix,g_u_dict)
for i in range(args.epochs):
    train(args,user_train_loader,group_train_loader,model,i+1,group_train_loader)
    evaluate(model,groupTest,groupNeg,5,'group')
    evaluate(model,groupTest,groupNeg,10,'group')