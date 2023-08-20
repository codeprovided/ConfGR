from os import device_encoding
from statistics import mode
import torch
import numpy as np
import pandas as pd
import math
import heapq
def train(args,Userloader,Grouploader,model):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device)
    opt=torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for data in Userloader:
        opt.zero_grad()
        data=data
        userid=data[0].to(device)
        item_i_id=data[1].to(device)
        item_j_id=data[2].to(device)
        y_pos=model.modelUser(userid,item_i_id)
        y_neg=model.modelUser(userid,item_j_id)
        loss=torch.sum((y_pos-y_neg-1)**2)
        loss.backward()
        opt.step()
        # assert 1>2,'user success'
    for data in Grouploader:
        opt.zero_grad()
        data=data
        groupid=data[0].to(device)
        item_i_id=data[1].to(device)
        item_j_id=data[2].to(device)
        y_pos=model.modelGroup(groupid,item_i_id)
        y_neg=model.modelGroup(groupid,item_j_id)
        loss=torch.sum((y_pos-y_neg-1)**2)
        loss.backward()
        opt.step()
        # assert 1>2,'group success'
    for data in Grouploader:
        opt.zero_grad()
        data=data
        groupid=data[0].to(device)
        itemid=data[1].to(device)
        l=args.lam*model.modelssl(groupid,itemid)
        l.backward()
        opt.step()
        # assert 1>2,'ssl success'

def evaluate(model,testRatings, testNegatives, K, type_m):
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model=model.to(device)
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, K, type_m)
    hit, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    if type_m=='user':
        print('User@{} hit:{:.5f},ndcg:{:.5f}'.format(K,hit,ndcg))
    else:
        print('Group@{} hit:{:.5f},ndcg:{:.5f}'.format(K,hit,ndcg))

def evaluate_model( model, testRatings, testNegatives, K, type_m):
    
    hits, ndcgs = [], []

    for idx in range(len(testRatings)):
        (hr,ndcg) = eval_one_rating(model, testRatings, testNegatives, K, type_m, idx)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits, ndcgs)


def eval_one_rating(model, testRatings, testNegatives, K, type_m, idx):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rating = testRatings[idx]
    items = testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u)

    users_var = torch.from_numpy(users).to(device)
    items_var = torch.LongTensor(items).to(device)
    if type_m == 'group':
        predictions = model.modelGroup(users_var,items_var).cpu()
    elif type_m == 'user':
        predictions = model.modelUser(users_var,items_var).cpu()
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions.data.numpy()[i]
    items.pop()

    # Evaluate top rank list
    ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio( ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG( ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
