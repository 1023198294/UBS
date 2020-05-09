import pandas as pd
import numpy as np
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt
import random
UP = 0
LEFT = 1
OK = 2
batch_size = 100
rating_file = ''
HGN_path = ''
SAS_path = ''
CASER_path = ''
def read_user_rating_records(dataset):
    col_names = ['user_id', 'item_id', 'rating', 'timestamp']
    if dataset=='ML':
        sep = '::'
    else:
        sep = ','
    data_records = pd.read_csv(dir_path + rating_file, sep=sep, names=col_names, engine='python')
    return data_records
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
def remove_infrequent_items(data, min_counts=20):
    df = deepcopy(data)
    counts = df['item_id'].value_counts()
    df = df[df["item_id"].isin(counts[counts >= min_counts].index)]

    print("items with < {} interactoins are removed".format(min_counts))
    # print(df.describe())
    return df

def remove_infrequent_users(data, min_counts=5):
    df = deepcopy(data)
    counts = df['user_id'].value_counts()
    df = df[df["user_id"].isin(counts[counts >= min_counts].index)]

    print("users with < {} interactoins are removed".format(min_counts))
    # print(df.describe())
    return df
def convert_data(data):
    # for each user, sort by timestamps
    df = deepcopy(data)
    df_ordered = df.sort_values(['timestamp'], ascending=True)
    data = df_ordered.groupby('user_id')['item_id'].apply(list)
    #print(data)
    #time_l = df_ordered.groupby('user')['checkin_time'].apply(list)
    #print(time_l)
    print("succressfully created sequencial data! head:", data.head(5))
    unique_data = df_ordered.groupby('user_id')['item_id'].nunique()
    data = data[unique_data[unique_data >= 10].index]
    #print(data[:10])
    #print(len(data))
    return data
def seqence_similarity(seq1,seq2):
    len1 = len(seq1)
    len2 = len(seq2)
    dp = np.zeros((len1+1,len2+1))
    for i in range(len1 + 1):
        dp[i][0]=i
    for j in range(len2 + 1):
        dp[0][j]=j
    for i in range(1,len1+1):
        for j in range(1,len2+1):
            delta=0 if seq1[i-1]==seq2[j-1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i-1][j] + 1, dp[i][j - 1] + 1))
    length = max(len1,len2)
    distance = dp[len1][len2]
    similarity = (length-distance)/distance
    return similarity
def LCS(seq1,seq2):
    if (set(seq1).intersection(set(seq2)))==set():
        return []
    m = [[0 for x in range(len(seq2)+1)] for y in range(len(seq1)+1)]
    d = [[None for x in range(len(seq2)+1)] for y in range(len(seq1)+1)]
    for p1 in range(len(seq1)):
        for p2 in range(len(seq2)):
            if seq1[p1] == seq2[p2]:
                m[p1+1][p2+1] = m[p1][p2]+1
                d[p1+1][p2+1] = OK
            elif m[p1+1][p2]>m[p1][p2+1]:
                m[p1+1][p2+1]=m[p1+1][p2]
                d[p1+1][p2+1] = LEFT
            else:
                m[p1+1][p2+1]=m[p1][p2+1]
                d[p1+1][p2+1] = UP
    (p1,p2) = (len(seq1),len(seq2))
    index = []
    while m[p1][p2]:
        direction = d[p1][p2]
        #print(p1, p2)
        #print(direction)
        if direction == OK:
            index.append((p1-1,p2-1))
            p1 -= 1
            p2 -= 1
        elif direction == LEFT:
            p2 -= 1
        elif direction == UP:
            p1 -= 1
    index.reverse()
    return index


def merging(seq1,seq2,index):
    index_first = [x[0] for x in index]
    index_second = [x[1] for x in index]
    new_seq = []
    parts = len(index)
    p1 = 0
    p2 = 0
    for i in range(parts):
        idx1 = index_first[i]
        idx2 = index_second[i]
        new_seq+=seq1[p1:idx1]+seq2[p2:idx2]+[seq1[idx1]]
        p1 = idx1+1
        p2 = idx2+1

    new_seq+=seq1[p1:len(seq1)]+seq2[p2:len(seq2)]
    return new_seq
#datasets = ['Beauty','Game']
Ks = [1,3,5,0]
datasets = ['Beauty','Game','ML']
dir_path = './preprocess/data/'
K = 0
hists = []
for dataset in datasets:
        print('Now processing dataset {} with K={}'.format(dataset,K))
        if dataset == 'Beauty':
            rating_file = 'ratings_Beauty.csv'
        elif dataset=='Game':
            rating_file = 'ratings_Toys_and_Games.csv'
        elif dataset=='ML':
            rating_file = 'ratings.dat'
        data_records = read_user_rating_records(dataset)
        data_records['user_id'] = data_records['user_id'].astype(str)
        data_records.loc[data_records.rating < 4, 'rating'] = 0
        data_records.loc[data_records.rating >= 4, 'rating'] = 1
        data_records = data_records[data_records.rating > 0]
        print(len(data_records['user_id'].unique()), len(data_records['item_id'].unique()))

        filtered_data = remove_infrequent_users(data_records, 5)
        filtered_data = remove_infrequent_items(filtered_data, 5)
        print('num of users:{}, num of items:{}'.format(len(filtered_data['user_id'].unique()), len(filtered_data['item_id'].unique())))
        seq_data = convert_data(filtered_data)
        len_list = []
        for items in seq_data:
            len_list.append(len(items))
        hists.append(len_list)
plt.figure()
plt.subplot(1,3,1)
plt.hist(hists[0],100,normed=1)
plt.title('Beauty',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('probability',fontsize=20)
plt.xlabel('sequence length',fontsize=20)
plt.subplot(1,3,2)
plt.hist(hists[1],100,normed=1)
plt.title('Game',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('probability',fontsize=20)
plt.xlabel('sequence length',fontsize=20)
plt.subplot(1,3,3)
plt.hist(hists[2],100,normed=1)
plt.title('MovieLens',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('probability',fontsize=20)
plt.xlabel('sequence length',fontsize=20)
plt.show()
