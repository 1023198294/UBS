import pandas as pd
import numpy as np
from copy import deepcopy
import pickle
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
datasets = ['Beauty','Game']
Ks = [10]
#datasets = ['ML']
dir_path = './preprocess/data/'
for K in Ks:
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
        batch_num = len(seq_data)//batch_size
        index = 0
        series = []
        for i in range(0,batch_num+1):
            #print(index)
            end = index + 100 if index + 100 <= len(seq_data) else len(seq_data)
            series.append(seq_data[index:end])
            index += 100
        #print(series)
        merged_sequence = pd.Series()
        for b in range(batch_num+1):
            for k in range(K):
                MAX = 0
                MAX_A,MAX_B = -1,-1
                for i  in range(len(series[b])):
                    for j in range(i):
                        #print(i,j)
                        if i != j:
                            #if dataset=='ML':
                            #    #print(series[b].iloc[i])
                            #    similarity = seqence_similarity(series[b].iloc[i], series[b].iloc[j])
                            #    if similarity > MAX:
                            #        MAX = similarity
                            #        MAX_A = i
                            #        MAX_B = j
                            #        print('update similarity, MAX S={} comes from {} and {}'.format(MAX, MAX_A, MAX_B))

                            similarity = seqence_similarity(series[b][i],series[b][j])
                            if similarity>MAX:
                                MAX = similarity
                                MAX_A = i
                                MAX_B = j
                                print('update similarity, MAX S={} comes from {} and {}'.format(MAX,MAX_A,MAX_B))
                seq1 = series[b][MAX_A]
                seq2 = series[b][MAX_B]
                user_A = series[b].index[MAX_A]
                user_B = series[b].index[MAX_B]
                series[b] = series[b].drop([user_A,user_B])
                if len(seq1)<len(seq2):
                    index = LCS(seq2,seq1)
                    seq = merging(seq2,seq1,index)
                    series[b][user_B]=seq
                else:
                    index = LCS(seq1,seq2)
                    seq = merging(seq1,seq2,index)
                    series[b][user_A] = seq
                print('In batch {}, merging step {}, user {} and {} are merged'.format(b,k+1,user_A,user_B))
        for b in range(batch_num+1):
            #print(b)
            merged_sequence = pd.concat([merged_sequence,series[b]])
        #print(len(merged_sequence),len(seq_data))
        user_item_dict = merged_sequence.to_dict()
        user_mapping = []
        item_set = set()
        for user_id, item_list in merged_sequence.iteritems():
            user_mapping.append(user_id)
            for item_id in item_list:
                item_set.add(item_id)
        item_mapping = list(item_set)
        def generate_inverse_mapping(data_list):
            inverse_mapping = dict()
            for inner_id, true_id in enumerate(data_list):
                inverse_mapping[true_id] = inner_id
            return inverse_mapping

        def convert_to_inner_index(user_records, user_mapping, item_mapping):
            inner_user_records = []
            user_inverse_mapping = generate_inverse_mapping(user_mapping)
            item_inverse_mapping = generate_inverse_mapping(item_mapping)

            for user_id in range(len(user_mapping)):
                real_user_id = user_mapping[user_id]
                item_list = list(user_records[real_user_id])
                for index, real_item_id in enumerate(item_list):
                    item_list[index] = item_inverse_mapping[real_item_id]
                inner_user_records.append(item_list)

            return inner_user_records, user_inverse_mapping, item_inverse_mapping

        inner_data_records, user_inverse_mapping, item_inverse_mapping = convert_to_inner_index(user_item_dict, user_mapping, item_mapping)
        if dataset=='Beauty':
            HGN_path = 'Amazon/Beauty'
            SAS_path = './SASRec/data/Beauty'
            CASER_path = './caser/datasets/Amazon_Beauty'
        if dataset=='Game':
            HGN_path = 'Amazon/Game'
            SAS_path = './SASRec/data/Game'
            CASER_path = './caser/datasets/Amazon_Game'
        if dataset=='ML':
            HGN_path = 'ML'
            SAS_path = './SASRec/data/Game'
            CASER_path = './caser/datasets/ml1m'
#HGN
        save_obj(inner_data_records, './data/dataset/{}/K={}_BS={}_item_sequences'.format(HGN_path,K,batch_size))
        save_obj(user_mapping, './data/dataset/{}/K={}_BS={}_user_mapping'.format(HGN_path,K,batch_size))
        save_obj(item_mapping, './data/dataset/{}/K={}_BS={}_item_mapping'.format(HGN_path,K,batch_size))
#SASRec
        #item_mapping = np.load('item_mapping.pkl')
        #user_mapping = np.load('user_mapping.pkl')

        f = open('{}_K={}.txt'.format(SAS_path,K), 'w')
        for user in range(1, len(user_mapping) + 1):
            for item in inner_data_records[user - 1]:
                f.write('%d %d\n' % (user, item))
        f.close()
#Caser
        mapping = []
        user = 0
        for userlist in inner_data_records:
            user += 1
            for item in userlist:
                mapping.append((user, item))
        mlen = len(mapping)
        test_size = 0.25
        indices = list(range(mlen))
        test_indices = np.sort(random.sample(indices, int(mlen * test_size)))
        train_indices = [x for x in indices if x not in test_indices]
        train_mapping = [mapping[idx] for idx in train_indices]
        test_mapping = [mapping[idx] for idx in test_indices]
        ftest = open('{}/test_K={}.txt'.format(CASER_path,K), 'w')
        for (user, item) in test_mapping:
            ftest.write('%d %d 1\n' % (user, item))
        ftest.close()
        ftrain = open('{}/train_K={}.txt'.format(CASER_path,K), 'w')
        for (user, item) in train_mapping:
            ftrain.write('%d %d 1\n' % (user, item))
        ftrain.close()