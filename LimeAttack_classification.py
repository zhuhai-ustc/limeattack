#from nltk.corpus import stopwords
import pickle
import copy
from tqdm import *
import math
import argparse
import os
import numpy as np
import dataloader
from train_classifier import Model
import criteria
import random
import tensorflow as tf
import tensorflow_hub as hub
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from train_classifier import Model
from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig
from generate_embedding_mat import generate_embedding_mat
import generateCosSimMat
from scipy.spatial.distance import pdist
import datetime
import tqdm
import argparse
parser = argparse.ArgumentParser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from lime.lime_text import LimeTextExplainer
def my_token(a):
    return a.split()
filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves',',','.','?',"!","n't"]
filter_words = list(set(filter_words))
stopwords = filter_words
def lime_word_ranking(text,predictor,true_label):
    data = " ".join(text)
    class_names = ['negative','positive']
    explainer = LimeTextExplainer(class_names=class_names,bow=False,split_expression=my_token)
    exp = explainer.explain_instance(data,predictor,num_features=len(text),labels=[true_label],num_samples=len(text))
    ant = exp.as_index(true_label)
    aa = []
    word_index_dict = {}
    word_dict= exp.as_list(true_label)
    for i in ant:
        word_index_dict[i[0]]=i[1]
    word_index_dict = sorted(word_index_dict.items(),key=lambda x:x[0])
    for i in word_index_dict:
        aa.append(i[-1])
    query = 0
    return aa,word_dict,query

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


USE = hub.load('https://tfhub.dev/google/universal-sentence-encoder')
def semantic_sim(text1, text2):
    #return 1
    a = USE([text1,text2])
    simi = 1.0 - pdist([a['outputs'][0], a['outputs'][1]], 'cosine') 
    return simi.item()


# loading model
class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        print(pretrained_dir)
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses)
        self.model = self.model.to(device)

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)

    def lime_text_pred(self, text_data, batch_size=32):
        aa = []
        for i in text_data:
            aa.append(i.split(" "))
        text_data = aa
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)
        hard_label = (torch.cat(probs_all,dim=0)>0.5)+0
        aa = torch.cat(probs_all,dim=0)
        hard_label = torch.zeros(aa.shape).numpy()
        maxvalue=torch.argmax(aa,dim=1).cpu().numpy()
        for i in range(len(hard_label)):
            hard_label[i][maxvalue[i]]=1
        return hard_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, text_a) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features

    def transform_text(self, data, batch_size=32):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(data,
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader


def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


def csim_matrix(lst_revs ,embeddings, word2idx_vocab):
    ''' Create a cosine similarity matrix only for words in reviews

    Identify all the words in the reviews that are not stop words
    Use embedding matrix to create a cosine similarity submatrix just for these words
    Columns of this cosine similarity matrix correspond to the original words in the embedding
    rows of the matrix correspond to the words in the reviews
    '''
    reviews = [d[0] for d in lst_revs]
    all_words = set()

    for r in reviews:
        all_words = all_words.union(set([str(word) for word in r if str(word) in word2idx_vocab and str(word) not in stopwords]))

    word2idx_rev={}
    idx2word_rev={}
    p=0
    embeddings_rev_words=[]
    for word in all_words:
        word2idx_rev[str(word)] = p
        idx2word_rev[p]=str(word)
        p+=1
        embeddings_rev_words.append(embeddings[word2idx_vocab[str(word)]])

    embeddings_rev_words=np.array(embeddings_rev_words)
    cos_sim = np.dot(embeddings_rev_words, embeddings.T)

    return cos_sim, word2idx_rev,idx2word_rev

def get_synonym_list(ori_text, stop_words_set, cos_sim, idx2word_vocab, word2idx_rev, synonym_num=50, synonym_sim=0.5):
    text_len = len(ori_text)
    synonym_list = [None]*text_len
    pos_ls = criteria.get_pos(ori_text)
    for wd_i in range(len(ori_text)):
        _text = ori_text.copy()
        wd = ori_text[wd_i]
        if wd in stop_words_set or wd not in word2idx_rev:
            synonym_list[wd_i] = []
        else:
            synonym_words, _ = pick_most_similar_words_batch([word2idx_rev[wd]], cos_sim, idx2word_vocab, synonym_num, synonym_sim)
            new_texts = [_text[:wd_i]+[syn]+_text[1+wd_i:] for syn in synonym_words[0]]
            synonyms_pos_ls = [criteria.get_pos(tmp_text[max(wd_i - 4, 0):wd_i + 5])[min(4, wd_i)] \
                                                   if text_len > 10 else criteria.get_pos(tmp_text)[wd_i] for tmp_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[wd_i], synonyms_pos_ls))
            yes_syn = [synonym_words[0][i] for i in range(len(synonym_words[0])) if pos_mask[i]]
            synonym_list[wd_i] = yes_syn
    synonym_len = [len(i) for i in synonym_list]
    w_select_probs = np.array(synonym_len)/sum(synonym_len)
    return synonym_list, w_select_probs


def issuccess(predictor,batch_size,all_text_ls,text_ls,true_label,num_queries,k=2):
    tmp_prob = []
    success_text_ls = []
    _probs = predictor(all_text_ls, batch_size=batch_size)
    _probs_argmax = torch.argmax(_probs, dim=-1).cpu().numpy()
    np_true_label = np.array([true_label]*len(all_text_ls))
    success_text_ls = list(np.array(all_text_ls)[np_true_label!=_probs_argmax])
    
    if len(success_text_ls) == 0:
        num_queries += len(all_text_ls)
        semantic_sims = []
        for simi_text in all_text_ls:
            semantic_sims.append(semantic_sim(" ".join(text_ls)," ".join(simi_text)))
        semantic_sims = torch.tensor(semantic_sims)
        a,idx1 = torch.sort(semantic_sims,descending=True)
        k_text_ls = []
        tmp_k = []
        if len(idx1)>k:
            for i in idx1[:int(k/3)]:
                k_text_ls.append(all_text_ls[i])
            for i in idx1[-int(k/3):]:
                k_text_ls.append(all_text_ls[i])
            b = idx1.numpy()
            b = b[int(k/3):-int(k/3)]
            np.random.shuffle(b)
            for i in b[:int(k/3)]:
                k_text_ls.append(all_text_ls[i])
        else:
            for i in idx1:
                k_text_ls.append(all_text_ls[i])
        
        return [0,k_text_ls,tmp_k],num_queries

    semantic = []
    bb = success_text_ls[0].tolist()
    num_queries+=all_text_ls.index(bb)
    for i in success_text_ls:
        semantic.append(semantic_sim(" ".join(i)," ".join(text_ls)))
    return [1,text_ls,num_queries,success_text_ls[semantic.index(max(semantic))]],num_queries


def auto_beamsearch(predictor,batch_size,text_ls,true_label,num_queries,k_text_ls,words_perturb_idx,start,synonym_words,k=2):
    new_tmp = []
    tmp = []
    for i in range(k):
        if i==0:
            tmp = k_text_ls
        else:
            tmp = new_tmp
            new_tmp = []
        for text in tmp:
            #new_tmp = []
            new_tmp.append(text)

            for j in synonym_words[start+i]:
                aa = text.copy()
                aa[words_perturb_idx[start+i]]  = j
                new_tmp.append(aa)
        suc,num_queries = issuccess(predictor,batch_size,new_tmp,text_ls,true_label,num_queries,k)
        if suc[0] == 1:
            return suc,num_queries
        new_tmp = suc[1]

    return suc,num_queries
#===================================attack=============================================
def attack(text_id,fail,text_ls,true_label,predictor,lime_pred,stop_word_set,word2idx, idx2word, cos_sim,
synonyms_num=50,sim_score_threshold=0, batch_size=32,import_score_threshold=0.005,pos=1,k=2):
    less = 0
    len_text = len(text_ls)
    num_queries = 0
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        fail[text_id] = 0
        return  [2, orig_label, orig_label, 0]
    else:
        import_scores,lime_word,query = lime_word_ranking(text_ls,lime_pred,orig_label.item())
        num_queries+=query
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            if score > import_score_threshold and text_ls[idx] not in stop_word_set:
                words_perturb.append((idx, text_ls[idx]))
        word_perturb_text_idx = [idx for idx, word in words_perturb]
        word_list = [word for idx, word in words_perturb]
        synonym_words = []

        for i in range(len(word_list)):
            if word_list[i] in word2idx:
                t , _ = pick_most_similar_words_batch( [word2idx[word_list[i]]], cos_sim, idx2word, synonyms_num,sim_score_threshold)
                synonym_words.append(t[0])
            else:
                synonym_words.append([])
        tmp_syn = []
        rank_synonyms_word = []
        new_word_1 =[]
        new_word_2 = []
        for i in range(len(word_perturb_text_idx)):
            if synonym_words[i]!=[]:
                new_word_2.append(word_perturb_text_idx[i])
                new_word_1.append(word_list[i])
                tmp_syn.append(synonym_words[i])
        word_perturb_text_idx = new_word_2
        word_list = new_word_1
        synonym_words = tmp_syn
        if len(word_perturb_text_idx)==0 or len(synonym_words) == 0:
            fail[text_id] = 4
            return [0]
        if len(word_perturb_text_idx)<k:
            suc,num_queries = auto_beamsearch(predictor,batch_size,text_ls,true_label,num_queries,[text_ls],word_perturb_text_idx,0,synonym_words,k=len(word_perturb_text_idx))
            if suc[0] == 0:
                fail[text_id] = 1
            return suc
        suc,num_queries = auto_beamsearch(predictor,batch_size,text_ls,true_label,num_queries,[text_ls],word_perturb_text_idx,0,synonym_words,k=k)
        if suc[0] == 1:
            return suc
        k_text_ls = suc[1]
        for start in range(k,len(word_perturb_text_idx),k):
            if  len(word_perturb_text_idx)<start+k:
                suc,num_queries = auto_beamsearch(predictor,batch_size,text_ls,true_label,num_queries,k_text_ls,word_perturb_text_idx,start,synonym_words,k=len(word_perturb_text_idx)-start)
                if suc[0] == 1:
                    return suc
                fail[text_id] = 2
                return [0]
            suc,num_queries = auto_beamsearch(predictor,batch_size,text_ls,true_label,num_queries,k_text_ls,word_perturb_text_idx,start,synonym_words,k=k)
            if suc[0] == 1:
                return suc
            k_text_ls = suc[1]
        fail[text_id] = 3
        return [0]

def test_beam(a):
    sim = [0]
    change_rates = [0]
    count = 0
    change_num = 0
    atk=0
    change_all = 0
    query = [0]
    for i in a:
        if i[0]!=0:
            count+=1
        if i[0] ==1:
            sim.append(semantic_sim(" ".join(i[-1])," ".join(i[1])))
            change_num += np.sum(np.array(i[-1]) != np.array(i[1]))
            change_all+=len(i[1])
            change_rates.append(np.sum(np.array(i[-1]) != np.array(i[1])) / len(i[1]))
            query.append(i[2])
            if len(i[1])+i[2]<=100 and (np.sum(np.array(i[-1]) != np.array(i[1])) / len(i[1]))<=0.1:
                atk+=1
    print("acc:{},sim:{},change_rate:{},new_change_rate:{},query:{},atk:{}".format(count/len(a),np.mean(sim),np.mean(change_rates),change_num/change_all,np.mean(query),atk))


def run_attack():
    ## Required parameters
    parser.add_argument("--dataset_path",
                        type=str,
                        default='data/mr',
                        # required=True,
                        help="Which dataset to attack.")  # TODO

    parser.add_argument("--target_model",
                        type=str,
                        default='bert',
                        # required=True,
                        choices=['wordLSTM', 'bert', 'wordCNN','roberta'],
                        help="Target models for text classification: wordcnn, word Lstm ")   # TODO

    parser.add_argument("--nclasses",
                        type=int,
                        default=2,
                        help="How many classes for classification.")

    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='/data/word_embeddings_path/glove.6B.200d.txt',
                        help="path to the word embeddings for the target model, CNN and LSTM is need")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        required=False,
                        default="/data/counter/counter-fitted-vectors.txt",
                        help="path to the counter-fitting embeddings used to find synonyms")

    parser.add_argument("--sim",
                        type=float,
                        default=0,
                        help="sim threshold")


    parser.add_argument("--syn_num",
                        type=int,
                        default=50,
                        help="syn num")

    parser.add_argument("--k",
                        type=int,
                        default=10,
                        help="beam size")

    parser.add_argument("--pos",
                            type=int,
                            default=0,
                            help="pos check")

    args = parser.parse_args()
    texts, labels = dataloader.read_corpus(args.dataset_path)

    data = list(zip(texts, labels))
    print("Cos sim import finished!")
    embeddings, word2idx_vocab, idx2word_vocab = generate_embedding_mat(args.counter_fitting_embeddings_path)
    cos_sim, word2idx_rev, idx2word_rev = csim_matrix(data, embeddings, word2idx_vocab)
    # Load the saved model usin state dic
    if args.target_model == "wordCNN":
        default_model_path = "/model/"
        if 'imdb' in args.dataset_path:
            default_model_path += 'imdb'
        elif 'mr' in args.dataset_path:
            default_model_path += 'cnn_mr/mr'
        elif 'yelp' in args.dataset_path:
            default_model_path += 'cnn_yelp/yelp'
        elif 'ag' in args.dataset_path:
            default_model_path += 'cnn_ag/cnn-ag'
            args.nclasses = 4
        elif 'sst' in args.dataset_path:
            default_model_path += 'cnn_sst/sst'
            args.nclasses = 2
        elif 'yahoo' in args.dataset_path:
            default_model_path += 'cnn_yahoo/cnn'
            args.nclasses = 10
        checkpoint = torch.load(default_model_path, map_location='cuda:0')
        try:
            model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=150, cnn=True).to(device)
            model.load_state_dict(checkpoint)
        except:
            model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=100, cnn=True).to(device)
            model.load_state_dict(checkpoint)
    elif args.target_model == "wordLSTM":
        default_model_path = "/model/"
        if 'imdb' in args.dataset_path:
            default_model_path += 'imdb'
        elif 'mr' in args.dataset_path:
            default_model_path += 'lstm_mr/mr'
        elif 'yelp' in args.dataset_path:
            default_model_path += 'yelp'
        elif 'ag' in args.dataset_path:
            default_model_path += 'lstm_ag/ag'
            args.nclasses = 4
        elif 'sst' in args.dataset_path:
            default_model_path += 'lstm_sst/sst'
            args.nclasses = 2
        elif 'yahoo' in args.dataset_path:
            default_model_path += 'lstm_yahoo/lstm'
            args.nclasses = 10
        model = Model(args.word_embeddings_path, nclasses=args.nclasses).cuda()
        checkpoint = torch.load(default_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == "bert":
        default_model_path = "/model/"
        if 'imdb' in args.dataset_path:
            default_model_path += 'imdb'
            max_seq_length = 256
        elif 'mr' in args.dataset_path:
            default_model_path += 'bert_mr/mr'
            max_seq_length = 128
        elif 'yelp' in args.dataset_path:
            default_model_path += 'yelp'
            max_seq_length = 256
        elif 'ag' in args.dataset_path:
            default_model_path += 'bert_ag/ag'
            args.nclasses = 4
            max_seq_length = 256
        elif 'sst' in args.dataset_path:
            default_model_path += 'bert_sst/bert_sst'
            args.nclasses = 2
            max_seq_length = 256
        elif 'yahoo' in args.dataset_path:
            default_model_path += 'bert_yahoo/yahoo'
            args.nclasses = 10
            max_seq_length = 256
        model = NLI_infer_BERT(default_model_path, nclasses=args.nclasses, max_seq_length=max_seq_length)

    predictor = model.text_pred
    lime_pred = model.lime_text_pred
    print('Start attacking...')
    starttime = datetime.datetime.now()
    orig_texts = []
    lime_result = list() 
    ours_result = list()
    fail  = {}
    print("model:{},data:{}".format(args.target_model,args.dataset_path))
    print("para->sim:{},synonyms_num:{},k:{}".format(args.sim,args.syn_num,args.k))
    jj = 0
    for idx ,(text,true_label) in enumerate(data):
        if torch.argmax(predictor([text]).squeeze())==true_label:
            jj+=1
    print("origin acc:{}".format(jj/len(data)))
    for idx, (text, true_label) in enumerate(tqdm.tqdm(data)):
        if idx % 100 == 0 and idx != 0:
            test_beam(lime_result)
            datasetname = args.dataset_path.split("/")[-1]
            save_name = '/data/' + args.target_model+"-"+datasetname+"-"+str(args.sim)+"-"+str(args.syn_num)+"-"+str(args.k)+str(args.sim)
            save_obj(lime_result,save_name)
        orig_texts.append(text)
        re = attack(idx,fail,text, true_label, predictor, lime_pred,filter_words, word2idx_rev, idx2word_vocab, cos_sim,
                    import_score_threshold=-100, sim_score_threshold=args.sim, synonyms_num=args.syn_num,batch_size=32,pos = args.pos, k=args.k)
        lime_result.append(re)
    endtime = datetime.datetime.now()
    print("time:{}".format((endtime - starttime).seconds / 60))
    test_beam(lime_result)
    lime_result.append([(endtime - starttime).seconds / 60])
    lime_result.append(fail)
    datasetname = args.dataset_path.split("/")[-1]
    save_name = '/data/' + args.target_model+"-"+datasetname+"-"+str(args.sim)+"-"+str(args.syn_num)+"-"+str(args.k)+str(args.sim)
    save_obj(lime_result,save_name)


if __name__ == "__main__":
    run_attack()