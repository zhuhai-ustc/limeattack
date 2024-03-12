import pickle
import argparse
import os
import numpy as np
import dataloader
import criteria
import torch
import torch.nn as nn
from train_classifier import Model,NLI_infer_BERT
from generate_embedding_mat import generate_embedding_mat,csim_matrix
import datetime
import tqdm
from lime.lime_text import LimeTextExplainer

parser = argparse.ArgumentParser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def split_token(seq):
    return seq.split()

#lime word ranking 
def lime_word_ranking(text,predictor,true_label):
    data = " ".join(text)
    class_names = ['negative','positive']
    explainer = LimeTextExplainer(class_names=class_names,bow=False,split_expression=split_token)
    exp = explainer.explain_instance(data,predictor,num_features=len(text),labels=[true_label],num_samples=len(text))
    ant = exp.as_index(true_label)
    import_scores = []
    word_index_dict = {}
    word_dict= exp.as_list(true_label)
    for i in ant:
        word_index_dict[i[0]]=i[1]
    word_index_dict = sorted(word_index_dict.items(),key=lambda x:x[0])
    for i in word_index_dict:
        import_scores.append(i[-1])
    return import_scores,word_dict

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# semantic function
#import tensorflow as tf
#import tensorflow_hub as hub
#from scipy.spatial.distance import pdist
#USE = hub.load('https://tfhub.dev/google/universal-sentence-encoder')
def semantic_sim(text1, text2):
    return 1
    a = USE([text1,text2])
    simi = 1.0 - pdist([a['outputs'][0], a['outputs'][1]], 'cosine') 
    return simi.item()





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



def issuccess(predictor,batch_size,all_text_ls,text_ls,true_label,num_queries,k=2,query_budget=100):
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
        a,idx = torch.sort(semantic_sims,descending=True)
        k_text_ls = []
        tmp_k = []
        if len(idx)>k:
            for i in idx[:int(k/3)]:
                k_text_ls.append(all_text_ls[i])
            for i in idx[-int(k/3):]:
                k_text_ls.append(all_text_ls[i])
            idx = idx.numpy()[int(k/3):-int(k/3)]
            #b = b[int(k/3):-int(k/3)]
            np.random.shuffle(idx)
            for i in idx[:int(k/3)]:
                k_text_ls.append(all_text_ls[i])
        else:
            for i in idx:
                k_text_ls.append(all_text_ls[i])
        
        return [0,k_text_ls,tmp_k],num_queries

    semantic = []
    bb = success_text_ls[0].tolist()
    num_queries+=all_text_ls.index(bb)
    for i in success_text_ls:
        semantic.append(semantic_sim(" ".join(i)," ".join(text_ls)))
    if num_queries>query_budget:
        return [1,text_ls,num_queries,success_text_ls[semantic.index(max(semantic))]],num_queries
    return [1,text_ls,num_queries,success_text_ls[semantic.index(max(semantic))]],num_queries


def auto_beamsearch(predictor,batch_size,text_ls,true_label,num_queries,k_text_ls,words_perturb_idx,start,synonym_words,k=2,query_budget=100):
    curr = []
    tmp = []
    for i in range(k):
        if i==0:
            tmp = k_text_ls
        else:
            tmp = curr
            curr = []
        for text in tmp:
            #new_tmp = []
            curr.append(text)

            for j in synonym_words[start+i]:
                aa = text.copy()
                aa[words_perturb_idx[start+i]]  = j
                curr.append(aa)
        suc,num_queries = issuccess(predictor,batch_size,curr,text_ls,true_label,num_queries,k,query_budget)
        if suc[0] == 1:
            return suc,num_queries
        curr = suc[1]

    return suc,num_queries

#===================================attack=============================================
def attack(text_id,fail,text_ls,true_label,predictor,lime_pred,stop_word_set,word2idx, idx2word, cos_sim,synonyms_num=50,sim_score_threshold=0, batch_size=32,import_score_threshold=0.005,pos=1,k=2,query_budget=100):
    num_queries = 0
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    if true_label != orig_label:
        fail[text_id] = 0
        return  [2, orig_label, 0]
    else:
        import_scores,lime_word = lime_word_ranking(text_ls,lime_pred,orig_label.item())
        #num_queries+=query
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
            suc,num_queries = auto_beamsearch(predictor,batch_size,text_ls,true_label,num_queries,[text_ls],word_perturb_text_idx,0,synonym_words,k=len(word_perturb_text_idx),query_budget=query_budget)
            if suc[0] == 0:
                fail[text_id] = 1
            return suc
        
        suc,num_queries = auto_beamsearch(predictor,batch_size,text_ls,true_label,num_queries,[text_ls],word_perturb_text_idx,0,synonym_words,k=k,query_budget=query_budget)
        if suc[0] == 1:
            return suc

        k_text_ls = suc[1]
        for start in range(k,len(word_perturb_text_idx),k):
            if  len(word_perturb_text_idx)<start+k:
                suc,num_queries = auto_beamsearch(predictor,batch_size,text_ls,true_label,num_queries,k_text_ls,word_perturb_text_idx,start,synonym_words,k=len(word_perturb_text_idx)-start,query_budget=query_budget)
                if suc[0] == 1:
                    return suc
                fail[text_id] = 2
                return [0]
            suc,num_queries = auto_beamsearch(predictor,batch_size,text_ls,true_label,num_queries,k_text_ls,word_perturb_text_idx,start,synonym_words,k=k,query_budget=query_budget)
            if suc[0] == 1:
                return suc
            k_text_ls = suc[1]
        fail[text_id] = 3
        return [0]

def evaluation(res,query_budget):
    sim = [0]
    change_num = 0
    atk=0
    change_all = 0
    query = [0]
    for i in res:
        if i[0] == 2:
            atk+=1
            #count+=1
        if i[0] == 1:
            sim.append(semantic_sim(" ".join(i[-1])," ".join(i[1])))
            change_num += np.sum(np.array(i[-1]) != np.array(i[1]))
            change_all+=len(i[1])
            if len(i[1])+i[2]<=query_budget and (np.sum(np.array(i[-1]) != np.array(i[1])) / len(i[1]))<=0.1: #query budget and perturb limit
                atk+=1
                query.append(i[2])
    print("asr:{},sim:{},pert:{},query:{}".format(atk,np.mean(sim),change_num/change_all,np.mean(query)))


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
    
    parser.add_argument("--model_path",
                        type=str,
                        default='cnn/cnn_mr',
                        help="Target models for text classification: wordcnn, word Lstm ")   # TODO

    parser.add_argument("--nclasses",
                        type=int,
                        default=2,
                        help="How many classes for classification.")

    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='/mnt/d2/sxy2/TextHacker/data/embedding/glove.6B.200d.txt',
                        help="path to the word embeddings for the target model, CNN and LSTM is need")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        required=False,
                        default="/mnt/d2/sxy2/TextHacker/data/aux_files/counter-fitted-vectors.txt",
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

    parser.add_argument("--query_budget",
                            type=int,
                            default=100,
                            help="query_budget")

    args = parser.parse_args()
    texts, labels = dataloader.read_corpus(args.dataset_path)

    data = list(zip(texts, labels))
    print("Cos sim import finished!")
    embeddings, word2idx_vocab, idx2word_vocab = generate_embedding_mat(args.counter_fitting_embeddings_path)
    cos_sim, word2idx_rev, idx2word_rev = csim_matrix(data, embeddings, word2idx_vocab)
    # Load the saved model usin state dic
    if args.target_model == "wordCNN":
        default_model_path = "/mnt/d2/sxy2/TextHacker/data/model/"
        if 'ag' in args.dataset_path:
            default_model_path = os.path.join(default_model_path,args.model_path)
            args.nclasses = 4
        elif 'yahoo' in args.dataset_path:
            default_model_path = os.path.join(default_model_path,args.model_path)
            args.nclasses = 10
        else:
            default_model_path = os.path.join(default_model_path,args.model_path)
           
        checkpoint = torch.load(default_model_path, map_location='cuda:0')
        try:
            model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=150, cnn=True).to(device)
            model.load_state_dict(checkpoint)
        except:
            model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=100, cnn=True).to(device)
            model.load_state_dict(checkpoint)

    elif args.target_model == "wordLSTM":
        default_model_path = "/model/"
        if 'ag' in args.dataset_path:
            default_model_path = os.path.join(default_model_path,args.model_path)
            args.nclasses = 4
        elif 'yahoo' in args.dataset_path:
            default_model_path = os.path.join(default_model_path,args.model_path)
            args.nclasses = 10
        else:
            default_model_path = os.path.join(default_model_path,args.model_path)
        model = Model(args.word_embeddings_path, nclasses=args.nclasses).cuda()
        checkpoint = torch.load(default_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == "bert":
        #default_model_path = "/model/"
        default_model_path = "/mnt/d2/sxy2/TextHacker/data/model/bert/mr/mr_example/checkpoint-epoch-1"
        if 'ag' in args.dataset_path:
            default_model_path = os.path.join(default_model_path,args.model_path)
            args.nclasses = 4
        elif 'yahoo' in args.dataset_path:
            default_model_path = os.path.join(default_model_path,args.model_path)
            args.nclasses = 10
        else:
            default_model_path = os.path.join(default_model_path,args.model_path)
        max_seq_length = 256
        model = NLI_infer_BERT(default_model_path, nclasses=args.nclasses, max_seq_length=max_seq_length)

    predictor = model.text_pred
    lime_pred = model.lime_text_pred
    print('Start attacking...')
    starttime = datetime.datetime.now()
    print("model:{},data:{}".format(args.target_model,args.dataset_path))
    print("para->sim:{},synonyms_num:{},k:{}".format(args.sim,args.syn_num,args.k))
    right = 0
    for idx,(text,true_label) in enumerate(data):
        if torch.argmax(predictor([text]).squeeze())==true_label:
            right+=1
    print("origin acc:{}".format(right/len(data)))

    orig_texts = []
    attack_result = []
    fail  = {}
    
    
    for idx, (text, true_label) in enumerate(tqdm.tqdm(data)):
        if idx % 100 == 0 and idx != 0:
            evaluation(attack_result,args.query_budget)
        orig_texts.append(text)
        res = attack(idx,fail,text, true_label, predictor, lime_pred,criteria.filter_words, word2idx_rev, idx2word_vocab, cos_sim,import_score_threshold=-100, sim_score_threshold=args.sim, synonyms_num=args.syn_num,batch_size=32,pos = args.pos, k=args.k,query_budget = args.query_budget)
        attack_result.append(res)

    endtime = datetime.datetime.now()
    print("time:{}".format((endtime - starttime).seconds / 60))
    evaluation(attack_result,args.query_budget)
    attack_result.append([(endtime - starttime).seconds / 60])
    attack_result.append(fail)
    datasetname = args.dataset_path.split("/")[-1]
    save_name = '/data/' + args.target_model+"-"+datasetname+"-"+str(args.sim)+"-"+str(args.syn_num)+"-"+str(args.k)+str(args.sim)
    #save_obj(lime_result,save_name)


if __name__ == "__main__":
    run_attack()
