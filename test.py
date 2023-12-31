import json
import numpy as np
import time
import logging
import argparse
from IR_Model import BM25, dense
# BM25, dense
# 输入query，返回由 (分数， id) 组成的列表，（100项，按分数降序）

def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return obj

def get_top30_golden_labels(processed: bool = False, label_path="data/label_top30_dict.json",
                             save_path="data/top30_golden_labels.json", golden_path="data/top30_golden_labels.json"):
    # 得到前30的标注值，按照标注值的降序排列
    # 返回id_to_label以及golden_labels
    if processed:
        with open(label_path, 'r') as f:
            label_dict = json.load(f)
        with open(golden_path, 'r') as g:
            golden_labels = json.load(g)
        return label_dict, golden_labels
    else:
        with open(label_path, 'r') as f:
            label_dict = json.load(f)
            golden_labels = {key: sorted(value.values(), reverse=True)
                                for key, value in label_dict.items()}
        with open(save_path, 'w') as f:
            json.dump(golden_labels, f)
        return label_dict, golden_labels

def compute_nDCG(res, labels, id_to_label):
    # 计算nDCG@k
    # res: 检索结果，id值，list
    # labels: 标注值，按照降序排列，list
    # id_to_label: id到标注值的映射，没有的项表示0，dict

    idcg_values = [labels[i]/np.log2(i+2) for i in range(30)]
    idcg_5 = sum(idcg_values[:5])
    idcg_10 = sum(idcg_values[:10])
    idcg_30 = sum(idcg_values)

    dcg_values = [id_to_label[res[i]] /
                    np.log2(i+2) if res[i] in id_to_label.keys() else 0 for i in range(30)]
    dcg_5 = sum(dcg_values[:5])
    dcg_10 = sum(dcg_values[:10])
    dcg_30 = sum(dcg_values)

    ndcg_5 = dcg_5/idcg_5
    ndcg_10 = dcg_10/idcg_10
    ndcg_30 = dcg_30/idcg_30

    return (ndcg_5, ndcg_10, ndcg_30)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--weight',default=1.0,type=float)
    args=parser.parse_args()

    logger=logging.getLogger("IR_test")
    logger.setLevel(logging.INFO)

    fh=logging.FileHandler("test.log")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info(f"args: {args.__dict__}")

    query_path="data/query.json"

    id_to_label, golden_labels = get_top30_golden_labels()
    querys = load_json(query_path)

    ndcg_5 = []
    ndcg_10 = []
    ndcg_30 = []


    logger.info("BM25: ")
    logger.info("     Querying...")
    bm25_res=[]
    start=time.time()
    for query in querys:
        res=BM25(query)
        bm25_res.append(res)
    end=time.time()
    logger.info(f"     Finish. Time spent: {round(end-start,3)}")

    logger.info("     Computing nDCG...")
    for i in zip(querys, bm25_res):
        ndcg = compute_nDCG(
            [j[1] for j in i[1]], golden_labels[str(i[0]['ridx'])], id_to_label[str(i[0]['ridx'])])
        ndcg_5.append(ndcg[0])
        ndcg_10.append(ndcg[1])
        ndcg_30.append(ndcg[2])
    logger.info("     nDCG@5: "+str(sum(ndcg_5)/len(ndcg_5)))
    logger.info("     nDCG@10: "+str(sum(ndcg_10)/len(ndcg_10)))
    logger.info("     nDCG@30: "+str(sum(ndcg_30)/len(ndcg_30)))

    ndcg_5.clear()
    ndcg_10.clear()
    ndcg_30.clear()

    logger.info("Dense: ")
    logger.info("     Querying...")
    dense_res=[]
    start=time.time()
    for query in querys:
        res=dense(query)
        dense_res.append(res)
    end=time.time()
    logger.info(f"     Finish. Time spent: {round(end-start,3)}")

    logger.info("     Computing nDCG...")
    for i in zip(querys, dense_res):
        ndcg = compute_nDCG(
            [j[1] for j in i[1]], golden_labels[str(i[0]['ridx'])], id_to_label[str(i[0]['ridx'])])
        ndcg_5.append(ndcg[0])
        ndcg_10.append(ndcg[1])
        ndcg_30.append(ndcg[2])
    logger.info("     nDCG@5: "+str(sum(ndcg_5)/len(ndcg_5)))
    logger.info("     nDCG@10: "+str(sum(ndcg_10)/len(ndcg_10)))
    logger.info("     nDCG@30: "+str(sum(ndcg_30)/len(ndcg_30)))

    ndcg_5.clear()
    ndcg_10.clear()
    ndcg_30.clear()

    logger.info("Weighted average: ")
    
    weighted_res=[]
    for res in zip(bm25_res,dense_res):
        dense_id_to_score={entry[1]:entry[0] for entry in res[1]}
        weighted_id_score=[(entry[1],entry[0]+dense_id_to_score[entry[1]]) for entry in res[0]]
        sorted_weight_id=sorted(weighted_id_score,reverse=True,key=lambda x:x[1])
        sorted_id=[entry[0] for entry in sorted_weight_id]
        weighted_res.append(sorted_id)
        
    logger.info("     Computing nDCG...")
    for i in zip(querys, weighted_res):
        ndcg = compute_nDCG(
            i[1], golden_labels[str(i[0]['ridx'])], id_to_label[str(i[0]['ridx'])])
        ndcg_5.append(ndcg[0])
        ndcg_10.append(ndcg[1])
        ndcg_30.append(ndcg[2])
    logger.info("     nDCG@5: "+str(sum(ndcg_5)/len(ndcg_5)))
    logger.info("     nDCG@10: "+str(sum(ndcg_10)/len(ndcg_10)))
    logger.info("     nDCG@30: "+str(sum(ndcg_30)/len(ndcg_30)))