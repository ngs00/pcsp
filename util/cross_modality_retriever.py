import torch
import json
from torch.utils.data import DataLoader
from util.data import collate_fn


class RetrievalResult:
    def __init__(self, embs, graph_embs, retrieved_ids):
        self.embs = embs
        self.graph_embs = graph_embs
        self.retrieved_ids = retrieved_ids

    def calc_acc(self, gt_ids):
        num_correct = 0

        for i in range(0, len(gt_ids)):
            if gt_ids[i] in self.retrieved_ids[i]:
                num_correct += 1

        return num_correct / len(gt_ids)

    def save(self, path_result_file, gt_ids=None):
        retrieval_result = dict()

        if gt_ids is None:
            for i in range(0, len(self.retrieved_ids)):
                retrieval_result[i] = {
                    'emb': self.embs[i].tolist(),
                    'graph_emb': self.graph_embs[i].tolist(),
                    'retrieved_ids': self.retrieved_ids[i]
                }
        else:
            for i in range(0, len(self.retrieved_ids)):
                retrieval_result[i] = {
                    'emb': self.embs[i].tolist(),
                    'graph_emb': self.graph_embs[i].tolist(),
                    'retrieved_ids': self.retrieved_ids[i],
                    'gt_ids': gt_ids[i]
                }

        with open(path_result_file, 'w', encoding='utf-8') as fp:
            json.dump(retrieval_result, fp)


class GraphRetriever:
    def __init__(self, retriever_model):
        self.retriever_model = retriever_model
        self.retriever_model.eval()

    def retrieve(self, dataset, k):
        data_ids = [d.data_id for d in dataset.data]
        embs, graph_embs = self.retriever_model.encode(DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_fn))
        loader_data_embs = DataLoader(embs, batch_size=8192, shuffle=False)
        loader_graph_embs = DataLoader(graph_embs, batch_size=8192, shuffle=False)

        sim_idx = list()
        for de in loader_data_embs:
            sims = list()
            for ge in loader_graph_embs:
                sims.append(self.retriever_model.calc_sims(de, ge))
            _, sim_idx_batch = torch.topk(torch.cat(sims, dim=1), k, dim=1)
            sim_idx.append(sim_idx_batch)
        sim_idx = torch.cat(sim_idx, dim=0)

        retrieved_ids = list()
        for i in range(0, len(dataset)):
            retrieved_ids.append([data_ids[idx] for idx in sim_idx[i]])

        return RetrievalResult(embs.cpu(), graph_embs[sim_idx].cpu(), retrieved_ids)
