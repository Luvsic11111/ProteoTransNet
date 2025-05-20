import torch
import random
import numpy as np
from copy import deepcopy
from model import Net
from eval import infer_result, save_result
from data import prepare_dataloader, partition_data, adjacency
import warnings

warnings.filterwarnings('ignore')


class Args:
    def __init__(self):
        # Default arguments
        self.data_path = "data/"
        self.source_data = "u3.csv"
        self.target_data = "s3.csv"
        self.source_preprocess = "Standard"
        self.target_preprocess = "Standard"
        self.reliability_threshold = 0.95
        self.align_loss_epoch = 1
        self.prototype_momentum = 0.9
        self.early_stop_acc = 0.99
        self.max_iteration = 30
        self.novel_type = False
        self.batch_size = 5
        self.train_epoch = 40
        self.learning_rate = 5e-5
        self.random_seed = 2222
        self.umap_plot = True  # Enable UMAP plotting by default


def main(args):
    (
        source_dataset,
        source_dataloader_train,
        source_dataloader_eval,
        target_dataset,
        target_dataloader_train,
        target_dataloader_eval,
        protein_num,
        type_num,
        label_map,
        source_adata,
        target_adata,
    ) = prepare_dataloader(args)

    source_dataloader_eval_all = deepcopy(source_dataloader_eval)
    target_dataloader_eval_all = deepcopy(target_dataloader_eval)
    if args.novel_type:
        target_adj = adjacency(target_dataset.tensors[0])
    else:
        target_adj = None

    source_label = source_dataset.tensors[1]
    count = torch.unique(source_label, return_counts=True, sorted=True)[1]
    ce_weight = 1.0 / count
    ce_weight = ce_weight / ce_weight.sum() * type_num
    ce_weight = ce_weight.cuda()

    print("======= Training Start =======")

    net = Net(protein_num, type_num, ce_weight, args).cuda()
    preds, prob_feat, prob_logit = net.run(
        source_dataloader_train,
        source_dataloader_eval,
        target_dataloader_train,
        target_dataloader_eval,
        target_adj,
        args,
    )

    for iter in range(args.max_iteration):
        (
            source_dataloader_train,
            source_dataloader_eval,
            target_dataloader_train,
            target_dataloader_eval,
            source_dataset,
            target_dataset,
        ) = partition_data(
            preds,
            prob_feat,
            prob_logit,
            source_dataset,
            target_dataset,
            args,
        )

        # Iteration convergence check
        if target_dataset.__len__() <= args.batch_size:
            break
        print("======= Iteration:", iter, "=======")

        source_label = source_dataset.tensors[1]
        count = torch.unique(source_label, return_counts=True, sorted=True)[1]
        ce_weight = 1.0 / count
        ce_weight = ce_weight / ce_weight.sum() * type_num
        ce_weight = ce_weight.cuda()

        net = Net(protein_num, type_num, ce_weight, args).cuda()
        preds, prob_feat, prob_logit = net.run(
            source_dataloader_train,
            source_dataloader_eval,
            target_dataloader_train,
            target_dataloader_eval,
            target_adj,
            args,
        )
    print("======= Training Done =======")

    features, predictions, reliabilities = infer_result(
        net, source_dataloader_eval_all, target_dataloader_eval_all, args
    )
    save_result(
        features,
        predictions,
        reliabilities,
        label_map,
        type_num,
        source_adata,
        target_adata,
        args,
    )


if __name__ == "__main__":
    # Initialize default arguments
    args = Args()

    # Randomization
    torch.manual_seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Run main function
    main(args)