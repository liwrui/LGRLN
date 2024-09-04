import os
import glob
import torch
import argparse
import numpy as np
from torch_geometric.loader import DataLoader
from gravit.utils.parser import get_cfg
from gravit.utils.logger import get_logger
import model as M
from gravit.datasets import GraphDataset
from gravit.utils.formatter import get_formatting_data_dict, get_formatted_preds
from gravit.utils.eval_tool import get_eval_score
from gravit.utils.vs import avg_splits
import time


def evaluate(cfg):
    """
    Run the evaluation process given the configuration
    """

    # Input and output paths
    path_graphs = os.path.join(cfg['root_data'], f'graphs/{cfg["graph_name"]}')
    path_result = os.path.join(cfg['root_result'], f'{cfg["exp_name"]}')
    if cfg['split'] is not None:
        path_graphs = os.path.join(path_graphs, f'split{cfg["split"]}')
        path_result = os.path.join(path_result, f'split{cfg["split"]}')

    # Prepare the logger
    logger = get_logger(path_result, file_name='eval')
    logger.info(cfg['exp_name'])
    logger.info(path_result)
    # Build a model and prepare the data loaders
    logger.info('Preparing a model and data loaders')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = M.SPELL(cfg, cfg['t_emb']).to(device)
    print(model)
    val_loader = DataLoader(GraphDataset(os.path.join(path_graphs, 'val')))
    num_val_graphs = len(val_loader)

    # Load the trained model
    logger.info('Loading the trained model')
    state_dict = torch.load(os.path.join(path_result, 'ckpt_best.pt'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # Load the feature files to properly format the evaluation results
    logger.info('Retrieving the formatting dictionary')
    data_dict = get_formatting_data_dict(cfg)

    # Run the evaluation process
    logger.info('Evaluation process started')

    ts = []

    with torch.no_grad():
        for i, data in enumerate(val_loader, 1):
            g = data.g.tolist()
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)

            c = None
            if cfg['use_spf']:
                c = data.c.to(device)

            t = time.time()
            logits = model(x, edge_index, edge_attr, c)
            # Change the format of the model output
            preds = get_formatted_preds(cfg, logits, g, data_dict)

            ts.append(time.time() - t)

            logger.info(f'[{i:04d}|{num_val_graphs:04d}] processed')
    return ts


if __name__ == "__main__":
    """
    Evaluate the trained model from the experiment "exp_name"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data',     type=str,   help='Root directory to the data', default='./data')
    parser.add_argument('--root_result',   type=str,   help='Root directory to output', default='./results')
    parser.add_argument('--dataset',       type=str,   help='Name of the dataset')
    parser.add_argument('--exp_name',      type=str,   help='Name of the experiment', required=True)
    parser.add_argument('--eval_type',     type=str,   help='Type of the evaluation', required=True)
    parser.add_argument('--split',         type=int,   help='Split to evaluate')
    parser.add_argument('--all_splits',    action='store_true',   help='Evaluate all splits')

    args = parser.parse_args()

    path_result = os.path.join(args.root_result, args.exp_name)
    if not os.path.isdir(path_result):
        raise ValueError(f'Please run the training experiment "{args.exp_name}" first')

    results = []
    if args.all_splits:
        results = glob.glob(os.path.join(path_result, "*", "cfg.yaml"))
    else:
        if args.split:
            path_result = os.path.join(path_result, f'split{args.split}')
            if not os.path.isdir(path_result):
                raise ValueError(f'Please run the training experiment "{args.exp_name}" first')

        results.append(os.path.join(path_result, 'cfg.yaml'))

    all_ts = []
    for result in results:
        args.cfg = result
        cfg = get_cfg(args)
        all_ts += evaluate(cfg)

    print(np.mean(np.array(all_ts[1:])))

