import os
import glob
import torch
import argparse
import numpy as np
from scipy.stats import stats, rankdata
from torch.utils.data import DataLoader
from gravit.utils.parser import get_cfg
from gravit.utils.logger import get_logger
import model as M
from videoxum_dataset import VideoXumDataset
from gravit.utils.formatter import get_formatting_data_dict
from gravit.utils.vs import avg_splits
from tqdm import tqdm


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
    val_loader = DataLoader(VideoXumDataset('val', 'blip', tau=cfg['tau'], device=device))
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

    f1_max = []
    f1_mean = []
    rho = []
    tau = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader, 1)):
            x, y, e, e_attr = data
            x = x[0]
            y = y[0]
            e = e[0]
            e_attr = e_attr[0]
            c = None
            if cfg['use_spf']:
                c = data.c.to(device)

            logits = model(x, e, e_attr, c)

            # Change the format of the model output
            preds = torch.sigmoid(logits.squeeze().cpu()).numpy()

            # logger.info(f'[{i:04d}|{num_val_graphs:04d}] processed')

            gt_score = torch.mean(y, dim=0).detach().cpu().numpy()

            rho_coeff, _ = stats.spearmanr(preds, gt_score)
            tau_coeff, _ = stats.kendalltau(rankdata(-preds), rankdata(-gt_score))

            pred = np.percentile(preds, 37.6)
            pred = (preds > pred).astype(int)

            y = y.detach().cpu().numpy()

            tp = pred[None, :] * y
            precision = np.sum(tp, 1) / np.sum(pred)
            recall = np.sum(tp, 1) / np.sum(y, 1)

            f1 = (2 * precision * recall * 100 / (precision + recall  + 1e-10))

            f1_mean.append(np.mean(f1))
            f1_max.append(np.max(f1))
            rho.append(rho_coeff)
            tau.append(tau_coeff)


    # Compute the evaluation score
    logger.info(f'Computing the evaluation score')
    # print(preds_all)

    f1_max = np.array(f1_max)
    f1_mean = np.array(f1_mean)
    tau = np.array(tau)
    rho = np.array(rho)

    print(f"f1_max: {np.mean(f1_max)}, f1_mean: {np.mean(f1_mean)}, tau: {np.mean(tau)}, rho: {np.mean(rho)}")


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

    results = []
    if args.all_splits:
        results = glob.glob(os.path.join(path_result, "*", "cfg.yaml"))
    else:
        if not os.path.isdir(path_result):
            raise ValueError(f'Please run the training experiment "{args.exp_name}" first')

        results.append(os.path.join(path_result, 'cfg.yaml'))

    for result in results:
        args.cfg = result
        cfg = get_cfg(args)
        evaluate(cfg)

