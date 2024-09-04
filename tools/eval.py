import os
import glob

import h5py
import torch
import argparse
import numpy as np
from scipy.stats import stats, rankdata

from gravit.utils.eval_tool import run_evaluation_asd, run_evaluation_al
from torch_geometric.loader import DataLoader
from gravit.utils.parser import get_cfg
from gravit.utils.logger import get_logger
import model as M
from gravit.datasets import GraphDataset
from gravit.utils.formatter import get_formatting_data_dict, get_formatted_preds
# from gravit.utils.eval_tool import get_eval_score
from gravit.utils.vs import avg_splits, knapsack


def get_eval_score(cfg, preds):
    """
    Compute the evaluation score
    """

    # Path to the annotations
    path_annts = os.path.join(cfg['root_data'], 'annotations')

    eval_type = cfg['eval_type']
    str_score = ""
    if eval_type == "VS_max" or eval_type == "VS_avg":

        path_dataset = os.path.join(cfg['root_data'], f'annotations/{cfg["dataset"]}/eccv16_dataset_{cfg["dataset"].lower()}_google_pool5.h5')
        with h5py.File(path_dataset, 'r') as hdf:

            all_f1_scores = []
            all_taus = []
            all_rhos = []
            for video, scores in preds:

                n_samples = hdf.get(video + '/n_steps')[()]
                n_frames = hdf.get(video + '/n_frames')[()]
                gt_segments = np.array(hdf.get(video + '/change_points'))
                gt_samples = np.array(hdf.get(video + '/picks'))
                gt_scores = np.array(hdf.get(video + '/gtscore'))
                user_summaries = np.array(hdf.get(video + '/user_summary'))

                # Take scores from sampled frames to all frames
                gt_samples = np.append(gt_samples, [n_frames - 1]) # To account for last frames within loop
                frame_scores = np.zeros(n_frames, dtype=np.float32)
                for idx in range(n_samples):
                    frame_scores[gt_samples[idx]:gt_samples[idx + 1]] = scores[idx]

                # Calculate segments' avg score and length
                # (Segment_X = video[frame_A:frame_B])
                n_segments = len(gt_segments)
                s_scores = np.empty(n_segments)
                s_lengths = np.empty(n_segments, dtype=np.int32)
                for idx in range(n_segments):
                    s_lengths[idx] = gt_segments[idx][1] - gt_segments[idx][0] + 1
                    s_scores[idx] = (frame_scores[gt_segments[idx][0]:gt_segments[idx][1]].mean())

                # Select for max importance
                final_len = int(n_frames * 0.15) # 15% of total length
                segments = knapsack.fill_knapsack(final_len, s_scores, s_lengths)

                # Mark frames from selected segments
                sum_segs = np.zeros((len(segments), 2), dtype=int)
                pred_summary = np.zeros(n_frames, dtype=np.int8)
                for i, seg in enumerate(segments):
                    pred_summary[gt_segments[seg][0]:gt_segments[seg][1]] = 1
                    sum_segs[i][0] = gt_segments[seg][0]
                    sum_segs[i][1] = gt_segments[seg][1]

                # Calculate F1-Score per user summary
                user_summary = np.zeros(n_frames, dtype=np.int8)
                n_user_sums = user_summaries.shape[0]
                f1_scores = np.empty(n_user_sums)

                for u_sum_idx in range(n_user_sums):
                    user_summary[:n_frames] = user_summaries[u_sum_idx]

                    # F-1
                    tp = pred_summary & user_summary
                    precision = sum(tp)/sum(pred_summary)
                    recall = sum(tp)/sum(user_summary)

                    if (precision + recall) == 0:
                        f1_scores[u_sum_idx] = 0
                    else:
                        f1_scores[u_sum_idx] = (2 * precision * recall * 100 / (precision + recall))

                # Correlation Metrics
                pred_imp_score = np.array(scores)
                ref_imp_scores = gt_scores
                rho_coeff, _ = stats.spearmanr(pred_imp_score, ref_imp_scores)
                tau_coeff, _ = stats.kendalltau(rankdata(-pred_imp_score), rankdata(-ref_imp_scores))

                all_taus.append(tau_coeff)
                all_rhos.append(rho_coeff)

                # Calculate one F1-Score from all user summaries
                if eval_type == "VS_max":
                    # print(f1_scores)
                    f1 = max(f1_scores)
                else:
                    f1 = np.mean(f1_scores)

                all_f1_scores.append(f1)

        f1_score = sum(all_f1_scores) / len(all_f1_scores)
        tau = sum(all_taus) / len(all_taus)
        rho = sum(all_rhos) / len(all_rhos)

        str_score = f"F1-Score = {f1_score}, Tau = {tau}, Rho = {rho}"
    return str_score



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
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = M.SPELL(cfg, cfg['t_emb']).to(device)
    print(model)
    val_loader = DataLoader(GraphDataset(os.path.join(path_graphs, 'val')))
    num_val_graphs = len(val_loader)

    # Init
    #x_dummy = torch.tensor(np.array(np.random.rand(10, 1024), dtype=np.float32), dtype=torch.float32).to(device)
    #node_source_dummy = np.random.randint(10, size=5)
    #node_target_dummy = np.random.randint(10, size=5)
    #edge_index_dummy = torch.tensor(np.array([node_source_dummy, node_target_dummy], dtype=np.int64), dtype=torch.long).to(device)
    #signs = np.sign(node_source_dummy - node_target_dummy)
    #edge_attr_dummy = torch.tensor(signs, dtype=torch.float32).to(device)
    #model(x_dummy, edge_index_dummy, edge_attr_dummy, None)

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

    preds_all = []
    with torch.no_grad():
        for i, data in enumerate(val_loader, 1):
            g = data.g.tolist()
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)

            # t = torch.tensor(range(x.shape[0]))
            # t = torch.stack(
            #     [torch.cos(t / 1.), torch.cos(t / 2), torch.cos(t / 3), torch.cos(t / 5), torch.cos(t / 7),
            #      torch.cos(t / 11)]
            # ).to(x.device).transpose(0, 1)
            # x = torch.concatenate([x, t], dim=1)

            c = None
            if cfg['use_spf']:
                c = data.c.to(device)

            logits = model(x, edge_index, edge_attr, c)

            # print(logits)
            # print(logits.shape)

            # Change the format of the model output
            preds = get_formatted_preds(cfg, logits, g, data_dict)

            # print(preds)

            # print(preds)
            # print(preds.shape)

            preds_all.extend(preds)

            logger.info(f'[{i:04d}|{num_val_graphs:04d}] processed')

    # Compute the evaluation score
    logger.info(f'Computing the evaluation score')
    # print(preds_all)
    eval_score = get_eval_score(cfg, preds_all)

    logger.info(f'{cfg["eval_type"]} evaluation finished: {eval_score}\n')
    return eval_score


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

    all_eval_results = []
    for result in results:
        args.cfg = result
        cfg = get_cfg(args)
        all_eval_results.append(evaluate(cfg))

    if "VS" in args.eval_type and args.all_splits:
        avg_splits.print_results(all_eval_results)
