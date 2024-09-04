import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from gravit.utils.parser import get_args, get_cfg
from gravit.utils.logger import get_logger
from gravit.models import get_loss_func
from videoxum_dataset import VideoXumDataset
import model as M
from tqdm import tqdm


@torch.no_grad()
def get_label(pred, usr_sum, loss_func, num=5, eps=0.5, y=None):
    usr_sum = usr_sum.to(pred.device)
    losses = [loss_func(pred[:, 0], y).item() for y in usr_sum]
    idxs = sorted(range(usr_sum.shape[0]), key=lambda x: losses[x])
    idxs = idxs[:min([len(idxs), num])]

    if y is None:
        y = (1 - eps ) * torch.mean(usr_sum[idxs], dim=0, keepdim=False) + eps * torch.mean(usr_sum, dim=0, keepdim=False)
    else:
        y = (1 - eps ) * torch.mean(usr_sum[idxs], dim=0, keepdim=False) + eps * y[:, 0]
    return y[:, None].detach()



def train(cfg):
    """
    Run the training process given the configuration
    """

    # Input and output paths
    path_graphs = os.path.join(cfg['root_data'], f'graphs/{cfg["graph_name"]}')
    path_result = os.path.join(cfg['root_result'], f'{cfg["exp_name"]}')
    if cfg['split'] is not None:
        path_graphs = os.path.join(path_graphs, f'split{cfg["split"]}')
        path_result = os.path.join(path_result, f'split{cfg["split"]}')
    os.makedirs(path_result, exist_ok=True)

    # Prepare the logger and save the current configuration for future reference
    logger = get_logger(path_result, file_name='train')
    logger.info(cfg['exp_name'])
    logger.info('Saving the configuration file')
    with open(os.path.join(path_result, 'cfg.yaml'), 'w') as f:
        yaml.dump({k: v for k, v in cfg.items() if v is not None}, f, default_flow_style=False, sort_keys=False)

    # Build a model and prepare the data loaders
    logger.info('Preparing a model and data loaders')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = M.SPELL(cfg, cfg['t_emb']).to(device)
    # print(model)
    train_loader = DataLoader(VideoXumDataset('train', 'blip', tau=cfg['tau']), batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(VideoXumDataset('test', 'blip', tau=cfg['tau']))

    # Prepare the experiment
    loss_func = get_loss_func(cfg)
    loss_func_val = get_loss_func(cfg, 'val')
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['sch_param'])

    # Run the training process
    logger.info('Training process started')

    epoch_best = 0

    min_loss_val = float('inf')
    for epoch in range(1, cfg['num_epoch']+1):
        model.train()

        # Train for a single epoch
        loss_sum = 0.
        for data in tqdm(train_loader):
            optimizer.zero_grad()

            x, y, e, e_attr = data
            x = x[0]
            y = y[0]
            e = e[0]
            e_attr = e_attr[0]

            c = None
            if cfg['use_spf']:
                c = data.c.to(device)

            logits = model(x, e, e_attr, c)

            y = get_label(logits, y, loss_func, cfg['k'], cfg['eps'], None)

            loss = loss_func(logits, y)
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()

        # Adjust the learning rate
        scheduler.step()

        loss_train = loss_sum / len(train_loader)

        # Get the validation loss
        loss_val = val(val_loader, cfg['use_spf'], model, device, loss_func_val)

        # Save the best-performing checkpoint
        if loss_val < min_loss_val:
            min_loss_val = loss_val
            epoch_best = epoch
            torch.save(model.state_dict(), os.path.join(path_result, 'ckpt_best.pt'))

        # Log the losses for every epoch
        logger.info(f'Epoch [{epoch:03d}|{cfg["num_epoch"]:03d}] loss_train: {loss_train:.4f}, loss_val: {loss_val:.4f}, best: epoch {epoch_best:03d}')

    logger.info('Training finished')


def val(val_loader, use_spf, model, device, loss_func):
    """
    Run a single validation process
    """

    model.eval()
    loss_sum = 0
    with torch.no_grad():
        for data in tqdm(val_loader):
            x, y, e, e_attr = data
            x = x[0]
            y = y[0]
            e = e[0]
            e_attr = e_attr[0]
            c = None
            if use_spf:
                c = data.c.to(device)

            logits = model(x, e, e_attr, c)
            y = get_label(logits, y, loss_func, cfg['k'], cfg['eps'], None)
            loss = loss_func(logits, y)
            loss_sum += loss.item()

    return loss_sum / len(val_loader)


if __name__ == "__main__":
    args = get_args()
    cfg = get_cfg(args)

    train(cfg)
