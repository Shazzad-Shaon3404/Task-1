import os, numpy as np, pandas as pd
import mne
from sklearn.preprocessing import LabelEncoder
from moabb.paradigms import MotorImagery
from moabb.datasets import BNCI2015_001, BNCI2014_001
try:
    from moabb.datasets import Lee2019_MI
except Exception:
    from moabb.datasets import Lee2019 as Lee2019_MI
if "CFG" not in globals():
    CFG = dict(
        sfreq=250, fmin=4.0, fmax=36.0, tmin=0.0, tmax=3.0,
        notch=None, apply_car=True, zscore=True, zclip=5.0,
        fast_dev_run=True, fast_subjects=3
    )
datasets = ["BNCI2015_001", "BNCI2014_001", "Lee2019_MI"]
def trialwise_zscore(X, clip=5.0):
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True) + 1e-6
    Xn = (X - mu) / sd
    if clip is not None:
        Xn = np.clip(Xn, -clip, clip)
    return Xn.astype(np.float32)
def load_eeg_dataset(name):
    if name == "BNCI2015_001":
        ds = BNCI2015_001(); n_classes = 2
    elif name == "BNCI2014_001":
        ds = BNCI2014_001(); n_classes = 4
    elif name == "Lee2019_MI":
        ds = Lee2019_MI(); n_classes = 2
    else:
        raise ValueError(name)
    paradigm = MotorImagery(
        n_classes=n_classes,
        resample=CFG["sfreq"],
        fmin=CFG["fmin"], fmax=CFG["fmax"],
        tmin=CFG["tmin"], tmax=CFG["tmax"],
    )
    epochs, y_raw, meta = paradigm.get_data(ds, return_epochs=True)
    if isinstance(epochs, list):
        epochs = mne.concatenate_epochs(epochs)
    if CFG.get("notch", None) is not None:
        epochs._data = mne.filter.notch_filter(
            epochs.get_data(), Fs=CFG["sfreq"], freqs=[CFG["notch"]], verbose=False
        )
    if CFG.get("apply_car", True):
        epochs = epochs.copy().set_eeg_reference("average", projection=False, verbose=False)
    X = epochs.get_data().astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(np.array(y_raw)).astype(np.int64)
    class_names = list(le.classes_)
    meta = meta.reset_index(drop=True)
    info = epochs.info
    if CFG.get("fast_dev_run", False):
        keep = sorted(meta["subject"].unique())[:CFG.get("fast_subjects", 3)]
        m = meta["subject"].isin(keep).values
        X, y, meta = X[m], y[m], meta.loc[m].reset_index(drop=True)
    if CFG.get("zscore", True):
        X = trialwise_zscore(X, clip=CFG.get("zclip", 5.0))
    return X, y, meta, info, n_classes, class_names
print("Setup OK. datasets =", datasets)
import os, numpy as np, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

from braindecode.models import EEGNet, ShallowFBCSPNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_model(model_name, n_chans, n_times, n_classes):
    if model_name in ["EEGNetv4", "EEGNet"]:
        model_name = "EEGNet"
    if model_name == "EEGNet":
        return EEGNet(n_chans=n_chans, n_outputs=n_classes, n_times=n_times, final_conv_length="auto")
    elif model_name == "ShallowFBCSPNet":
        return ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_classes, n_times=n_times, final_conv_length="auto")
    else:
        raise ValueError(model_name)
def grad_snapshot(model):
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(float(p.grad.detach().norm(2).cpu().item()))
    norms = sorted(norms, reverse=True)
    return norms[:200] if norms else [0.0]
def run_one_fold(model_name, dn, mode, fold, X, y, meta, n_classes, logdir, outdir, max_epochs):
    groups = make_groups(meta, mode=mode)
    splits = min(CFG["n_splits"], len(np.unique(groups)))
    splitter = GroupKFold(n_splits=splits)

    all_splits = list(splitter.split(X, y, groups=groups))
    tr, te = all_splits[fold]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + fold)
    tr_sub, va_sub = next(sss.split(np.zeros(len(tr)), y[tr]))
    tr2 = tr[tr_sub]
    va  = tr[va_sub]
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(Xt[tr2], yt[tr2]), batch_size=CFG["batch_size"], shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xt[va],  yt[va]),  batch_size=CFG["batch_size"]*2, shuffle=False)
    test_loader  = DataLoader(TensorDataset(Xt[te],  yt[te]),  batch_size=CFG["batch_size"]*2, shuffle=False)
    model = build_model(model_name, n_chans=X.shape[1], n_times=X.shape[2], n_classes=n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    best_val = -1.0
    best_state = None
    bad = 0
    snap_epochs = {"early": 0, "mid": max_epochs//2, "late": max_epochs-1}
    snaps = {}
    def eval_loader(loader):
        model.eval()
        ys, ps, ls = [], [], []
        losses = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                losses.append(loss.item())

                pred = logits.argmax(1).cpu().numpy()
                ys.append(yb.cpu().numpy())
                ps.append(pred)
                ls.append(logits.cpu().numpy())
        ys = np.concatenate(ys); ps = np.concatenate(ps); ls = np.concatenate(ls)
        m = compute_metrics(ys, ps, logits=ls)
        return float(np.mean(losses)) if losses else float("nan"), m, ys, ps, ls
    for ep in range(max_epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        if ep in snap_epochs.values():
            xb, yb = next(iter(train_loader))
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            tag = [k for k, v in snap_epochs.items() if v == ep][0]
            snaps[tag] = grad_snapshot(model)
        val_loss, m_val, *_ = eval_loader(val_loader)
        test_loss, m_te, *_ = eval_loader(test_loader)
        writer.add_scalar("loss/train", train_loss, ep)
        writer.add_scalar("loss/val",   val_loss,   ep)
        writer.add_scalar("loss/test",  test_loss,  ep)
        writer.add_scalar("bal_acc/val",  m_val["bal_acc"], ep)
        writer.add_scalar("bal_acc/test", m_te["bal_acc"],  ep)
        writer.add_scalar("macro_f1/val",  m_val["macro_f1"], ep)
        writer.add_scalar("macro_f1/test", m_te["macro_f1"],  ep)
        writer.add_scalar("mcc/val",   m_val["mcc"], ep)
        writer.add_scalar("mcc/test",  m_te["mcc"], ep)
        writer.add_scalar("kappa/val", m_val["kappa"], ep)
        writer.add_scalar("kappa/test",m_te["kappa"], ep)
        writer.add_scalar("auc/val",   m_val["auc"], ep)
        writer.add_scalar("auc/test",  m_te["auc"], ep)
        writer.flush()
        if m_val["bal_acc"] > best_val + 1e-4:
            best_val = m_val["bal_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= (6 if CFG["fast_dev_run"] else CFG["patience"]):
                break
    writer.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    _, m, ys, ps, ls = eval_loader(test_loader)
    cm = confusion_matrix(ys, ps)
    plt.figure(figsize=(4,4))
    plt.imshow(cm)
    plt.title(f"{dn} | {mode} | {model_name} | fold{fold}")
    plt.xlabel("pred"); plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"cm_{model_name}_fold{fold}.png"), dpi=220)
    plt.close()
    plt.figure(figsize=(7,3))
    for tag in ["early","mid","late"]:
        if tag in snaps:
            plt.plot(snaps[tag][:60], label=tag)
    plt.legend()
    plt.title(f"Grad norms ({dn} | {mode} | {model_name} | fold{fold})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"grad_{model_name}_fold{fold}.png"), dpi=220)
    plt.close()
    return m, model, (ys, ps, ls)
import os, numpy as np, torch, matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.manifold import TSNE
def get_penultimate_features(model, x):
    last_lin = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_lin = m
    feats = {"z": None}
    if last_lin is None:
        return None, model(x)
    def hook(mod, inp, out):
        feats["z"] = inp[0].detach()
    h = last_lin.register_forward_hook(hook)
    logits = model(x)
    h.remove()
    return feats["z"], logits
def plot_tsne(Z, y, title, outpath):
    Z2 = TSNE(n_components=2, perplexity=30, init="pca",
              learning_rate="auto", random_state=SEED).fit_transform(Z)
    plt.figure(figsize=(6,5))
    for c in np.unique(y):
        m = (y == c)
        plt.scatter(Z2[m,0], Z2[m,1], s=8, alpha=0.7, label=str(c))
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()
def tsne_three_stages(dn, mode, model_name, n_samples=800):
    model, info, X, y = saved_models[(dn, mode, model_name)]
    model = model.to(DEVICE).eval()
    idx = np.random.RandomState(SEED).choice(np.arange(len(y)),
                                            size=min(n_samples, len(y)),
                                            replace=False)
    Xs = torch.tensor(X[idx], dtype=torch.float32).to(DEVICE)
    ys = y[idx]
    raw = X[idx][:, :, ::4].reshape(len(idx), -1)
    with torch.no_grad():
        hid, logits = get_penultimate_features(model, Xs)
        hidden = hid.detach().cpu().numpy() if hid is not None else logits.detach().cpu().numpy()
        out = logits.detach().cpu().numpy()
    outdir = os.path.join(CFG["out_root"], dn, model_name, mode, "tsne")
    os.makedirs(outdir, exist_ok=True)
    plot_tsne(raw, ys,    f"{dn}|{mode}|raw",    os.path.join(outdir, "tsne_raw.png"))
    plot_tsne(hidden, ys, f"{dn}|{mode}|hidden", os.path.join(outdir, "tsne_hidden.png"))
    plot_tsne(out, ys,    f"{dn}|{mode}|logits", os.path.join(outdir, "tsne_logits.png"))
    print("Saved t-SNE plots in:", outdir)
