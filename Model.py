def grad_snapshot(model):
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(float(p.grad.detach().norm(2).cpu().item()))
    norms = sorted(norms, reverse=True)
    return norms[:200] if norms else [0.0]
def run_one_fold(model_name, dn, mode, fold, X, y, meta, n_classes, logdir, outdir, max_epochs):
    groups = make_groups(meta, mode=mode)
    splitter = GroupKFold(n_splits=min(CFG["n_splits"], len(np.unique(groups))))
    all_splits = list(splitter.split(X, y, groups=groups))
    tr, te = all_splits[fold]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + fold)
    tr_sub, va_sub = next(sss.split(np.zeros(len(tr)), y[tr]))
    tr2 = tr[tr_sub]
    va = tr[va_sub]
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(Xt[tr2], yt[tr2]), batch_size=CFG["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(Xt[va], yt[va]), batch_size=CFG["batch_size"] * 2, shuffle=False)
    test_loader = DataLoader(TensorDataset(Xt[te], yt[te]), batch_size=CFG["batch_size"] * 2, shuffle=False)
    model = build_model(model_name, n_chans=X.shape[1], n_times=X.shape[2], n_classes=n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    writer = SummaryWriter(logdir)
    best_val = -1.0
    best_state = None
    bad = 0
    snap_epochs = {"early": 0, "mid": max_epochs // 2, "late": max_epochs - 1}
    snaps = {}
    def eval_loader(loader):
        model.eval()
        ys, ps, ls = [], [], []
        losses = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE);
                yb = yb.to(DEVICE)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                losses.append(loss.item())
                pred = logits.argmax(1).cpu().numpy()
                ys.append(yb.cpu().numpy())
                ps.append(pred)
                ls.append(logits.cpu().numpy())
        ys = np.concatenate(ys);
        ps = np.concatenate(ps);
        ls = np.concatenate(ls)
        m = compute_metrics(ys, ps, logits=ls)
        return float(np.mean(losses)) if losses else float("nan"), m, ys, ps, ls
    for ep in range(max_epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE);
            yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        if ep in snap_epochs.values():
            xb, yb = next(iter(train_loader))
            xb = xb.to(DEVICE);
            yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            tag = [k for k, v in snap_epochs.items() if v == ep][0]
            snaps[tag] = grad_snapshot(model)
        val_loss, m_val, *_ = eval_loader(val_loader)
        test_loss, m_te, *_ = eval_loader(test_loader)
        writer.add_scalar("loss/train", train_loss, ep)
        writer.add_scalar("loss/val", val_loss, ep)
        writer.add_scalar("loss/test", test_loss, ep)
        writer.add_scalar("bal_acc/val", m_val["bal_acc"], ep)
        writer.add_scalar("bal_acc/test", m_te["bal_acc"], ep)
        writer.add_scalar("macro_f1/val", m_val["macro_f1"], ep)
        writer.add_scalar("macro_f1/test", m_te["macro_f1"], ep)
        writer.add_scalar("mcc/val", m_val["mcc"], ep)
        writer.add_scalar("mcc/test", m_te["mcc"], ep)
        writer.add_scalar("kappa/val", m_val["kappa"], ep)
        writer.add_scalar("kappa/test", m_te["kappa"], ep)
        writer.add_scalar("auc/val", m_val["auc"], ep)
        writer.add_scalar("auc/test", m_te["auc"], ep)
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
    test_loss, m, ys, ps, ls = eval_loader(test_loader)
    os.makedirs(outdir, exist_ok=True)
    cm = confusion_matrix(ys, ps)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm)
    plt.title(f"{dn} | {mode} | {model_name} | fold{fold}")
    plt.xlabel("pred");
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"cm_{model_name}_fold{fold}.png"), dpi=220)
    plt.close()
    plt.figure(figsize=(7, 3))
    for tag in ["early", "mid", "late"]:
        if tag in snaps:
            plt.plot(snaps[tag][:60], label=tag)
    plt.legend()
    plt.title(f"Grad norms ({dn} | {mode} | {model_name} | fold{fold})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"grad_{model_name}_fold{fold}.png"), dpi=220)
    plt.close()
    return m, model, (ys, ps, ls)
def run_experiment():
    results = []
    saved_models = {}
    for dn in datasets:
        X, y, meta, info, n_classes, classes = load_eeg_dataset(dn)
        for mode in ["inter-subject", "inter-session"]:
            groups = make_groups(meta, mode=mode)
            splits = min(CFG["n_splits"], len(np.unique(groups)))
            if splits < 2:
                print("[SKIP]", dn, mode, "not enough groups")
                continue
            for model_name in ["EEGNet", "ShallowFBCSPNet"]:
                fold_list = [0] if CFG["fast_dev_run"] else list(range(splits))
                fold_metrics = []
                for fold in fold_list:
                    logdir = os.path.join(CFG["runs_root"], dn, model_name, mode, f"fold{fold}")
                    outdir = os.path.join(CFG["out_root"], dn, model_name, mode)
                    m, model, _ = run_one_fold(
                        model_name, dn, mode, fold, X, y, meta, n_classes,
                        logdir=logdir, outdir=outdir,
                        max_epochs=(8 if CFG["fast_dev_run"] else CFG["max_epochs"])
                    )
                    fold_metrics.append(m)
                    saved_models[(dn, mode, model_name)] = (model, info, X, y)

                def agg(key):
                    vals = [mm[key] for mm in fold_metrics]
                    return float(np.nanmean(vals)), float(np.nanstd(vals))

                row = {"dataset": dn, "setting": mode, "model": model_name, "n_folds": len(fold_metrics)}
                for k in ["bal_acc", "acc", "macro_f1", "mcc", "kappa", "auc"]:
                    mu, sd = agg(k)
                    row[f"{k}_mean"] = mu
                    row[f"{k}_std"] = sd
                results.append(row)
    df = pd.DataFrame(results)
    csv_path = os.path.join(CFG["out_root"], "table5_like_results.csv")
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)
    return df, saved_models
df_results, saved_models = run_experiment()
df_results
def make_groups(meta, mode="inter-subject"):
    if mode == "inter-subject":
        g = meta["subject"].astype(str).to_numpy()
    elif mode == "inter-session":
        if "session" in meta.columns:
            g = (meta["subject"].astype(str) + "_" + meta["session"].astype(str)).to_numpy()
        elif "run" in meta.columns:
            g = (meta["subject"].astype(str) + "_" + meta["run"].astype(str)).to_numpy()
        else:
            g = meta["subject"].astype(str).to_numpy()
    else:
        raise ValueError(mode)
    gid, _ = pd.factorize(g)
    return gid
def compute_metrics(y_true, y_pred, logits=None):
    out = {}
    out["bal_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    out["acc"] = float(accuracy_score(y_true, y_pred))
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    out["kappa"] = float(cohen_kappa_score(y_true, y_pred))
    if logits is not None and logits.shape[1] == 2:
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
        try:
            out["auc"] = float(roc_auc_score(y_true, probs))
        except Exception:
            out["auc"] = float("nan")
    else:
        out["auc"] = float("nan")
    return out
def build_model(model_name, n_chans, n_times, n_classes):
    if model_name == "EEGNetv4":
        return EEGNetv4(n_chans=n_chans, n_outputs=n_classes, input_window_samples=n_times, final_conv_length="auto")
    elif model_name == "ShallowFBCSPNet":
        return ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_classes, input_window_samples=n_times, final_conv_length="auto")
    else:
        raise ValueError(model_name)
def get_penultimate_features(model, x):
    last_lin = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_lin = m
    feats = {"z": None}
    if last_lin is None:
        logits = model(x)
        return None, logits
    def hook(mod, inp, out):
        feats["z"] = inp[0].detach()
    h = last_lin.register_forward_hook(hook)
    logits = model(x)
    h.remove()
    return feats["z"], logits
from braindecode.models import EEGNet, ShallowFBCSPNet

def build_model(model_name, n_chans, n_times, n_classes):
    if model_name == "EEGNet":
        return EEGNet(
            n_chans=n_chans,
            n_outputs=n_classes,
            n_times=n_times,
            final_conv_length="auto",
        )
    elif model_name == "ShallowFBCSPNet":
        return ShallowFBCSPNet(
            n_chans=n_chans,
            n_outputs=n_classes,
            n_times=n_times,
            final_conv_length="auto",
        )
    else:
        raise ValueError(model_name)
CACHE = {}
for dn in datasets:
    X, y, meta, info, n_classes, class_names = load_eeg_dataset(dn)
    CACHE[dn] = dict(X=X, y=y, meta=meta, info=info,
                     n_classes=n_classes, class_names=class_names)
    print("cached:", dn, X.shape, "classes:", class_names)
    SEED = 42
    df_results, saved_models = run_experiment_cached()
    df_results