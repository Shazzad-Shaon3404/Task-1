def make_domain_ids(meta, mode="subject"):
    if mode == "subject":
        dom = meta["subject"].to_numpy()
    else:
        if "session" in meta.columns:
            dom = (meta["subject"].astype(str) + "_" + meta["session"].astype(str)).to_numpy()
        else:
            dom = (meta["subject"].astype(str) + "_" + meta.get("run", 0).astype(str)).to_numpy()
    # map to 0..K-1
    dom_ids, _ = pd.factorize(dom)
    return dom_ids.astype(np.int64), int(dom_ids.max()+1)

def train_eval_table5_like(ds_name, ds_obj, epochs=20, n_splits=5):
    X, y, meta, info, classes = load_mi(ds_obj)
    n_chans = X.shape[1]
    n_classes = len(np.unique(y))
    results = {}
    for mode in ["inter-session", "inter-subject"]:
        dom_ids, num_domains = make_domain_ids(meta, mode=("session" if mode=="inter-session" else "subject"))
        num_domains = min(num_domains, 64)
        gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(dom_ids))))
        bals, f1s = [], []
        for fold, (tr, te) in enumerate(gkf.split(X, y, groups=dom_ids), start=1):
            tr = np.array(tr); np.random.RandomState(SEED+fold).shuffle(tr)
            va_size = max(1, int(0.2*len(tr)))
            va = tr[:va_size]; tr2 = tr[va_size:]
            Xt = torch.tensor(X, dtype=torch.float32)
            yt = torch.tensor(y, dtype=torch.long)
            dt = torch.tensor(dom_ids, dtype=torch.long)
            train_ds = TensorDataset(Xt[tr2].unsqueeze(1), yt[tr2], dt[tr2])
            val_ds   = TensorDataset(Xt[va].unsqueeze(1),  yt[va],  dt[va])
            test_ds  = TensorDataset(Xt[te].unsqueeze(1),  yt[te],  dt[te])
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False)
            test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False)
            model = TSMNetLite(n_chans=n_chans, n_classes=n_classes, n_filters=16,
                               bimap_dim=8, num_domains=num_domains).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            logdir = f"/content/runs_tsm_lite/{ds_name}/{mode}/fold{fold}"
            writer = SummaryWriter(logdir)
            snap_epochs = {"early":0, "mid":epochs//2, "late":epochs-1}
            snaps = {}
            best_val = -1
            best_state = None
            for ep in range(epochs):
                model._orthonormalize_W()
                tr_loss, tr_bal, tr_f1, _, _ = run_epoch(model, train_loader, opt)
                va_loss, va_bal, va_f1, _, _ = run_epoch(model, val_loader, None)
                te_loss, te_bal, te_f1, _, _ = run_epoch(model, test_loader, None)
                writer.add_scalar("loss/train", tr_loss, ep)
                writer.add_scalar("loss/val", va_loss, ep)
                writer.add_scalar("loss/test", te_loss, ep)
                writer.add_scalar("bal_acc/train", tr_bal, ep)
                writer.add_scalar("bal_acc/val", va_bal, ep)
                writer.add_scalar("bal_acc/test", te_bal, ep)
                writer.add_scalar("f1/train", tr_f1, ep)
                writer.add_scalar("f1/val", va_f1, ep)
                writer.add_scalar("f1/test", te_f1, ep)
                if ep in snap_epochs.values():
                    xb, yb, db = next(iter(train_loader))
                    xb=xb.to(DEVICE); yb=yb.to(DEVICE); db=db.to(DEVICE)
                    opt.zero_grad()
                    logits,_ = model(xb, db)
                    loss = F.cross_entropy(logits, yb)
                    loss.backward()
                    snaps[[k for k,v in snap_epochs.items() if v==ep][0]] = grad_snapshot(model)
                if va_bal > best_val:
                    best_val = va_bal
                    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            writer.close()
            if best_state is not None:
                model.load_state_dict(best_state)
            te_loss, te_bal, te_f1, ytrue, ypred = run_epoch(model, test_loader, None)
            bals.append(te_bal); f1s.append(te_f1)
            ds_out = os.path.join(OUTDIR, ds_name)
            os.makedirs(ds_out, exist_ok=True)
            cm = confusion_matrix(ytrue, ypred)
            plt.figure(figsize=(4,4)); plt.imshow(cm); plt.title(f"{ds_name} {mode} fold{fold} CM")
            plt.tight_layout(); plt.savefig(os.path.join(ds_out, f"cm_{mode}_fold{fold}.png"), dpi=200); plt.close()
            plt.figure(figsize=(8,3))
            for tag in ["early","mid","late"]:
                if tag in snaps:
                    plt.plot(snaps[tag][:60], label=tag)
            plt.legend(); plt.title(f"{ds_name} {mode} grad norms")
            plt.tight_layout(); plt.savefig(os.path.join(ds_out, f"grad_{mode}_fold{fold}.png"), dpi=200); plt.close()
        results[f"{mode}_bal_mean"] = float(np.mean(bals))
        results[f"{mode}_bal_std"]  = float(np.std(bals))
        results[f"{mode}_f1_mean"]  = float(np.mean(f1s))
        results[f"{mode}_f1_std"]   = float(np.std(f1s))
        results[f"{mode}_splits"]   = len(bals)
    results["dataset"] = ds_name
    results["classes"] = ",".join(classes)
    return results

!pkill -f tensorboard || true
%load_ext tensorboard
%tensorboard --logdir /content/runs_task1
