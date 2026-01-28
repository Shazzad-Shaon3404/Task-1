from sklearn.preprocessing import LabelEncoder
CFG = dict(
    sfreq=250,
    fmin=4.0, fmax=36.0,
    tmin=0.0, tmax=3.0,
    apply_car=True,
    zscore=True, zclip=5.0,
    notch=None,
    batch_size=64,
    max_epochs=60,
    patience=12,
    lr=1e-3,
    weight_decay=1e-4,
    n_splits=5,
    fast_dev_run=True,
    fast_subjects=3,
    out_root="/content/eeg_task1_outputs",
    runs_root="/content/runs_task1",
)
os.makedirs(CFG["out_root"], exist_ok=True)
os.makedirs(CFG["runs_root"], exist_ok=True)
def load_dataset(name):
    if name == "BNCI2015_001":
        ds = BNCI2015_001()
        n_classes = 2
    elif name == "BNCI2014_001":
        ds = BNCI2014_001()
        n_classes = 4
    elif name == "Lee2019_MI":
        ds = Lee2019_MI()
        n_classes = 2
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
    if CFG["notch"] is not None:
        epochs._data = mne.filter.notch_filter(
            epochs.get_data(), Fs=CFG["sfreq"], freqs=[CFG["notch"]], verbose=False
        )
    if CFG["apply_car"]:
        epochs = epochs.copy().set_eeg_reference("average", projection=False, verbose=False)
    X = epochs.get_data().astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(np.array(y_raw)).astype(np.int64)
    class_names = list(le.classes_)
    meta = meta.reset_index(drop=True)
    if CFG["fast_dev_run"]:
        keep = sorted(meta["subject"].unique())[:CFG["fast_subjects"]]
        m = meta["subject"].isin(keep).values
        X, y, meta = X[m], y[m], meta.loc[m].reset_index(drop=True)
    if CFG["zscore"]:
        X = trialwise_zscore(X, clip=CFG["zclip"])
    try:
        montage = mne.channels.make_standard_montage("standard_1005")
        epochs.info.set_montage(montage, on_missing="ignore")
    except Exception:
        pass
    info = epochs.info
    return X, y, meta, info, n_classes, class_names

for dn in datasets:
    X, y, meta, info, nc, class_names = load_dataset(dn)
    print(dn, "X:", X.shape, "classes:", class_names, "subjects:", meta["subject"].nunique())
