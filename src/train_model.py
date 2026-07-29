import os
import numpy as np
import mne
from mne import create_info
from mne.io import RawArray
from mne.preprocessing import ICA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
from scipy.signal import welch, hilbert, coherence
from scipy.linalg import eigh
from scipy.stats import entropy as shannon_entropy
import pywt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
# -------------------------
# Constants
# -------------------------
SAMPLE_RATE = 256
WINDOW_SIZE = 5 * SAMPLE_RATE  # 5s
STEP_SIZE = 2 * SAMPLE_RATE    # 2s
CHANNELS = ['Fp1', 'F3', 'C3', 'P3', 'O1',
            'F7', 'T3', 'T5', 'Fz', 'Fp2',
            'F4', 'C4', 'P4', 'O2', 'F8',
            'T4', 'T6', 'Cz', 'Pz']

# -------------------------
# Build adjacency matrix
# -------------------------
def build_adjacency_matrix():
    montage = mne.channels.make_standard_montage("standard_1020")
    pos = montage.get_positions()["ch_pos"]
    coords = np.array([pos[ch] for ch in CHANNELS])
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    sigma = np.mean(dist)
    W = np.exp(-dist ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)
    return W

ADJ_MATRIX = build_adjacency_matrix()

# -------------------------
# Features
# -------------------------
def compute_gft_features(segment, adjacency_matrix):
    D = np.diag(adjacency_matrix.sum(axis=1))
    L = D - adjacency_matrix
    eigvals, eigvecs = eigh(L)
    gft = eigvecs.T @ segment
    feats = []
    for node_signal in gft:
        feats.extend([np.mean(node_signal), np.std(node_signal),
                      np.min(node_signal), np.max(node_signal)])
    return np.array(feats)

def compute_complexity_features(segment):
    hjorth, higuchi = [], []
    for ch in segment:
        d1, d2 = np.diff(ch), np.diff(np.diff(ch))
        v0, v1, v2 = np.var(ch), np.var(d1), np.var(d2)
        mob = np.sqrt(v1 / v0) if v0 != 0 else 0
        comp = np.sqrt(v2 / v1) / mob if v1 != 0 and mob != 0 else 0
        hjorth.extend([v0, mob, comp])
        # Higuchi FD
        kmax, Lk = 5, []
        for k in range(1, kmax + 1):
            Lm = []
            for m in range(k):
                Lmk, nmax = 0, int(np.floor((len(ch) - m) / k))
                for j in range(1, nmax):
                    Lmk += abs(ch[m + j*k] - ch[m + (j-1)*k])
                if nmax > 0:
                    Lmk = (Lmk * (len(ch)-1)) / (nmax * k)
                Lm.append(Lmk)
            Lk.append(np.mean(Lm))
        logLk, logk = np.log(Lk), np.log(1.0/np.arange(1, kmax+1))
        higuchi.append(np.polyfit(logk, logLk, 1)[0])
    return np.array(hjorth + higuchi)

def compute_bandpower_features(segment, sf=256):
    bands = [(0.5,4), (4,8), (8,13), (13,30), (30,45)]
    feats = []
    for ch in segment:
        freqs, psd = welch(ch, sf, nperseg=sf*2)
        for lo, hi in bands:
            idx = (freqs >= lo) & (freqs <= hi)
            feats.append(np.trapz(psd[idx], freqs[idx]))
    return np.array(feats)

# -------------------------
# ICA
# -------------------------
def fit_ica(data):
    """Fit ICA on concatenated training EEG data using IIR filters."""
    info = create_info(CHANNELS, SAMPLE_RATE, ch_types="eeg")
    raw = RawArray(data, info, verbose=False)

    # Use IIR filters (safe for short windows)
    raw.filter(1.0, 45.0, method="iir", verbose=False)
    raw.notch_filter(50.0, method="iir", verbose=False)

    ica = ICA(
        n_components=min(15, raw.info["nchan"] - 1),
        random_state=97,
        max_iter="auto",
        verbose=False
    )
    ica.fit(raw)
    return raw.get_data(), ica

def compute_coherence_features(segment, sf=256):
    """Compute mean coherence between all channel pairs."""
    feats = []
    n_channels = segment.shape[0]
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            f, Cxy = coherence(segment[i], segment[j], fs=sf, nperseg=sf*2)
            feats.append(np.mean(Cxy))  # average coherence across frequencies
    return np.array(feats)

def apply_ica_to_data(data, ica):
    """Apply a pre-fitted ICA object to new EEG segment."""
    info = create_info(CHANNELS, SAMPLE_RATE, ch_types="eeg")
    raw = RawArray(data, info, verbose=False)

    raw.filter(1.0, 45.0, method="iir", verbose=False)
    raw.notch_filter(50.0, method="iir", verbose=False)

    return ica.apply(raw.copy(), verbose=False).get_data()
# -------------------------
# Segmentation
# -------------------------
def segment_eeg(eeg_data, label, subj_id):
    segments, labels, ids = [], [], []
    for start in range(0, eeg_data.shape[1]-WINDOW_SIZE+1, STEP_SIZE):
        seg = eeg_data[:, start:start+WINDOW_SIZE]
        segments.append(seg)
        labels.append(label)
        ids.append(subj_id)
    return segments, labels, ids

# -------------------------
# Label inference
# -------------------------
def infer_label_from_filename(fname):
    name = fname.upper()
    if name.startswith("MDD"): return 1
    if name.startswith("H"): return 0
    return None

def compute_plv_segment(eeg_segment):
    analytic = hilbert(eeg_segment, axis=1)
    phases = np.angle(analytic)
    C, T = phases.shape
    plvs = []
    for i in range(C):
        for j in range(i+1, C):
            pdiff = phases[i] - phases[j]
            plv = np.abs(np.mean(np.exp(1j * pdiff)))
            plvs.append(plv)
    return np.asarray(plvs, dtype=np.float32)

def wavelet_feats_channel(sig, wavelet="db4", level=4):
    coeffs = pywt.wavedec(sig, wavelet, level=level, mode="symmetric")
    feats = []
    for c in coeffs:
        mean, std, energy = np.mean(c), np.std(c), np.sum(c ** 2)
        p = np.abs(c) + 1e-12
        p /= np.sum(p)
        ent = shannon_entropy(p, base=2)
        feats.extend([mean, std, energy, ent])
    return np.asarray(feats, dtype=np.float32)

def compute_wavelet_segment(eeg_segment, wavelet="db4", level=4):
    feats_all = [wavelet_feats_channel(eeg_segment[ch], wavelet, level) for ch in range(eeg_segment.shape[0])]
    return np.concatenate(feats_all, axis=0).astype(np.float32)

# -------------------------
# Process EDF
# -------------------------
def process_edf_dataset(input_dir):
    all_segments, all_y, all_ids = [], [], []
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.edf') and 'task' in f.lower()]
    for i,f in enumerate(sorted(files),1):
        label = infer_label_from_filename(f)
        if label is None: continue
        subj_id = os.path.splitext(f)[0]
        print(f"[{i}/{len(files)}] Processing {f} label={label}")
        try:
            raw = mne.io.read_raw_edf(os.path.join(input_dir,f), preload=True, verbose=False)
            raw.pick_types(eeg=True)
            raw.resample(SAMPLE_RATE)
            data = raw.get_data()
            if data.shape[0] > len(CHANNELS):
                data = data[:len(CHANNELS)]
            if data.shape[1] < WINDOW_SIZE:
                continue
            segs, ys, ids = segment_eeg(data, label, subj_id)
            all_segments.extend(segs)
            all_y.extend(ys)
            all_ids.extend(ids)
        except Exception as e:
            print(f"❌ {f}: {e}")
    return np.array(all_segments), np.array(all_y), np.array(all_ids)

# -------------------------
# Feature extraction
# -------------------------
def extract_handcrafted_features(segments):
    feats = []
    for seg in segments:
        gft = compute_gft_features(seg, ADJ_MATRIX)  # 1D
        comp = compute_complexity_features(seg)      # 1D
        bp = compute_bandpower_features(seg)         # 1D
        coh = compute_coherence_features(seg)        # ✅ NEW
        plv = compute_plv_segment(seg).ravel()       # ensure 1D
        wav = compute_wavelet_segment(seg, "db4", 4).ravel()  # ensure 1D

        # concatenate all into single feature vector
        feats.append(np.concatenate([gft, comp, bp, coh, plv, wav]))
    return np.array(feats)


# -------------------------
# Model
# -------------------------
def build_model(seq_shape, static_shape):
    # Sequence branch (CNN + LSTM)
    seq_in = layers.Input(shape=seq_shape)
    x = layers.Conv1D(64, kernel_size=5, activation='relu')(seq_in)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)

    # Handcrafted features branch
    static_in = layers.Input(shape=(static_shape,))
    y = layers.Dense(128, activation='relu')(static_in)
    y = layers.Dropout(0.3)(y)

    # Fusion
    combined = layers.concatenate([x, y])
    z = layers.Dense(64, activation='relu')(combined)
    out = layers.Dense(1, activation='sigmoid')(z)

    # Build model
    model = models.Model(inputs=[seq_in, static_in], outputs=out)
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    data_dir = "files"
    segments, y, subj_ids = process_edf_dataset(data_dir)
    print(f"Total segments: {len(y)}")

    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

    all_preds, all_true = [], []
    fold_metrics = {"acc": [], "prec": [], "rec": [], "f1": []}
    fold = 1

    for train_idx, test_idx in sgkf.split(segments, y, subj_ids):
        X_train_raw, X_test_raw = segments[train_idx], segments[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ---------------- ICA ----------------
        print(f"🔹 Fitting ICA on fold {fold} train data...")
        data_concat = np.hstack(X_train_raw)
        _, ica = fit_ica(data_concat)

        X_train_raw = np.array([apply_ica_to_data(seg, ica) for seg in X_train_raw])
        X_test_raw = np.array([apply_ica_to_data(seg, ica) for seg in X_test_raw])

        # ---------------- Features ----------------
        X_train_feats = extract_handcrafted_features(X_train_raw)
        X_test_feats = extract_handcrafted_features(X_test_raw)

        scaler = StandardScaler().fit(X_train_feats)
        X_train_feats = scaler.transform(X_train_feats)
        X_test_feats = scaler.transform(X_test_feats)

        selector = SelectKBest(f_classif, k=min(500, X_train_feats.shape[1])).fit(X_train_feats, y_train)
        X_train_feats = selector.transform(X_train_feats)
        X_test_feats = selector.transform(X_test_feats)

        # ---------------- Sequences ----------------
        X_train_seq = X_train_raw.transpose(0, 2, 1)
        X_test_seq = X_test_raw.transpose(0, 2, 1)

        # ---------------- Model ----------------
        model = build_model((X_train_seq.shape[1], X_train_seq.shape[2]), X_train_feats.shape[1])

        # class weights
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights = dict(zip(np.unique(y_train), class_weights))

        model.fit([X_train_seq, X_train_feats], y_train,
                  validation_data=([X_test_seq, X_test_feats], y_test),
                  epochs=20, batch_size=32, verbose=1,
                  class_weight=class_weights)

        # save fold model
        model.save(f"model_fold{fold}.h5")

        # ---------------- Evaluation per fold ----------------
        probs = model.predict([X_test_seq, X_test_feats]).ravel()
        preds = (probs > 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        fold_metrics["acc"].append(acc)
        fold_metrics["prec"].append(prec)
        fold_metrics["rec"].append(rec)
        fold_metrics["f1"].append(f1)

        print(f"✅ Fold {fold} acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

        all_preds.extend(preds)
        all_true.extend(y_test)
        fold += 1

    # -------------------- Overall CV Report --------------------
    print("\n=== Overall CV Report (pooled predictions) ===")
    print(classification_report(all_true, all_preds))

    cm = confusion_matrix(all_true, all_preds)
    print("Confusion Matrix:\n", cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Cross-Validation Confusion Matrix")
    plt.show()

    # ROC should use probs (not hard labels!)
    fpr, tpr, _ = roc_curve(all_true, all_preds)  # you can also store probs during CV
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Cross-Validation ROC Curve")
    plt.legend()
    plt.show()

    # -------------------- Mean ± Std Metrics --------------------
    print("\n=== Mean ± Std across folds ===")
    for m in fold_metrics:
        mean = np.mean(fold_metrics[m])
        std = np.std(fold_metrics[m])
        print(f"{m.upper()}: {mean:.4f} ± {std:.4f}")

    # -------------------- Final model on all data --------------------
    import joblib

    print("\n🔹 Training final model on all data...")
    data_concat = np.hstack(segments)
    _, ica_full = fit_ica(data_concat)
    segments_ica = np.array([apply_ica_to_data(seg, ica_full) for seg in segments])

    # === Handcrafted features ===
    full_feats = extract_handcrafted_features(segments_ica)

    # Save & apply scaler
    scaler = StandardScaler().fit(full_feats)
    full_feats = scaler.transform(full_feats)
    joblib.dump(scaler, "scaler.pkl")  # ✅ save scaler

    # Save & apply feature selector
    selector = SelectKBest(f_classif, k=min(100, full_feats.shape[1])).fit(full_feats, y)
    full_feats = selector.transform(full_feats)
    joblib.dump(selector, "selector.pkl")  # ✅ save selector

    # Sequence input
    X_seq = segments_ica.transpose(0, 2, 1)

    # Build & train model
    final_model = build_model((X_seq.shape[1], X_seq.shape[2]), full_feats.shape[1])
    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    class_weights = dict(zip(np.unique(y), class_weights))

    final_model.fit([X_seq, full_feats], y, epochs=10, batch_size=32, class_weight=class_weights)

    # Save model
    final_model.save("model_final.h5")
    print("✅ Final model + preprocessing saved (model_final.h5, scaler.pkl, selector.pkl)")
