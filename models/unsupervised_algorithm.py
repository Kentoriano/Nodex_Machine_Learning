import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# Features used for clustering
FEATURES = [
    'BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT',
    'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE'
]

OPTIMAL_K = 4

# Descriptive label for each cluster based on centroid analysis
CLUSTER_LABELS = {
    0: "Inactive / Low Activity",
    1: "Cash Advance Users",
    2: "Active Buyers",
    3: "High Balance Accumulators"
}


def _plot_to_base64(fig):
    """Convert a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#212529', edgecolor='none', dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


def _plot_elbow(k_range, inertias, optimal_k):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#212529')
    ax.set_facecolor('#1a1e24')

    ax.plot(k_range, inertias, 'o-', color='#0dcaf0',
            linewidth=2.5, markersize=7, markerfacecolor='white')
    ax.axvline(x=optimal_k, color='#ff6b6b', linestyle='--',
               linewidth=2, label=f'Optimal K = {optimal_k}')

    ax.set_xlabel('Number of Clusters (K)', color='white', fontsize=11)
    ax.set_ylabel('Inertia  (WCSS)', color='white', fontsize=11)
    ax.set_title('Elbow Method — Finding the Optimal K', color='white', fontsize=13, pad=12)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#343a40', labelcolor='white', fontsize=10)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#495057')
    ax.xaxis.set_tick_params(colors='white')
    ax.yaxis.set_tick_params(colors='white')

    return _plot_to_base64(fig)


def _plot_clusters(df_clean, labels, feature_x, feature_y, centroids_df):
    colors = ['#0dcaf0', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8']

    fig, ax = plt.subplots(figsize=(9, 5), facecolor='#212529')
    ax.set_facecolor('#1a1e24')

    unique_labels = sorted(set(labels))
    for i in unique_labels:
        mask = labels == i
        ax.scatter(
            df_clean.loc[mask, feature_x],
            df_clean.loc[mask, feature_y],
            c=colors[i % len(colors)], alpha=0.45, s=18,
            label=f'Cluster {i} — {CLUSTER_LABELS.get(i, "")}'
        )

    ax.scatter(
        centroids_df[feature_x], centroids_df[feature_y],
        c='white', s=220, marker='X', zorder=6,
        edgecolors='black', linewidths=1.2, label='Centroids'
    )

    ax.set_xlabel(feature_x, color='white', fontsize=10)
    ax.set_ylabel(feature_y, color='white', fontsize=10)
    ax.set_title(f'K-Means Clusters: {feature_x}  vs  {feature_y}',
                 color='white', fontsize=12, pad=10)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#343a40', labelcolor='white', fontsize=8,
              loc='upper right', framealpha=0.8)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#495057')

    return _plot_to_base64(fig)


def _plot_cluster_sizes(sizes_dict):
    colors = ['#0dcaf0', '#ff6b6b', '#51cf66', '#ffd43b']
    labels_list = [f'Cluster {k}\n{CLUSTER_LABELS.get(k,"")}' for k in sizes_dict.keys()]
    values = list(sizes_dict.values())

    fig, ax = plt.subplots(figsize=(7, 4), facecolor='#212529')
    ax.set_facecolor('#1a1e24')
    bars = ax.bar(labels_list, values, color=colors[:len(values)],
                  width=0.5, edgecolor='#343a40', linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                str(val), ha='center', va='bottom', color='white', fontsize=10)

    ax.set_ylabel('Number of Customers', color='white', fontsize=10)
    ax.set_title('Customers per Cluster', color='white', fontsize=12, pad=10)
    ax.tick_params(colors='white')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#495057')

    return _plot_to_base64(fig)


def run_analysis():
    """Run the full K-Means pipeline and return all results as a dict."""
    # ── 1. Load & clean ─────────────────────────────────────────
    df_raw = pd.read_csv('data/CC GENERAL.csv')
    df_model = df_raw.drop(columns=['CUST_ID'])

    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(df_model), columns=df_model.columns
    )

    # ── 2. Scale ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed[FEATURES])

    # ── 3. Elbow method ─────────────────────────────────────────
    k_range = list(range(1, 11))
    inertias = []
    for k in k_range:
        km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_tmp.fit(X_scaled)
        inertias.append(round(km_tmp.inertia_, 2))

    elbow_chart = _plot_elbow(k_range, inertias, OPTIMAL_K)

    # ── 4. Fit K-Means with optimal K ────────────────────────────
    km = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # ── 5. Metrics ───────────────────────────────────────────────
    sil_score = round(silhouette_score(X_scaled, labels), 4)
    inertia = round(km.inertia_, 2)

    # ── 6. Centroids in original scale ───────────────────────────
    centroids_scaled = km.cluster_centers_
    # Reconstruct full feature matrix for inverse_transform
    dummy = np.zeros((OPTIMAL_K, len(FEATURES)))
    for i, feat in enumerate(FEATURES):
        dummy[:, i] = centroids_scaled[:, i]
    centroids_original = scaler.inverse_transform(dummy)
    centroids_df = pd.DataFrame(centroids_original, columns=FEATURES).round(2)
    centroids_df.index.name = 'Cluster'

    # ── 7. Cluster summary ───────────────────────────────────────
    df_result = X_imputed[FEATURES].copy()
    df_result['Cluster'] = labels
    cluster_summary = df_result.groupby('Cluster')[FEATURES].mean().round(2)
    cluster_sizes = {int(k): int(v) for k, v in
                     zip(*np.unique(labels, return_counts=True))}

    # ── 8. Scatter plots ─────────────────────────────────────────
    scatter_balance_purchases = _plot_clusters(
        df_result, labels, 'BALANCE', 'PURCHASES', centroids_df)
    scatter_credit_payments = _plot_clusters(
        df_result, labels, 'CREDIT_LIMIT', 'PAYMENTS', centroids_df)

    bar_chart = _plot_cluster_sizes(cluster_sizes)

    # ── 9. Prepare centroid rows for template ────────────────────
    centroid_rows = []
    for idx, row in centroids_df.iterrows():
        centroid_rows.append({
            'cluster': idx,
            'label': CLUSTER_LABELS.get(idx, f'Cluster {idx}'),
            'feats': {feat: row[feat] for feat in FEATURES}
        })

    summary_rows = []
    for idx, row in cluster_summary.iterrows():
        summary_rows.append({
            'cluster': idx,
            'label': CLUSTER_LABELS.get(idx, f'Cluster {idx}'),
            'size': cluster_sizes.get(idx, 0),
            'pct': round(cluster_sizes.get(idx, 0) / len(labels) * 100, 1),
            'feats': {feat: row[feat] for feat in FEATURES}
        })

    return {
        'k':                   OPTIMAL_K,
        'n_samples':           len(df_result),
        'n_features':          len(FEATURES),
        'features':            FEATURES,
        'silhouette':          sil_score,
        'inertia':             inertia,
        'null_count':          int(df_raw.isnull().sum().sum()),
        'elbow_chart':         elbow_chart,
        'scatter1':            scatter_balance_purchases,
        'scatter2':            scatter_credit_payments,
        'bar_chart':           bar_chart,
        'cluster_sizes':       cluster_sizes,
        'centroid_rows':       centroid_rows,
        'summary_rows':        summary_rows,
        'k_range':             k_range,
        'inertias':            inertias,
        'k_inertia_pairs':     list(zip(k_range, inertias)),
    }
