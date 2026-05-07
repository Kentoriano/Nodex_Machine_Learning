from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io



def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def calculate_distances(X, centroids):
    distances = []
    for point in X:
        dist_point = []
        for centroid in centroids:
            distance = np.linalg.norm(point - centroid)
            dist_point.append(distance)
        distances.append(dist_point)
    return np.array(distances)


def recalculate_centroids(X, labels, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
        else:
            new_centroid = X[np.random.randint(0, len(X))]
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

def calculate_variance(X, centroids, labels):
    variance = 0
    for i, point in enumerate(X):
        centroid = centroids[labels[i]]
        variance += np.linalg.norm(point - centroid) ** 2
    return variance

def run_kmeans_steps():

    # 🔹 Load data
    data = pd.read_csv("data/CC GENERAL.csv")

    # 🔹 Preprocessing
    data = data.drop("CUST_ID", axis=1)
    data = data.fillna(data.mean())
    data = data.drop_duplicates()

    # Feature selection
    data = data[[
        "BALANCE",
        "BALANCE_FREQUENCY",
        "PURCHASES",
        "PAYMENTS"
    ]]

    data = data[data["BALANCE"] < data["BALANCE"].quantile(0.99)]
    data = data[data["PURCHASES"] < data["PURCHASES"].quantile(0.99)]

    data = data.reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)


    centroids = X_scaled[[0, 50, 100]]

    k = 3
    iterations_data = []


    for iteration in range(3):

        distances = calculate_distances(X_scaled, centroids)
        labels = np.argmin(distances, axis=1)

        # 🔹 Build table
        table = pd.DataFrame(X_scaled, columns=[
            "BALANCE",
            "BALANCE_FREQUENCY",
            "PURCHASES",
            "PAYMENTS"
        ])

        table["Dist_C1"] = distances[:, 0]
        table["Dist_C2"] = distances[:, 1]
        table["Dist_C3"] = distances[:, 2]
        table["Cluster"] = labels

        table = table.head(30)

  
        variance = calculate_variance(X_scaled, centroids, labels)

        iterations_data.append({
            "iteration": iteration + 1,
            "centroids": centroids.tolist(),
            "table": table.to_dict(orient="records"),
            "variance": float(variance)
        })

     
        centroids = recalculate_centroids(X_scaled, labels, k)


    variances = [it["variance"] for it in iterations_data]


    plt.figure()
    plt.plot(range(1, len(variances)+1), variances, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Variance")
    plt.title("Variance Evolution")
    variance_img = fig_to_base64()



    plt.figure()
    plt.scatter(X_scaled[:, 0], X_scaled[:, 2], c=labels)
    plt.xlabel("BALANCE (scaled)")
    plt.ylabel("PURCHASES (scaled)")
    plt.title("Final Clusters")
    clusters_img = fig_to_base64()

    return {
        "initial_data": data.head(30).to_dict(orient="records"),
        "iterations": iterations_data,
        "variances": variances,
        "variance_img": variance_img,
        "clusters_img": clusters_img,
        "final_message": "Variance decreases across iterations, indicating improved clustering and convergence."

    }
