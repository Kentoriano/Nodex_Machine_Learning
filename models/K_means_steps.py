from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "CC GENERAL.csv")

data = pd.read_csv(file_path)
data = data.drop("CUST_ID", axis=1)
data = data.fillna(data.mean())
data = data.drop_duplicates()

data = data.drop("ONEOFF_PURCHASES", axis=1)
data = data.drop("INSTALLMENTS_PURCHASES", axis=1)
data = data.drop("CASH_ADVANCE", axis=1)
data = data.drop("PURCHASES_FREQUENCY", axis=1)
data = data.drop("ONEOFF_PURCHASES_FREQUENCY", axis=1)
data = data.drop("PURCHASES_INSTALLMENTS_FREQUENCY", axis=1)
data = data.drop("CASH_ADVANCE_FREQUENCY", axis=1)
data = data.drop("CASH_ADVANCE_TRX", axis=1)
data = data.drop("PURCHASES_TRX", axis=1)
data = data.drop("CREDIT_LIMIT", axis=1)
data = data.drop("MINIMUM_PAYMENTS", axis=1)
data = data.drop("PRC_FULL_PAYMENT", axis=1)


data = data.reset_index(drop=True) 
data = data.sample(500, random_state=42)



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

    # Remove outliers
    data = data[data["BALANCE"] < data["BALANCE"].quantile(0.99)]
    data = data[data["PURCHASES"] < data["PURCHASES"].quantile(0.99)]

    data = data.reset_index(drop=True)

    # 🔹 Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # 🔹 Initial centroids (MANUAL selection)
    centroids = X_scaled[[0, 50, 100]]

    k = 3
    iterations_data = []

    # 🔹 Run ONLY 3 iterations
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

        # 🔹 Variance
        variance = calculate_variance(X_scaled, centroids, labels)

        iterations_data.append({
            "iteration": iteration + 1,
            "centroids": centroids.tolist(),
            "table": table.to_dict(orient="records"),
            "variance": float(variance)
        })

        # 🔹 Update centroids
        centroids = recalculate_centroids(X_scaled, labels, k)

    # 🔹 Final comparison
    variances = [it["variance"] for it in iterations_data]

    variances = [it["variance"] for it in iterations_data]

    plt.figure()
    plt.plot(range(1, len(variances)+1), variances, marker='o')

    plt.xlabel("Iteration")
    plt.ylabel("Variance")
    plt.title("Variance Evolution")

    plt.savefig("static/img/variance.png")
    plt.close()


    plt.figure()

    plt.scatter(
        X_scaled[:, 0],   # BALANCE
        X_scaled[:, 2],   # PURCHASES
        c=labels
    )

    plt.xlabel("BALANCE (scaled)")
    plt.ylabel("PURCHASES (scaled)")
    plt.title("Final Clusters")

    plt.savefig("static/img/clusters.png")
    plt.close()

    return {
        "initial_data": data.head(30).to_dict(orient="records"),  
        "iterations": iterations_data,
        "variances": variances,
        "variance_img": "img/variance.png",
        "clusters_img": "img/clusters.png",
        "final_message": "Variance decreases across iterations, indicating improved clustering and convergence."
    }

 