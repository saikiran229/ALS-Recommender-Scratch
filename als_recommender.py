import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ALSRecommender:
    def __init__(self, k=15, lambda_reg=0.20, epochs=10):
        """
        Matrix Factorization using ALS with User and Item Biases.
        """
        self.k = k
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.U = None 
        self.V = None 
        self.b_u = None # User biases
        self.b_i = None # Item biases
        self.global_mean = 0

    def fit(self, R, test_data):
        self.N, self.M = R.shape
        
        np.random.seed(42)
        self.U = np.random.normal(scale=1.0 / self.k, size=(self.N, self.k))
        self.V = np.random.normal(scale=1.0 / self.k, size=(self.M, self.k))
        
        self.b_u = np.zeros(self.N)
        self.b_i = np.zeros(self.M)
        self.global_mean = R[R > 0].mean() if np.any(R > 0) else 3.5

        train_history = []
        test_history = []

        print("\n--- Starting ALS Training Loop with Biases ---")
        for epoch in range(1, self.epochs + 1):
            
            # Update Users and User Biases
            V_T_V = np.dot(self.V.T, self.V)
            I_k = np.eye(self.k) * self.lambda_reg
            
            for u in range(self.N):
                rated_idx = np.where(R[u, :] > 0)[0]
                if len(rated_idx) == 0: continue
                
                V_u = self.V[rated_idx, :]
                R_u = R[u, rated_idx] - self.global_mean - self.b_i[rated_idx]
                
                A = np.dot(V_u.T, V_u) + I_k
                b = np.dot(V_u.T, R_u)
                self.U[u, :] = np.linalg.solve(A, b)
                
                self.b_u[u] = np.mean(R[u, rated_idx] - self.global_mean - np.dot(V_u, self.U[u, :]) - self.b_i[rated_idx])

            # Update Items and Item Biases
            U_T_U = np.dot(self.U.T, self.U)
            
            for i in range(self.M):
                rated_idx = np.where(R[:, i] > 0)[0]
                if len(rated_idx) == 0: continue
                
                U_i = self.U[rated_idx, :]
                R_i = R[rated_idx, i] - self.global_mean - self.b_u[rated_idx]
                
                A = np.dot(U_i.T, U_i) + I_k
                b = np.dot(U_i.T, R_i)
                self.V[i, :] = np.linalg.solve(A, b)
                
                self.b_i[i] = np.mean(R[rated_idx, i] - self.global_mean - np.dot(U_i, self.V[i, :]) - self.b_u[rated_idx])

            train_rmse = self.calculate_train_rmse(R)
            test_rmse = self.evaluate(test_data)
            train_history.append(train_rmse)
            test_history.append(test_rmse)
            
            print(f"Epoch {epoch:02d}/{self.epochs:02d} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")

        return train_history, test_history

    def calculate_train_rmse(self, R):
        user_idx, item_idx = np.where(R > 0)
        predictions = self.global_mean + self.b_u[user_idx] + self.b_i[item_idx] + np.sum(self.U[user_idx] * self.V[item_idx], axis=1)
        return np.sqrt(np.mean((R[user_idx, item_idx] - predictions) ** 2))

    def evaluate(self, test_df):
        user_indices = test_df['user_idx'].values
        item_indices = test_df['item_idx'].values
        actuals = test_df['rating'].values
        
        predictions = self.global_mean + self.b_u[user_indices] + self.b_i[item_indices] + np.sum(self.U[user_indices] * self.V[item_indices], axis=1)
        predictions = np.clip(predictions, 1.0, 5.0)
        
        return np.sqrt(np.mean((actuals - predictions) ** 2))


def download_and_extract_movielens():
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = "ml-1m.zip"
    data_file = "ml-1m/ratings.dat"
    
    if not os.path.exists(data_file):
        print("Downloading MovieLens 1M dataset (approx. 6MB)...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Extraction complete.")
    return data_file


def run_naive_svd_baseline(R_train, test_df):
    print("\nRunning Naive SVD Baseline...")
    global_mean = R_train[R_train > 0].mean()
    R_filled = R_train.copy()
    R_filled[R_filled == 0] = global_mean
    
    U, Sigma, Vt = np.linalg.svd(R_filled, full_matrices=False)
    
    k = 20
    Sigma_k = np.diag(Sigma[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    
    R_hat_svd = np.dot(np.dot(U_k, Sigma_k), Vt_k)
    
    user_indices = test_df['user_idx'].values
    item_indices = test_df['item_idx'].values
    actuals = test_df['rating'].values
    
    predictions = R_hat_svd[user_indices, item_indices]
    predictions = np.clip(predictions, 1.0, 5.0)
    
    svd_rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    print(f"Naive SVD Baseline Test RMSE: {svd_rmse:.4f}")
    return svd_rmse


if __name__ == "__main__":
    data_path = download_and_extract_movielens()
    
    print("Loading data into Pandas DataFrame...")
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(data_path, sep='::', engine='python', names=columns)
    df = df.drop('timestamp', axis=1)
    
    df['user_idx'] = df['user_id'].astype('category').cat.codes
    df['item_idx'] = df['item_id'].astype('category').cat.codes
    
    N = df['user_idx'].nunique()
    M = df['item_idx'].nunique()
    print(f"Unique Users (N): {N} | Unique Movies (M): {M}")

    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(df))
    split_point = int(len(df) * 0.8)
    
    train_df = df.iloc[shuffled_indices[:split_point]].copy()
    test_df = df.iloc[shuffled_indices[split_point:]].copy()
    
    test_df = test_df[test_df['user_idx'].isin(train_df['user_idx']) & test_df['item_idx'].isin(train_df['item_idx'])]
    
    R_train = np.zeros((N, M))
    R_train[train_df['user_idx'].values, train_df['item_idx'].values] = train_df['rating'].values
    print(f"Training Matrix built. Sparsity: {100 * (1 - len(train_df)/(N*M)):.2f}%")

    # Run the Bias-Optimized Model
    als_model = ALSRecommender(k=10, lambda_reg=0.1, epochs=20)
    train_rmse_hist, test_rmse_hist = als_model.fit(R_train, test_df)

    svd_baseline_rmse = run_naive_svd_baseline(R_train, test_df)
    
    final_als_rmse = test_rmse_hist[-1]
    pct_improvement = ((svd_baseline_rmse - final_als_rmse) / svd_baseline_rmse) * 100
    
    print(f"\nDeliverable Metric: Achieved Bias-Optimized ALS RMSE of {final_als_rmse:.4f} vs SVD Baseline of {svd_baseline_rmse:.4f}")
    print(f"Performance Gain: {pct_improvement:.2f}% improvement over baseline.")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_rmse_hist) + 1), train_rmse_hist, label='ALS Train RMSE', marker='o')
    plt.plot(range(1, len(test_rmse_hist) + 1), test_rmse_hist, label='ALS Test RMSE', marker='s')
    plt.axhline(y=svd_baseline_rmse, color='r', linestyle='--', label=f'Naive SVD Baseline ({svd_baseline_rmse:.3f})')
    plt.title('ALS Recommender Training Convergence vs SVD Baseline')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig('als_vs_svd_performance12345.png')
    print("\nPerformance comparison plot saved as 'als_vs_svd_performance.png'.")