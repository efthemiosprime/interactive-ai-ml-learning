import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function RecommendationSystemTutorial() {
  const [selectedStep, setSelectedStep] = useState('step1');
  const [selectedFramework, setSelectedFramework] = useState('pytorch');

  const steps = {
    step1: {
      title: 'Step 1: Data Collection & Preprocessing',
      description: 'Gather user-item interaction data and prepare it for recommendation models',
      code: `# Step 1: Data Collection & Preprocessing
import pandas as pd
import numpy as np
from collections import defaultdict

# 1.1 Collect User-Item Interaction Data
# Example: Movie ratings dataset (like MovieLens)
# Columns: userId, movieId, rating, timestamp

# Sample data structure
ratings_data = {
    'userId': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'movieId': [101, 102, 103, 101, 104, 102, 103, 105, 101, 103],
    'rating': [5, 4, 5, 3, 4, 5, 4, 3, 4, 5],
    'timestamp': [1234567890, 1234567900, 1234567910, 1234567920, 
                   1234567930, 1234567940, 1234567950, 1234567960, 
                   1234567970, 1234567980]
}

df_ratings = pd.DataFrame(ratings_data)
print("Ratings Data:")
print(df_ratings.head())

# 1.2 Load Movie Metadata (Optional but useful)
movies_data = {
    'movieId': [101, 102, 103, 104, 105],
    'title': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 'Avatar'],
    'genres': ['Action|Sci-Fi', 'Action|Sci-Fi|Thriller', 'Sci-Fi|Drama', 
               'Action|Crime|Drama', 'Action|Adventure|Sci-Fi']
}

df_movies = pd.DataFrame(movies_data)
print("\\nMovies Metadata:")
print(df_movies.head())

# 1.3 Create User-Item Matrix
# Linear Algebra: Represent interactions as a matrix
# Rows = Users, Columns = Items, Values = Ratings

def create_user_item_matrix(df_ratings):
    """
    Create user-item interaction matrix
    
    Matrix R where:
    - R[i, j] = rating user i gave to item j
    - R[i, j] = 0 if user i hasn't rated item j
    """
    # Pivot table: users as rows, items as columns
    user_item_matrix = df_ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        fill_value=0  # 0 means no rating
    )
    
    return user_item_matrix

user_item_matrix = create_user_item_matrix(df_ratings)
print("\\nUser-Item Matrix:")
print(user_item_matrix)
print(f"\\nMatrix Shape: {user_item_matrix.shape}")
print(f"Users: {user_item_matrix.shape[0]}, Items: {user_item_matrix.shape[1]}")

# 1.4 Handle Missing Values and Sparsity
def analyze_sparsity(user_item_matrix):
    """
    Analyze matrix sparsity
    Sparsity = Percentage of zero entries (no ratings)
    """
    total_entries = user_item_matrix.shape[0] * user_item_matrix.shape[1]
    non_zero_entries = np.count_nonzero(user_item_matrix.values)
    sparsity = (1 - non_zero_entries / total_entries) * 100
    
    print(f"\\nSparsity Analysis:")
    print(f"Total entries: {total_entries}")
    print(f"Non-zero entries: {non_zero_entries}")
    print(f"Sparsity: {sparsity:.2f}%")
    
    return sparsity

sparsity = analyze_sparsity(user_item_matrix)

# 1.5 Normalize Ratings
# Normalize ratings to handle different user rating scales
def normalize_ratings(user_item_matrix, method='mean_centering'):
    """
    Normalize ratings
    
    Methods:
    - mean_centering: Subtract user's mean rating
    - z_score: Standardize to z-scores
    """
    normalized_matrix = user_item_matrix.copy()
    
    if method == 'mean_centering':
        # Subtract each user's mean rating
        user_means = normalized_matrix.mean(axis=1)
        normalized_matrix = normalized_matrix.sub(user_means, axis=0)
        # Replace NaN (from users with no ratings) with 0
        normalized_matrix = normalized_matrix.fillna(0)
    
    elif method == 'z_score':
        # Standardize: (x - mean) / std
        user_means = normalized_matrix.mean(axis=1)
        user_stds = normalized_matrix.std(axis=1)
        normalized_matrix = normalized_matrix.sub(user_means, axis=0)
        normalized_matrix = normalized_matrix.div(user_stds, axis=0)
        normalized_matrix = normalized_matrix.fillna(0)
    
    return normalized_matrix

normalized_matrix = normalize_ratings(user_item_matrix, method='mean_centering')
print("\\nNormalized Matrix (Mean-Centered):")
print(normalized_matrix)

# 1.6 Split Data into Train/Test Sets
from sklearn.model_selection import train_test_split

def split_ratings(df_ratings, test_size=0.2, random_state=42):
    """
    Split ratings into train and test sets
    Important: Use stratified split to maintain user distribution
    """
    # Group by user to ensure each user appears in both sets
    train_data = []
    test_data = []
    
    for user_id in df_ratings['userId'].unique():
        user_ratings = df_ratings[df_ratings['userId'] == user_id]
        
        # Split user's ratings
        user_train, user_test = train_test_split(
            user_ratings,
            test_size=test_size,
            random_state=random_state
        )
        
        train_data.append(user_train)
        test_data.append(user_test)
    
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    return train_df, test_df

train_df, test_df = split_ratings(df_ratings, test_size=0.2)
print(f"\\nTrain set: {len(train_df)} ratings")
print(f"Test set: {len(test_df)} ratings")

# 1.7 Create Mappings
# Map user/item IDs to matrix indices
def create_mappings(df_ratings):
    """
    Create mappings between original IDs and matrix indices
    """
    unique_users = sorted(df_ratings['userId'].unique())
    unique_items = sorted(df_ratings['movieId'].unique())
    
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
    
    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
    idx_to_item = {idx: item_id for item_id, idx in item_to_idx.items()}
    
    return user_to_idx, item_to_idx, idx_to_user, idx_to_item

user_to_idx, item_to_idx, idx_to_user, idx_to_item = create_mappings(df_ratings)
print(f"\\nMappings created:")
print(f"Users: {len(user_to_idx)}, Items: {len(item_to_idx)}")`
    },
    step2: {
      title: 'Step 2: Collaborative Filtering - User-Based',
      description: 'Implement user-based collaborative filtering using cosine similarity',
      code: `# Step 2: Collaborative Filtering - User-Based
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# 2.1 User-Based Collaborative Filtering
# Core Idea: Users with similar preferences will like similar items
# 
# Algorithm:
# 1. Find users similar to target user (using cosine similarity)
# 2. Predict rating = weighted average of similar users' ratings
# 
# Formula: r̂_{u,i} = r̄_u + Σ_{v∈N(u)} sim(u,v) · (r_{v,i} - r̄_v) / Σ_{v∈N(u)} |sim(u,v)|
# where:
# - r̂_{u,i} = predicted rating user u gives to item i
# - r̄_u = average rating of user u
# - N(u) = set of users similar to user u
# - sim(u,v) = similarity between users u and v

def calculate_user_similarity(user_item_matrix, method='cosine'):
    """
    Calculate similarity between users
    
    Methods:
    - cosine: Cosine similarity (most common)
    - pearson: Pearson correlation
    - euclidean: Euclidean distance (inverted)
    """
    if method == 'cosine':
        # Cosine Similarity: cos(θ) = (A · B) / (||A|| × ||B||)
        # Measures angle between user rating vectors
        similarity_matrix = cosine_similarity(user_item_matrix.values)
    
    elif method == 'pearson':
        # Pearson Correlation: Measures linear relationship
        similarity_matrix = np.corrcoef(user_item_matrix.values)
        # Replace NaN with 0
        similarity_matrix = np.nan_to_num(similarity_matrix)
    
    return similarity_matrix

# Example calculation:
# User 1 ratings: [5, 4, 5, 0, 0]
# User 2 ratings: [3, 0, 4, 4, 0]
# 
# Cosine similarity:
# dot_product = 5×3 + 4×0 + 5×4 + 0×4 + 0×0 = 15 + 0 + 20 + 0 + 0 = 35
# ||user1|| = √(5² + 4² + 5²) = √(25 + 16 + 25) = √66 ≈ 8.12
# ||user2|| = √(3² + 4² + 4²) = √(9 + 16 + 16) = √41 ≈ 6.40
# cosine = 35 / (8.12 × 6.40) ≈ 0.674

user_similarity = calculate_user_similarity(user_item_matrix, method='cosine')
print("User Similarity Matrix:")
print(user_similarity)

# 2.2 Find Similar Users
def find_similar_users(user_id, user_similarity, user_to_idx, top_k=5):
    """
    Find top-k most similar users to a given user
    """
    user_idx = user_to_idx[user_id]
    
    # Get similarities for this user
    similarities = user_similarity[user_idx]
    
    # Sort by similarity (descending), exclude self (similarity = 1.0)
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    # Map back to user IDs
    similar_users = [(idx_to_user[idx], similarities[idx]) 
                     for idx in similar_indices]
    
    return similar_users

# Example: Find users similar to user 1
similar_to_user1 = find_similar_users(1, user_similarity, user_to_idx, top_k=3)
print(f"\\nUsers similar to User 1:")
for user_id, similarity in similar_to_user1:
    print(f"  User {user_id}: similarity = {similarity:.3f}")

# 2.3 Predict Rating - User-Based
def predict_rating_user_based(user_id, item_id, user_item_matrix, 
                               user_similarity, user_to_idx, item_to_idx, 
                               top_k=5):
    """
    Predict rating using user-based collaborative filtering
    
    Formula: r̂_{u,i} = r̄_u + Σ_{v∈N(u)} sim(u,v) · (r_{v,i} - r̄_v) / Σ_{v∈N(u)} |sim(u,v)|
    """
    user_idx = user_to_idx[user_id]
    item_idx = item_to_idx[item_id]
    
    # Get user's average rating
    user_ratings = user_item_matrix.iloc[user_idx].values
    user_mean = np.mean(user_ratings[user_ratings > 0]) if np.any(user_ratings > 0) else 0
    
    # Find similar users
    similarities = user_similarity[user_idx]
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    # Calculate weighted average
    numerator = 0
    denominator = 0
    
    for similar_user_idx in similar_indices:
        similarity = similarities[similar_user_idx]
        
        # Skip if similarity is 0 or negative
        if similarity <= 0:
            continue
        
        # Get similar user's rating for this item
        similar_user_ratings = user_item_matrix.iloc[similar_user_idx].values
        similar_user_mean = np.mean(similar_user_ratings[similar_user_ratings > 0]) if np.any(similar_user_ratings > 0) else 0
        
        rating = user_item_matrix.iloc[similar_user_idx, item_idx]
        
        # Only use if user has rated this item
        if rating > 0:
            numerator += similarity * (rating - similar_user_mean)
            denominator += abs(similarity)
    
    # Predict rating
    if denominator > 0:
        predicted_rating = user_mean + (numerator / denominator)
    else:
        predicted_rating = user_mean
    
    # Clip to valid rating range (e.g., 1-5)
    predicted_rating = np.clip(predicted_rating, 1, 5)
    
    return predicted_rating

# Example: Predict rating for User 1, Movie 104
predicted = predict_rating_user_based(
    1, 104, user_item_matrix, user_similarity, 
    user_to_idx, item_to_idx, top_k=3
)
print(f"\\nPredicted rating (User 1, Movie 104): {predicted:.2f}")

# 2.4 Generate Recommendations - User-Based
def recommend_items_user_based(user_id, user_item_matrix, user_similarity,
                               user_to_idx, item_to_idx, idx_to_item, top_n=10):
    """
    Generate top-N recommendations for a user
    """
    user_idx = user_to_idx[user_id]
    user_ratings = user_item_matrix.iloc[user_idx].values
    
    # Get items user hasn't rated yet
    unrated_items = [item_idx for item_idx, rating in enumerate(user_ratings) 
                     if rating == 0]
    
    # Predict ratings for unrated items
    predictions = []
    for item_idx in unrated_items:
        item_id = idx_to_item[item_idx]
        predicted_rating = predict_rating_user_based(
            user_id, item_id, user_item_matrix, user_similarity,
            user_to_idx, item_to_idx
        )
        predictions.append((item_id, predicted_rating))
    
    # Sort by predicted rating (descending)
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-N
    return predictions[:top_n]

# Example: Recommend top 3 items for User 1
recommendations = recommend_items_user_based(
    1, user_item_matrix, user_similarity,
    user_to_idx, item_to_idx, idx_to_item, top_n=3
)
print(f"\\nTop 3 Recommendations for User 1:")
for item_id, predicted_rating in recommendations:
    print(f"  Movie {item_id}: predicted rating = {predicted_rating:.2f}")`
    },
    step3: {
      title: 'Step 3: Collaborative Filtering - Item-Based',
      description: 'Implement item-based collaborative filtering using item similarity',
      code: `# Step 3: Collaborative Filtering - Item-Based
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 3.1 Item-Based Collaborative Filtering
# Core Idea: Items similar to items a user liked will also be liked
# 
# Algorithm:
# 1. Find items similar to items user has rated
# 2. Predict rating = weighted average of similar items' ratings
# 
# Formula: r̂_{u,i} = Σ_{j∈R(u)} sim(i,j) · r_{u,j} / Σ_{j∈R(u)} |sim(i,j)|
# where:
# - r̂_{u,i} = predicted rating user u gives to item i
# - R(u) = set of items user u has rated
# - sim(i,j) = similarity between items i and j
# - r_{u,j} = rating user u gave to item j

def calculate_item_similarity(user_item_matrix, method='cosine'):
    """
    Calculate similarity between items
    
    Transpose matrix: items become rows, users become columns
    Then calculate similarity between item vectors
    """
    # Transpose: items as rows, users as columns
    item_user_matrix = user_item_matrix.T
    
    if method == 'cosine':
        similarity_matrix = cosine_similarity(item_user_matrix.values)
    
    elif method == 'pearson':
        similarity_matrix = np.corrcoef(item_user_matrix.values)
        similarity_matrix = np.nan_to_num(similarity_matrix)
    
    return similarity_matrix

item_similarity = calculate_item_similarity(user_item_matrix, method='cosine')
print("Item Similarity Matrix:")
print(item_similarity)

# 3.2 Find Similar Items
def find_similar_items(item_id, item_similarity, item_to_idx, idx_to_item, top_k=5):
    """
    Find top-k most similar items to a given item
    """
    item_idx = item_to_idx[item_id]
    
    # Get similarities for this item
    similarities = item_similarity[item_idx]
    
    # Sort by similarity (descending), exclude self
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    # Map back to item IDs
    similar_items = [(idx_to_item[idx], similarities[idx]) 
                     for idx in similar_indices]
    
    return similar_items

# Example: Find items similar to Movie 101
similar_to_movie101 = find_similar_items(101, item_similarity, item_to_idx, idx_to_item, top_k=3)
print(f"\\nItems similar to Movie 101:")
for item_id, similarity in similar_to_movie101:
    print(f"  Movie {item_id}: similarity = {similarity:.3f}")

# 3.3 Predict Rating - Item-Based
def predict_rating_item_based(user_id, item_id, user_item_matrix,
                              item_similarity, user_to_idx, item_to_idx, top_k=5):
    """
    Predict rating using item-based collaborative filtering
    
    Formula: r̂_{u,i} = Σ_{j∈R(u)} sim(i,j) · r_{u,j} / Σ_{j∈R(u)} |sim(i,j)|
    """
    user_idx = user_to_idx[user_id]
    item_idx = item_to_idx[item_id]
    
    # Get user's ratings
    user_ratings = user_item_matrix.iloc[user_idx].values
    
    # Get items user has rated
    rated_items = [idx for idx, rating in enumerate(user_ratings) if rating > 0]
    
    # Get similarities for target item
    item_similarities = item_similarity[item_idx]
    
    # Calculate weighted average
    numerator = 0
    denominator = 0
    
    for rated_item_idx in rated_items:
        similarity = item_similarities[rated_item_idx]
        
        # Skip if similarity is 0 or negative
        if similarity <= 0:
            continue
        
        rating = user_ratings[rated_item_idx]
        numerator += similarity * rating
        denominator += abs(similarity)
    
    # Predict rating
    if denominator > 0:
        predicted_rating = numerator / denominator
    else:
        # Fallback: use average rating
        predicted_rating = np.mean(user_ratings[user_ratings > 0]) if np.any(user_ratings > 0) else 0
    
    # Clip to valid rating range
    predicted_rating = np.clip(predicted_rating, 1, 5)
    
    return predicted_rating

# Example: Predict rating for User 1, Movie 104
predicted = predict_rating_item_based(
    1, 104, user_item_matrix, item_similarity,
    user_to_idx, item_to_idx, top_k=3
)
print(f"\\nPredicted rating (User 1, Movie 104): {predicted:.2f}")

# 3.4 Generate Recommendations - Item-Based
def recommend_items_item_based(user_id, user_item_matrix, item_similarity,
                               user_to_idx, item_to_idx, idx_to_item, top_n=10):
    """
    Generate top-N recommendations for a user using item-based CF
    """
    user_idx = user_to_idx[user_id]
    user_ratings = user_item_matrix.iloc[user_idx].values
    
    # Get items user hasn't rated yet
    unrated_items = [item_idx for item_idx, rating in enumerate(user_ratings) 
                     if rating == 0]
    
    # Predict ratings for unrated items
    predictions = []
    for item_idx in unrated_items:
        item_id = idx_to_item[item_idx]
        predicted_rating = predict_rating_item_based(
            user_id, item_id, user_item_matrix, item_similarity,
            user_to_idx, item_to_idx
        )
        predictions.append((item_id, predicted_rating))
    
    # Sort by predicted rating (descending)
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-N
    return predictions[:top_n]

# Example: Recommend top 3 items for User 1
recommendations = recommend_items_item_based(
    1, user_item_matrix, item_similarity,
    user_to_idx, item_to_idx, idx_to_item, top_n=3
)
print(f"\\nTop 3 Recommendations for User 1 (Item-Based):")
for item_id, predicted_rating in recommendations:
    print(f"  Movie {item_id}: predicted rating = {predicted_rating:.2f}")

# 3.5 Comparison: User-Based vs Item-Based
# 
# User-Based CF:
# - Pros: Better for diverse user preferences, handles new items well
# - Cons: Computationally expensive, sparse user-user matrix
# 
# Item-Based CF:
# - Pros: More stable (items change less than users), faster
# - Cons: Struggles with new items (cold start problem)
# 
# Generally, Item-Based CF performs better in practice (Amazon uses it)`
    },
    step4: {
      title: 'Step 4: Matrix Factorization (SVD)',
      description: 'Implement matrix factorization using Singular Value Decomposition',
      code: `# Step 4: Matrix Factorization (SVD)
import numpy as np
from scipy.sparse.linalg import svds

# 4.1 Matrix Factorization Concept
# 
# Goal: Decompose user-item matrix R into lower-dimensional matrices
# 
# Formula: R ≈ U × S × V^T
# where:
# - R: User-Item matrix (m × n)
# - U: User features matrix (m × k)
# - S: Singular values (diagonal matrix, k × k)
# - V: Item features matrix (n × k)
# - k: Number of latent factors (k << min(m, n))
# 
# Prediction: r̂_{u,i} = U[u] · S · V[i]^T

def matrix_factorization_svd(user_item_matrix, n_factors=5):
    """
    Factorize user-item matrix using SVD
    
    SVD decomposes matrix into:
    - U: User embeddings (latent factors)
    - S: Singular values (importance of factors)
    - V: Item embeddings (latent factors)
    """
    # Convert to numpy array
    R = user_item_matrix.values
    
    # Perform SVD
    # Note: For sparse matrices, use scipy.sparse.linalg.svds
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    
    # Keep only top n_factors
    U_k = U[:, :n_factors]
    S_k = np.diag(S[:n_factors])
    Vt_k = Vt[:n_factors, :]
    
    # Reconstruct matrix
    R_predicted = U_k @ S_k @ Vt_k
    
    return U_k, S_k, Vt_k, R_predicted

# Example: Factorize with 3 latent factors
U, S, Vt, R_predicted = matrix_factorization_svd(user_item_matrix, n_factors=3)
print("User Embeddings (U):")
print(U)
print(f"\\nShape: {U.shape}")
print("\\nSingular Values (S):")
print(S)
print("\\nItem Embeddings (V^T):")
print(Vt)

# 4.2 Predict Rating - Matrix Factorization
def predict_rating_mf(user_id, item_id, U, S, Vt, user_to_idx, item_to_idx):
    """
    Predict rating using matrix factorization
    
    Formula: r̂_{u,i} = U[u] · S · V[i]^T
    """
    user_idx = user_to_idx[user_id]
    item_idx = item_to_idx[item_id]
    
    # Get user and item embeddings
    user_embedding = U[user_idx, :]  # (k,)
    item_embedding = Vt[:, item_idx]  # (k,)
    
    # Calculate prediction: user_embedding · S · item_embedding
    prediction = user_embedding @ S @ item_embedding
    
    # Clip to valid rating range
    prediction = np.clip(prediction, 1, 5)
    
    return prediction

# Example: Predict rating for User 1, Movie 104
predicted = predict_rating_mf(1, 104, U, S, Vt, user_to_idx, item_to_idx)
print(f"\\nPredicted rating (User 1, Movie 104): {predicted:.2f}")

# 4.3 Generate Recommendations - Matrix Factorization
def recommend_items_mf(user_id, user_item_matrix, U, S, Vt,
                       user_to_idx, item_to_idx, idx_to_item, top_n=10):
    """
    Generate top-N recommendations using matrix factorization
    """
    user_idx = user_to_idx[user_id]
    user_ratings = user_item_matrix.iloc[user_idx].values
    
    # Get items user hasn't rated yet
    unrated_items = [item_idx for item_idx, rating in enumerate(user_ratings) 
                     if rating == 0]
    
    # Predict ratings for unrated items
    predictions = []
    for item_idx in unrated_items:
        item_id = idx_to_item[item_idx]
        predicted_rating = predict_rating_mf(
            user_id, item_id, U, S, Vt, user_to_idx, item_to_idx
        )
        predictions.append((item_id, predicted_rating))
    
    # Sort by predicted rating (descending)
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-N
    return predictions[:top_n]

# Example: Recommend top 3 items for User 1
recommendations = recommend_items_mf(
    1, user_item_matrix, U, S, Vt,
    user_to_idx, item_to_idx, idx_to_item, top_n=3
)
print(f"\\nTop 3 Recommendations for User 1 (Matrix Factorization):")
for item_id, predicted_rating in recommendations:
    print(f"  Movie {item_id}: predicted rating = {predicted_rating:.2f}")

# 4.4 Understanding Latent Factors
# 
# Latent factors are hidden features that explain user preferences
# Examples:
# - Factor 1: Action vs Drama preference
# - Factor 2: Sci-Fi vs Romance preference
# - Factor 3: Recent vs Classic movies
# 
# User embeddings: How much user likes each factor
# Item embeddings: How much item belongs to each factor
# 
# Example interpretation:
# User 1 embedding: [0.8, 0.6, 0.2]
#   → Likes action (0.8), likes sci-fi (0.6), prefers recent (0.2)
# Movie 101 embedding: [0.9, 0.7, 0.3]
#   → High action (0.9), high sci-fi (0.7), recent (0.3)
# Prediction = 0.8×0.9 + 0.6×0.7 + 0.2×0.3 = 0.72 + 0.42 + 0.06 = 1.2
# (After scaling, this becomes a rating like 4.2)

# 4.5 Choosing Number of Factors
# 
# Trade-off:
# - Too few factors: Underfitting, poor predictions
# - Too many factors: Overfitting, poor generalization
# 
# Common approach: Use cross-validation to find optimal k
# 
# Rule of thumb: k ≈ √(min(m, n))
# where m = number of users, n = number of items`
    },
    step5: {
      title: 'Step 5: Deep Learning Recommendation System',
      description: 'Build neural network-based recommendation system using embeddings',
      code: `# Step 5: Deep Learning Recommendation System
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 5.1 Neural Collaborative Filtering (NCF)
# 
# Architecture:
# - User Embedding Layer
# - Item Embedding Layer
# - Multi-Layer Perceptron (MLP)
# - Output Layer (prediction)
# 
# Formula: r̂_{u,i} = MLP(concat(user_embedding, item_embedding))

class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering Model
    
    Combines matrix factorization with deep learning
    """
    def __init__(self, num_users, num_items, embedding_dim=50, hidden_dims=[64, 32, 16]):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        layers = []
        input_dim = embedding_dim * 2  # Concatenated user + item embeddings
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())  # Output between 0 and 1
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        """
        Forward pass
        
        Parameters:
        - user_ids: Tensor of user IDs (batch_size,)
        - item_ids: Tensor of item IDs (batch_size,)
        
        Returns:
        - predictions: Tensor of predicted ratings (batch_size, 1)
        """
        # Get embeddings
        user_emb = self.user_embedding(user_ids)  # (batch_size, embedding_dim)
        item_emb = self.item_embedding(item_ids)  # (batch_size, embedding_dim)
        
        # Concatenate embeddings
        concat_emb = torch.cat([user_emb, item_emb], dim=1)  # (batch_size, 2*embedding_dim)
        
        # Pass through MLP
        output = self.mlp(concat_emb)  # (batch_size, 1)
        
        # Scale to rating range (1-5)
        output = output * 4 + 1  # Maps [0, 1] to [1, 5]
        
        return output

# 5.2 Dataset Class
class RatingDataset(Dataset):
    """
    PyTorch Dataset for ratings
    """
    def __init__(self, df_ratings, user_to_idx, item_to_idx):
        self.user_ids = [user_to_idx[uid] for uid in df_ratings['userId']]
        self.item_ids = [item_to_idx[iid] for iid in df_ratings['movieId']]
        self.ratings = df_ratings['rating'].values
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return (
            torch.LongTensor([self.user_ids[idx]])[0],
            torch.LongTensor([self.item_ids[idx]])[0],
            torch.FloatTensor([self.ratings[idx]])[0]
        )

# 5.3 Training Function
def train_ncf_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    """
    Train Neural Collaborative Filtering model
    
    Loss Function: Mean Squared Error (MSE)
    Optimizer: Adam
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for user_ids, item_ids, ratings in train_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            
            # Forward pass
            predictions = model(user_ids, item_ids).squeeze()
            loss = criterion(predictions, ratings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in val_loader:
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.to(device)
                
                predictions = model(user_ids, item_ids).squeeze()
                loss = criterion(predictions, ratings)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

# 5.4 Create Data Loaders
def create_data_loaders(train_df, test_df, user_to_idx, item_to_idx, batch_size=32):
    """Create PyTorch data loaders"""
    train_dataset = RatingDataset(train_df, user_to_idx, item_to_idx)
    val_dataset = RatingDataset(test_df, user_to_idx, item_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# 5.5 Complete Training Pipeline
def train_deep_learning_recommender(train_df, test_df, user_to_idx, item_to_idx,
                                    num_users, num_items, embedding_dim=50):
    """
    Complete pipeline to train deep learning recommender
    """
    # Create model
    model = NeuralCollaborativeFiltering(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        hidden_dims=[64, 32, 16]
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_df, test_df, user_to_idx, item_to_idx
    )
    
    # Train model
    print("Training Neural Collaborative Filtering model...")
    train_losses, val_losses = train_ncf_model(model, train_loader, val_loader)
    
    return model, train_losses, val_losses

# Example usage:
# model, train_losses, val_losses = train_deep_learning_recommender(
#     train_df, test_df, user_to_idx, item_to_idx,
#     num_users=len(user_to_idx),
#     num_items=len(item_to_idx),
#     embedding_dim=50
# )

# 5.6 Generate Recommendations - Deep Learning
def recommend_items_dl(user_id, user_item_matrix, model, user_to_idx, item_to_idx,
                      idx_to_item, top_n=10, device='cpu'):
    """
    Generate top-N recommendations using deep learning model
    """
    model.eval()
    user_idx = user_to_idx[user_id]
    user_ratings = user_item_matrix.iloc[user_idx].values
    
    # Get items user hasn't rated yet
    unrated_items = [item_idx for item_idx, rating in enumerate(user_ratings) 
                     if rating == 0]
    
    # Predict ratings for unrated items
    predictions = []
    
    with torch.no_grad():
        for item_idx in unrated_items:
            item_id = idx_to_item[item_idx]
            
            user_tensor = torch.LongTensor([user_idx]).to(device)
            item_tensor = torch.LongTensor([item_idx]).to(device)
            
            predicted_rating = model(user_tensor, item_tensor).item()
            predictions.append((item_id, predicted_rating))
    
    # Sort by predicted rating (descending)
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-N
    return predictions[:top_n]`
    },
    step6: {
      title: 'Step 6: Evaluation Metrics',
      description: 'Evaluate recommendation system performance using various metrics',
      code: `# Step 6: Evaluation Metrics
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 6.1 Rating Prediction Metrics
# 
# These metrics evaluate how well the model predicts ratings

def calculate_rmse(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE)
    
    Formula: RMSE = √(Σ(ŷ_i - y_i)² / n)
    
    Lower is better. Penalizes large errors more.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def calculate_mae(y_true, y_pred):
    """
    Mean Absolute Error (MAE)
    
    Formula: MAE = (1/n) Σ|ŷ_i - y_i|
    
    Lower is better. Less sensitive to outliers than RMSE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    return mae

# Example:
# True ratings: [5, 4, 3, 5, 2]
# Predicted ratings: [4.5, 4.2, 3.1, 4.8, 2.2]
# 
# RMSE calculation:
# Squared errors: [(4.5-5)², (4.2-4)², (3.1-3)², (4.8-5)², (2.2-2)²]
#                = [0.25, 0.04, 0.01, 0.04, 0.04]
# Mean = (0.25 + 0.04 + 0.01 + 0.04 + 0.04) / 5 = 0.076
# RMSE = √0.076 ≈ 0.276
# 
# MAE calculation:
# Absolute errors: [|4.5-5|, |4.2-4|, |3.1-3|, |4.8-5|, |2.2-2|]
#                 = [0.5, 0.2, 0.1, 0.2, 0.2]
# MAE = (0.5 + 0.2 + 0.1 + 0.2 + 0.2) / 5 = 0.24

# 6.2 Ranking Metrics
# 
# These metrics evaluate the quality of recommendations (top-N lists)

def calculate_precision_at_k(y_true, y_pred, k=10):
    """
    Precision@K: Fraction of recommended items that are relevant
    
    Formula: Precision@K = |Relevant items in top-K| / K
    
    Higher is better (0-1 range).
    """
    # Get top-K predicted items
    top_k_pred = set(y_pred[:k])
    
    # Get relevant items (items user actually rated highly, e.g., >= 4)
    relevant_items = set([item for item, rating in y_true.items() if rating >= 4])
    
    # Calculate precision
    if len(top_k_pred) == 0:
        return 0.0
    
    relevant_recommended = len(top_k_pred & relevant_items)
    precision = relevant_recommended / len(top_k_pred)
    
    return precision

def calculate_recall_at_k(y_true, y_pred, k=10):
    """
    Recall@K: Fraction of relevant items that are recommended
    
    Formula: Recall@K = |Relevant items in top-K| / |All relevant items|
    
    Higher is better (0-1 range).
    """
    # Get top-K predicted items
    top_k_pred = set(y_pred[:k])
    
    # Get relevant items
    relevant_items = set([item for item, rating in y_true.items() if rating >= 4])
    
    # Calculate recall
    if len(relevant_items) == 0:
        return 0.0
    
    relevant_recommended = len(top_k_pred & relevant_items)
    recall = relevant_recommended / len(relevant_items)
    
    return recall

def calculate_f1_at_k(y_true, y_pred, k=10):
    """
    F1@K: Harmonic mean of Precision@K and Recall@K
    
    Formula: F1@K = 2 × (Precision@K × Recall@K) / (Precision@K + Recall@K)
    
    Higher is better (0-1 range).
    """
    precision = calculate_precision_at_k(y_true, y_pred, k)
    recall = calculate_recall_at_k(y_true, y_pred, k)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Example:
# User's true ratings: {101: 5, 102: 4, 103: 3, 104: 5, 105: 2}
# Recommended items (top-3): [104, 101, 106]
# Relevant items (rating >= 4): {101, 102, 104}
# 
# Precision@3:
# Relevant in top-3: {104, 101} (2 items)
# Precision = 2/3 = 0.667
# 
# Recall@3:
# Relevant in top-3: {104, 101} (2 items)
# Total relevant: {101, 102, 104} (3 items)
# Recall = 2/3 = 0.667
# 
# F1@3:
# F1 = 2 × (0.667 × 0.667) / (0.667 + 0.667) = 0.667

# 6.3 Mean Average Precision (MAP)
def calculate_map(y_true_list, y_pred_list, k=10):
    """
    Mean Average Precision (MAP@K)
    
    Average precision across all users
    
    Formula: MAP@K = (1/|U|) Σ_u AP@K(u)
    where AP@K(u) = (1/|R_u|) Σ_{i=1}^K P@i × rel(i)
    """
    ap_scores = []
    
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        relevant_items = set([item for item, rating in y_true.items() if rating >= 4])
        
        if len(relevant_items) == 0:
            continue
        
        top_k_pred = y_pred[:k]
        precision_scores = []
        
        relevant_count = 0
        for i, item in enumerate(top_k_pred, 1):
            if item in relevant_items:
                relevant_count += 1
                precision = relevant_count / i
                precision_scores.append(precision)
        
        if len(precision_scores) > 0:
            ap = np.mean(precision_scores)
            ap_scores.append(ap)
    
    if len(ap_scores) == 0:
        return 0.0
    
    map_score = np.mean(ap_scores)
    return map_score

# 6.4 Normalized Discounted Cumulative Gain (NDCG)
def calculate_ndcg_at_k(y_true, y_pred, k=10):
    """
    Normalized Discounted Cumulative Gain (NDCG@K)
    
    Measures ranking quality, giving more weight to top positions
    
    Formula: NDCG@K = DCG@K / IDCG@K
    where:
    - DCG@K = Σ_{i=1}^K rel(i) / log₂(i+1)
    - IDCG@K = DCG@K of ideal ranking
    """
    # Get top-K predicted items
    top_k_pred = y_pred[:k]
    
    # Get relevance scores (ratings)
    relevance_scores = [y_true.get(item, 0) for item in top_k_pred]
    
    # Calculate DCG
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
    
    # Calculate IDCG (ideal DCG - sorted by relevance descending)
    ideal_relevance = sorted([rating for rating in y_true.values()], reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
    
    # Calculate NDCG
    if idcg == 0:
        return 0.0
    
    ndcg = dcg / idcg
    return ndcg

# Example:
# True ratings: {101: 5, 102: 4, 103: 3}
# Predicted order: [101, 103, 102]
# 
# DCG@3:
# DCG = 5/log₂(2) + 3/log₂(3) + 4/log₂(4)
#     = 5/1 + 3/1.585 + 4/2
#     = 5 + 1.893 + 2 = 8.893
# 
# IDCG@3 (ideal order: [101, 102, 103]):
# IDCG = 5/log₂(2) + 4/log₂(3) + 3/log₂(4)
#      = 5/1 + 4/1.585 + 3/2
#      = 5 + 2.524 + 1.5 = 9.024
# 
# NDCG@3 = 8.893 / 9.024 ≈ 0.986

# 6.5 Evaluate Recommendation System
def evaluate_recommendation_system(model, test_df, user_item_matrix,
                                  user_to_idx, item_to_idx, idx_to_item,
                                  predict_function, top_k=10):
    """
    Comprehensive evaluation of recommendation system
    """
    # Rating prediction metrics
    y_true_ratings = []
    y_pred_ratings = []
    
    # Ranking metrics
    precision_scores = []
    recall_scores = []
    f1_scores = []
    ndcg_scores = []
    
    # Evaluate for each user in test set
    for user_id in test_df['userId'].unique():
        user_test = test_df[test_df['userId'] == user_id]
        
        # Get true ratings for this user
        user_true_ratings = {}
        for _, row in user_test.iterrows():
            user_true_ratings[row['movieId']] = row['rating']
        
        # Generate recommendations
        recommendations = predict_function(
            user_id, user_item_matrix, model, user_to_idx, item_to_idx, idx_to_item, top_k
        )
        
        # Extract predicted items and ratings
        predicted_items = [item_id for item_id, _ in recommendations]
        predicted_ratings = [rating for _, rating in recommendations]
        
        # Rating prediction metrics
        for _, row in user_test.iterrows():
            item_id = row['movieId']
            true_rating = row['rating']
            
            # Find predicted rating
            pred_rating = next((r for i, r in recommendations if i == item_id), None)
            if pred_rating:
                y_true_ratings.append(true_rating)
                y_pred_ratings.append(pred_rating)
        
        # Ranking metrics
        precision = calculate_precision_at_k(user_true_ratings, predicted_items, top_k)
        recall = calculate_recall_at_k(user_true_ratings, predicted_items, top_k)
        f1 = calculate_f1_at_k(user_true_ratings, predicted_items, top_k)
        ndcg = calculate_ndcg_at_k(user_true_ratings, predicted_items, top_k)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        ndcg_scores.append(ndcg)
    
    # Calculate averages
    metrics = {
        'RMSE': calculate_rmse(y_true_ratings, y_pred_ratings) if y_true_ratings else None,
        'MAE': calculate_mae(y_true_ratings, y_pred_ratings) if y_true_ratings else None,
        'Precision@K': np.mean(precision_scores),
        'Recall@K': np.mean(recall_scores),
        'F1@K': np.mean(f1_scores),
        'NDCG@K': np.mean(ndcg_scores)
    }
    
    return metrics

# Example usage:
# metrics = evaluate_recommendation_system(
#     model, test_df, user_item_matrix,
#     user_to_idx, item_to_idx, idx_to_item,
#     recommend_items_dl, top_k=10
# )
# print("Evaluation Metrics:")
# for metric, value in metrics.items():
#     print(f"  {metric}: {value:.4f}")`
    },
    step7: {
      title: 'Step 7: Complete Recommendation System Integration',
      description: 'Integrate all components into a complete production-ready system',
      code: `# Step 7: Complete Recommendation System Integration
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

class CompleteRecommendationSystem:
    """
    Complete recommendation system integrating all components:
    1. Data preprocessing
    2. User-based CF
    3. Item-based CF
    4. Matrix Factorization
    5. Deep Learning
    6. Evaluation
    7. Production deployment
    """
    def __init__(self, method='hybrid'):
        """
        Initialize recommendation system
        
        Methods:
        - 'user_based': User-based collaborative filtering
        - 'item_based': Item-based collaborative filtering
        - 'matrix_factorization': SVD-based matrix factorization
        - 'deep_learning': Neural Collaborative Filtering
        - 'hybrid': Combine multiple methods
        """
        self.method = method
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.U = None
        self.S = None
        self.Vt = None
        self.dl_model = None
        self.user_to_idx = None
        self.item_to_idx = None
        self.idx_to_user = None
        self.idx_to_item = None
        
    def load_and_preprocess_data(self, df_ratings, df_movies=None):
        """Step 1: Load and preprocess data"""
        print("Step 1: Loading and preprocessing data...")
        
        # Create mappings
        self.user_to_idx, self.item_to_idx, self.idx_to_user, self.idx_to_item = create_mappings(df_ratings)
        
        # Create user-item matrix
        self.user_item_matrix = create_user_item_matrix(df_ratings)
        
        # Store movie metadata if provided
        self.df_movies = df_movies
        
        print(f"  Users: {len(self.user_to_idx)}, Items: {len(self.item_to_idx)}")
        print(f"  Matrix shape: {self.user_item_matrix.shape}")
        
        return self.user_item_matrix
    
    def train_user_based_cf(self):
        """Step 2: Train user-based collaborative filtering"""
        print("Step 2: Training user-based CF...")
        self.user_similarity = calculate_user_similarity(self.user_item_matrix, method='cosine')
        print("  User similarity matrix computed")
    
    def train_item_based_cf(self):
        """Step 3: Train item-based collaborative filtering"""
        print("Step 3: Training item-based CF...")
        self.item_similarity = calculate_item_similarity(self.user_item_matrix, method='cosine')
        print("  Item similarity matrix computed")
    
    def train_matrix_factorization(self, n_factors=50):
        """Step 4: Train matrix factorization"""
        print(f"Step 4: Training matrix factorization (n_factors={n_factors})...")
        self.U, self.S, self.Vt, _ = matrix_factorization_svd(
            self.user_item_matrix, n_factors=n_factors
        )
        print(f"  Factorization complete: U{self.U.shape}, S{self.S.shape}, Vt{self.Vt.shape}")
    
    def train_deep_learning(self, train_df, test_df, embedding_dim=50, epochs=10):
        """Step 5: Train deep learning model"""
        print("Step 5: Training deep learning model...")
        
        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)
        
        self.dl_model, _, _ = train_deep_learning_recommender(
            train_df, test_df, self.user_to_idx, self.item_to_idx,
            num_users, num_items, embedding_dim=embedding_dim
        )
        print("  Deep learning model trained")
    
    def recommend(self, user_id, top_n=10, method=None):
        """
        Generate recommendations for a user
        
        Parameters:
        - user_id: User ID
        - top_n: Number of recommendations
        - method: Override default method ('user_based', 'item_based', 'mf', 'dl', 'hybrid')
        """
        method = method or self.method
        
        if method == 'user_based':
            recommendations = recommend_items_user_based(
                user_id, self.user_item_matrix, self.user_similarity,
                self.user_to_idx, self.item_to_idx, self.idx_to_item, top_n
            )
        
        elif method == 'item_based':
            recommendations = recommend_items_item_based(
                user_id, self.user_item_matrix, self.item_similarity,
                self.user_to_idx, self.item_to_idx, self.idx_to_item, top_n
            )
        
        elif method == 'mf':
            recommendations = recommend_items_mf(
                user_id, self.user_item_matrix, self.U, self.S, self.Vt,
                self.user_to_idx, self.item_to_idx, self.idx_to_item, top_n
            )
        
        elif method == 'dl':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            recommendations = recommend_items_dl(
                user_id, self.user_item_matrix, self.dl_model,
                self.user_to_idx, self.item_to_idx, self.idx_to_item, top_n, device
            )
        
        elif method == 'hybrid':
            # Combine multiple methods
            recommendations = self._hybrid_recommend(user_id, top_n)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Add movie titles if available
        if self.df_movies is not None:
            recommendations_with_titles = []
            for item_id, rating in recommendations:
                movie_info = self.df_movies[self.df_movies['movieId'] == item_id]
                title = movie_info['title'].values[0] if len(movie_info) > 0 else f"Movie {item_id}"
                recommendations_with_titles.append((item_id, title, rating))
            return recommendations_with_titles
        
        return recommendations
    
    def _hybrid_recommend(self, user_id, top_n):
        """Hybrid recommendation combining multiple methods"""
        # Get recommendations from different methods
        user_based_recs = recommend_items_user_based(
            user_id, self.user_item_matrix, self.user_similarity,
            self.user_to_idx, self.item_to_idx, self.idx_to_item, top_n * 2
        )
        
        item_based_recs = recommend_items_item_based(
            user_id, self.user_item_matrix, self.item_similarity,
            self.user_to_idx, self.item_to_idx, self.idx_to_item, top_n * 2
        )
        
        mf_recs = recommend_items_mf(
            user_id, self.user_item_matrix, self.U, self.S, self.Vt,
            self.user_to_idx, self.item_to_idx, self.idx_to_item, top_n * 2
        )
        
        # Combine and re-rank
        combined_scores = {}
        for item_id, score in user_based_recs:
            combined_scores[item_id] = combined_scores.get(item_id, 0) + score * 0.3
        
        for item_id, score in item_based_recs:
            combined_scores[item_id] = combined_scores.get(item_id, 0) + score * 0.3
        
        for item_id, score in mf_recs:
            combined_scores[item_id] = combined_scores.get(item_id, 0) + score * 0.4
        
        # Sort by combined score
        sorted_recommendations = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_n]
        
        return sorted_recommendations
    
    def evaluate(self, test_df, top_k=10):
        """Evaluate system performance"""
        print("Evaluating recommendation system...")
        
        metrics = evaluate_recommendation_system(
            self, test_df, self.user_item_matrix,
            self.user_to_idx, self.item_to_idx, self.idx_to_item,
            self.recommend, top_k=top_k
        )
        
        return metrics
    
    def save_model(self, filepath):
        """Save trained model to disk"""
        model_data = {
            'method': self.method,
            'user_item_matrix': self.user_item_matrix,
            'user_similarity': self.user_similarity,
            'item_similarity': self.item_similarity,
            'U': self.U,
            'S': self.S,
            'Vt': self.Vt,
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'idx_to_user': self.idx_to_user,
            'idx_to_item': self.idx_to_item
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.method = model_data['method']
        self.user_item_matrix = model_data['user_item_matrix']
        self.user_similarity = model_data['user_similarity']
        self.item_similarity = model_data['item_similarity']
        self.U = model_data['U']
        self.S = model_data['S']
        self.Vt = model_data['Vt']
        self.user_to_idx = model_data['user_to_idx']
        self.item_to_idx = model_data['item_to_idx']
        self.idx_to_user = model_data['idx_to_user']
        self.idx_to_item = model_data['idx_to_item']
        
        print(f"Model loaded from {filepath}")

# 7.1 Complete Usage Example
def run_complete_recommendation_system():
    """Run complete recommendation system pipeline"""
    
    # Initialize system
    system = CompleteRecommendationSystem(method='hybrid')
    
    # Step 1: Load data
    system.load_and_preprocess_data(df_ratings, df_movies)
    
    # Step 2-4: Train models
    system.train_user_based_cf()
    system.train_item_based_cf()
    system.train_matrix_factorization(n_factors=5)
    
    # Step 5: Train deep learning (optional)
    # system.train_deep_learning(train_df, test_df)
    
    # Step 6: Generate recommendations
    recommendations = system.recommend(user_id=1, top_n=10)
    print("\\n=== RECOMMENDATIONS FOR USER 1 ===")
    for item_id, title, rating in recommendations:
        print(f"  {title}: {rating:.2f}")
    
    # Step 7: Evaluate
    metrics = system.evaluate(test_df, top_k=10)
    print("\\n=== EVALUATION METRICS ===")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    system.save_model('recommendation_model.pkl')
    
    return system

# Run the complete system
# system = run_complete_recommendation_system()`
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200 mb-4">
        <h2 className="text-2xl font-bold text-blue-900 mb-2">Recommendation System - Complete Tutorial</h2>
        <p className="text-blue-800">
          A comprehensive step-by-step guide to building AI-powered recommendation systems. 
          Learn collaborative filtering (user-based and item-based), matrix factorization (SVD), 
          deep learning approaches, evaluation metrics, and complete system integration.
        </p>
      </div>

      {/* Framework Selector */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Select Framework
        </label>
        <select
          value={selectedFramework}
          onChange={(e) => setSelectedFramework(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
        >
          <option value="pytorch">PyTorch</option>
          <option value="tensorflow">TensorFlow</option>
        </select>
      </div>

      {/* Step Selector */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Select Tutorial Step
        </label>
        <select
          value={selectedStep}
          onChange={(e) => setSelectedStep(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
        >
          <option value="step1">Step 1: Data Collection & Preprocessing</option>
          <option value="step2">Step 2: Collaborative Filtering - User-Based</option>
          <option value="step3">Step 3: Collaborative Filtering - Item-Based</option>
          <option value="step4">Step 4: Matrix Factorization (SVD)</option>
          <option value="step5">Step 5: Deep Learning Recommendation System</option>
          <option value="step6">Step 6: Evaluation Metrics</option>
          <option value="step7">Step 7: Complete System Integration</option>
        </select>
      </div>

      {/* Step Content */}
      <div className="bg-white rounded-lg p-6 shadow-md">
        <h3 className="text-xl font-bold text-gray-900 mb-2">
          {steps[selectedStep].title}
        </h3>
        <p className="text-gray-600 mb-4">
          {steps[selectedStep].description}
        </p>
        
        <div className="mt-4">
          <SyntaxHighlighter language="python" style={vscDarkPlus} showLineNumbers>
            {steps[selectedStep].code}
          </SyntaxHighlighter>
        </div>
      </div>
    </div>
  );
}

