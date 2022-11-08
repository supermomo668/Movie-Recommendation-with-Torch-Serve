import numpy as np, torch
import os, logging, pickle


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_movies, n_factors=60):
        super().__init__()
        # create user embeddings
        self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=True)
        # create item embeddings
        self.movie_factors = torch.nn.Embedding(n_movies, n_factors, sparse=True)

    def forward(self, data):
        user, movie = data[:,0], data[:,1]
        # matrix multiplication
        return (self.user_factors(user)*self.movie_factors(movie)).sum(1)
    
    def predict(self, user, movie):
        return self.forward(user, movie)

