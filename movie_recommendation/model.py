import numpy as np, torch
import os, logging, pickle

from ts.torch_handler.base_handler import BaseHandler

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
    
    
# custom handler file
class ModelHandler(BaseHandler):

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.top_k = 5

    def initialize(self, context):
        from model import MatrixFactorization
        self.manifest = context.manifest
        properties = context.system_properties
        
       # load metadata
        mapping_file = os.path.join(properties.get("model_dir"), "id_mapping.pkl")   
        with open(mapping_file, 'rb') as f:
            dic = pickle.load(f)
        self.user_map = dic['user']
        self.movie_map = dic['movie']
        # self.users = dic['users']
        # self.movies = dic['movies']
        self.n_users, self.n_movies = len(self.user_map), len(self.movie_map)
        
        # load model
        self.model = MatrixFactorization(n_users, n_movies)
        self.model.load_state_dict(torch.load( os.path.join(properties.get("model_dir"), self.manifest['model']['serializedFile'])))
        self.model.eval()

        self.initialized = True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        """
        # Get user id
        input_ = data[0].get("data")
        if input_ is None: input_ = data[0].get("body")
        user_id = input_['user_id']
        # if user not exist, return null
        if user_id not in self.user_map.keys():
            return None
        # get user index
        user_idx = self.user_map[user_id]
        # construct combination of user with all movies
        user_idxs = np.ones(self.n_movies) * user_idx
        movie_idxs = np.arange(self.n_movies)
        data = torch.tensor(np.vstack([user_idxs, movie_idxs]).T, dtype=torch.long)
        return data


    def inference(self, model_input):
        """
        Predict ratings.
        """
        # predict rating for all movies
        with torch.no_grad():
            model_output = self.model.forward(model_input)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        """
        # sort the ratings, return indexes in descending order
        ratings = inference_output.numpy()
        inds = np.argsort(ratings)[::-1]
        # get name for 5 highest rating movie
        movie_names = [self.movie_map[i] for i in movie_inds[:self.top_k]inds]
        return [movie_names]

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        """
        model_input = self.preprocess(data)
        # if user_id not exist
        if model_input is None: 
            return ['invalid user_id']
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
