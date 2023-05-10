# Databricks notebook source
# MAGIC %md
# MAGIC ## Building a Simple Recommender System
# MAGIC In this notebook, we generate some dummy data for user interactions with products, and pivot the data to get a user-product matrix with ratings. We then calculate the cosine similarity matrix between the products, and define a function recommend_products that takes in the user ID, current product ID, and number of recommendations, and returns a list of recommended products based on the similarity scores between the features of the products.

# COMMAND ----------

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import mlflow.sklearn
import mlflow.pyfunc
import numpy as np


# Define a list of users and products
users = ['User1', 'User2', 'User3', 'User4', 'User5', 'User6', 'User7', 'User8', 'User9', 'User10']
products = ['Product1', 'Product2', 'Product3', 'Product4', 'Product5', 'Product6', 'Product7', 'Product8', 'Product9', 'Product10']

# Generate 100 rows of data with random ratings
data = pd.DataFrame({
    'user': np.random.choice(users, 100),
    'product': np.random.choice(products, 100),
    'rating': np.random.randint(1, 6, 100)
}).drop_duplicates(['user', 'product'])

# Pivot the data to get a user-product matrix with ratings
user_product_matrix = data.pivot(index='user', columns='product', values='rating').fillna(0)

# Calculate the cosine similarity matrix between the products
product_similarity_matrix = cosine_similarity(user_product_matrix.T)

# Define a function to recommend products
def recommend_products(user_id, product_id, n_recommendations):
    # Get the index of the current product
    product_index = user_product_matrix.columns.get_loc(product_id)

    # Calculate the similarity scores between the current product and all other products
    similarity_scores = list(enumerate(product_similarity_matrix[product_index]))

    # Sort the similarity scores in descending order and return the top n_recommendations products
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_products = [(user_product_matrix.columns[i], score) for i, score in sorted_scores if i != product_index][:n_recommendations]
    
    return recommended_products



# COMMAND ----------

# MAGIC %md
# MAGIC ### Wrap the custom model into a MLFlow pyfunc class
# MAGIC Wrapping the recommender model in Pyfunc allows it to be serialized and loaded as a Python function using the MLflow Pyfunc API, making it easy to deploy and use in a production environment. It also allows for input and output schemas to be defined, making it easier to integrate the model with other systems.

# COMMAND ----------

# Variable
model_name = "jn_product_recommender"

# Define a Pyfunc model class
class RecommenderModel(mlflow.pyfunc.PythonModel):
    def __init__(self, product_similarity_matrix, user_product_matrix):
        self.product_similarity_matrix = product_similarity_matrix
        self.user_product_matrix = user_product_matrix

    def recommend_products(self, user_id, current_product_id, n_recommendations):
        # Get the index of the current product
        product_index = self.user_product_matrix.columns.get_loc(current_product_id)

        # Calculate the similarity scores between the current product and all other products
        similarity_scores = list(enumerate(self.product_similarity_matrix[product_index]))

        # Sort the similarity scores in descending order and return the top n_recommendations products
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        recommended_products = [(self.user_product_matrix.columns[i], score) for i, score in sorted_scores if i != product_index][:n_recommendations]

        return recommended_products

    def predict(self, context, input_df):
        output = []
        for _, row in input_df.iterrows():
            user_id = row['user_id']
            current_product_id = row['current_product_id']
            n_recommendations = row['n_recommendations']

            recommended_products = self.recommend_products(user_id, current_product_id, n_recommendations)
            
            # Append the list of recommended products to the output
            output.append([(product, score) for product, score in recommended_products])

        return {"predictions": output}

# Create an instance of the RecommenderModel class
model = RecommenderModel(product_similarity_matrix, user_product_matrix)

# Log the model with an example input
with mlflow.start_run(run_name='content-based-recommender'):
    # Log the model parameters
    mlflow.log_param('n_recommendations', 3)

    # Log the Pyfunc model
    mlflow.pyfunc.log_model(
        'model',
        python_model=model,
        conda_env={
            'channels': ['defaults'],
            'dependencies': [
                'python=3.10.6',
                'scikit-learn=1.1.1',
                'pandas=1.4.4',
                'numpy=1.21.5',
                'cloudpickle=2.0.0',
                {
                    'pip': [
                        'mlflow==2.2.1'
                    ]
                }
            ]
        },
        input_example=pd.DataFrame([{
            'user_id': 'User1',
            'current_product_id': 'Product1',
            'n_recommendations': 3
        }]),
        registered_model_name=model_name

    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use the model for inference
# MAGIC To use the model for inference, we load the serialized Pyfunc model using mlflow.pyfunc.load_model(). Then, we define an example input and use the loaded model to make a prediction using the predict() function, which returns the recommended products for the given user and current product.

# COMMAND ----------


# Load the model using the mlflow.pyfunc API and make a prediction
import mlflow.pyfunc

# Load model as a PyFuncModel.
model_version = "9"
model_uri = f"models:/{model_name}/{model_version}"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Define an example input for prediction
input_example = pd.DataFrame([{
            'user_id': 'User1',
            'current_product_id': 'Product1',
            'n_recommendations': 3
        }, 
        {'user_id': 'User1',
            'current_product_id': 'Product1',
            'n_recommendations': 3
        }])

# Make a prediction using the loaded model
output = loaded_model.predict(input_example)

print(output)

# COMMAND ----------


