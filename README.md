# Product Recommender with MLFlow
This repository illustrates how to build a very simple content-based recommender system that takes in the user and the current product they are viewing, and returns three similar products that could interest them. The main goal here is to unveil the usage of MLFlow on Databricks to track experiments, register the model and deploy an online inference endpoint with code only.

# Dataset
For this example, we generate some dummy data for user interactions with products, where each row represents a user-product interaction with a rating between 1 and 5. We will then use the cosine similarity metric to calculate the similarity between the features of the products and recommend the top three products with the highest similarity score to the current product.

# Setup
1. Clone this to your Databricks workspace. For more guidance, check the documentation on [how to add a git repo on Databricks](https://docs.databricks.com/repos/git-operations-with-repos.html).
2. Run the first notebook to [Build a Simple Recommender System.py](https://github.com/julie-nguyen-ds/product-recommender-with-mlflow/blob/main/Build%20a%20Simple%20Recommender%20System.py)
3. Run the second notebook to [Deploy Endpoint with Inference Table.py](https://github.com/julie-nguyen-ds/product-recommender-with-mlflow/blob/main/Deploy%20Endpoint%20with%20Inference%20Table.py). Note that inference table are in private preview at the moment and might not be available in your workspace.
