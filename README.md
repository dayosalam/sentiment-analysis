# BERT pre trained model for sentiment analysis

This project aims to perform sentiment analysis using BERT (Bidirectional Encoder Representations from Transformers), a state-of-the-art pre-trained deep learning model for natural language processing (NLP). The project involves training and fine-tuning BERT on a dataset of movie reviews to predict the sentiment of a given review on a scale of one to five, one being extremely bad and five being extremely great.

The dataset used in this project was scrapped from a resturant comment section on YELP which contains 50 resturant reviews.

Methodology:
The BERT model is pre-trained on a large corpus of text data and then fine-tuned on the resturant review dataset to classify the sentiment of each review. The pre-trained BERT model is loaded using the Hugging Face library, and then a neural network is added on top of it for classification.
