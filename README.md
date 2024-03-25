# GESI Content Analysis Dashboard

This repository contains datasets, machine learning models, and a Streamlit application script required to build and deploy a dashboard for Gender Equality and Social Inclusion (GESI) content analysis.

## Overview

The GESI Content Analysis Dashboard is a tool for monitoring and analyzing online content related to gender and social issues. It utilizes natural language processing models to classify content sentiment, domain, and discrimination level and presents the analysis in an interactive format.

## Repository Structure

- `dataset/`: This directory contains datasets used for the analysis. It includes annotated social media and news content in English, Sinhala, and Tamil.
- `models/`: This directory contains pre-trained machine learning models for sentiment analysis and content classification.
- `app.py`: The main Streamlit application script that uses the datasets and models to render the dashboard.
- `prediction_sinhala.py`: A Python script with functions to make predictions using the Sinhala language model.
- `requirements.txt`: A file with all the necessary Python libraries required to run the Streamlit app.

## Data

The datasets included in the `data/` directory are:

- `reviewed_social_media_english.csv`
- `reviewed_news_english.csv`
- `tamil_social_media.csv`
- `tamil_news.csv`

Each dataset contains various fields such as content, sentiment, discrimination, domain, and more.

## Models

The `models/` directory contains:

- Pre-trained model files for sentiment analysis and content classification.
- A configuration file or training script for reproducing the models.

## Running the Dashboard

To run the dashboard on your local machine:

1. Clone this repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the command `streamlit run app.py`.

Make sure you have Python installed on your machine and a virtual environment set up if required.

## Deployment

The dashboard is designed to be deployed on Hugging Face Spaces. To deploy:

1. Fork this repository into your Hugging Face account.
2. Set up a new Space and link it to your forked repository.
3. The Space will automatically detect the `app.py` as the Streamlit app script and deploy it.

For any custom deployment steps or troubleshooting, refer to the [Hugging Face documentation](https://huggingface.co/docs).

## License

This project is open-source and available under the MIT License.

## Contributing

Contributions to improve the dashboard or models are welcome. Please fork the repository, make your changes, and submit a pull request.


