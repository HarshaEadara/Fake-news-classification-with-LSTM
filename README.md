# Fake News Classification with LSTM
This program aims to analyze data, perform Exploratory Data Analysis (EDA), and classify news articles as fake or real using an LSTM model. By leveraging the power of deep learning and natural language processing (NLP), the project provides an effective solution for detecting misinformation in the form of fake news.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Analysis Performed](#analysis-performed)
- [Tools and Technologies](#tools-and-technologies)
- [Workflow](#workflow)
- [Additional Features](#additional-features)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)

## Project Overview

Fake news is a form of disinformation spread via traditional or online media. In recent years, the proliferation of fake news has posed significant challenges to societies worldwide, leading to misinformation and confusion. This project addresses this issue by building a machine learning pipeline to analyze news headlines and classify them as fake or real. The primary model employed is a Long Short-Term Memory (LSTM) network, which is highly suitable for sequence prediction tasks involving textual data.

## Dataset

The dataset used is the ISOT Fake News Dataset, which contains:
- **Article title:** The headline of the news article.
- **Article text:** The main body of the article.
- **Label:** A binary label indicating whether the news is "REAL" or "FAKE."
- **Date of publication:** The publication date of the article.

The dataset consists of over 12,600 articles in each category. Real news articles were obtained from Reuters.com, while fake news articles were collected from unreliable websites flagged by Politifact and Wikipedia. This dataset provides a balanced and diverse set of examples for model training and evaluation.

### Data Availability

The data for this project is available in the data folder of the repository, containing the following files:
- `fake.zip`: Compressed file containing data on fake news articles.
- `true.zip`: Compressed file containing data on real news articles.
Unzip these files before running the project to access the dataset.

## Data Preprocessing

The preprocessing steps applied to the dataset include:

1. Removal of duplicate entries.
2. One-hot encoding of categorical variables.
3. Normalizing numerical features.
4. Preparing the data for model training and evaluation.

## Analysis Performed

- **Exploratory Data Analysis (EDA):** An initial investigation to understand the dataset structure, identify patterns, and detect anomalies.
- **Data Visualization:** Visualizations, including word clouds and distribution plots, were created to summarize and understand the data better.
- **LSTM Model Analysis:** The core part of the project, where an LSTM model was designed, trained, and tested to classify fake news.

## Tools and Technologies

The project leverages the following technologies:

- **Programming Languages:** Python
- **Libraries and Frameworks:**
   - Data Manipulation: Pandas, NumPy
   - Visualization: Matplotlib, Seaborn
   - Natural Language Processing: NLTK
   - Machine Learning: Scikit-learn
   - Deep Learning: TensorFlow, Keras
- **Visualization**: Data visualizations for exploratory data analysis (EDA).
- **Jupyter Notebook**: Development and analysis environment.

These tools provide robust capabilities for data analysis, visualization, and deep learning model development, ensuring a seamless workflow from data preparation to model evaluation.

## Workflow

1. **Import Libraries and Dataset:** Import all necessary libraries such as Pandas, NumPy, Matplotlib, and TensorFlow, and load the dataset for analysis.
2. **Perform EDA:** Explore the dataset to gain insights and identify trends.
3. **Data Cleaning:** Handle missing values, remove duplicates, and preprocess textual data by removing noise like stopwords, punctuation, and special characters.
4. **Visualize Cleaned Dataset:** Create visualizations like word clouds and bar charts to understand the distribution of words and article lengths.
5. **Prepare Data:**
   - **Tokenization:** Convert text data into sequences of integers.
   - **Padding:** Standardize the input sequence length for compatibility with the LSTM model.
6. **Build and Train LSTM Model:**
   - Design an LSTM neural network architecture with layers optimized for text sequence classification.
   - Train the model on the dataset, ensuring the model generalizes well to unseen data.
7. **Assess Trained Model Performance:**
   - Evaluate the model's accuracy, precision, recall, and F1-score.


## Additional Features
- **Word Clouds:** Word clouds were generated to visualize the most frequent words in fake and real news articles, providing intuitive insights into the dataset.
- **Confusion Matrix:** A confusion matrix was plotted to assess the model's performance in terms of true positives, true negatives, false positives, and false negatives.
- **Performance Metrics:** Detailed metrics, including precision, recall, and F1-score, were computed to provide a comprehensive evaluation of the model.

## Results 
The trained LSTM model achieved an impressive accuracy of 99.12% on the test dataset. This exceptional result highlights the model's ability to distinguish between real and fake news articles with high precision.
- **Effectiveness:** The model demonstrated consistent performance across various metrics, including precision, recall, and F1-score, making it highly reliable for practical applications.
- **Applications:** Its capabilities position it as a valuable tool for media organizations, fact-checkers, and social media platforms to detect and curb the spread of misinformation.
- **Real-world Impact:** By identifying fake news effectively, this system can contribute to reducing public confusion and promoting informed decision-making among audiences.

### Future Work

This project provides a solid foundation for fake news classification. Future enhancements could include:
- Extending the model to handle multilingual datasets.
- Incorporating attention mechanisms to improve model interpretability.
- Exploring transformer-based models like BERT for improved accuracy.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/HarshaEadara/Fake-news-classification-with-LSTM.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Fake-news-classification-with-LSTM
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Extract the dataset files:
   ```bash
   unzip data/fake.zip -d data/
   unzip data/true.zip -d data/
   ```
5. Open the Jupyter Notebook:
   ```bash
   jupyter notebook FakeNewsClassification_with_LSTM.ipynb
   ```
6. Run the notebook cells sequentially to reproduce the analysis.

## Contributing

Contributions to this project are welcome. If you'd like to suggest improvements or report issues, please open an issue or submit a pull request on the repository. Let's collaborate to make this project even better!

