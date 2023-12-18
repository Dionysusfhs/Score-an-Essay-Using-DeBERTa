# Score an Essay v1.7

#### Introduction

Our group used the pre-trained large language model **DeBERTa** based on **Transformer** to complete the English composition scoring task using the given data set.

Research motivation:

Using AI to replace manual grading of English compositions has become a more successful task in the field of NLP(natural language processing). The code implemented by the graph neural network and the data set used for this task have been open sourced on the Kaggle platform. We speculate that a pre-trained model using the Transformer architecture may achieve better results than GNN.

Achievements:

The MCRMSE score we achieved (lower is better) was 0.4422. It is close to the gold medal project's score (0.4334) of the competition on the kaggle platform, and far beats the GNN(graph neural network) project's score (0.7139).

Significance:

We often find that after writing an English composition, it is difficult to find other people who can help grade the composition. This project can help students who are learning English writing have a quick and rough understanding of the writing level of an article.

#### Dataset

kaggle competition link：
https://www.kaggle.com/competitions/feedback-prize-english-language-learning/

Dataset introduction:

The data set (ELLIPSE corpus) consists of argumentative essays written by English Language Learners (ELLs) in grades 8 to 12. The essays are scored on six analytical indicators: coherence, grammar, vocabulary, phrasing, grammar and convention.
Each indicator represents a component of composition level, and the higher the score, the higher the composition level on that indicator. Scores range from 1.0 to 5.0 in increments of 0.5. Your task is to predict the score of these six metrics for each article in the test set.

Dataset file and field information:

train.csv - The training set, including the full_text of each article, is identified by a unique text_id.
test.csv - For test data, only the full_text of the article and its text_id are provided.
sample_submission.csv - A submission file saved in the correct format.

#### Model

After the input text is segmented, the hidden states are extracted through the DeBERTa backbone model. These hidden states are passed to the mean pooling module to generate a fixed-length text representation. This text representation is sent to the fully connected neural network to output the prediction results.

<img src="./model.png" title="模型结构图" width=400 >

#### Installation

1.  Download dataset

Download the dataset from <a href="https://www.kaggle.com/competitions/feedback-prize-english-language-learning/data">https://www.kaggle.com/competitions/feedback-prize-english-language-learning/data</a>

Put it in this path <strong>.\input\feedback-prize-english-language-learning</strong>

2.  Install dependent libraries

If you use conda to manage the environment, you need to activate the environment first
<code>conda activate EnvironmentName</code>

Use the command line in the project path to execute the following command to install dependent libraries
<code>pip install -r requirement.txt</code>

3.  Use a compiler that supports jupyter notebook to open this project. It is recommended to use vscode with the notebook plug-in installed.

#### Usage

1. train.ipynb is the training code, providing training and simple testing functions.
2. test.ipynb is the test code and implements some examples of using the model.
3. demo.ipynb implements a GUI interface, and articles can be entered for rating at any time.

<img src="./gui.png" title="GUI输出效果图" width=400 >
