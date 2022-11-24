# Project : Twitter-Speech-Detection

The goal of this research is to utilise machine learning to classify text into three unique categories: "hate speech," "offensive language," and "Neither Hateful Nor Offensive".

# Tools Utilized:
Pandas, NumPy, Scikit-Learn, Lemmatizer, nlpaug synonym text augmentation, Word2Vec with Fasttext pretrained compressed model, Machine Learning Algorithms, SMOTE, SMOTETomek Links

# Streamlit App Link:
https://mansi67-twitter-speech-detection-app-7bpr2g.streamlit.app/

# Demo:

__Home Page__
![image](https://user-images.githubusercontent.com/105342764/203835552-2b8ad88e-7b5f-4e70-aaa5-470a2e413ade.png)

![image](https://user-images.githubusercontent.com/105342764/203836247-44cda1a1-469f-454c-9777-dd685e2c24c5.png)

![image](https://user-images.githubusercontent.com/105342764/203837306-f6b8bfbe-c2aa-45a5-9588-39715e611c45.png)

![image](https://user-images.githubusercontent.com/105342764/203837459-2ff85a4f-089a-40ea-8a60-f62b698c0b75.png)

![image](https://user-images.githubusercontent.com/105342764/203837583-8e3201e4-c755-4825-8d81-8fbdb4b4cfd3.png)

# Steps:
1. Links, punctuations were removed, contractions were fixed, spell corrector was applied.
2. After initial preprocessing, artificial data was generated.
3. Stopwords were removed, lemmatization applied.
4. Word2Vectorization applied
5. Model Build with and without Oversampling


# Thoughts:
The dataset was poorly balanced and contained less than 6% of Hateful Tweet examples and more than 75% Offensive Tweets. I tried text augmentation to balance Hateful Tweets but was not of great help. The model can be improved if we add more Hateful and Neutral Tweets to make the dataset richer. 
Logistic Regression was working better than the rest in termms of F1 Score with or without oversampling. Oversampling was hardly making any difference in this dataset as the data was extremely imbalanced.

# References:
Dataset originates from the paper cited below and can be found at: https://github.com/t-davidson/hate-speech-and-offensive-language.

Davidson, T., Warmsley, D., Macy, M. and Weber, I., 2017. Automated Hate Speech Detection and the Problem of Offensive Language. ArXiv. https://arxiv.org/pdf/1703.04009.pdf
