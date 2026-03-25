# Fake News Detection Web App

This project is a simple web application built to classify news as real or fake using basic machine learning techniques. I developed this project as a second-year B.Tech student to understand how Natural Language Processing (NLP) and machine learning models can be applied to real-world problems.

---

## Project Overview

The application takes news text as input from the user and predicts whether the news is real or fake. It also provides a confidence score for the prediction.

---

## How it Works

1. The input text is cleaned by removing special characters, converting to lowercase, and removing stopwords.
2. The cleaned text is converted into numerical form using TF-IDF vectorization.
3. A trained machine learning model (Naive Bayes) analyzes the text.
4. The result is displayed on the web interface along with a confidence score.

---

## Tech Stack

* Python
* Flask
* Scikit-learn
* Pandas
* NLTK
* HTML

---

## Project Structure

```
fake-news-detector/
│
├── app.py
├── model.py
├── Fake.csv
├── True.csv
├── model.pkl
├── vectorizer.pkl
│
└── templates/
    └── index.html
```

---

## How to Run

1. Train the model:

```
python3 model.py
```

2. Run the application:

```
python3 app.py
```

3. Open in browser:

```
http://127.0.0.1:5001
```

---

## Limitations

* The model does not verify facts; it only identifies patterns in text.
* Short or unclear input may lead to incorrect predictions.
* Accuracy depends on the quality of the dataset used for training.

---

## What I Learned

* Basics of Natural Language Processing
* How machine learning models work with text data
* Integrating a trained model with a Flask web application
* Debugging issues in both backend and model

---

## Future Improvements

* Improve model accuracy with better techniques
* Use more advanced models
* Improve the user interface
* Deploy the application online

---

## Conclusion

This project helped me understand the complete workflow of building a machine learning application, from data preprocessing to deploying it using a web framework. It was a good learning experience and a step towards building more advanced projects.
