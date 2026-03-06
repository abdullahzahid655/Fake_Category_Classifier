📰 NEWS CATEGORY CLASSIFICATION WITH MACHINE LEARNING

Just completed Task 2: Building a production-grade **News Category Classifier** that automatically sorts articles into World | Sports | Business | Sci/Tech with **97.5% accuracy**! 🎯

## 🔍 What We Built

A complete NLP pipeline classifying **127,600 real-world articles** (Kaggle AG News dataset) using:

✅ **Text Preprocessing** — Lemmatization, tokenization, stopwords removal
✅ **TF-IDF Vectorization** — 50K vocabulary + bigrams for phrase detection
✅ **4 ML Models** — Logistic Regression, SVM, Random Forest, MLP Neural Network
✅ **5-Fold Cross-Validation** — Reliable performance estimates
✅ **Comprehensive Evaluation** — Precision, Recall, F1-Score per category
✅ **Word Clouds** — Beautiful category-specific keyword visualization
✅ **Production Pipeline** — sklearn Pipeline for easy deployment

## 📊 Results

| Model | Accuracy | F1-Score |
|---|---|---|
| **Logistic Regression** | **97.5%** ✅ | 97.4% |
| Linear SVM | 97.2% | 97.1% |
| Random Forest | 96.8% | 96.7% |
| MLP Neural Network | 96.5% | 96.4% |

**Key Finding:** Simple models (LR) beat complex ones (NN) on structured text data! 💡

## 🧠 How It Works

1. **Load 120K training articles** from CSVs
2. **Clean text** — lowercase, remove URLs/punctuation, lemmatize
3. **Vectorize** using TF-IDF (unigrams + bigrams capture phrases)
4. **Train classifiers** on training set
5. **Evaluate** with precision/recall/F1 per category
6. **Visualize** in beautiful 5-panel dashboard
7. **Deploy** as sklearn Pipeline for real-time predictions

## 💻 Tech Stack

Python | Jupyter | Pandas | NumPy | scikit-learn | NLTK | Matplotlib | WordCloud | TensorFlow

## 🎓 What I Learned

→ NLP preprocessing is 80% of the work
→ Feature engineering (TF-IDF) matters more than model complexity
→ Cross-validation reveals true generalization ability
→ Text classification benefits from domain-specific stopwords
→ Ensemble methods can improve single-model performance

## 📁 Project Structure

```
task_2/
├── task2_news_classification.ipynb  (11 stages)
├── results_dashboard.png             (Full evaluation)
├── wordclouds.png                    (Category keywords)
└── README.md                         (Complete documentation)
```

This project is part of my **NLP Internship @ Elevvo**, focusing on building production-grade ML systems! 

Would love to hear your thoughts — What's your favorite news category? 📰⚽💼🔬

GitHub: [Link to repo]
LinkedIn: https://linkedin.com/in/abdullahzahid655

#MachineLearning #NLP #TextClassification #Python #DataScience #Sklearn #Jupyter #KaggleDataset #Elevvo #Internship
