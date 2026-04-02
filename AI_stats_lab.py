import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


# =========================
# Q1 Naive Bayes
# =========================

def naive_bayes_mle_spam():

    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])

    test_email = "win cash prize now"

    # Tokenize
    tokenized = [text.split() for text in texts]

    # Vocabulary
    vocab = set(word for doc in tokenized for word in doc)

    # Priors
    priors = {}
    for c in [0, 1]:
        priors[c] = np.sum(labels == c) / len(labels)

    # Word probabilities (MLE, no smoothing)
    word_probs = {0: {}, 1: {}}

    for c in [0, 1]:
        class_docs = [tokenized[i]
                      for i in range(len(labels)) if labels[i] == c]
        all_words = [word for doc in class_docs for word in doc]
        total_words = len(all_words)

        word_counts = Counter(all_words)

        for word in vocab:
            word_probs[c][word] = word_counts[word] / \
                total_words if total_words > 0 else 0.0

    # Prediction
    test_words = test_email.split()

    log_probs = {}

    for c in [0, 1]:
        log_prob = np.log(priors[c])

        for word in test_words:
            prob = word_probs[c].get(word, 0)

            # Avoid log(0) → assign very small value
            if prob == 0:
                prob = 1e-9

            log_prob += np.log(prob)

        log_probs[c] = log_prob

    prediction = max(log_probs, key=log_probs.get)

    return priors, word_probs, prediction


# =========================
# Q2 KNN
# =========================

def knn_iris(k=3, test_size=0.2, seed=0):

    # Load dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Euclidean distance
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # Predict one point
    def predict_one(x):
        distances = [euclidean_distance(x, x_train) for x_train in X_train]
        k_indices = np.argsort(distances)[:k]
        k_labels = y_train[k_indices]
        return Counter(k_labels).most_common(1)[0][0]

    # Predictions
    train_predictions = np.array([predict_one(x) for x in X_train])
    test_predictions = np.array([predict_one(x) for x in X_test])

    # Accuracy
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    return train_accuracy, test_accuracy, test_predictions
