**Project Report: Math Question Classification**

**1. Overview**

The goal of this task was to build a classical machine learning model
capable of classifying math questions into subdomains (Algebra,
Calculus, Geometry, etc.) using the provided JSON dataset.

**2. Approach & Design Choices**

**2.1 Handling the Data (The \"Zip\" Strategy)**

The dataset came as a zip file containing thousands of individual JSONs.

- **Initial thought:** Extract everything to a folder.

- **Correction:** Windows file extraction for thousands of tiny files is
  very slow.

- **Solution:** I wrote the script to use Python\'s zipfile module to
  read the files directly from the archive into memory. This drastically
  reduced the setup time and makes the script portable---you don\'t need
  to manually unzip anything to run it.

**2.2 Feature Engineering (My attempt to fix)**

This was the biggest challenge. My first attempt used standard natural
language processing cleaning (keeping only A-Z and 0-9).

- **The Problem:** The model was stuck at around 60% accuracy. It was
  confusing \"Algebra\" with \"Geometry\" because it couldn\'t see the
  symbols. \"Solve 2x\" and \"Find angle ABC\" looked almost identical
  once the symbols were removed.

- **My fix:** I rewrote the regex cleaner to **preserve mathematical
  syntax**. I specifically kept operators (+, -, =, \^) and delimiters
  ({}, ()) and added spaces around them. This allowed the TF-IDF(Term
  frequency inverse document frequency) vectorizer to treat symbols as
  distinct \"words\" which immediately boosted accuracy.

**2.3 Model Selection**

I went with **Logistic Regression** paired with **TF-IDF
Vectorization**.

- **Why TF-IDF?** Math problems are defined by specific keywords. Words
  like \"integral,\" \"derivative,\" or \"hypotenuse\" are easy
  giveaways. TF-IDF is perfect for capturing the weight of these unique
  terms.

- **Why Logistic Regression?** It's a solid base for text
  classification. It's fast, handles multi-class problems well, and most
  importantly for the interactive mode it gives us probability scores so
  we can see how confident the model is.

**2.4 Tuning**

I didn\'t want to just guess the hyperparameters, so I used
GridSearchCV.

- I tuned C (regularization strength) and min_df (minimum word
  frequency).

- The search preferred higher regularization, which makes sense given
  that topics like \"Prealgebra\" and \"Algebra\" have very fuzzy
  boundaries.

**3. Performance & Observations**

I ran a few quick tests to confirm my choices:

1.  Baseline- Removed all symbols . Hence struggled with geometry and
    algebra distinction

2.  Improved- Kept math symbols and helped me understand that the
    symbols where the key features.

3.  Merged- Combined pre algebra and algebra and showed that most errors
    arose due to the difficulty level parameter.

Confusion Matrix:

The model rarely confuses distinct topics like Calculus and Probability.
The majority of errors are between Prealgebra, Algebra, and Intermediate
Algebra, which is expected since they share nearly the same vocabulary
(e.g., \"solve\", \"x\", \"equation\").

**4. Future Work**

If I had more time, I would explore:

- **Deep Learning:** A transformer like BERT or MathBERT would
  understand the *context* of a question better than a bag-of-words
  approach.

**5. How to Run**

1.  Make sure dataset.zip is in the same folder as the script.

2.  Run python solution.py.

3.  The script will train the model, print the evaluation metrics, and
    then drop you into an **Interactive Mode** where you can type your
    own math questions to test it.
