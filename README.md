# Main Ideas of the Text Summarization Code

## 1. Objective
- Automatically generate summaries for text documents by identifying key sentences.
- Utilize machine learning (SVM) combined with linguistic features to determine sentences relevant to the summary.

## 2. Overall Workflow
- **Data Reading**: Process original text files and reference summaries from input directories.
- **Feature Extraction**: Analyze sentences based on linguistic features to create feature vectors.
- **Model Training**: Use SVM to learn sentence classification based on reference data.
- **Prediction**: Apply the model to select important sentences for the summary.
- **Output Generation**: Save selected sentences to a summary file.

## 3. Linguistic Features
- **Indicator Phrases**: Detect phrases like "in conclusion" or "to summarize" common in summaries.
- **Sentence Length**: Prioritize sentences longer than 5 words, typically containing more information.
- **Sentence Position**: Assess importance based on position (beginning, middle, or end of the document).
- **Thematic Words**: Use TF-IDF to identify significant words in the document.
- **Proper Nouns**: Detect sentences with multiple proper nouns, often carrying specific information.

## 4. Techniques Used
- **Natural Language Processing (NLP)**:
  - Parse HTML/XML with BeautifulSoup.
  - Perform POS tagging with NLTK.
  - Compute TF-IDF with Scikit-learn.
- **Machine Learning**:
  - SVM model with balanced sample weights for sentence classification.
  - Customize SVM parameters (C, kernel) to optimize performance.

## 5. Applications
- Automate document summarization, reducing manual processing time.
- Support content analysis for large datasets (e.g., DUC dataset).
- Easily extensible to other text types.

## 6. Output
- A summary file containing sentences predicted as important by the model.
- Ensures concise summaries that retain the core ideas of the original document.

## NOTE: 
- Need to improve the model, because the output just correct about **40%**
