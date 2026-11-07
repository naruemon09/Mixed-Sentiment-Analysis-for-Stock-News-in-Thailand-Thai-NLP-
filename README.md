# Mixed Sentiment Analysis for Thai Stock News

Thai financial headlines often carry two signals at once, such as *"ราคาทองคำปรับตัวสูง ความเชื่อมั่นไม่มาก"* (gold price rises, but confidence is low). This project classifies the overall sentiment of such mixed sentences and compares classical machine-learning models with a fine-tuned Thai transformer, **WangchanBERTa**.

Computer Science research project, Ramkhamhaeng University.

## What's in this repo

| File | What it does |
|---|---|
| `sentiment_wangchan.ipynb` | Fine-tunes WangchanBERTa for binary sentiment (negative / positive), evaluates it and runs predictions on new text |
| `sentiment_analytics.py` | Classical pipeline: Thai text cleaning, tokenisation, TF-IDF features and Logistic Regression, Naive Bayes or SVM for three labels (negative / neutral / positive) |

## Approach

### 1. Classical baseline (`sentiment_analytics.py`)

1. **Cleaning:** normalise Thai text with PyThaiNLP, remove emoji and symbols, keep numbers and currency signs
2. **Tokenisation:** PyThaiNLP `newmm` word segmentation, with a custom step that keeps amounts such as `0.37%` or `2.5฿` as one token
3. **Features:** TF-IDF on unigrams and bigrams (5,000 features), then chi-squared selection of the top 2,000
4. **Models:** Logistic Regression, Multinomial Naive Bayes and linear SVM, with class weighting where supported
5. **Output:** classification report, confusion matrix, and a saved `joblib` pipeline for predicting new headlines

### 2. WangchanBERTa (`sentiment_wangchan.ipynb`)

- Base model: [`airesearch/wangchanberta-base-att-spm-uncased`](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased)
- Data: 2,015 sentences (1,008 positive, 1,007 negative), stratified 80/20 split → 1,612 train / 403 test
- Training: Hugging Face `Trainer`, max length 128, batch size 8, learning rate 2e-5, up to 12 epochs, best checkpoint by accuracy

## Results

WangchanBERTa on the 403-sentence test split:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 1.000 | 0.990 | 0.995 | 201 |
| Positive | 0.990 | 1.000 | 0.995 | 202 |
| **Accuracy** | | | **0.995** | 403 |

Only 2 of 403 test sentences were misclassified, both negative sentences that contain a positive-sounding word: *"หุ้นไทยความท้าทายสร้างการเติบโต"* ("growth") and *"การลงทุนเพื่อลดหย่อนภาษีตกต่ำฟื้นตัวช้า"* ("recovery").

<!-- TODO: add the accuracy / F1 of Logistic Regression, Naive Bayes and SVM here for comparison -->

## Limitations

These numbers should be read with care:

- **The sentences are short and follow a few repeated patterns**, so the test split is very similar to the training data. Accuracy on real news articles will be lower.
- **The test split was also used to pick the best checkpoint**, so it isn't a fully independent hold-out set. A separate validation split would give a fairer estimate.
- **Informal sentences outside the dataset style are harder.** In the notebook, *"ชอบบรรยากาศในหุ้นตัวนี้มากมาก"* ("I really like this stock") was predicted as negative.
- **The datasets are not included in the repository.** The notebook expects `mixed_sentiment_2.xlsx` and the script expects an Excel file with `paragraph` and `label` columns.

## Run it

```bash
pip install pandas numpy scikit-learn pythainlp emoji matplotlib seaborn joblib openpyxl
pip install torch transformers datasets   # for the notebook

# classical models: set data_file in main() to your Excel file first
python sentiment_analytics.py

# transformer
jupyter notebook sentiment_wangchan.ipynb
```

## Next steps

- Hold out a separate validation set and report test results only once
- Evaluate on real headlines collected after the training data
- Train the three-class (neutral) version with WangchanBERTa and compare it with the classical models on the same split
- Add a `requirements.txt` and publish a small sample of the dataset

## Tech stack

Python · PyThaiNLP · scikit-learn · Hugging Face Transformers · PyTorch · WangchanBERTa · pandas
