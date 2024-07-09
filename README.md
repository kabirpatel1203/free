import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import defaultdict
import nltk
import re

nltk.download('stopwords')
from nltk.corpus import stopwords

# Function to normalize text
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Function to paraphrase text using PEGASUS model
def paraphrase_text(text, model, tokenizer):
    inputs = tokenizer.encode("paraphrase: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased_text

# Function to extract aspects using topic modeling
def extract_aspects(texts, n_topics=5, n_words=10):
    vectorizer = CountVectorizer(stop_words='english')
    text_matrix = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(text_matrix)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
        topics.append(" ".join(top_words))
    return topics

# Function to perform ABSA
def perform_absa(df, text_column, n_topics=5):
    # Load paraphrasing model and tokenizer
    paraphrase_model_name = "tuner007/pegasus_paraphrase"
    paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(paraphrase_model_name)
    paraphrase_tokenizer = AutoTokenizer.from_pretrained(paraphrase_model_name)

    # Normalize and paraphrase text
    df['cleaned_text'] = df[text_column].apply(normalize_text)
    df['paraphrased_text'] = df['cleaned_text'].apply(lambda x: paraphrase_text(x, paraphrase_model, paraphrase_tokenizer))

    # Extract aspects using topic modeling
    aspects = extract_aspects(df['paraphrased_text'].tolist(), n_topics=n_topics)

    # Load the model and tokenizer for ABSA
    MODEL_DIR = "./local_model_directory/aspect_analysis"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Initialize the ABSA pipeline with the local model
    absa = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

    def analyze_text(text):
        results = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
        for aspect in aspects:
            inputs = f"{aspect} [SEP] {text}"
            output = absa(inputs)
            for item in output:
                sentiment = item['label'].lower()
                results[aspect][sentiment] = item['score']
        return results

    # Apply ABSA to each comment
    df['absa_results'] = df['paraphrased_text'].apply(analyze_text)

    # Extract sentiment scores for each aspect
    for aspect in aspects:
        df[f'{aspect}_positive'] = df['absa_results'].apply(lambda x: x[aspect]['positive'])
        df[f'{aspect}_negative'] = df['absa_results'].apply(lambda x: x[aspect]['negative'])
        df[f'{aspect}_neutral'] = df['absa_results'].apply(lambda x: x[aspect]['neutral'])

    # Determine overall sentiment for each aspect
    for aspect in aspects:
        df[f'{aspect}_sentiment'] = df[[f'{aspect}_positive', f'{aspect}_negative', f'{aspect}_neutral']].idxmax(axis=1)
        df[f'{aspect}_sentiment'] = df[f'{aspect}_sentiment'].apply(lambda x: x.replace(f'{aspect}_', ''))

    # Identify strengths and weaknesses
    df['strengths'] = df.apply(lambda row: [aspect for aspect in aspects if row[f'{aspect}_sentiment'] == 'positive'], axis=1)
    df['weaknesses'] = df.apply(lambda row: [aspect for aspect in aspects if row[f'{aspect}_sentiment'] == 'negative'], axis=1)

    # Calculate overall strengths and weaknesses
    overall_strengths = df['strengths'].explode().value_counts()
    overall_weaknesses = df['weaknesses'].explode().value_counts()

    # Calculate average sentiment scores for each aspect
    avg_sentiments = {}
    for aspect in aspects:
        avg_positive = df[f'{aspect}_positive'].mean()
        avg_negative = df[f'{aspect}_negative'].mean()
        avg_sentiments[aspect] = {'avg_positive': avg_positive, 'avg_negative': avg_negative}

    # Add summary statistics to the DataFrame
    df.attrs['overall_strengths'] = overall_strengths.to_dict()
    df.attrs['overall_weaknesses'] = overall_weaknesses.to_dict()
    df.attrs['avg_sentiments'] = avg_sentiments

    return df

# Example usage
sample_data = [
    "The product quality is excellent, but the delivery was delayed.",
    "Customer support was very helpful, though the price is a bit high.",
    "I love the product features, but the service could be improved.",
    "Quick delivery, but the product didn't meet my expectations.",
    "Great value for money, and the support team is fantastic!",
    "The product is okay, but not worth the high price.",
    "Excellent service and prompt delivery, but the product quality is average.",
    "The support team is responsive, but the product lacks some key features.",
    "Fast delivery and good product, but the after-sales service is poor.",
    "High-quality product, but the website is difficult to navigate.",
    "The price is reasonable, but the delivery took too long.",
    "Product exceeded expectations, though customer service was hard to reach.",
    "Good overall experience, but the product manual is confusing.",
    "The product design is innovative, but it's not very user-friendly.",
    "Satisfied with the purchase, but the return policy is too strict."
]

df = pd.DataFrame(sample_data, columns=['comment'])
df_results = perform_absa(df, 'comment')

# Debugging prints
print("Columns in the resulting DataFrame:", df_results.columns)

# Printing the sample of ABSA results
print("Sample of ABSA results:")
print(df_results[['comment'] + [f'{aspect}_sentiment' for aspect in df_results.attrs['avg_sentiments'].keys()] + ['strengths', 'weaknesses']].head())

print("\nOverall Strengths:", df_results.attrs['overall_strengths'])
print("\nOverall Weaknesses:", df_results.attrs['overall_weaknesses'])
print("\nAverage Sentiments:")
for aspect, scores in df_results.attrs['avg_sentiments'].items():
    print(f"{aspect}: Avg Positive: {scores['avg_positive']:.2f}, Avg Negative: {scores['avg_negative']:.2f}")
