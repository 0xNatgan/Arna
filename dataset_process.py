from transformers import pipeline
import pandas as pd
from tqdm import tqdm

def generate_sentiment_dataset(input_path, output_path):
    """Generates sentiment labels for the dataset using a pre-trained sentiment analysis model."""
    sentiment_pipeline = pipeline("sentiment-analysis", device=0)  # GPU if available

    df = pd.read_csv(input_path)

    sentiments = []
    for text in tqdm(df['Description'], desc="Analyzing sentiment"):
        try:
            result = sentiment_pipeline(str(text)[:512])[0]
            # Convert to numeric label: POSITIVE=2, NEUTRAL=1, NEGATIVE=0
            if result['label'] == 'POSITIVE':
                sentiments.append(2)
            elif result['label'] == 'NEGATIVE':
                sentiments.append(0)
            else:
                sentiments.append(1)
        except:
            sentiments.append(1)  # Default to neutral

    df['Sentiment'] = sentiments
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

# Generate the labels
generate_sentiment_dataset(
    '/home/knwldgosint/Documents/School5/Advanced Neural network/project/Arna/dataset/test.csv',
    '/home/knwldgosint/Documents/School5/Advanced Neural network/project/Arna/dataset/test_with_sentiment.csv'
)