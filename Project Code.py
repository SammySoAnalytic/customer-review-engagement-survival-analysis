# ==============================================================================
# CELL 1: Check GPU and Install Required Packages
# ==============================================================================

# Checking if GPU is available
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Installing required packages
!pip install scipy==1.11.4
!pip install lifelines==0.27.8
!pip install vaderSentiment==3.3.2
!pip install transformers==4.36.2
!pip install torch==2.1.2
!pip install pandas numpy matplotlib seaborn
!pip install accelerate==0.25.0

# ==============================================================================
# CELL 2: Import Libraries and Mount Google Drive
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Mount Google Drive to access your data files
from google.colab import drive
drive.mount('/content/drive')

import torch
import gc
import os
import formulaic
import transformers, tokenizers
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from transformers import pipeline
from tqdm import tqdm

# Set style for consistent visualizations
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
})
sns.set_style("whitegrid")

# Color palette for consistency
PALETTE = {
    'High Sentiment': '#1f78b4',
    'Low Sentiment': '#e31a1c',
    'Positive': '#1f78b4',
    'Neutral': '#a6cee3',
    'Negative': '#e31a1c',
}

# ==============================================================================
# CELL 3: Load Your Analyzer Class
# ==============================================================================
class ReviewEngagementSurvivalAnalysis:
    """
    Enhanced survival analysis class optimized for Google Colab environment.
    """

    def __init__(self, study_end_date='2023-12-31'):
        self.study_end_date = pd.Timestamp(study_end_date)
        self.vader_analyzer = SentimentIntensityAnalyzer()

        # Initialize transformer model with Colab-specific settings
        print("Loading transformer model...")
        try:
            self.transformer_sentiment = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                max_length=512,
                truncation=True
            )
            print("✓ Transformer model loaded successfully")
        except Exception as e:
            print(f"⚠ Transformer model loading failed: {e}")
            print("Falling back to VADER-only sentiment analysis")
            self.transformer_sentiment = None

        # Storage for different buffer period analyses
        self.buffer_analyses = {}

    def upload_data_from_drive(self):
        """
        Helper function to mount Google Drive and load data files.
        """
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("✓ Google Drive mounted successfully")

            # Update these paths to match your Drive structure
            business_file = '/content/drive/MyDrive/business.json'
            review_file = '/content/drive/MyDrive/review.json'

            return business_file, review_file

        except Exception as e:
            print(f"Drive mount error: {e}")
            print("Please upload files manually to Colab session storage")
            return None, None

    def load_sample_data_for_testing(self, n_businesses=1000, n_reviews_per_business=50):
        """
        Generate sample data for testing the pipeline when real data isn't available.
        """
        print(f"Generating sample data: {n_businesses} businesses, ~{n_reviews_per_business} reviews each")

        np.random.seed(42)

        # Generate sample businesses
        us_states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']

        businesses_data = []
        for i in range(n_businesses):
            businesses_data.append({
                'business_id': f'biz_{i}',
                'name': f'Restaurant {i}',
                'categories': 'Restaurants, Food',
                'state': np.random.choice(us_states),
                'stars': np.random.normal(3.5, 0.8),
                'review_count': np.random.randint(10, 500),
                'is_open': np.random.choice([0, 1], p=[0.2, 0.8])
            })

        businesses = pd.DataFrame(businesses_data)

        # Generate sample reviews
        reviews_data = []
        review_id = 0

        for business_id in businesses['business_id']:
            n_reviews = np.random.randint(5, n_reviews_per_business)
            start_date = pd.Timestamp('2018-01-01')
            end_date = pd.Timestamp('2023-12-31')

            # Generate review dates
            dates = pd.date_range(start_date, end_date, periods=n_reviews)

            for date in dates:
                # Generate realistic review text and ratings
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.35, 0.3])

                # Generate text based on rating
                if rating >= 4:
                    text_options = [
                        "Great food and excellent service! Highly recommend.",
                        "Amazing experience, will definitely come back!",
                        "Fantastic restaurant with delicious food.",
                        "Outstanding service and quality food.",
                        "Love this place! Great atmosphere and food."
                    ]
                elif rating == 3:
                    text_options = [
                        "Decent food, nothing special but okay.",
                        "Average experience, might try again.",
                        "Food was alright, service could be better.",
                        "Not bad, but not great either.",
                        "Mediocre food and service."
                    ]
                else:
                    text_options = [
                        "Terrible food and poor service. Very disappointed.",
                        "Would not recommend. Food was cold and tasteless.",
                        "Worst dining experience ever. Avoid this place.",
                        "Poor quality food and rude staff.",
                        "Disgusting food and horrible service."
                    ]

                reviews_data.append({
                    'review_id': f'review_{review_id}',
                    'business_id': business_id,
                    'stars': rating,
                    'date': date,
                    'text': np.random.choice(text_options),
                    'user_id': f'user_{np.random.randint(1, 10000)}'
                })
                review_id += 1

        reviews = pd.DataFrame(reviews_data)

        print(f"Generated sample dataset: {len(businesses)} businesses, {len(reviews)} reviews")

        self.restaurants = businesses
        self.reviews = reviews

        return businesses, reviews

    def analyze_sentiment_colab_optimized(self, text, max_length=300):
        """
        Colab-optimized sentiment analysis with memory management.
        """
        if pd.isna(text) or len(text) == 0:
            return 0.0

        # Truncate text for memory efficiency
        text = str(text)[:max_length]

        try:
            # VADER sentiment (always available)
            vader_scores = self.vader_analyzer.polarity_scores(text)
            vader_sentiment = vader_scores['compound']

            # Transformer sentiment (if available)
            if self.transformer_sentiment is not None:
                try:
                    transformer_result = self.transformer_sentiment(text)[0]

                    # Convert to numerical scale
                    label_mapping = {
                        'LABEL_0': -1,  # Negative
                        'LABEL_1': 0,   # Neutral
                        'LABEL_2': 1    # Positive
                    }

                    transformer_sentiment = label_mapping.get(
                        transformer_result['label'], 0
                    ) * transformer_result['score']

                    # Combined sentiment score
                    combined_sentiment = (vader_sentiment + transformer_sentiment) / 2
                    return combined_sentiment

                except Exception:
                    # Fall back to VADER only
                    return vader_sentiment
            else:
                return vader_sentiment

        except Exception:
            return 0.0

    def process_sentiment_batch_colab(self, sample_size=50000):
        """
        Process sentiment analysis in batches optimized for Colab memory constraints.
        """
        print("Processing sentiment analysis in batches...")

        # Sample reviews if dataset is too large
        if len(self.reviews) > sample_size:
            print(f"Sampling {sample_size} reviews from {len(self.reviews)} total")
            reviews_sample = self.reviews.sample(n=sample_size, random_state=42)
        else:
            reviews_sample = self.reviews.copy()

        # Process in smaller batches for memory efficiency
        batch_size = 1000
        sentiment_scores = []

        for i in tqdm(range(0, len(reviews_sample), batch_size), desc="Processing sentiment"):
            batch = reviews_sample.iloc[i:i+batch_size]
            batch_scores = batch['text'].apply(self.analyze_sentiment_colab_optimized)
            sentiment_scores.extend(batch_scores.tolist())

            # Memory cleanup every few batches
            if i % (batch_size * 5) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        reviews_sample['sentiment_score'] = sentiment_scores

        # Classify sentiment
        reviews_sample['sentiment_category'] = pd.cut(
            reviews_sample['sentiment_score'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['Negative', 'Neutral', 'Positive']
        )

        self.reviews_with_sentiment = reviews_sample
        print(f"Sentiment analysis completed for {len(reviews_sample)} reviews")

        return reviews_sample

    def create_survival_dataset(self, buffer_days=30):
        """Create survival dataset with engagement duration as the time variable.

        Parameters:
        -----------
        buffer_days : int
            Minimum days between first and last review to include restaurant

        Returns:
        --------
        pd.DataFrame : Survival dataset with engagement duration and covariates
        """
        print(f"Creating survival dataset with {buffer_days} day buffer...")

        if not hasattr(self, 'reviews_with_sentiment'):
            print("⚠ Running sentiment analysis first...")
            self.process_sentiment_batch_colab()

        # Group reviews by business to calculate engagement metrics
        business_stats = []

        for business_id, group in tqdm(self.reviews_with_sentiment.groupby('business_id'),
                                     desc="Processing businesses"):

            # Calculate engagement duration
            first_review = group['date'].min()
            last_review = group['date'].max()
            engagement_duration = (last_review - first_review).days + 1

            # Apply buffer filter
            if engagement_duration < buffer_days:
                continue

            # Calculate sentiment metrics
            avg_sentiment = group['sentiment_score'].mean()
            sentiment_volatility = group['sentiment_score'].std()

            # Handle single review case
            if pd.isna(sentiment_volatility):
                sentiment_volatility = 0.0

            # Calculate other metrics
            review_count = len(group)
            avg_star_rating = group['stars'].mean()

            # Get business info
            business_info = self.restaurants[self.restaurants['business_id'] == business_id].iloc[0]

            business_stats.append({
                'business_id': business_id,
                'engagement_duration': engagement_duration,
                'event': 1,  # All businesses have observable engagement end
                'avg_sentiment': avg_sentiment,
                'sentiment_volatility': sentiment_volatility,
                'review_count': review_count,
                'avg_star_rating': avg_star_rating,
                'business_stars': business_info.get('stars', 3.5),
                'business_review_count': business_info.get('review_count', 0),
                'is_open': business_info.get('is_open', 1),
                'state': business_info.get('state', 'Unknown'),
                'first_review_date': first_review,
                'last_review_date': last_review
            })

        survival_df = pd.DataFrame(business_stats)

        print(f"✓ Survival dataset created: {len(survival_df)} businesses")
        print(f"✓ Mean engagement duration: {survival_df['engagement_duration'].mean():.1f} days")
        print(f"✓ Median engagement duration: {survival_df['engagement_duration'].median():.1f} days")
        print(f"✓ Mean sentiment: {survival_df['avg_sentiment'].mean():.3f}")

        # Store for buffer analysis
        self.buffer_analyses[buffer_days] = survival_df

        return survival_df

    def perform_kaplan_meier_analysis(self, survival_df, sentiment_threshold=None):
        """
        Perform Kaplan-Meier survival analysis comparing high vs low sentiment groups.

        Parameters:
        -----------
        survival_df : pd.DataFrame
            Survival dataset from create_survival_dataset()
        sentiment_threshold : float, optional
            Threshold to split high/low sentiment groups (default: median)
        """
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        import matplotlib.pyplot as plt

        if sentiment_threshold is None:
            sentiment_threshold = survival_df['avg_sentiment'].median()

        # Create sentiment groups
        survival_df['sentiment_group'] = survival_df['avg_sentiment'].apply(
            lambda x: 'High Sentiment' if x >= sentiment_threshold else 'Low Sentiment'
        )

        print(f"Sentiment threshold: {sentiment_threshold:.3f}")
        print(f"High sentiment group: {(survival_df['sentiment_group'] == 'High Sentiment').sum()} businesses")
        print(f"Low sentiment group: {(survival_df['sentiment_group'] == 'Low Sentiment').sum()} businesses")

        # Fit Kaplan-Meier curves
        kmf = KaplanMeierFitter()

        plt.figure(figsize=(12, 8))

        # High sentiment group
        high_sentiment = survival_df[survival_df['sentiment_group'] == 'High Sentiment']
        kmf.fit(high_sentiment['engagement_duration'],
                high_sentiment['event'],
                label='High Sentiment')
        kmf.plot_survival_function(ci_show=True)

        # Low sentiment group
        low_sentiment = survival_df[survival_df['sentiment_group'] == 'Low Sentiment']
        kmf.fit(low_sentiment['engagement_duration'],
                low_sentiment['event'],
                label='Low Sentiment')
        kmf.plot_survival_function(ci_show=True)

        plt.title('Customer Review Engagement Duration by Sentiment Level', fontsize=14, fontweight='bold')
        plt.xlabel('Days of Review Engagement', fontsize=12)
        plt.ylabel('Probability of Continued Engagement', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Log-rank test
        results = logrank_test(
            high_sentiment['engagement_duration'],
            low_sentiment['engagement_duration'],
            high_sentiment['event'],
            low_sentiment['event']
        )

        print("\n" + "="*60)
        print("LOG-RANK TEST RESULTS")
        print("="*60)
        print(f"Test statistic: {results.test_statistic:.4f}")
        print(f"p-value: {results.p_value:.6f}")
        print(f"Significant difference: {'Yes' if results.p_value < 0.05 else 'No'}")

        # Store results
        self.kaplan_meier_results = {
            'sentiment_threshold': sentiment_threshold,
            'logrank_test': results,
            'survival_df_grouped': survival_df
        }

        return results

    def perform_cox_regression(self, survival_df):
        """
        Perform Cox Proportional Hazards regression analysis.

        Parameters:
        -----------
        survival_df : pd.DataFrame
            Survival dataset from create_survival_dataset()
        """
        from lifelines import CoxPHFitter

        # Prepare data for Cox regression
        cox_data = survival_df[[
            'engagement_duration', 'event',
            'avg_sentiment', 'sentiment_volatility',
            'review_count', 'avg_star_rating'
        ]].copy()

        # Log transform review count to handle skewness
        cox_data['log_review_count'] = np.log1p(cox_data['review_count'])

        # Standardize continuous variables for better interpretation
        continuous_vars = ['avg_sentiment', 'sentiment_volatility',
                          'log_review_count', 'avg_star_rating']

        for var in continuous_vars:
            cox_data[f'{var}_std'] = (cox_data[var] - cox_data[var].mean()) / cox_data[var].std()

        # Fit Cox model
        cph = CoxPHFitter()
        cox_features = ['avg_sentiment_std', 'sentiment_volatility_std',
                       'log_review_count_std', 'avg_star_rating_std']

        cph.fit(cox_data[['engagement_duration', 'event'] + cox_features],
                duration_col='engagement_duration',
                event_col='event')

        print("\n" + "="*80)
        print("COX PROPORTIONAL HAZARDS REGRESSION RESULTS")
        print("="*80)
        print("Note: Hazard Ratio < 1 = Lower risk of engagement cessation (longer engagement)")
        print("      Hazard Ratio > 1 = Higher risk of engagement cessation (shorter engagement)")
        print("-"*80)

        # Display results with interpretation
        summary = cph.summary
        for idx, row in summary.iterrows():
            var_name = idx.replace('_std', '').replace('_', ' ').title()
            hr = row['exp(coef)']
            p_val = row['p']
            ci_lower = row['exp(coef) lower 95%']
            ci_upper = row['exp(coef) upper 95%']

            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

            print(f"{var_name:20} HR: {hr:.3f} [{ci_lower:.3f}-{ci_upper:.3f}] p={p_val:.4f}{significance}")

            if hr < 1:
                effect = f"reduces engagement cessation risk by {(1-hr)*100:.1f}%"
            else:
                effect = f"increases engagement cessation risk by {(hr-1)*100:.1f}%"
            print(f"{' '*20} → {effect}")
            print()

        print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
        print(f"Concordance Index: {cph.concordance_index_:.3f}")
        print(f"Log-likelihood ratio test p-value: {cph.log_likelihood_ratio_test().p_value:.6f}")

        # Store results
        self.cox_results = {
            'model': cph,
            'summary': summary,
            'concordance': cph.concordance_index_
        }

        return cph

    def generate_comprehensive_report(self, buffer_days=60):
        """
        Generate a comprehensive analysis report.
        """
        print("="*100)
        print("COMPREHENSIVE CUSTOMER REVIEW ENGAGEMENT SURVIVAL ANALYSIS REPORT")
        print("="*100)

        # Create survival dataset
        survival_df = self.create_survival_dataset(buffer_days=buffer_days)

        print(f"\n📊 DATASET OVERVIEW")
        print("-"*50)
        print(f"Analysis Period: 2018-01-01 to 2023-12-31")
        print(f"Total Restaurants: {len(survival_df):,}")
        print(f"Total Reviews Analyzed: {len(self.reviews_with_sentiment):,}")
        print(f"Buffer Period: {buffer_days} days minimum engagement duration")

        # Descriptive statistics
        print(f"\n📈 ENGAGEMENT DURATION STATISTICS")
        print("-"*50)
        print(f"Mean Engagement Duration: {survival_df['engagement_duration'].mean():.1f} days")
        print(f"Median Engagement Duration: {survival_df['engagement_duration'].median():.1f} days")
        print(f"Standard Deviation: {survival_df['engagement_duration'].std():.1f} days")
        print(f"Range: {survival_df['engagement_duration'].min():.0f} - {survival_df['engagement_duration'].max():.0f} days")

        # Sentiment statistics
        print(f"\n😊 SENTIMENT ANALYSIS OVERVIEW")
        print("-"*50)
        print(f"Mean Sentiment Score: {survival_df['avg_sentiment'].mean():.3f}")
        print(f"Mean Sentiment Volatility: {survival_df['sentiment_volatility'].mean():.3f}")

        sentiment_categories = self.reviews_with_sentiment['sentiment_category'].value_counts()
        for category, count in sentiment_categories.items():
            percentage = (count / len(self.reviews_with_sentiment)) * 100
            print(f"{category} Reviews: {count:,} ({percentage:.1f}%)")

        # Perform Kaplan-Meier analysis
        print(f"\n⏱️ KAPLAN-MEIER SURVIVAL ANALYSIS")
        print("-"*50)
        km_results = self.perform_kaplan_meier_analysis(survival_df)

        # Perform Cox regression
        print(f"\n📊 COX PROPORTIONAL HAZARDS ANALYSIS")
        cox_model = self.perform_cox_regression(survival_df)

        # Key insights
        print(f"\n🔍 KEY INSIGHTS")
        print("-"*50)

        # Sentiment impact
        high_sentiment_median = survival_df[survival_df['avg_sentiment'] >= survival_df['avg_sentiment'].median()]['engagement_duration'].median()
        low_sentiment_median = survival_df[survival_df['avg_sentiment'] < survival_df['avg_sentiment'].median()]['engagement_duration'].median()

        print(f"• High sentiment restaurants maintain engagement for {high_sentiment_median:.0f} days (median)")
        print(f"• Low sentiment restaurants maintain engagement for {low_sentiment_median:.0f} days (median)")
        print(f"• Difference: {high_sentiment_median - low_sentiment_median:.0f} days longer for high sentiment")

        if hasattr(self, 'cox_results'):
            avg_sent_hr = self.cox_results['summary'].loc['avg_sentiment_std', 'exp(coef)']
            vol_hr = self.cox_results['summary'].loc['sentiment_volatility_std', 'exp(coef)']

            print(f"• Each standard deviation increase in sentiment reduces cessation risk by {(1-avg_sent_hr)*100:.1f}%")
            print(f"• Each standard deviation increase in volatility changes cessation risk by {(vol_hr-1)*100:.1f}%")

        print(f"\n✅ ANALYSIS COMPLETE")
        print("="*100)

        return survival_df
    
# ==============================================================================
# CELL 4: Load Your Data Files
# ==============================================================================

# Update these paths to match your Drive structure
business_file = '/content/drive/MyDrive/business.json'
review_file = '/content/drive/MyDrive/review.json'

# Initialize analyzer
print("Initializing analyzer...")
analyzer = ReviewEngagementSurvivalAnalysis()

# Load data
print("Loading business data...")
import json
businesses = []
with open(business_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            business = json.loads(line)
            businesses.append(business)
        except:
            continue

analyzer.restaurants = pd.DataFrame(businesses)

# Filter for US restaurants
us_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
             'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
             'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
             'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
             'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

restaurant_categories = ['restaurant', 'food', 'pizza', 'cafe', 'bar', 'diner']

analyzer.restaurants = analyzer.restaurants[
    (analyzer.restaurants['state'].isin(us_states)) &
    (analyzer.restaurants['categories'].str.lower().str.contains('|'.join(restaurant_categories), na=False))
]

print(f"Filtered to {len(analyzer.restaurants):,} US restaurants")

# Load reviews (with GPU memory management)
print("Loading review data...")
restaurant_ids = set(analyzer.restaurants['business_id'])
reviews = []

with open(review_file, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f):
        if line_num % 100000 == 0:
            print(f"  Processed {line_num:,} review lines...")
            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        try:
            review = json.loads(line)
            if review['business_id'] in restaurant_ids:
                review_date = pd.Timestamp(review['date'])
                if pd.Timestamp('2018-01-01') <= review_date <= pd.Timestamp('2023-12-31'):
                    reviews.append(review)
        except:
            continue

analyzer.reviews = pd.DataFrame(reviews)
analyzer.reviews['date'] = pd.to_datetime(analyzer.reviews['date'])

print(f"Loaded {len(analyzer.reviews):,} relevant reviews")

# ==============================================================================
# CELL 5: Run Sentiment Analysis with GPU
# ==============================================================================

# Monitor GPU memory before processing
if torch.cuda.is_available():
    print(f"GPU Memory before sentiment analysis: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

# Run sentiment analysis with GPU acceleration
analyzer.transformer_sentiment = None  # Disable transformer
analyzer.process_sentiment_batch_colab(sample_size=10000000)  # Process all reviews

# Monitor GPU memory after processing
if torch.cuda.is_available():
    print(f"GPU Memory after sentiment analysis: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
    torch.cuda.empty_cache()
    print("GPU cache cleared")

# ==============================================================================
# CELL 6: Create Survival Dataset
# ==============================================================================

# Create survival dataset
survival_df = analyzer.create_survival_dataset(buffer_days=60)

print(f"\nSurvival dataset created:")
print(f"Total restaurants: {len(survival_df):,}")
print(f"Mean engagement duration: {survival_df['engagement_duration'].mean():.1f} days")
print(f"Median engagement duration: {survival_df['engagement_duration'].median():.1f} days")

# ==============================================================================
# CELL 7: Save Results to Drive
# ==============================================================================

# Save processed data to Drive for future use
output_path = '/content/drive/MyDrive/yelp_analysis_results/'
import os
os.makedirs(output_path, exist_ok=True)

# Save survival dataset
survival_df.to_csv(f'{output_path}survival_dataset.csv', index=False)
print(f"Saved survival dataset to {output_path}survival_dataset.csv")

# Save reviews with sentiment
analyzer.reviews_with_sentiment.to_csv(f'{output_path}reviews_with_sentiment.csv', index=False)
print(f"Saved sentiment analysis to {output_path}reviews_with_sentiment.csv")

# ==============================================================================
# CELL 8: Run Your Analysis Code
# ==============================================================================

# It will use the GPU-processed data

# Import required libraries for analysis
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# Set visualization style
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
})
sns.set_style("whitegrid")

# ==============================================================================
# CELL 9: Load Analysis Results
# ==============================================================================

# Load the analysis results from your analyzer
survival_df = analyzer.buffer_analyses[60].copy()  # Using 60-day buffer analysis

print(f"Dataset Overview:")
print(f"Total restaurants analyzed: {len(survival_df):,}")
print(f"Mean engagement duration: {survival_df['engagement_duration'].mean():.1f} days")
print(f"Median engagement duration: {survival_df['engagement_duration'].median():.1f} days")
print(f"Mean sentiment: {survival_df['avg_sentiment'].mean():.3f}")
print(f"Mean sentiment volatility: {survival_df['sentiment_volatility'].mean():.3f}")

# ==============================================================================
# CELL 10: Engagement Duration Distribution Analysis
# ==============================================================================

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(survival_df['engagement_duration'], bins=40, kde=True, ax=ax, color='#4c78a8')
ax.set_title('Distribution of Customer Review Engagement Duration')
ax.set_xlabel('Engagement Duration (days)')
ax.set_ylabel('Number of Restaurants')

# Add median and mean lines
med = survival_df['engagement_duration'].median()
mean = survival_df['engagement_duration'].mean()
ax.axvline(med, color='black', linestyle='--', linewidth=1.2, label=f'Median = {med:.0f}')
ax.axvline(mean, color='gray', linestyle=':', linewidth=1.2, label=f'Mean = {mean:.0f}')
ax.legend()
plt.tight_layout()
plt.savefig('fig_engagement_duration_dist.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# CELL 11: Sentiment Analysis Results
# ==============================================================================

# Create sentiment groups based on median
sentiment_threshold = survival_df['avg_sentiment'].median()
survival_df['sentiment_group'] = np.where(
    survival_df['avg_sentiment'] >= sentiment_threshold,
    'High Sentiment',
    'Low Sentiment'
)

# Summary statistics by sentiment group
print("\nSentiment Group Statistics:")
for group in ['High Sentiment', 'Low Sentiment']:
    data = survival_df[survival_df['sentiment_group'] == group]
    print(f"\n{group} (n={len(data)}):")
    print(f"  Mean engagement: {data['engagement_duration'].mean():.1f} days")
    print(f"  Median engagement: {data['engagement_duration'].median():.1f} days")
    print(f"  Mean sentiment: {data['avg_sentiment'].mean():.3f}")
    print(f"  Mean volatility: {data['sentiment_volatility'].mean():.3f}")

# ==============================================================================
# CELL 12: Kaplan-Meier Survival Analysis
# ==============================================================================

kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(10, 6))

# Plot survival curves for each sentiment group
for label, color in [('High Sentiment', PALETTE['High Sentiment']),
                     ('Low Sentiment', PALETTE['Low Sentiment'])]:
    data = survival_df[survival_df['sentiment_group'] == label]
    kmf.fit(durations=data['engagement_duration'],
            event_observed=data['event'],
            label=f'{label} (n={len(data)})')
    kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2)

ax.set_title('Customer Review Engagement Duration by Sentiment Group (KM Curves)')
ax.set_xlabel('Days of Review Engagement')
ax.set_ylabel('Probability of Continued Engagement')

# Perform log-rank test
hi = survival_df[survival_df['sentiment_group'] == 'High Sentiment']
lo = survival_df[survival_df['sentiment_group'] == 'Low Sentiment']
lr = logrank_test(
    hi['engagement_duration'],
    lo['engagement_duration'],
    event_observed_A=hi['event'],
    event_observed_B=lo['event']
)

# Add test results to plot
ax.text(0.65, 0.15, f'Log-rank p = {lr.p_value:.3e}',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.9))

plt.tight_layout()
plt.savefig('fig_km_sentiment_groups.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# CELL 13: Cox Proportional Hazards Regression
# ==============================================================================

# Prepare data for Cox regression
cox_data = survival_df[['engagement_duration', 'event', 'avg_sentiment',
                        'sentiment_volatility', 'review_count', 'avg_star_rating']].copy()

# Log transform review count
cox_data['log_review_count'] = np.log1p(cox_data['review_count'])

# Standardize continuous variables
continuous_vars = ['avg_sentiment', 'sentiment_volatility', 'log_review_count', 'avg_star_rating']
for var in continuous_vars:
    cox_data[f'{var}_std'] = (cox_data[var] - cox_data[var].mean()) / cox_data[var].std()

# Fit Cox model
cph = CoxPHFitter()
cox_features = ['avg_sentiment_std', 'sentiment_volatility_std',
                'log_review_count_std', 'avg_star_rating_std']

cph.fit(cox_data[['engagement_duration', 'event'] + cox_features],
        duration_col='engagement_duration',
        event_col='event')

# Print results
print("\n" + "="*80)
print("COX PROPORTIONAL HAZARDS REGRESSION RESULTS")
print("="*80)
print("Note: Hazard Ratio < 1 = Lower risk of engagement cessation (longer engagement)")
print("      Hazard Ratio > 1 = Higher risk of engagement cessation (shorter engagement)")
print("-"*80)

# Display results with interpretation
summary = cph.summary
for idx, row in summary.iterrows():
    var_name = idx.replace('_std', '').replace('_', ' ').title()
    hr = row['exp(coef)']
    p_val = row['p']
    ci_lower = row['exp(coef) lower 95%']
    ci_upper = row['exp(coef) upper 95%']

    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

    print(f"{var_name:20} HR: {hr:.3f} [{ci_lower:.3f}-{ci_upper:.3f}] p={p_val:.4f}{significance}")

    if hr < 1:
        effect = f"reduces engagement cessation risk by {(1-hr)*100:.1f}%"
    else:
        effect = f"increases engagement cessation risk by {(hr-1)*100:.1f}%"
    print(f"{' '*20} → {effect}")
    print()

print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
print(f"Concordance Index: {cph.concordance_index_:.3f}")
print(f"Log-likelihood ratio test p-value: {cph.log_likelihood_ratio_test().p_value:.6f}")

# Store results
analyzer.cox_results = {
    'model': cph,
    'summary': summary,
    'concordance': cph.concordance_index_
}

# ==============================================================================
# CELL 14: Forest Plot for Cox Model Results
# ==============================================================================

from matplotlib.collections import LineCollection

# Prepare data for forest plot
summ = cph.summary.copy()
rename = {
    'avg_sentiment_std': 'Avg sentiment (std)',
    'sentiment_volatility_std': 'Sentiment volatility (std)',
    'log_review_count_std': 'Log review count (std)',
    'avg_star_rating_std': 'Avg review star (std)'
}
summ = summ.loc[rename.keys()]
summ['Var'] = summ.index.map(rename)
summ['HR'] = summ['exp(coef)']
summ['CI_lo'] = summ['exp(coef) lower 95%']
summ['CI_hi'] = summ['exp(coef) upper 95%']

fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(summ))

# Plot hazard ratios with confidence intervals
ax.errorbar(summ['HR'], y,
            xerr=[summ['HR'] - summ['CI_lo'], summ['CI_hi'] - summ['HR']],
            fmt='o', color='#333333', ecolor='#666666',
            elinewidth=2, capsize=4)

ax.axvline(1.0, color='red', linestyle='--', alpha=0.6, label='HR = 1 (no effect)')
ax.set_yticks(y)
ax.set_yticklabels(summ['Var'])
ax.set_xlabel('Hazard Ratio (engagement cessation risk)')
ax.set_title('Cox PH Model – Hazard Ratios with 95% CI')
ax.legend(loc='lower right')

# Add effect direction hints
for i, row in summ.iterrows():
    effect = "↓ risk / longer engagement" if row['HR'] < 1 else "↑ risk / shorter engagement"
    ax.text(max(row['CI_hi'], 1.05), y[summ.index.get_loc(i)], effect, va='center')

plt.tight_layout()
plt.savefig('fig_cox_forest.png', dpi=300, bbox_inches='tight')
plt.show()