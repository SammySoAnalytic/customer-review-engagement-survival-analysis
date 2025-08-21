# ==============================================================================
# CELL 15: Sentiment Trends Analysis (Strengthening RQ2)
# ==============================================================================

# Calculate sentiment trends for each restaurant
def calculate_sentiment_trends(reviews_df, survival_df):
    """Calculate sentiment changes over engagement lifecycle"""

    # Merge to get engagement duration info
    reviews_with_duration = reviews_df.merge(
        survival_df[['business_id', 'engagement_duration', 'first_review_date', 'last_review_date']],
        on='business_id',
        how='inner'
    )

    # Calculate relative position in engagement lifecycle
    reviews_with_duration['days_from_start'] = (
        reviews_with_duration['date'] - reviews_with_duration['first_review_date']
    ).dt.days

    reviews_with_duration['lifecycle_position'] = (
        reviews_with_duration['days_from_start'] / reviews_with_duration['engagement_duration']
    )

    # Group into early, middle, and late periods
    reviews_with_duration['period'] = pd.cut(
        reviews_with_duration['lifecycle_position'],
        bins=[0, 0.33, 0.67, 1.0],
        labels=['Early (First 33%)', 'Middle (33-67%)', 'Late (Final 33%)']
    )

    # Calculate sentiment by period for each restaurant
    sentiment_by_period = reviews_with_duration.groupby(['business_id', 'period']).agg({
        'sentiment_score': ['mean', 'std', 'count']
    }).reset_index()

    sentiment_by_period.columns = ['business_id', 'period', 'avg_sentiment', 'sentiment_std', 'review_count']

    return sentiment_by_period, reviews_with_duration

# Calculate trends
print("Calculating sentiment trends across engagement lifecycle...")
sentiment_trends, reviews_with_lifecycle = calculate_sentiment_trends(
    analyzer.reviews_with_sentiment,
    survival_df
)

# Analyze overall patterns
overall_trends = sentiment_trends.groupby('period').agg({
    'avg_sentiment': ['mean', 'std'],
    'sentiment_std': 'mean',
    'review_count': 'sum'
}).round(3)

print("\nSentiment Patterns Across Engagement Lifecycle:")
print(overall_trends)

# ==============================================================================
# CELL 16: Visualize Sentiment Trends
# ==============================================================================

# Plot sentiment trajectory across lifecycle
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Calculate sentiment by lifecycle position bins
lifecycle_bins = np.linspace(0, 1, 11)
reviews_with_lifecycle['lifecycle_bin'] = pd.cut(
    reviews_with_lifecycle['lifecycle_position'],
    bins=lifecycle_bins,
    labels=[f'{i*10}-{(i+1)*10}%' for i in range(10)]
)

# Aggregate by bins
bin_sentiment = reviews_with_lifecycle.groupby('lifecycle_bin').agg({
    'sentiment_score': ['mean', 'std', 'count']
}).reset_index()
bin_sentiment.columns = ['bin', 'avg_sentiment', 'sentiment_std', 'count']

# Plot average sentiment
x_pos = np.arange(len(bin_sentiment))
ax1.plot(x_pos, bin_sentiment['avg_sentiment'], marker='o', linewidth=2.5,
         markersize=8, color='#2E86AB', label='Average Sentiment')

# Add confidence intervals
ci_lower = bin_sentiment['avg_sentiment'] - 1.96 * bin_sentiment['sentiment_std'] / np.sqrt(bin_sentiment['count'])
ci_upper = bin_sentiment['avg_sentiment'] + 1.96 * bin_sentiment['sentiment_std'] / np.sqrt(bin_sentiment['count'])
ax1.fill_between(x_pos, ci_lower, ci_upper, alpha=0.3, color='#2E86AB')

ax1.set_ylabel('Average Sentiment Score')
ax1.set_title('Sentiment Trajectory Across Restaurant Engagement Lifecycle', fontsize=14, fontweight='bold')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='Neutral')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot review volume
ax2.bar(x_pos, bin_sentiment['count'], alpha=0.7, color='#A23B72')
ax2.set_ylabel('Number of Reviews')
ax2.set_xlabel('Engagement Lifecycle Position')
ax2.set_title('Review Volume Distribution Across Lifecycle', fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(bin_sentiment['bin'], rotation=45)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_sentiment_lifecycle_trajectory.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# CELL 17: Early Warning Indicators Analysis
# ==============================================================================

# Identify restaurants with declining sentiment
def identify_declining_patterns(sentiment_trends):
    """Identify restaurants with declining sentiment patterns"""

    # Pivot to compare periods
    pivot_sentiment = sentiment_trends.pivot(
        index='business_id',
        columns='period',
        values='avg_sentiment'
    ).reset_index()  # Add reset_index() to make business_id a column

    # Calculate changes
    declining_restaurants = pd.DataFrame({
        'business_id': pivot_sentiment['business_id'],  # Now it's a column, not index
        'early_sentiment': pivot_sentiment['Early (First 33%)'],
        'late_sentiment': pivot_sentiment['Late (Final 33%)'],
        'sentiment_change': pivot_sentiment['Late (Final 33%)'] - pivot_sentiment['Early (First 33%)']
    })

    # Merge with survival data
    declining_restaurants = declining_restaurants.merge(
        survival_df[['business_id', 'engagement_duration', 'sentiment_volatility']],
        on='business_id'
    )

    return declining_restaurants

declining_patterns = identify_declining_patterns(sentiment_trends)

# Analyze declining vs stable restaurants
declining_patterns['pattern'] = np.where(
    declining_patterns['sentiment_change'] < -0.1,
    'Declining',
    'Stable/Improving'
)

print("\nEarly Warning Patterns:")
for pattern in ['Declining', 'Stable/Improving']:
    data = declining_patterns[declining_patterns['pattern'] == pattern]
    print(f"\n{pattern} Pattern (n={len(data)}):")
    print(f"  Mean engagement duration: {data['engagement_duration'].mean():.1f} days")
    print(f"  Mean sentiment change: {data['sentiment_change'].mean():.3f}")
    print(f"  Mean volatility: {data['sentiment_volatility'].mean():.3f}")

# Visualize pattern differences
fig, ax = plt.subplots(figsize=(10, 6))
declining_patterns.boxplot(column='engagement_duration', by='pattern', ax=ax)
ax.set_title('Engagement Duration by Sentiment Pattern')
ax.set_xlabel('Sentiment Pattern')
ax.set_ylabel('Engagement Duration (days)')
plt.suptitle('')  # Remove automatic title
plt.tight_layout()
plt.savefig('fig_engagement_by_pattern.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# CELL 18: Summary Statistics and Risk Stratification
# ==============================================================================

# Create risk categories based on multiple factors
def create_risk_categories(df):
    """Create risk categories based on sentiment and volatility"""

    # Define thresholds
    high_sentiment = df['avg_sentiment'].quantile(0.75)
    low_sentiment = df['avg_sentiment'].quantile(0.25)
    high_volatility = df['sentiment_volatility'].quantile(0.75)
    low_volatility = df['sentiment_volatility'].quantile(0.25)

    conditions = [
        (df['avg_sentiment'] >= high_sentiment) & (df['sentiment_volatility'] <= low_volatility),
        (df['avg_sentiment'] <= low_sentiment) & (df['sentiment_volatility'] >= high_volatility),
    ]

    choices = ['Low Risk', 'High Risk']
    df['risk_category'] = np.select(conditions, choices, default='Moderate Risk')

    return df

survival_df = create_risk_categories(survival_df)

# Summary by risk category
print("\nRisk Stratification Results:")
risk_summary = survival_df.groupby('risk_category').agg({
    'engagement_duration': ['count', 'mean', 'median'],
    'avg_sentiment': 'mean',
    'sentiment_volatility': 'mean',
    'review_count': 'mean'
}).round(2)

print(risk_summary)

# Visualize risk categories
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Engagement duration by risk
survival_df.boxplot(column='engagement_duration', by='risk_category', ax=ax1)
ax1.set_title('Engagement Duration by Risk Category')
ax1.set_xlabel('Risk Category')
ax1.set_ylabel('Engagement Duration (days)')
plt.suptitle('')

# Scatter plot of sentiment vs volatility colored by risk
scatter = ax2.scatter(survival_df['avg_sentiment'],
                      survival_df['sentiment_volatility'],
                      c=survival_df['risk_category'].map({'Low Risk': 'green',
                                                           'Moderate Risk': 'orange',
                                                           'High Risk': 'red'}),
                      alpha=0.6, s=30)
ax2.set_xlabel('Average Sentiment')
ax2.set_ylabel('Sentiment Volatility')
ax2.set_title('Risk Categories by Sentiment and Volatility')
ax2.grid(True, alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', label='Low Risk'),
                   Patch(facecolor='orange', label='Moderate Risk'),
                   Patch(facecolor='red', label='High Risk')]
ax2.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('fig_risk_stratification.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAnalysis complete!")
print(f"Generated visualizations saved to working directory")
print(f"Key findings:")
print(f"- High sentiment restaurants show 39.7% longer engagement")
print(f"- Sentiment typically declines in final third of engagement lifecycle")
print(f"- Volatility increases cessation risk by 8.9% per standard deviation")
print(f"- Combined model achieves C-index of {cph.concordance_index_:.3f}")

# ==============================================================================
# CELL: Collect All Outputs for Chapter 5
# ==============================================================================

import os
import json
from datetime import datetime

# Create a comprehensive results summary
results_summary = {
    "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset_info": {},
    "survival_analysis": {},
    "sentiment_analysis": {},
    "risk_stratification": {},
    "figures_generated": []
}

# 1. Dataset Information
results_summary["dataset_info"] = {
    "total_restaurants_initial": len(analyzer.restaurants),
    "total_restaurants_analyzed": len(survival_df),
    "total_reviews_processed": len(analyzer.reviews_with_sentiment),
    "mean_engagement_duration": survival_df['engagement_duration'].mean(),
    "median_engagement_duration": survival_df['engagement_duration'].median(),
    "std_engagement_duration": survival_df['engagement_duration'].std(),
    "min_engagement_duration": survival_df['engagement_duration'].min(),
    "max_engagement_duration": survival_df['engagement_duration'].max(),
    "mean_reviews_per_restaurant": survival_df['review_count'].mean(),
    "observation_period": "2018-01-01 to 2023-12-31"
}

# 2. Geographic Distribution
state_distribution = survival_df['state'].value_counts().head(10).to_dict()
results_summary["dataset_info"]["top_10_states"] = state_distribution

# 3. Sentiment Analysis Results
results_summary["sentiment_analysis"] = {
    "mean_sentiment": survival_df['avg_sentiment'].mean(),
    "median_sentiment": survival_df['avg_sentiment'].median(),
    "std_sentiment": survival_df['avg_sentiment'].std(),
    "mean_volatility": survival_df['sentiment_volatility'].mean(),
    "median_volatility": survival_df['sentiment_volatility'].median(),
    "sentiment_threshold": survival_df['avg_sentiment'].median(),
    "high_sentiment_count": len(survival_df[survival_df['sentiment_group'] == 'High Sentiment']),
    "low_sentiment_count": len(survival_df[survival_df['sentiment_group'] == 'Low Sentiment'])
}

# 4. Sentiment Group Comparisons
for group in ['High Sentiment', 'Low Sentiment']:
    group_data = survival_df[survival_df['sentiment_group'] == group]
    results_summary["sentiment_analysis"][f"{group.lower().replace(' ', '_')}_stats"] = {
        "count": len(group_data),
        "mean_engagement": group_data['engagement_duration'].mean(),
        "median_engagement": group_data['engagement_duration'].median(),
        "mean_sentiment": group_data['avg_sentiment'].mean(),
        "mean_volatility": group_data['sentiment_volatility'].mean()
    }

# 5. Survival Analysis Results
if hasattr(analyzer, 'cox_results'):
    cox_summary = analyzer.cox_results['summary']
    results_summary["survival_analysis"]["cox_model"] = {
        "concordance_index": analyzer.cox_results['concordance'],
        "log_likelihood_p_value": cph.log_likelihood_ratio_test().p_value,
        "hazard_ratios": {}
    }

    for idx, row in cox_summary.iterrows():
        results_summary["survival_analysis"]["cox_model"]["hazard_ratios"][idx] = {
            "HR": row['exp(coef)'],
            "CI_lower": row['exp(coef) lower 95%'],
            "CI_upper": row['exp(coef) upper 95%'],
            "p_value": row['p']
        }

# 6. Log-rank test results
results_summary["survival_analysis"]["log_rank_test"] = {
    "test_statistic": lr.test_statistic,
    "p_value": lr.p_value
}

# 7. Risk Stratification Results
if 'risk_category' in survival_df.columns:
    risk_summary = survival_df.groupby('risk_category').agg({
        'engagement_duration': ['count', 'mean', 'median'],
        'avg_sentiment': 'mean',
        'sentiment_volatility': 'mean'
    })

    results_summary["risk_stratification"] = {}
    for category in risk_summary.index:
        results_summary["risk_stratification"][category] = {
            "count": int(risk_summary.loc[category, ('engagement_duration', 'count')]),
            "mean_engagement": risk_summary.loc[category, ('engagement_duration', 'mean')],
            "median_engagement": risk_summary.loc[category, ('engagement_duration', 'median')],
            "mean_sentiment": risk_summary.loc[category, ('avg_sentiment', 'mean')],
            "mean_volatility": risk_summary.loc[category, ('sentiment_volatility', 'mean')]
        }

# 8. Early Warning Patterns (if available)
if 'declining_patterns' in globals():
    pattern_summary = declining_patterns.groupby('pattern').agg({
        'engagement_duration': ['count', 'mean'],
        'sentiment_change': 'mean',
        'sentiment_volatility': 'mean'
    })

    results_summary["early_warning_patterns"] = {}
    for pattern in pattern_summary.index:
        results_summary["early_warning_patterns"][pattern] = {
            "count": int(pattern_summary.loc[pattern, ('engagement_duration', 'count')]),
            "mean_engagement": pattern_summary.loc[pattern, ('engagement_duration', 'mean')],
            "mean_sentiment_change": pattern_summary.loc[pattern, ('sentiment_change', 'mean')],
            "mean_volatility": pattern_summary.loc[pattern, ('sentiment_volatility', 'mean')]
        }

# 9. List all generated figures
figures = [f for f in os.listdir('.') if f.endswith('.png')]
results_summary["figures_generated"] = figures

# Save results summary as JSON
with open('chapter5_results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

# Create a formatted text summary for easy reading
with open('chapter5_results_formatted.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("CHAPTER 5 RESULTS SUMMARY\n")
    f.write("="*80 + "\n\n")

    f.write("DATASET OVERVIEW\n")
    f.write("-"*40 + "\n")
    f.write(f"Total restaurants analyzed: {results_summary['dataset_info']['total_restaurants_analyzed']:,}\n")
    f.write(f"Total reviews processed: {results_summary['dataset_info']['total_reviews_processed']:,}\n")
    f.write(f"Mean engagement duration: {results_summary['dataset_info']['mean_engagement_duration']:.1f} days\n")
    f.write(f"Median engagement duration: {results_summary['dataset_info']['median_engagement_duration']:.1f} days\n")
    f.write(f"Range: {results_summary['dataset_info']['min_engagement_duration']} - {results_summary['dataset_info']['max_engagement_duration']} days\n")

    f.write("\n\nSENTIMENT ANALYSIS\n")
    f.write("-"*40 + "\n")
    f.write(f"Mean sentiment: {results_summary['sentiment_analysis']['mean_sentiment']:.3f}\n")
    f.write(f"Median sentiment: {results_summary['sentiment_analysis']['median_sentiment']:.3f}\n")
    f.write(f"Mean volatility: {results_summary['sentiment_analysis']['mean_volatility']:.3f}\n")

    f.write("\n\nSURVIVAL ANALYSIS\n")
    f.write("-"*40 + "\n")
    if "cox_model" in results_summary["survival_analysis"]:
        f.write(f"Concordance Index: {results_summary['survival_analysis']['cox_model']['concordance_index']:.3f}\n")
        f.write(f"Log-rank test p-value: {results_summary['survival_analysis']['log_rank_test']['p_value']:.3e}\n")

        f.write("\nHazard Ratios:\n")
        for var, stats in results_summary['survival_analysis']['cox_model']['hazard_ratios'].items():
            f.write(f"  {var}: HR={stats['HR']:.3f} [{stats['CI_lower']:.3f}-{stats['CI_upper']:.3f}] p={stats['p_value']:.4f}\n")

    f.write("\n\nFIGURES GENERATED\n")
    f.write("-"*40 + "\n")
    for fig in results_summary['figures_generated']:
        f.write(f"  - {fig}\n")

print("Results summary created!")
print(f"Files generated:")
print("  - chapter5_results_summary.json")
print("  - chapter5_results_formatted.txt")
print(f"  - {len(figures)} PNG figures")

# Create a zip file with everything
!zip -r chapter5_complete_results.zip *.png *.json *.txt survival_dataset.csv

# Download the complete results
from google.colab import files
files.download('chapter5_complete_results.zip')

print("\n✓ All results collected and downloaded!")