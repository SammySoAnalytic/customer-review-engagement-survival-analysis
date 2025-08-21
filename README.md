# Customer Review Engagement Duration Analysis

**MSc Data Science Thesis - Coventry University (2024/25)**  
**Author:** Samuel Oyakhilome Kwasi Nyarkotey  
**Supervisors:** Prof. David Oyebisi, Prof. Ian Dunwell

## Abstract

This study examines the relationship between customer sentiment patterns and review engagement duration in restaurants using sentiment analysis and survival modelling techniques on Yelp data (51,063 restaurants, 2.2M reviews, 2018-2023).

## Key Findings

- **Sentiment Paradox**: Higher sentiment associated with shorter engagement (HR = 1.036)
- **Volume Dominance**: Review volume strongest predictor (58.9% risk reduction per SD)
- **Volatility as Vitality**: Moderate sentiment volatility extends engagement (HR = 0.945)
- **Natural Decline**: Universal 19% sentiment decline across lifecycle

## Repository Contents

- `src/` - Core analysis code (VADER sentiment + Cox regression)
- `data/` - Processed datasets and results
- `notebooks/` - Jupyter analysis notebooks
- `figures/` - Publication-quality visualisations
- `docs/` - Methodology and results documentation

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/customer-review-engagement-survival-analysis.git
cd customer-review-engagement-survival-analysis

# Install dependencies
pip install -r requirements.txt

# Run analysis
python src/Project_Code.py
