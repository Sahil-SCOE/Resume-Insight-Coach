# Resume Dataset Visualization Insights

## Dataset Summary

- Total resumes: 400
- Total categories: 10
- Resumes per category: 40 each

## Key Observation

The dataset is perfectly balanced. That means the model is less likely to be biased toward a single category during training, which is useful for classification experiments.

## Keyword Trends

The synthetic resume text contains strong role-specific keywords such as:
- Data and analytics: Python, SQL, statistics, data
- Engineering roles: Java, engineering, developer, software
- Business and operations: business, analyst, marketing, HR

## Dashboard Highlights

The dashboard focuses on three important views:
1. Category distribution
2. Resume length distribution by category
3. Top keyword frequency

## Why this matters

This dataset is a solid beginner-friendly demo for:
- resume category classification
- text analytics
- dashboard storytelling
- simple AI-assisted resume coaching workflows

## Local Run Flow

```bash
python generate_dataset.py
python model/train_model.py
streamlit run src/app.py
streamlit run src/resume_dashboard.py
```
