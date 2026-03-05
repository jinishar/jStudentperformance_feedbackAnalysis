# Smart Student Feedback & Performance Dashboard

An interactive Streamlit dashboard that analyzes student academic performance and feedback sentiment using Natural Language Processing (NLP).

This project combines marks data and student feedback to generate insights about:

- Academic performance
- Student satisfaction
- At-risk students
- Course improvement areas

Built using Python, Streamlit, Plotly, and NLTK.

---

# Project Overview

Universities collect large amounts of data such as:

- Student marks
- Student attendance
- Student feedback

However, these datasets are often analyzed separately.

This dashboard integrates both academic data and feedback sentiment analysis to help educators understand:

- How student satisfaction impacts academic performance
- Which courses need improvement
- Which students may need support

---

# Features

## Academic Performance Dashboard
- Student-wise performance visualization
- Grade distribution charts
- Subject-wise average marks
- Attendance vs performance analysis
- Top-performing students
- At-risk students detection

---

## Feedback Sentiment Analysis

Uses VADER Sentiment Analysis (NLTK) to analyze student feedback.

Features include:

- Positive / Neutral / Negative sentiment detection
- Sentiment score calculation
- Keyword extraction
- Word cloud visualization
- Individual feedback viewer

---

## Combined Insights

The system merges academic performance and sentiment analysis to generate insights like:

- Relationship between feedback sentiment and academic scores
- Branch-wise sentiment comparison
- Students failing with negative feedback
- High-performing but dissatisfied students
- Data-driven recommendations for faculty

---

# Tech Stack

| Technology | Purpose |
|-----------|--------|
| Python | Core programming |
| Streamlit | Interactive dashboard |
| Pandas | Data processing |
| Plotly | Data visualization |
| NLTK | Sentiment analysis |
| WordCloud | Keyword visualization |
| Matplotlib | Plot rendering |

---

# Project Structure

```
student-feedback-dashboard/
│
├── app.py
├── sample_data.csv
├── feedback_data.csv
├── requirements.txt
└── README.md
```

---

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

---

# Dashboard Sections

### Performance Dashboard
Shows academic analytics including:

- student performance
- grade distribution
- subject averages
- attendance analysis

### Feedback Analysis
Analyzes student feedback using NLP:

- sentiment detection
- keyword extraction
- word clouds

### Combined Insights
Merges marks and feedback to produce:

- sentiment vs performance insights
- branch-level comparisons
- at-risk student detection

---

# Example Insights Generated

The system automatically identifies:

- Students failing with negative feedback
- High performers who are dissatisfied
- Branches with highest negative feedback
- Relationship between student sentiment and academic performance

---

# Use Cases

This project can help:

- Universities analyze course feedback
- Faculty improve teaching quality
- Administrators identify struggling students
- Improve student academic outcomes

---

# Future Improvements

Possible enhancements include:

- Machine learning model to predict student performance
- Instructor performance analytics
- Real-time feedback collection
- Student dropout prediction
- LMS integration
