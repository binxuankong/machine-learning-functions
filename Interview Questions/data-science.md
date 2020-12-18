# Data Science Questions

## Machine Learning

### What are some of the steps for data wrangling and data cleaning before applying machine learning algorithms?
- Data profiling: understanding the dataset using `shape`, `describe`, etc
- Data visualizations: visualize data with histograms, boxplots, scatterplots to better understand the relationships between variables and to identify potential outliers
- Syntax error: making sure there are no white spaces, letter casing is consistent, typos
- Standardization/normalization: ensure different scales of different variables don't negatively impact the performance of model
- Handling null values: deleting rows with null values altogether, replacing null values with mean/median/mode, predicting the values

### What is precision and recall?
**Recall** - what proportion of actual positives are identified correctly? <br>
Recall = TP / (TP + FN) <br>
**Precision** - what proportion of positive identifications are actually correct? <br>
Precision = TP / (TP + FP)

### What are the differences between supervised and unsupervised learning?
| Supervised Learning | Unsupervised learning |
| --- | --- |
| Uses known and labeled data as input | Uses unlabeled data as input |
| Has feedback mechanism | No feedback mechanism |
| Examples: decision trees, logistic regression, support vector machine | Examples: k-means clustering, hierarchical clustering |

### Why is Naive Bayes a bad classifier? How would you improve a spam detection algorithm that uses Naive Bayes?
The major drawback of Naive Bayes is that it holds a strong assumption in that the features are assumed to be uncorrelated with one another, which is typically never the case. One way to improve such an algorithm that uses Naive Bayes is by decorrelating the features so that the assumption holds true.