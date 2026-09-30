# DS605 Lab 6: Feature Extraction and Machine Learning with Image and Text Data

**Name:** Akanksha Dasani  
**Student ID:** 202618062  
**Course:** DS605 - Fundamentals of Machine Learning  
**Assignment:** Lab Assignment 6  

---

## Objective
The goal of this lab is to extract handcrafted numerical features from raw images and structured text data, and train traditional machine learning models for binary classification without using any deep learning or pretrained embeddings.

---

## Part A: Image Feature Extraction and Classification

### Dataset
- Asphalt Crack Dataset containing 400 images (200 crack images and 200 non-crack images).

### Methodology
1. **Preprocessing:** 
   - Loaded images using OpenCV (`cv2`).
   - Resized each image to a uniform size of $128 \times 128$ pixels.
   - Converted RGB images to single-channel grayscale using standard luminance weighting.
2. **Feature Extraction:**
   - **Intensity statistics (NumPy):** Calculated mean brightness, contrast (standard deviation), median intensity, dark-pixel ratio (pixels < 80), and bright-pixel ratio (pixels > 180).
   - **Edge features (OpenCV Canny):** Applied Canny edge detection with thresholds 50 and 150. Counted total edge pixels and computed edge density (edge count / total pixels).
   - Combined all extracted features into a single table (`extracted_image_features.csv`) with labels (1 for Crack, 0 for Non-Crack).
3. **Model Training:**
   - Standardized the feature matrix using `StandardScaler`.
   - Performed an 80-20 stratified train-test split.
   - Trained and compared Logistic Regression, Support Vector Machine (RBF kernel), and Random Forest.

### Results
| Model | Accuracy | Precision | Recall | F1-Score | Training Time (s) | Prediction Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.8875 | 0.8947 | 0.8750 | 0.8846 | ~0.005 | ~0.0004 |
| SVM (RBF Kernel) | 0.9125 | 0.9231 | 0.9000 | 0.9114 | ~0.003 | ~0.0010 |
| Random Forest | 0.9375 | 0.9487 | 0.9250 | 0.9367 | ~0.120 | ~0.0060 |

### Observations
- Cracked asphalt surfaces have noticeably higher edge density and higher contrast compared to smooth pavement because cracks introduce sharp dark edges and uneven shadows.
- Random Forest performed best on this tabular feature set because it naturally handles non-linear boundaries between intensity and edge statistics.

---

## Part B: Text Classification (Email Spam Detection)

### Dataset
- Kaggle Email Spam dataset with 5,172 emails and 3,000 word count columns plus a `Prediction` target column (0 for ham, 1 for spam).
- **Note:** As instructed by the professor, TF-IDF vectorization was not used. The models were evaluated directly on the word count (frequency) representation.

### Methodology
1. Inspected class balance: roughly 71% non-spam (ham) and 29% spam.
2. Verified no missing values across the 3,000 features.
3. Separated features and labels, then split 80% train and 20% test (stratified).
4. Trained Multinomial Naive Bayes and Logistic Regression.

### Results
| Model | Features | Accuracy | Precision | Recall | F1-Score | Training Time (s) | Prediction Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Multinomial Naive Bayes | 3,000 | 0.9478 | 0.8833 | 0.9458 | 0.9135 | ~0.006 | ~0.001 |
| Logistic Regression | 3,000 | 0.9671 | 0.9387 | 0.9492 | 0.9439 | ~0.150 | ~0.002 |

### Observations
- Multinomial Naive Bayes is extremely fast and effective for word count data.
- Logistic Regression achieved slightly higher precision and F1-score because it learns specific positive weights for spam trigger words and negative weights for typical work/ham words.

---

## Part C: Improving the Representation

### Change Made
Applied **Chi-Square ($\chi^2$) feature selection** (`SelectKBest`) on the email word counts to reduce the vocabulary from 3,000 words down to the top 500 most informative words.

### Trade-off Comparison
| Representation | Number of Features | Accuracy | F1-Score | Training Time (s) |
|---|:---:|:---:|:---:|:---:|
| Original (All Words) | 3,000 | 0.9478 | 0.9135 | 0.0061s |
| Improved (Top 500 Chi-2) | 500 | 0.9449 | 0.9085 | 0.0018s |

### Discussion
- **Dimensionality vs. Performance:** Reducing the feature count by over 83% (from 3,000 to 500) resulted in virtually the same classification accuracy (less than 0.3% difference).
- **Computation Time:** Training and prediction were roughly 3 times faster on the reduced feature set.
- Top informative words identified by Chi-Square included words like *pills, free, money, prescription, deal, meter, gas*, which clearly separate spam offers from regular Enron email exchanges.

---

## Files in this Repository
- `202618062_lab06.ipynb`: Main Jupyter notebook with all code and outputs.
- `extracted_image_features.csv`: Extracted feature table for all 400 asphalt images.
- `canny_edge_visualization.png`: Sample RGB, grayscale, and Canny edge comparison plot.
- `image_confusion_matrices.png`: Confusion matrices for Part A models.
- `text_confusion_matrices.png`: Confusion matrices for Part B models.
- `email_class_distribution.png`: Spam vs. ham distribution bar chart.