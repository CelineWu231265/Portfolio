# Data-Driven Risk Assessment for NAC Breda Football Club

## Overview
This project was completed during **Year 1, Block B**.  
In collaboration with **NAC Breda**, the professional football club of Breda, the project focused on developing a **data-driven risk assessment model** to help improve team performance and overall success.

The study analyzed player behavior using historical disciplinary data such as **red and yellow cards** to classify players into different **risk categories**.  
By identifying patterns in player conduct, the project aimed to provide NAC Breda with insights to support **strategic decisions** on performance management and player development.

---

## Objectives
- Analyze historical player data to identify behavioral and performance patterns.  
- Develop a **machine-learning-based risk assessment system** that classifies players as **low-risk**, **risky**, or **high-risk**.  
- Evaluate and compare multiple models to determine the most reliable approach.  
- Address **ethical considerations** regarding data use and privacy.  
- Provide **actionable recommendations** to support data-driven decision-making for NAC Breda.  

---

## Methodology
1. **Data Exploration & Cleaning:**  
   - Dataset: 16 535 records and 115 features (numerical + categorical).  
   - Imputed missing values using mean, median, and mode where appropriate.  
   - Encoded categorical variables (e.g., player position, footedness) for ML compatibility.  
   - Defined a new target variable, **Risk Assessment**, derived from disciplinary data.  

2. **Exploratory Data Analysis (EDA):**  
   - Computed statistical summaries (mean, median, variance) for red/yellow cards.  
   - Used **boxplots**, **bar charts**, and **correlation matrices** to visualize trends.  
   - Identified that ≈ 8 % of players fall into the high-risk category.  

3. **Machine Learning:**  
   - Tested multiple models: **Random Forest** and **Gradient Boosting Trees**.  
   - Performed **hyperparameter tuning** (Grid Search + Random Search).  
   - Evaluated models using **accuracy**, **confusion matrix**, **classification report**, and **cross-validation**.  

4. **Model Selection:**  
   - Random Forest chosen for its **stability and consistency** across validation sets.  
   - Final accuracy: ≈ 69 %, showing meaningful predictive capability despite weak correlations.  

---

## Technologies & Tools
- **Programming:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, scikit-learn  
- **Visualization:** Matplotlib  
- **Tools:** Jupyter Notebook, GitHub  

---

## Key Features
- Custom **risk-assessment algorithm** based on disciplinary metrics.  
- Comparative model evaluation to ensure reliability and interpretability.  
- Visual exploration of correlations between behavior, age, and performance.  
- Recommendations for integrating analytics into **player-development strategy**.  

---

## Results
- Achieved ≈ 69 % accuracy using a tuned **Random Forest Classifier**.  
- Delivered a detailed analysis of **player-risk profiles** for NAC Breda.  
- Demonstrated how **machine learning can enhance sports performance analytics**.  

---

## Ethical Considerations
- Evaluated NAC Breda’s internal data-governance structure and GDPR compliance.  
- Recommended the creation of **ethical guidelines** and **data-protection policies**.  
- Emphasized transparency and accountability in using player data for analytics.  

---

## Recommendations
- Implement the developed **risk-assessment tool** for continuous monitoring.  
- Integrate results into player training and disciplinary strategies.  
- Establish data-ethics frameworks to ensure responsible analytics adoption.  

---

## Reflections
This project demonstrated the intersection of **data science and sports strategy**.  
It showcased how predictive analytics can offer tangible benefits to team management while underscoring the importance of **ethical and interpretable AI** in real-world contexts.  
Collaborating with NAC Breda provided practical insight into using **machine learning** for **performance optimization** within a professional sports environment.
