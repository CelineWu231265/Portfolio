# Image Classification and Human-Centered AI

## Overview
This project was completed during **Year 1, Block C** of the **Data Science and Artificial Intelligence** program at **Breda University of Applied Sciences**.  
It combined **deep learning**, **Explainable AI (XAI)**, and **human-centered design** to explore how AI models can make transparent and trustworthy predictions.  

The project involved developing an **image classification model** using a **Convolutional Neural Network (CNN)**, visualizing its decision process through **explainability tools**, and evaluating user understanding via **A/B testing** of a prototype interface.

---

## Objectives
- Train and evaluate a **CNN model** for image classification.  
- Apply **Explainable AI** techniques to visualize model decision-making.  
- Design a **user interface** that presents predictions clearly and ethically.  
- Conduct **A/B testing** to assess how design and explanations affect user trust.  
- Reflect on **ethical AI principles**, transparency, and user interaction.  

---

## Methodology
1. **Data Preparation:**  
   - Imported and preprocessed an image dataset (resizing, normalization, and augmentation).  
   - Split data into training, validation, and test sets to ensure model robustness.

2. **Model Development:**  
   - Built a **Convolutional Neural Network (CNN)** using TensorFlow/Keras.  
   - Tuned hyperparameters such as learning rate, batch size, and dropout rate.  
   - Compared model performance against a **baseline MLP** for benchmarking.

3. **Explainability & Evaluation:**  
   - Implemented **Grad-CAM** and **LIME** to visualize which image areas influenced model decisions.  
   - Evaluated results using **accuracy**, **precision**, **recall**, and **F1-score**.  
   - Analyzed misclassifications to identify potential model bias.

4. **Human-Centered Design & A/B Testing:**  
   - Created **two interface prototypes** in Figma (Version A and Version B).  
   - Conducted **A/B testing** with users to compare different ways of displaying AI explanations.  
   - Collected both **quantitative metrics** (e.g., task accuracy, response time) and **qualitative feedback** (user trust, perceived clarity).  

---

## Technologies & Tools
- **Programming:** Python  
- **Libraries:** TensorFlow, Keras, NumPy, Matplotlib, scikit-learn, LIME  
- **Design & Testing:** Figma, Google Forms / Excel  
- **Environment:** Jupyter Notebook  

---

## Key Features
- Custom **CNN model** for supervised image classification.  
- Integrated **Explainable AI (XAI)** visualizations using Grad-CAM and LIME.  
- Designed a **user-centered interface** showing model predictions transparently.  
- Conducted **A/B testing** to measure user trust and comprehension.  

---

## Results
- Achieved **~88% model accuracy**, outperforming baseline MLP models.  
- Explainability visualizations clearly illustrated which image features influenced predictions.  
- A/B testing revealed that **explanation-rich designs** significantly improved user confidence in the AI system.  
- Delivered a full workflow from **model development** to **human evaluation**.

---

## Reflections
This project demonstrated how **technical AI development** and **human-centered design** must work hand in hand.  
By combining **deep learning**, **XAI**, and **A/B testing**, I learned how to build systems that are not only accurate but also transparent and user-trustworthy.  
It strengthened my understanding of **ethical design**, **model interpretability**, and **the importance of human feedback** in AI deployment.
