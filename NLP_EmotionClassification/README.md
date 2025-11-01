# Emotion Classification with Natural Language Processing

## Overview
This project was completed during **Year 2, Block C** of the **Data Science and Artificial Intelligence** program at **Breda University of Applied Sciences**, in collaboration with the **Content Intelligence Agency (CIA)**.  
The focus of this block was on **Natural Language Processing (NLP)** — specifically, the design and evaluation of models that classify **emotions in Spanish dialogue transcripts** from the TV show *La Isla de las Tentaciones*.

The project explored **traditional machine learning**, **transformer-based models**, and **zero-shot classification**, with additional emphasis on **explainability** and **bias evaluation**.  
As part of the technical implementation, the team developed a complete **video transcription, translation, and emotion detection pipeline**, integrating multiple AI models into a single automated workflow.

---

## Project Aim
To build a multilingual NLP system capable of extracting and classifying emotional cues from real-world Spanish-language media, combining **transcription**, **translation**, and **emotion recognition** into one AI-powered pipeline.

---

## Objectives
- Preprocess and annotate Spanish text data for emotion classification.  
- Train and compare multiple models to evaluate accuracy and generalization.  
- Implement **transformer-based zero-shot emotion detection** using prompt engineering.  
- Integrate a **video-to-emotion pipeline** combining transcription, translation, and emotion recognition.  
- Analyze model performance, interpretability, and bias using **Explainable AI (XAI)** techniques.

---

## Research Context
Understanding emotion in media content is a growing field in AI, combining **linguistics**, **machine learning**, and **psychology**.  
The dataset, derived from *La Isla de las Tentaciones*, provided authentic conversational data with natural emotional variation — an ideal challenge for emotion recognition models.  
The project addressed key research questions such as:
- How accurately can transformer-based models identify emotional tone in Spanish dialogue?  
- How does translation impact emotion classification accuracy?  
- What biases or limitations emerge when applying multilingual models across languages?

---

## Methodology
The project followed a **data-driven experimentation workflow** consisting of:
1. **Data Preprocessing:** Cleaning and normalizing transcripts, including removing emojis, expanding contractions, and handling punctuation.  
2. **Model Comparison:** Evaluating traditional models (Logistic Regression, SVM) versus transformer-based approaches (BERT, DistilBERT, and XLM-RoBERTa).  
3. **Zero-Shot Classification:** Testing multilingual transformer models on emotion labels without retraining.  
4. **Explainability Analysis:** Using SHAP and attention visualization to interpret model behavior and potential biases.  
5. **Pipeline Integration:** Automating the end-to-end process from video transcription to emotion prediction.

---

## Technologies & Tools
- **Languages:** Python  
- **Libraries:** Transformers (Hugging Face), TensorFlow, Keras, scikit-learn, Pandas, NumPy, Matplotlib, SHAP  
- **APIs & Frameworks:** AssemblyAI, Deep Translator, PyTube, Regex, Emoji, Sacremoses  
- **Tools:** Jupyter Notebook, GitHub, Visual Studio Code  

---

## Pipeline Overview

### 1. Video Transcription
- Extracts audio from a YouTube video using **PyTube**.  
- Transcribes speech using **AssemblyAI’s Automatic Speech Recognition API**, producing timestamped text.

### 2. Translation
- Translates Spanish transcripts to English using a **pre-trained Hugging Face translation model**.  
- Outputs structured CSV files for downstream analysis.

### 3. Emotion Detection
- Preprocesses translated text with tokenization, cleaning, and padding.  
- Predicts emotions (e.g., joy, anger, sadness, surprise) using a **TensorFlow-based classifier**.  
- Saves results in `translated_emotions.csv` for reporting and visualization.

---

## Key Features
- End-to-end **AI pipeline** for multilingual video-based emotion analysis.  
- Comparative evaluation between **classical ML** and **transformer-based** NLP models.  
- **Zero-shot emotion classification** using multilingual transformer architectures.  
- **Explainable AI (XAI)** visualizations to interpret model predictions.  
- **Bias analysis** to assess fairness across languages and datasets.

---

## Results
- Developed a **functional multilingual NLP pipeline** integrating transcription, translation, and emotion classification.  
- Achieved strong performance with **transformer-based models**, surpassing traditional machine learning baselines.  
- Identified key sources of bias related to **translation semantics** and **dataset imbalance**.  
- Delivered an analytical report and visual presentation connecting technical findings to real-world media applications.

---

## How to Run

### 1. Configure API Access
Create an account at [AssemblyAI](https://www.assemblyai.com) and obtain an API key.  
Then add it to your environment file:

```bash
ASSEMBLYAI_API_KEY=<YOUR_KEY>
```

Alternatively, you can define the key directly in your script:

```python
aai.settings.api_key = API_KEY
```

---

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 3. Run the Pipeline
```bash
python pipeline.py
```

This will:
1. Download and transcribe the video.  
2. Translate the transcript.  
3. Classify emotions in each translated sentence.  
4. Save results to the `outputs/` folder.

---

## Example Output
- **extracted_sentences.csv** – raw transcript from video.  
- **translated_output.csv** – English translations.  
- **translated_emotions.csv** – predicted emotion labels per sentence.  

---

## Reflections
This project represented a turning point between **academic research and applied AI engineering**.  
By building a multilingual, explainable NLP system, the project bridged theoretical understanding with real-world implementation.  
It reinforced skills in **pipeline integration**, **model evaluation**, and **ethical AI deployment** — all crucial for the next step in my data science career.
