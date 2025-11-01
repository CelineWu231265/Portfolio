# %% [markdown]
# ## Task 7 
# 
# Goal
# - A well-documented script (prompt_engineering) that shows how you send the prompts to the LLM and process responses.
# - A prompt engineering log that summarizes your different prompt variations and how they performed.
# 
# How
# - Sign Up
# - Generate an API Key
# - Choose a Model >> start with model": "llama3.2:3b
# - Send Prompt Requests
# - Manage Context Length
# - Avoid Overloading the Service

# %%
import requests
import json
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# %%
# Load data and clean the Sentence column
data_file = '../Data/group 16_url1.xlsx'
df = pd.read_excel(data_file).dropna().drop_duplicates()
df['Sentence'] = df['Sentence'].str.replace('"', '', regex=False)

# Model configuration
TOKEN = "sk-1325adfbf63c47f4bf5fd888b144dff6" 
MODEL = "llama3.2:3b"
API_URL = "http://194.171.191.227:30080/api/chat/completions"

# %%
def chat_with_model(token, messages, context_length=100):
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {"model": MODEL, "messages": messages, "context_length": context_length}
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        if 'choices' in response_data and response_data['choices']:
            return response_data['choices'][0]['message']['content']
        else:
            return f"Error: Unexpected response format: {response_data}"
    except Exception as e:
        return f"Error: {e}"

# %% [markdown]
# # Prompt 1 - Simple sentences

# %%
sample_df = df.sample(n=20, random_state=42)

def prompt_json(sentence):
    """
    Instruct the LLM to output a JSON object with keys:
    "predicted_emotion": <one word>,
    "notes": <brief explanation>
    Do not include any additional text.
    """
    return f"""
You are an AI that classifies the emotional content of sentences.
Assign exactly one emotion from the following: [happiness, sadness, anger, surprise, fear, disgust, neutral].

The sentence is:
"{sentence}"

Respond with a valid JSON object exactly in the following format:
{{
  "predicted_emotion": "<one word>",
  "notes": "<brief explanation>"
}}
Do not include any additional text."""

# %%
def run_emotion_classification_json(df, token, prompt_function):
    results = []
    for idx in range(len(df)):
        sentence = df.iloc[idx]['Sentence']
        original_emotion = df.iloc[idx]['Emotion']
        prompt = prompt_function(sentence)
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
        response_text = chat_with_model(token, messages).strip()
        try:
            response_json = json.loads(response_text)
            predicted_emotion = response_json.get("predicted_emotion", "").strip().lower()
            notes = response_json.get("notes", "").strip()
        except Exception as err:
            predicted_emotion = ""
            notes = f"Error parsing JSON: {err}"
        results.append({
            "sentence": sentence,
            "original_emotion": original_emotion,
            "predicted_emotion": predicted_emotion,
            "notes": notes
        })
    return pd.DataFrame(results)

# %%
final_results = run_emotion_classification_json(sample_df, TOKEN, prompt_json)
print("Final Output (Sentence, Original Emotion, Predicted Emotion, Notes):")
print(final_results)


final_results.to_csv('prom1_emotion_classification_comparison.csv', index=False)


log_data = {
    "Run": ["Emotion Classification"],
    "Model": [MODEL],
    "Sample Size": [len(sample_df)],
    "API_URL": [API_URL],
    "Notes": ["Used JSON output prompt to obtain predicted_emotion and notes."]
}
log_df = pd.DataFrame(log_data)
log_df.to_csv('1_emotion_classification_run_log.csv', index=False)

# %%
def calculate_f1_score(df):
    df['predicted_emotion'] = df['predicted_emotion'].str.lower()
    df['original_emotion'] = df['original_emotion'].str.lower()
    f1 = f1_score(df['original_emotion'], df['predicted_emotion'], average='weighted')
    return f1

# Calculate f1 score
f1 = calculate_f1_score(final_results)
print(f"F1 Score: {f1:.2f}")


# %% [markdown]
# # Prompt 2 - Incorporates conversation context
# 
# For each sentence, it includes previous sentences as context

# %%
sample_df2 = df.sample(n=20, random_state=42)


ALLOWED_EMOTIONS = {"happiness", "sadness", "anger", "surprise", "fear", "disgust", "neutral"}

def prompt_complex(df, current_index, context_size=3):
    """
    Builds a complex prompt that incorporates previous context from the conversation.
    The prompt instructs the LLM to classify the current sentence based on the conversation so far.
    The output must be a valid JSON object with keys:
        "predicted_emotion": <one word>,
        "notes": <brief explanation>
    """
    context_sentences = []
    start = max(0, current_index - context_size)
    for i in range(start, current_index):
        context_sentences.append(f"{i+1}. \"{df.iloc[i]['Sentence']}\"")
    current_sentence = df.iloc[current_index]['Sentence']
    context_block = "\n".join(context_sentences)
    
    prompt = f"""
You are an AI trained to classify emotions in sentences from a Spanish dating show.
People are in intense, dramatic situations. The sentences follow the flow of the episode.
Your objective is to assign exactly one emotion from the following:
[happiness, sadness, anger, surprise, fear, disgust, neutral].

Context: The sentences below are extracted (and translated) from the show "La isla de las tentaciones". The participants are in a relationship and are separated from their partners to test their loyalty. Emotions are very intense and the participants are very vocal.
The sentences follow the episode in order. Here is the conversation so far:
{context_block}

Now classify the emotion of the following sentence:
"{current_sentence}"

Respond with a valid JSON object exactly in the following format:
{{
  "predicted_emotion": "<one word>",
  "notes": "<brief explanation>"
}}
Do not include any additional text.
"""
    return prompt

# %%
def run_emotion_classification_with_context(df, token, prompt_function, context_size=50):
    """
    Processes each sentence in the dataset using a prompt function that leverages conversation context.
    Returns a DataFrame with: sentence, original_emotion, predicted_emotion, and notes.
    """
    results = []
    for idx in range(len(df)):
        sentence = df.iloc[idx]['Sentence']
        original_emotion = df.iloc[idx]['Emotion']
        prompt = prompt_function(df, idx, context_size=context_size)
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
        response_text = chat_with_model(token, messages).strip()
        try:
            response_json = json.loads(response_text)
            predicted_emotion = response_json.get("predicted_emotion", "").strip().lower()
            notes = response_json.get("notes", "").strip()
        except Exception as err:
            predicted_emotion = ""
            notes = f"Error parsing JSON: {err}"
        results.append({
            "sentence": sentence,
            "original_emotion": original_emotion,
            "predicted_emotion": predicted_emotion,
            "notes": notes
        })
    return pd.DataFrame(results)

# Run the complex pipeline with context
final_results_complex = run_emotion_classification_with_context(sample_df2, TOKEN, prompt_complex, context_size=3)
print("Final Output (Complex Pipeline):")
print(final_results_complex)

# %%
# Save the results to a CSV file
final_results_complex.to_csv('prom2_emotion_classification_complex.csv', index=False)

# Log the run details
log_data = {
    "Run": ["Emotion Classification (Complex Pipeline)"],
    "Model": [MODEL],
    "Sample Size": [len(sample_df)],
    "API_URL": [API_URL],
    "Notes": ["Used complex JSON output prompt with context from previous sentences."]
}
log_df = pd.DataFrame(log_data)
log_df.to_csv('prom2_emotion_classification_complex_run_log.csv', index=False)


# %%
# Calculate F1 score for the complex pipeline
y_true = final_results_complex["original_emotion"].str.lower()
y_pred = final_results_complex["predicted_emotion"]

# Calculate macro-average F1-score
f1 = f1_score(y_true, y_pred, average='macro')
print(f"Macro-Average F1 Score: {f1:.2%}")

# classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=list(ALLOWED_EMOTIONS)))


# %% [markdown]
# # Prompt 3  
# few-shot examples and instructs the model to select the dominant emotion if the sentence appears to express more than one

# %%
sample_df3 = df.sample(n=20, random_state=42)


def prompt_json_p3_complex(sentence):
    """
    Complex Prompt 3: Detailed Few-Shot with Context and Dominance Instruction.
    The sentences are from the Spanish reality show "La isla de las tentaciones" (translated to English),
    known for its dramatic portrayal of couples, betrayal, and intense emotions.
    
    Provide three few-shot examples and instruct:
    - If a sentence suggests multiple emotions, choose the dominant one.
    - Output a valid JSON object with keys:
          "predicted_emotion": <one word>,
          "notes": <brief explanation including context, reasoning, and dominance selection>.
    Do not include any additional text.
    """
    return f"""
You are an AI that classifies the emotional content of sentences.
These sentences are extracted from the Spanish reality show "La isla de las tentaciones" (translated to English),
a show known for its dramatic portrayal of couples, betrayal, and intense emotions.

Assign exactly one emotion from: [happiness, sadness, anger, surprise, fear, disgust, neutral].
If the sentence expresses more than one emotion, select the dominant emotion that best reflects the overall feeling.

Below are three examples:
Example 1:
Sentence: "I just got a promotion at work!"
Correct Emotion: happiness
Explanation: The sentence clearly expresses joy and excitement due to a positive career event.
Example 2:
Sentence: "I feel so alone and abandoned."
Correct Emotion: sadness
Explanation: The sentence conveys deep sorrow and loneliness.
Example 3:
Sentence: "I can't believe how betrayed I feel after everything that happened!"
Correct Emotion: anger
Explanation: The sentence expresses intense anger and a sense of betrayal.

Now, classify the emotion of the following sentence:
"{sentence}"

Respond with a valid JSON object exactly in the following format:
{{
  "predicted_emotion": "<one word>",
  "notes": "<brief explanation including context and reasoning, indicating the dominant emotion if multiple are present>"
}}

Do not include any additional text.
"""

# %%
def run_emotion_classification_with_context(df, token, prompt_function, context_size=70):
    results = []
    for idx in range(len(df)):
        sentence = df.iloc[idx]['Sentence']
        original_emotion = df.iloc[idx]['Emotion']
        prompt = prompt_function(sentence)
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
        response_text = chat_with_model(token, messages).strip()
        try:
            response_json = json.loads(response_text)
            predicted_emotion = response_json.get("predicted_emotion", "").strip().lower()
            notes = response_json.get("notes", "").strip()
        except Exception as err:
            predicted_emotion = ""
            notes = f"Error parsing JSON: {err}"
        results.append({
            "sentence": sentence,
            "original_emotion": original_emotion,
            "predicted_emotion": predicted_emotion,
            "notes": notes
        })
    return pd.DataFrame(results)

# %%
final_results_complex = run_emotion_classification_with_context(sample_df3, TOKEN, prompt_json_p3_complex, context_size = 70)
print("Final Output (Complex Pipeline using Prompt):")
print(final_results_complex)

# Save results to CSV
final_results_complex.to_csv('prom3_emotion_classification_complex.csv', index=False)

# Create a log DataFrame summarizing the run
log_data = {
    "Run": ["Emotion Classification (Complex Pipeline - Prompt 3)"],
    "Model": [MODEL],
    "Sample Size": [len(sample_df3)],
    "API_URL": [API_URL],
    "Notes": ["Used complex JSON output prompt with three few-shot examples and context. Dominance instruction included."]
}
log_df = pd.DataFrame(log_data)
log_df.to_csv('prom3_emotion_classification_complex_run_log.csv', index=False)

# %%
# Evaluation: Calculate F1 score, classification report, and plot a confusion matrix.
y_true = final_results_complex["original_emotion"].str.lower()
y_pred = final_results_complex["predicted_emotion"]

f1_macro = f1_score(y_true, y_pred, average='macro')
print(f"Macro-Average F1 Score: {f1_macro:.2%}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=list(ALLOWED_EMOTIONS)))


