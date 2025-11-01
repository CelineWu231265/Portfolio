import assemblyai as aai
import os
import pandas as pd
from datetime import timedelta
import tiktoken

# Function to format milliseconds into HH:MM:SS,mmm
def format_timestamp(ms):
    td = timedelta(milliseconds=ms)
    return f"{str(td)[:-3].replace('.', ',')}"  # Format HH:MM:SS,mmm

# Function to count tokens using OpenAI's encoding
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")  # OpenAI GPT-4 tokenizer
    return len(encoding.encode(text))

# API token (set your API key here)
aai.settings.api_key = "4ba97f247dd44f86b2c51a29f14caa26"

# Audio file path
audio_file = r"C:\Users\emilp\Documents\GitHub\2024-25c-fai2-adsai-group-group16\Data\La isla de las tentaciones _ Temporada 7 capitulo 7 (1).mp3"

# Configure and transcribe
config = aai.TranscriptionConfig(language_code="es", speakers_expected=1, punctuate=True)
transcriber = aai.Transcriber(config=config)
transcript = transcriber.transcribe(audio_file)

if transcript.status == aai.TranscriptStatus.error:
    print(f"Transcription failed: {transcript.error}")
    exit(1)

sentences = transcript.get_sentences()

# Limiting to first 300 sentences
limited_sentences = sentences[:300]

# Creating DataFrame with timestamps and token count
data = {
    'Sentence Number': list(range(1, len(limited_sentences) + 1)),
    'Start Time': [format_timestamp(sentence.start) for sentence in limited_sentences],
    'End Time': [format_timestamp(sentence.end) for sentence in limited_sentences],
    'Text': [sentence.text for sentence in limited_sentences],
    'Token Count': [count_tokens(sentence.text) for sentence in limited_sentences]
}

df = pd.DataFrame(data)

if hasattr(transcript, 'words') and transcript.words:
    total_words = len(transcript.words)
    wer = transcript.utterances[0].wer if transcript.utterances else "N/A"

# Add WER as a single row at the bottom of the DataFrame
wer_df = pd.DataFrame({'Sentence Number': ['Overall'], 'Start Time': [''], 'End Time': [''], 'Text': ['Word Error Rate'], 'Token Count': [wer]})
df = pd.concat([df, wer_df], ignore_index=True)

# Saving to Excel
output_file = r"C:\Users\emilp\Downloads\transcribed_data_2.xlsx"
df.to_excel(output_file, index=False)

print(f"First 300 sentences with timestamps and WER saved to {output_file}")
