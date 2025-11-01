import whisper
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

# Load the Whisper model
model = whisper.load_model("turbo") 

# Path to audio file
raw_audio = "/Users/Buas/Desktop/Data Start/GitHub/Ano 2/2024-25c-fai2-adsai-group-group16/Data/La isla de las tentaciones Temporada 7 capitulo 7.mp3"

# Transcribe the audio
result = model.transcribe(raw_audio, language="es", word_timestamps=False)

# Function to format timestamps in HH:MM:SS format
def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Extract sentences with timestamps and token count
sentences = []
for segment in result["segments"]:
    start_time = format_timestamp(segment["start"])
    end_time = format_timestamp(segment["end"])
    text = segment["text"]
    tokens = word_tokenize(text, language="spanish")  # Tokenize text
    token_count = len(tokens)  # Count tokens
    sentences.append({
        "Start Time": start_time,
        "End Time": end_time,
        "Text": text,
        "Token Count": token_count
    })

# Limit to the first 300 sentences
less_sentences = sentences[:300]

# Create DataFrame
df = pd.DataFrame(less_sentences)
df.insert(0, "Sentence Number", range(1, len(df) + 1))  # Add sentence numbers

# Save to Excel
excel_path = "/Users/Buas/Desktop/Data Start/GitHub/Ano 2/2024-25c-fai2-adsai-group-group16/Data/transcription_whisper2.xlsx"
df.to_excel(excel_path, index=False)