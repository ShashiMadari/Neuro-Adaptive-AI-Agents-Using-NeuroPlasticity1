from processors.image_processor import process_image
from processors.audio_processor import process_audio
from processors.text_processor import process_text

# --- Add Metrics ---
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# --- Input Paths and Sample Text ---
image_path = "data/mc2.png"
audio_path = "data/einstein_formula.wav"
input_text = "explain this formula?"

# --- Reference (Ground Truth) Caption ---
reference_caption = "A little girl climbing into a wooden playhouse."   # <-- replace with your ground truth

# --- Process Each Modality Independently ---
img_result = process_image(image_path)
aud_result = process_audio(audio_path)
txt_result = process_text(input_text)

# --- Check Caption Accuracy ---
candidate = img_result['caption']
reference = [reference_caption.split()]   # tokenized reference
candidate_tokens = candidate.split()

# BLEU Score
smooth_fn = SmoothingFunction().method1
bleu_score = sentence_bleu(reference, candidate_tokens, smoothing_function=smooth_fn)

# ROUGE Score
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
rouge_scores = scorer.score(reference_caption, candidate)

# --- Display Results ---
print("\n🖼️ Image Output:")
print(" Emotion:", img_result['emotion'])
print(" Caption:", candidate)
print(" Importance:", img_result['importance'])
print(" ➡️ Caption Accuracy (BLEU):", round(bleu_score, 3))
print(" ➡️ Caption Accuracy (ROUGE):", {k: round(v.fmeasure, 3) for k, v in rouge_scores.items()})

print("\n🔊 Audio Output:")
print(" Transcribed:", aud_result['transcribed'])
print(" Emotion:", aud_result['emotion'])
print(" Importance:", aud_result['importance'])

print("\n📝 Text Output:")
print(" Summary:", txt_result['text_summary'])
print(" Emotion:", txt_result['emotion'])
print(" Importance:", txt_result['importance'])
