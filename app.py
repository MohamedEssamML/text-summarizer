import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from rouge_score import rouge_scorer

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

def summarize(text, model, tokenizer, max_length=150):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"], 
        num_beams=4, 
        max_length=max_length, 
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def evaluate_summary(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

def main():
    st.title("Text Summarization with BART")
    st.write("This app uses the BART model from Hugging Face to generate summaries of your text.")
    
    model, tokenizer = load_model()
    
    # Input options
    input_method = st.radio("Choose input method:", ("Enter text", "Upload file"))
    
    text = ""
    if input_method == "Enter text":
        text = st.text_area("Enter your text here:", height=200)
    else:
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if uploaded_file is not None:
            text = uploaded_file.read().decode("utf-8")
    
    if text:
        # Summary options
        col1, col2 = st.columns(2)
        with col1:
            summary_type = st.radio("Summary type:", ("Short", "Medium", "Long"))
        with col2:
            max_length = st.slider("Max length (tokens):", 50, 300, 150 if summary_type == "Medium" else (100 if summary_type == "Short" else 200))
        
        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                summary = summarize(text, model, tokenizer, max_length)
                
                st.subheader("Summary")
                st.write(summary)
                
                # Evaluation (if reference available)
                st.subheader("Evaluation")
                if input_method == "Enter text":
                    st.write("To evaluate the summary, please provide a reference summary:")
                    reference = st.text_area("Enter reference summary for evaluation:", height=100)
                    if reference:
                        scores = evaluate_summary(reference, summary)
                        st.write("ROUGE Scores:")
                        st.json(scores)
                else:
                    st.write("Upload a separate file with reference summary to enable evaluation.")

if __name__ == "__main__":
    main()