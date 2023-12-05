from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from download import text_file_convert
import os
from dotenv import load_dotenv

load_dotenv('config.env')
file_path = os.getenv('FILE_PATH')
output_path = os.getenv('OUTPUT_PATH')

def pegasus_summarizer(text,output_path):
    # Load pre-trained model and tokenizer
    model_name = 'google/pegasus-large'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)

    # Tokenize and generate summary
    inputs = tokenizer([text], return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=150, early_stopping=True)

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    text_file_convert(summary, "peg_summary", output_path)
    print("\nPegasus Model Text Summarization Executed!!")
    return summary

