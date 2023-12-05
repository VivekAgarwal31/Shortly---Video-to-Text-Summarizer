from download import text_file_convert
import pickle
import os
from dotenv import load_dotenv
from transformers import BartTokenizer
from transformers import BertModel
from transformers import BartForConditionalGeneration, BartTokenizer

load_dotenv('config.env')
file_path = os.getenv('FILE_PATH')
output_path = os.getenv('OUTPUT_PATH')

def summary_bart_v1(text,file_path,output_path):
    print("\nText Summarizing...")

    model_name = 'facebook/bart-large-cnn'
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    input_tensor = tokenizer.encode(text, return_tensors="pt", max_length=512,truncation=True)
    outputs_tensor = model.generate(input_tensor, max_length=160, min_length=120, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary_v1 = tokenizer.decode(outputs_tensor.squeeze(), skip_special_tokens = True)


    text_file_convert(summary_v1, "summary_bart_v1", output_path)
    print("\nBart Model Text Summarization Executed!!")
    return summary_v1




