import moviepy.editor as mp
import speech_recognition as sr
import os
from dotenv import load_dotenv
from input_text_cc import extract_data_cc
from input_text_ncc import extract_data_ncc
from text_clean import clean_data
from peg import pegasus_summarizer
from bart import summary_bart_v1

load_dotenv('config.env')
file_path = os.getenv('FILE_PATH')
output_path = os.getenv('OUTPUT_PATH')

def text_local(local,output_path):
    clip=mp.VideoFileClip(local)
    clip.audio.write_audiofile(output_path+"audio.wav")
    
    r=sr.Recognizer()
    with sr.AudioFile(output_path+"audio.wav") as source:
        audio=r.record(source)
        
    text_nc=r.recognize_google(audio)
    text=clean_data(text_nc)
    print("\nCleaned Text:\n",text.replace("""'""",""))
    
    bart = summary_bart_v1(text,file_path,output_path)
    peg = pegasus_summarizer(text, output_path)