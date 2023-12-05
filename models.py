from transcript import get_transcript_of_yt_video
from pytube import extract
from input_text_cc import extract_data_cc
from input_text_ncc import extract_data_ncc
from text_clean import clean_data
from peg import pegasus_summarizer
from bart import summary_bart_v1
import os
from dotenv import load_dotenv

load_dotenv('config.env')
file_path = os.getenv('FILE_PATH')
output_path = os.getenv('OUTPUT_PATH')

def text_extract_full(v_link,output_path):
  
  v_id = extract.video_id(v_link)
  transcript = get_transcript_of_yt_video(v_id)

  if transcript == '0':
    print("\nThis video doesn't contains subtitles")
    text = extract_data_ncc(v_link, output_path)
    return text

  else:
      print("\nThis video contains subtitles")
      text = extract_data_cc(v_link, output_path)
      return text


def model(v_link,file_path,output_path):

    ref = text_extract_full(v_link,output_path)
    text = clean_data(ref)
    print("\nCleaned Text:\n",text.replace("""'""",""))

    bart = summary_bart_v1(text,file_path,output_path)
    peg = pegasus_summarizer(text, output_path)
    

