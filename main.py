from models import model
from localvid import text_local
import validators
from translate import trans_switch
import os
from dotenv import load_dotenv

load_dotenv('config.env')
file_path = os.getenv('FILE_PATH')
output_path = os.getenv('OUTPUT_PATH')

def main():
  print("=============================================")
  print("                  SHORTLY                    ")
  print("           Video to Text Summarizer          ")
  print("        By Vivek Agarwal & Madhav Garg       ")
  print("=============================================")
  choice=input("Summarize from a local file or a video link? (l/v): ")
  if(choice=='v'):
    v_link = input("\nEnter the video link: ")
    valid=validators.url(v_link)
    if valid==True:
      print("\nUrl is valid")
      model(v_link,"A:/Video-To-Multi-Language-Text-Summarizer-main","A:/Video-To-Multi-Language-Text-Summarizer-main/outputs")
      trans_switch(output_path)
    else:
      print("\nInvalid url")
    
  elif(choice=='l'):
    local=input("Enter Video Path: ")
    text_local(local,output_path)
    trans_switch(output_path)
  else:
    print("Enter correct choice !")

main()
