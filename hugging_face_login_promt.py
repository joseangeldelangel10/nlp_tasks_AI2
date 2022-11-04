import os

def prompt_for_hugging_face_login():
    os.system('huggingface-cli login')

if __name__ == "__main__":
    prompt_for_hugging_face_login()