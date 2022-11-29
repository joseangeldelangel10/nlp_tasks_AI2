'''
This script calls three other scripts to:
1 - Evaluate the sentiment of a set of movie reviews (POSITIVE, NEGATIVE)
2 - Retrain a NER model using tagged tweets
3 - Evaluate the perfomance of two translation APIs
'''
import task_1.task_1 as t1
import task_2.task_2 as t2
import task_3.task_3 as t3
from task_3.api_keys import DEEPL_KEY
import os
import hugging_face_login_promt

if __name__ == "__main__":    
    hugging_face_login_promt.prompt_for_hugging_face_login()
    current_dir = os.getcwd()
    print(" ==================================================== ")
    print("                    EXCECUTING TASK 1                 ")
    print(" ==================================================== ")    
    task_1_script_dir = os.path.join(current_dir, "task_1")
    t1.task_1(calling_dir = task_1_script_dir)
    print(" ==================================================== ")
    print("                    EXCECUTING TASK 2                 ")
    print(" ==================================================== ")    
    t2.N_EXAMPLES_TO_TRAIN = 100    
    t2.N_TRAINING_EPOCHS = 20
    task_2_script_dir = os.path.join(current_dir, "task_2")
    t2.task_2(calling_dir = task_2_script_dir)
    print(" ==================================================== ")
    print("                    EXCECUTING TASK 3                 ")
    print(" ==================================================== ")    
    task_3_script_dir = os.path.join(current_dir, "task_3")
    t3.N_LINES_TO_TRANSLATE = 100
    t3.task_3(api_key=DEEPL_KEY, calling_dir = task_3_script_dir)