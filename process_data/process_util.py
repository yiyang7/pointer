import jsonlines
import os
import numpy as np
import re

# create a .story file and save it to the directory save_dir:
def create_story_file(input_dict, save_dir):
    '''
    input:
    input_dict: input dictionary, include information about its id, content, summary etc
    save_dir: a directory about where to save the story files
    '''
    dic_id = input_dict["id"]
    filename = os.path.join(save_dir, dic_id + ".story")
    file1 = open(filename,"w")
    file1.writelines(input_dict["content"]+'\n')
    file1.writelines('@highlight \n')
    file1.writelines(input_dict['summary'])
    file1.close()
    
    
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
    
    
# create a .txt file and save it to the directory save_dir using baseline summarization method
# (first three sentence):
def create_baseline_summarization_file(input_dict, dec_dir):
    '''
    input:
    input_dict: input dictionary, include information about its id, content, summary etc
    dec_dir: a directory where to save the summarization files
    '''
    content = input_dict["content"]
    #print(content)
    #print("\n")
    summarization_top3 = ' '.join(re.split(r'(?<=[.:;])\s', content)[:3])
    #print(summarization_top3)
    dic_id = input_dict["id"]
    filename = os.path.join(dec_dir, dic_id + "_decoded.txt")
    file1 = open(filename,"w")
    file1.writelines(summarization_top3)
    file1.close()
    
    
    
    
# create a .txt file and save it to the directory save_dir using baseline summarization method
# (first three sentence):
def create_reference_file(input_dict, ref_dir):
    '''
    input:
    input_dict: input dictionary, include information about its id, content, summary etc
    ref_dir: a directory about where to save the summarization files
    '''
    reference = input_dict["summary"].encode('utf-8')
    #print(type(reference))
    #content = input_dict["content"]
    #summarization_with_first_3_sentences = ' '.join(re.split(r'(?<=[.:;])\s', content)[:3])
    dic_id = input_dict["id"]
    filename = os.path.join(ref_dir, dic_id + "_reference.txt")
    file1 = open(filename,"w")
    file1.writelines(reference)
    file1.close()
    
    
    
# input_type: train/val/test
# input_str: name of a story
def create_list(subreddit_type, input_type, input_str):
    filename = os.path.join("/home/ubuntu/cs224u/new_"+ subreddit_type+"/", subreddit_type + input_type + "list.txt")
    f = open(filename,"w")
    f.writelines(input_str)
    f.close()    
 
