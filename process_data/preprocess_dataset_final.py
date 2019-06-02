from process_util import *
from random import sample

import os
import shutil
import glob




def create_preprocessed_story_file_include_subreddit(input_dict, save_dir):
    '''
    input:
    input_dict: input dictionary, include information about its id, content, summary etc
    save_dir: a directory about where to save the story files
    reference: https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908
    here we preprocessed the content and the summary of the story by:
    1) get rid of extra space tab
    2) filter out those whose summary is too short/content is too short
    3) delete special characters like [...]
    4) [potential] Stemming (spend/spent/spends...)
    5) [potential] Lemmatization (do/done/did)
    '''
    dic_id = input_dict["id"]
    content = input_dict["content"]
    summary = input_dict['summary']
    subreddit = input_dict["subreddit"]
    #print(type(summary.split()))
    if(len(summary.split()) > 3):
        # get rid of extra space tab
        content = re.sub('\s+', ' ', content).strip()
        summary = re.sub('\s+', ' ', summary).strip()    
        # get rid of words inside special characterss
        content = re.sub("[\(\[].*?[\)\]]", "", content)
        summary = re.sub("[\(\[].*?[\)\]]", "", summary)

        filename = os.path.join(save_dir, dic_id +'_'+subreddit+ ".story")
        file1 = open(filename,"w")
        # add the subreddit information before the summary
        file1.writelines(content+'\n')
        file1.writelines('@highlight \n')
        file1.writelines(subreddit+' '+summary)
        file1.close()
        
        
def create_final_list(subreddit_type, input_type, input_str):
    filename = os.path.join("/home/ubuntu/cs224u/processed_10_1k"+'/'+"processed_"+subreddit_type+"/", subreddit_type + input_type + "_list.txt")
    print(filename)
    f = open(filename,"w")
    f.writelines(input_str)
    f.close()          
       
        
          
       
        
        



def main():
    SubReddits_to_include_10 = ['relationships', 'legaladvice', 'nfl',  'pettyrevenge', 'atheismbot', 'ShouldIbuythisgame', 'ukpolitics', 'Dogtraining',  'AskHistorians', 'Anxiety']


    # read jsonl dataset
    src = "/home/ubuntu/cs224u/raw_reddit/tldr-training-data.jsonl"
    reader = jsonlines.open(src)
    for i in range(len(SubReddits_to_include_10)):
        print(i)
        # choose a subreddit
        input_subreddit = SubReddits_to_include_10[i]
        # create the directory to this corresponding dataset
        dst = "/home/ubuntu/cs224u/processed_10_1k_mymodel"+'/'+"processed_"+input_subreddit
        os.mkdir(dst)
        dst = "/home/ubuntu/cs224u/processed_10_1k_mymodel"+'/'+"processed_"+input_subreddit+"/"+input_subreddit+'_story'
        os.mkdir(dst)
        #os.mkdir(dst)
        # get a corresponding dataset
        count = 0
        dic_list = []
        reader = jsonlines.open(src)
        for dic in reader:
            if("subreddit" in dic.keys() and dic["subreddit"] == input_subreddit and 
               isEnglish(dic["summary"]) == True and  isEnglish(dic["content"]) == True):
                dic_list.append(dic)

        # create a small dataset if needed
        sample_list = sample(dic_list,1200)
        for dic in sample_list:
            create_preprocessed_story_file_include_subreddit(dic, dst)  

        #input_subreddit = "relationships"
        #dst = "/home/ubuntu/cs224u/new_relationships"+'/'+input_subreddit+'_story'
        result_list = os.listdir(dst)
        np.random.shuffle(result_list)
        size = len(result_list)

        train_list = result_list[0:int(0.8*size)-1]
        train_str = "\n".join(x for x in train_list)

        dev_list = result_list[int(0.8*size):int(0.9*size)-1]
        dev_str = "\n".join(x for x in dev_list)

        test_list = result_list[int(0.9*size): int(size)-1]
        test_str = "\n".join(x for x in test_list)
        # create three lists
        create_final_list(input_subreddit, "_train", train_str)
        create_final_list(input_subreddit, "_val", dev_str)
        create_final_list(input_subreddit, "_test", test_str)
        
        
        
        # combine all data into a file
    root_dir = "/home/ubuntu/cs224u/processed_10_1k_mymodel"
    # create a processed_combine_all file
    dst = "/home/ubuntu/cs224u/processed_10_1k_mymodel"+'/'+"processed_combine_all"
    os.mkdir(dst)
    dest = "/home/ubuntu/cs224u/processed_10_1k_mymodel"+'/'+"processed_combine_all/"+ "combine_all_story"
    os.mkdir(dest)


    for i in range(len(SubReddits_to_include_10)):
        # add all information into the processed_combine_all file from 
        input_subreddit = SubReddits_to_include_10[i]
        src = root_dir + "/processed_"+ input_subreddit + "/"+input_subreddit + "_story"
        src_files = os.listdir(src)
        for file_name in src_files:
            full_file_name = os.path.join(src, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dest)

 # combine all subreddit_train_list.txt data into a combine_train_list.txt
    # create a result outfile name combine_train_list.txt
    # combine_train_list.txt

    combine_train = os.path.join(dst, "combine_train_list.txt")
    print(combine_train)

    #"/home/ubuntu/cs224u/processed_10_1k/processed_" + input_subreddit+ "/"+input_subreddit+"_train_list.txt"

    for i in range(len(SubReddits_to_include_10)):
        # add all information into the processed_combine_all file from 
        input_subreddit = SubReddits_to_include_10[i]
        read_files = glob.glob("/home/ubuntu/cs224u/processed_10_1k_mymodel/processed_" + input_subreddit+ "/"+input_subreddit+"_train_list.txt")
        print(read_files)
        combine_train = os.path.join(dst, "combine_train_list.txt")
        with open(combine_train, "a") as outfile:
            for f in read_files:
                with open(f, "rb") as infile:
                    outfile.writelines(infile.read())
                outfile.writelines("\n")
          #  num_lines = sum(1 for line in open("/home/ubuntu/cs224u/processed_10_1k/processed_combine_all/combine_all_story/combine_train_list.txt"))
          #  print(num_lines)    




        # combine all subreddit_train_list.txt data into a combine_train_list.txt
        # create a result outfile name combine_train_list.txt
        # combine_train_list.txt

        combine_train = os.path.join(dst, "combine_val_list.txt")
        print(combine_train)

        #"/home/ubuntu/cs224u/processed_10_1k/processed_" + input_subreddit+ "/"+input_subreddit+"_train_list.txt"

        for i in range(len(SubReddits_to_include_10)):
            # add all information into the processed_combine_all file from 
            input_subreddit = SubReddits_to_include_10[i]
            read_files = glob.glob("/home/ubuntu/cs224u/processed_10_1k_mymodel/processed_" + input_subreddit+ "/"+input_subreddit+"_val_list.txt")
            print(read_files)
            combine_train = os.path.join(dst, "combine_val_list.txt")
            with open(combine_train, "a") as outfile:
                for f in read_files:
                    with open(f, "rb") as infile:
                        outfile.writelines(infile.read())
                    outfile.writelines("\n")
             #   num_lines = sum(1 for line in open("/home/ubuntu/cs224u/processed_10_1k/processed_combine_all/combine_all_story/combine_val_list.txt"))
             #   print(num_lines)

        # combine all subreddit_train_list.txt data into a combine_train_list.txt
        # create a result outfile name combine_train_list.txt
        # combine_train_list.txt

        combine_train = os.path.join(dst, "combine_test_list.txt")
        print(combine_train)

        #"/home/ubuntu/cs224u/processed_10_1k/processed_" + input_subreddit+ "/"+input_subreddit+"_train_list.txt"

        for i in range(len(SubReddits_to_include_10)):
            # add all information into the processed_combine_all file from 
            input_subreddit = SubReddits_to_include_10[i]
            read_files = glob.glob("/home/ubuntu/cs224u/processed_10_1k_mymodel/processed_" + input_subreddit+ "/"+input_subreddit+"_test_list.txt")
            print(read_files)
            combine_train = os.path.join(dst, "combine_test_list.txt")
            with open(combine_train, "a") as outfile:
                for f in read_files:
                    with open(f, "rb") as infile:
                        outfile.writelines(infile.read())
                outfile.writelines("\n")
            #    num_lines = sum(1 for line in open("/home/ubuntu/cs224u/processed_10_1k/processed_combine_all/combine_all_story/combine_test_list.txt"))
            #    print(num_lines)

                  

        
        

    
    
    
    

    
    
if __name__ == "__main__":
    main()
    
    