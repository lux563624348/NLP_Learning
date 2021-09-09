########################################################################
## 09/18/2019
## By Xiang Li,
## lux@gwu.edu
## 
########################################################################
## Usage python Word_Bigram_Model.py  the data has to be at the same directory with script. 
########################################################################
## Python 3.7+

import re
import numpy as np


def head_file(file_path, num_rows):
    ## num_rows has to be a 2 elements list
    with open(file_path, mode='r', newline='') as file:
        row_range=range(num_rows[0],num_rows[1]+1)
        i=0
        for line in file:
            if (i in row_range):               
                   print (line)
            i+=1
    return None

def filter_alphanumeric(word):
#\w matches any alphanumeric character
    merge_words_no_digit=''
    if (word!=''):
        all_match = re.findall('\w+', word)
        all_match = list(filter(None, all_match))
        merge_words=''
        for item in all_match:
            merge_words+=item
        ## Find All unicode, then all non digits
        merge_words=re.findall('[^(\_|\d)]', merge_words)
        for item in merge_words:
            merge_words_no_digit+=item        
    else:
        #print ("Warning, One word is empty.")
        return '' 
    return merge_words_no_digit.lower()

def return_word_list_from_file(Path_File):
    list_words=list()
    with open(Path_File,  mode='r', newline='') as file:
        for line in file:
            for word in (re.split("\s+", line.rstrip('\n'))):
                if (word !=''):
                    list_words.append(filter_alphanumeric(word))
## if different languague, above line has to be changed
    list_words = list(filter(None, list_words))
    return list_words

def return_sentence_list_from_file(Path_File):
    list_sentences=list()
    with open(Path_File,  mode='r', newline='') as file:
        for line in file:
            if (len(line)>1):
                sentence = line.rstrip('\n').rstrip('\r')
                #marked_sentence = #'<^> '+line.rstrip('\n').rstrip('\r')+' </s>'
                list_sentences.append(sentence)
    list_sentences = list(filter(None, list_sentences))
    return list_sentences

def return_vocabulary_from_sentence_list(sentence_list):
    total_vocabulary=set({'^','$'})
    for sentence in sentence_list:
        word_list = re.split("\s+", sentence.rstrip('\n'))
        for word in word_list:
            filtered_word = filter_alphanumeric(word)
            if(filtered_word!=''):
                total_vocabulary.add(filtered_word)
    return sorted(list(total_vocabulary))

def return_unigram_counts(sentence_list, vocabulary):
    count_matrix =np.zeros((len(vocabulary)))
    count_matrix += len(vocabulary) ######## Add-one smoothing
    ## Set value for sentence start <s>
    count_matrix[vocabulary.index('^')] += len(sentence_list)
    count_matrix[vocabulary.index('$')] += len(sentence_list)
    for tem_sentence in sentence_list:
        word_list = re.split("\s+", tem_sentence.rstrip('\n'))
        for word in word_list:
            filtered_word = filter_alphanumeric(word)
            if(filtered_word in vocabulary):
                word_index = vocabulary.index(filtered_word)
                count_matrix[word_index]+=1
    return count_matrix

def return_bigram_word_counts(sentence_list, vocabulary):
    count_matrix=np.zeros((len(vocabulary),len(vocabulary)))
    count_matrix+=1  ######## Add-one smoothing
    num_word=0
    for tem_sentence in sentence_list:
        word_list = re.split("\s+", tem_sentence.rstrip('\n'))
        word_list = list(filter(None, word_list))
        
        for i in range(0,len(word_list)-1):
            num_word+=1
            if (i==0):
                first_word = filter_alphanumeric(word_list[0])
                if(first_word in vocabulary):
                    count_matrix[vocabulary.index('^'), vocabulary.index(first_word)] += 1
            tem_bigram_count_pairs = word_list[i:i+2]
            first_word=filter_alphanumeric(tem_bigram_count_pairs[0])
            second_word=filter_alphanumeric(tem_bigram_count_pairs[1])
            
            if((first_word in vocabulary) & (second_word in vocabulary)):
                first_digit=vocabulary.index(first_word)
                second_digit=vocabulary.index(second_word)
                count_matrix[first_digit,second_digit]+=1
    print ("Number of Words: "+ str(num_word))
    return count_matrix

def Norm_Bigram(unigram_counts, bigram_counts, vocabulary): 
    norm_bigram=np.zeros((len(vocabulary),len(vocabulary)))
    for i in range(len(vocabulary)):
        norm_bigram[i,:]=bigram_counts[i,:]/unigram_counts[i]
    return norm_bigram    

def return_Probability_of_Sentence_Word(sentence, norm_bigram_counts, vocabulary):
    word_list = re.split("\s+", sentence.rstrip('\n'))
    Log_Probability_sentence = 0

    ## Filter Null Element
    word_list = list(filter(None, word_list))
    
    for i in range(0,len(word_list)-1):
        if (i==0):
            first_word=filter_alphanumeric(word_list[0])
            if(first_word in vocabulary):
                Log_Probability_sentence += np.log(norm_bigram_counts[vocabulary.index('^'), vocabulary.index(first_word)])
            else:
                ## assuming new word as a probability of 1/V
                Log_Probability_sentence += np.log(1/len(vocabulary)**2)
        tem_bigram_count_pairs = word_list[i:i+2]
        first_word=filter_alphanumeric(tem_bigram_count_pairs[0])
        second_word=filter_alphanumeric(tem_bigram_count_pairs[1])
        if((first_word in vocabulary) & (second_word in vocabulary)):
            first_index=vocabulary.index(first_word)
            second_index=vocabulary.index(second_word)
            Log_Probability_sentence += np.log(norm_bigram_counts[first_index, second_index])
            #print (first_word + " "+ second_word )
            #print (Log_Probability_sentence)
        else:
            ## assuming new word as a probability of 1/V
            Log_Probability_sentence += np.log(1/len(vocabulary)**2)
    return np.e**Log_Probability_sentence

def return_Probability_of_Sentence_trigram_Word(sentence, norm_bigram, vocabulary, p_unseen):
    word_list = re.split("\s+", sentence.rstrip('\n'))
    Log_Probability_sentence = 0 
    ## Filter Null Element
    word_list = list(filter(None, word_list))
    
    for i in range(0,len(word_list)-2):
        if (i==0):
            first_word=filter_alphanumeric(word_list[0])
            second_word=filter_alphanumeric(word_list[1])
            if ((first_word in vocabulary) & (second_word in vocabulary)):
                Log_Probability_sentence += np.log(norm_bigram[vocabulary.index('^'), vocabulary.index(first_word)])
                Log_Probability_sentence += np.log(norm_bigram[vocabulary.index(first_word),vocabulary.index(second_word)])
            else:
                Log_Probability_sentence += 2*np.log(p_unseen) # np.log(1/len(vocabulary)**2) #
        tem_trigram_count_pairs = word_list[i:i+3]
        first_word=filter_alphanumeric(tem_trigram_count_pairs[0])
        second_word=filter_alphanumeric(tem_trigram_count_pairs[1])
        third_word=filter_alphanumeric(tem_trigram_count_pairs[2])
        if((first_word in vocabulary) & (second_word in vocabulary) & (third_word in vocabulary)):
            first_index=vocabulary.index(first_word)
            second_index=vocabulary.index(second_word)
            third_index=vocabulary.index(third_word)                                                  
            Log_Probability_sentence += np.log(norm_bigram[first_index,second_index])    
            Log_Probability_sentence += np.log(norm_bigram[second_index,third_index])       
        else:
            Log_Probability_sentence += 2*np.log(p_unseen) #np.log(1/len(vocabulary)**2)
    return np.e**Log_Probability_sentence

def return_probability(Path_Test_Data, Norm_Bigram_counts, vocabulary):
    count_matrix=list()
    P_unseen=Norm_Bigram_counts.min()
    with open(Path_Test_Data,  mode='r', newline='') as file:
        i=0
        for line in file:
            results= return_Probability_of_Sentence_trigram_Word(line, Norm_Bigram_counts, vocabulary, P_unseen)
            count_matrix.append(results)
            i+=1
    return count_matrix
## Main
def test_with_word_model(path_train_data, path_test_data):  
    ## first generate sentence list
    sentence_list = return_sentence_list_from_file(path_train_data)
    ## building vocabulary
    vocabulary = return_vocabulary_from_sentence_list(sentence_list)

    raw_unigram_word_counts = return_unigram_counts(sentence_list, vocabulary)
    raw_bigram_word_counts  = return_bigram_word_counts(sentence_list,vocabulary)   
    norm_bigram_word_counts = Norm_Bigram(raw_unigram_word_counts, raw_bigram_word_counts, vocabulary)
    
    # test data output
    prob = return_probability(path_test_data, norm_bigram_word_counts, vocabulary)
    return prob

def Save_Results(En_prob,FR_prob,GR_prob, NAME):
    data_prob_output = np.transpose([En_prob,FR_prob,GR_prob])
    index_LANG=['EN','FR','GR']
    #out_index=list()
    file1 = open('Results_'+ NAME + '_Model.txt',"w")
    file1.writelines(['ID','LANG'])
    i=1
    for prob_row in data_prob_output[:]:
        index_l = list(prob_row).index(max(prob_row))
        #out_index.append([i,index_LANG[index_l]])
        file1.writelines([str(i),index_LANG[index_l]]) 
        i+=1
    file1.close() #to change file access modes
    
    #pd.DataFrame(out_index,columns=['ID','LANG']).set_index('ID').to_csv('Results_'+ NAME + '_Model.txt', sep="\t")
    print("Output Results can be found at the current directory!")
    print("Output Name is: "+ 'Results_'+ NAME + '_Model.txt')
    return data_prob_output


    
def main():
	Path_Data='./'
	Train_Data_Set=['EN.txt', 'FR.txt', 'GR.txt']
	Path_test_data = Path_Data+'LangID.test.txt'
	All_prob=list()
	for name_train in Train_Data_Set:
		Path_train_data = Path_Data+name_train
		print ("Train on " + name_train+ "......")
		All_prob.append(test_with_word_model(Path_train_data, Path_test_data))

	NAME='Word_Trigram'
	Save_Results(All_prob[0],All_prob[1],All_prob[2], NAME)
	return 0
	
print ("Start")
main()
print ("End")



