########################################################################
## 09/18/2019
## By Xiang Li,
## lux@gwu.edu
## 
########################################################################
## Usage python Letter_Bigram_Model.py  the data has to be at the same directory with script. 
########################################################################
## Python 3.7+
import re
import numpy as np


def return_letter_list(start_letter, end_letter):
    alpha=start_letter
    letter_list=list()
    for i in range(0,26):
        letter_list.append(alpha)
        if (alpha==end_letter):
            break
        alpha= chr(ord(alpha)+1)
    return letter_list

def filter_alphanumeric(word):
#\w matches any alphanumeric character
    if (word!=''):
        all_match = re.findall('\w+', word)
        all_match = list(filter(None, all_match))
        merge_words=''
        for item in all_match:
            merge_words+=item
        ## Find All unicode, then all non digits
        merge_words=re.findall('[^(\_|\d)]', merge_words)
        merge_words_no_digit=''
        for item in merge_words:
            merge_words_no_digit+=item        
    else:
        print ("Error!")
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
    list_words = given_word_marker(list_words)
    return list_words

def given_word_marker(word_list):
    marked_word_list=list()
    for word in word_list:
        marked_word_list.append('^'+word+'$')
    return marked_word_list


## Renturn Statistic Training Calculation
def return_alphabet_from_words_list(list_word):
    total_alphabet=set()
    for word in list_word:
        for letter in set(word):
            total_alphabet.add(letter)
    return sorted(list(total_alphabet))

def return_bigram_counts(word_list, alphabet):
    count_matrix=np.zeros((len(alphabet),len(alphabet)))
    alpha_a_digit=ord('a')
    num_word=0
    for tem_word in word_list:
        num_word+=1
        for i in range(len(tem_word)-1):
            tem_bigram_count_pairs = tem_word[i:i+2]
            first_letter=tem_bigram_count_pairs[0]
            second_letter=tem_bigram_count_pairs[1]
            ## Because marker ^ and $ take two space      
            ## convert to numberic value
            first_digit=alphabet.index(first_letter)
            second_digit=alphabet.index(second_letter)
            count_matrix[first_digit,second_digit]+=1
    print ("Number of Words: "+ str(num_word))
    return count_matrix

def return_unigram_counts(word_list, alphabet):
    count_matrix=np.zeros((len(alphabet)))
    alpha_a_digit=ord('a')
    for tem_word in word_list:
        for i in range(len(tem_word)):
            letter_value=alphabet.index(tem_word[i])
            ## Because marker ^ and $ take two space
            count_matrix[letter_value]+=1
    return count_matrix

def Norm_Bigram(unigram_counts, bigram_counts, alphabet):
    alpha_a_digit=ord('a')
    norm_bigram=np.zeros((len(alphabet),len(alphabet)))
    for i in range(len(alphabet)):
        norm_bigram[i,:]=bigram_counts[i,:]/unigram_counts[i]
    return norm_bigram
    
def return_Probability_of_word(input_word, norm_bigram_model, alphabet):
    Log_P_word=0
    filtered_input_word = filter_alphanumeric(input_word)
    if(filtered_input_word==''):
        Log_P_word+=0
    else:
        tem_word='^'+filtered_input_word+'$'
        for i in range(len(tem_word)-1):
            tem_bigram_count_pairs = tem_word[i:i+2]
            if((tem_bigram_count_pairs[0] in alphabet) & (tem_bigram_count_pairs[1] in alphabet)):
                first_digit=alphabet.index(tem_bigram_count_pairs[0])
                second_digit=alphabet.index(tem_bigram_count_pairs[1])
                if(norm_bigram_model[first_digit][second_digit]!=0):
                    Log_P_word += np.log(norm_bigram_model[first_digit][second_digit])
                else:
                    Log_P_word +=0
            else:
                Log_P_word+=0
    return np.e**(Log_P_word)

def return_Probability_of_Sentence(sentence, norm_bigram_counts, alphabet):
    sentence = re.split("\s+", sentence.rstrip('\n'))
    Log_Probability_sentence = 0
    ## Filter Null Element
    sentence = list(filter(None, sentence))
    for word in sentence[1:]:
        p_word=return_Probability_of_word(word, norm_bigram_counts,alphabet)
        if (p_word!=0):
            Log_Probability_sentence += np.log(p_word)
        else:
            Log_Probability_sentence += 0
    return np.e**Log_Probability_sentence

def return_probability(Path_Test_Data, Norm_Bigram_counts, Alphabet):
    count_matrix=list()
    with open(Path_Test_Data,  mode='r', newline='') as file:
        i=0
        for line in file:
            results= return_Probability_of_Sentence(line, Norm_Bigram_counts, Alphabet)
            count_matrix.append(results)
            i+=1
    return count_matrix

## 
def test_with_model(path_train_data, path_test_data):
    ## first generate words list
    test_word = return_word_list_from_file(path_train_data)
    ## build up alphabet with ^ and $
    alphabet = return_alphabet_from_words_list(test_word)

    raw_unigram_counts =return_unigram_counts(test_word,alphabet)
    raw_bigram_counts  =return_bigram_counts(test_word,alphabet)    
    norm_bigram_counts =Norm_Bigram(raw_unigram_counts, raw_bigram_counts, alphabet)

    # test data output
    prob = return_probability(path_test_data, norm_bigram_counts, alphabet)
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
		All_prob.append(test_with_model(Path_train_data, Path_test_data))

	NAME='Letter_Bigram'
	Save_Results(All_prob[0],All_prob[1],All_prob[2], NAME)
	return 0
	
print ("Start")
main()
print ("End")
