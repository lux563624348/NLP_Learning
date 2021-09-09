########################################################################
## 10/07/2019
## By Xiang Li,
## lux@gwu.edu
## 
########################################################################
## Usage python Word_Bigram_Model.py  the data has to be at the same directory with script. 
########################################################################
## Python 3.7+

import re
import numpy as np


def filter_alphanumeric(word):
    return word.lower()

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
    
def return_tag_from_words(word):
    # tagging_mark = "/"
    split_word_set = word.split("/")
    return split_word_set[len(split_word_set)-1] ### for some word, it may contains / that is not for tagging mark

def return_word_from_tagged_words(word):
    # tagging_mark = "/"
    split_word_set = word.split("/")
    return filter_alphanumeric(split_word_set[0])  ## For increasing speed
#'/'.join(split_word_set[0:len(split_word_set)-1]) ### for some word, it may contains / that is not for tagging mark

def return_Tagset_from_train_data(sentence_list):
    total_Tagset=set({'^','$'})
    for sentence in sentence_list:
        word_list = re.split("\s+", sentence.rstrip('\n'))
        word_list = list(filter(None, word_list))## drop out empty items
        for word in word_list:
            tag = return_tag_from_words(word)
            total_Tagset.add(tag)
    return dict(zip(sorted(list(total_Tagset)), np.arange(0,len(total_Tagset),1)))
#

def return_Vocabulary_from_tagged_data(sentence_list):
    total_vocabulary=set({'^','$'})
    for sentence in sentence_list:
        word_list = re.split("\s+", sentence.rstrip('\n'))
        word_list = list(filter(None, word_list))## drop out empty items
        for word in word_list:
            filtetered_word = return_word_from_tagged_words(word)
            total_vocabulary.add(filtetered_word)
    return dict(zip(sorted(list(total_vocabulary)), np.arange(0,len(total_vocabulary),1)))
#sorted(list(total_vocabulary))

def return_unigram_tag_counts(sentence_list, tag_set):
    count_matrix =np.zeros((len(tag_set)))
    #count_matrix += len(tag_set) ######## Add-one smoothing
    ## Set value for sentence start '^' end '$'
    count_matrix[tag_set['^']] += len(sentence_list)
    count_matrix[tag_set['$']] += len(sentence_list)
    for tem_sentence in sentence_list:
        word_list = re.split("\s+", tem_sentence.rstrip('\n'))
        word_list = list(filter(None, word_list))## drop out empty items
        for word in word_list:
            tag = return_tag_from_words(word)
            if(tag in tag_set):
                tag_index = tag_set[tag]
                count_matrix[tag_index]+=1
    return count_matrix

def return_bigram_tag_counts(sentence_list, tag_set):
    count_matrix=np.zeros((len(tag_set),len(tag_set)))
    #count_matrix+=1  ######## Add-one smoothing
    num_word=0
    for tem_sentence in sentence_list:
        word_list = re.split("\s+", tem_sentence.rstrip('\n'))
        word_list = list(filter(None, word_list))## drop out empty items   
        for i in range(0,len(word_list)-1):
            num_word+=1
            if (i==0):
                first_tag = return_tag_from_words(word_list[0])
                count_matrix[tag_set['^'], tag_set[first_tag]] += 1
            else:
                first_tag=return_tag_from_words(word_list[i])
                second_tag=return_tag_from_words(word_list[i+1])
            #if((first_word in vocabulary) & (second_word in vocabulary)):
                first_digit=tag_set[first_tag]
                second_digit=tag_set[second_tag]
                count_matrix[first_digit,second_digit]+=1
    print ("Number of Words: "+ str(num_word))
    return count_matrix

def return_bigram_tag2word_counts(sentence_list, tag_set, vocabulary):
    count_matrix=np.zeros((len(tag_set),len(vocabulary)+1))
    #count_matrix+=1  ######## Add-one smoothing
    i=0
    size_sentence=len(sentence_list)
    for tem_sentence in sentence_list:
        i+=1
        if (i%10000==1):
            print ("Traning Complete Rate: " + str(str(i/size_sentence)))
        word_list = re.split("\s+", tem_sentence.rstrip('\n'))
        word_list = list(filter(None, word_list))## drop out empty items        
        for word in word_list:
            tag = return_tag_from_words(word)
            if(tag in tag_set):
                filtered_word= return_word_from_tagged_words(word)
                tag_index = tag_set[tag]
                
                word_index = vocabulary[filtered_word] ### This process takes most of CPU time
                count_matrix[tag_index, word_index]+=1
            else:
                continue
    count_matrix[:,len(vocabulary)]+=1/len(vocabulary)
    return count_matrix


def return_probability_transition(tag_unigram_counts, bigram_tag_counts):
    size = len(tag_unigram_counts)
    Probability_transition = np.zeros((size, size))
    for i in range(len(tag_unigram_counts)):
        Probability_transition[i] = bigram_tag_counts[i]/tag_unigram_counts[i]
    return Probability_transition
    
def return_probability_emission(tag_unigram_counts, emission_counts):
    matrix_nrow = len(tag_unigram_counts)
    matrix_ncols = emission_counts.shape[1] ## number of columns
    Probability_emission = np.zeros((matrix_nrow, matrix_ncols))
    for i in range(len(tag_unigram_counts)):
        Probability_emission[i] = emission_counts[i]/tag_unigram_counts[i]
    return Probability_emission

def Save_Transition_Matrix(matrix, NAME, tag_dict):
    data_prob_output = matrix
    file1 = open('Q2_HMM_'+ NAME + '_Model.txt',"w")

    ## columns name
    file1.writelines(NAME+"\t")
    file1.writelines('\t'.join(map(str,list(tag_dict))))
    file1.writelines("\n")
    i=1
    for prob_row in data_prob_output:
        file1.writelines([list(tag_dict)[i-1],'\t'])
        file1.writelines('\t'.join(map(str,prob_row)))
        file1.writelines("\n")
        i+=1
    file1.close()
    print("Output Results can be found at the current directory!")
    print("Output Name is: "+ 'Q2_HMM_'+ NAME + '_Model.txt')
    return None

def Save_Emission_Matrix(matrix, NAME, tag_dict, vocabulary):
    data_prob_output = matrix
    file1 = open('Q2_HMM_'+ NAME + '_Model.txt',"w")

    file1.writelines(NAME+"\t")
    file1.writelines('\t'.join(map(str,list(vocabulary))))
    file1.writelines('\t'+"<OOV>"+"\n")

    i=1
    for prob_row in data_prob_output:
        file1.writelines([list(tag_dict)[i-1],'\t'])
        file1.writelines('\t'.join(map(str,prob_row)))
        file1.writelines("\n")
        i+=1
    file1.close()
    print("Output Results can be found at the current directory!")
    print("Output Name is: "+ 'Q2_HMM_'+ NAME + '_Model.txt')
    return None



def main():
	
	training_data = return_sentence_list_from_file('brown.train.tagged.txt')

	sentence_list = training_data#[0:10000]
	tag_dict = return_Tagset_from_train_data(sentence_list)
	vocabulary_dict = return_Vocabulary_from_tagged_data(sentence_list)

	tag_unigram_counts = return_unigram_tag_counts(sentence_list, tag_dict)
	bigram_tag_counts = return_bigram_tag_counts(sentence_list, tag_dict)
	emission_counts = return_bigram_tag2word_counts(sentence_list, tag_dict, vocabulary_dict)

	probability_transition = return_probability_transition(tag_unigram_counts, bigram_tag_counts)

	probability_emission = return_probability_emission(tag_unigram_counts, emission_counts)

	NAME="transition_matrix"
	Save_Transition_Matrix(probability_transition, NAME, tag_dict)
	NAME="emission_matrix"
	Save_Emission_Matrix(probability_emission, NAME, tag_dict, vocabulary_dict)

	
	return 0
	
print ("Start")
main()
print ("End")
