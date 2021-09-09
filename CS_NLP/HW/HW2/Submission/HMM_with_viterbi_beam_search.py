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


def filter_word(word):
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
    return filter_word(split_word_set[0])  ## For increasing speed
#'/'.join(split_word_set[0:len(split_word_set)-1]) ### for some word, it may contains / that is not for tagging mark
def return_Tagset_from_train_data(sentence_list):
    total_Tagset=set({'^','$'})
    for sentence in sentence_list:
        word_list = re.split("\s+", sentence.rstrip('\n'))
        word_list = list(filter(None, word_list))## drop out empty items
        for word in word_list:
            tag = return_tag_from_words(word)
            total_Tagset.add(tag)
    tag_list = sorted(list(filter(None, total_Tagset)))
    return dict(zip(tag_list, np.arange(0,len(tag_list),1)))
#
def return_Vocabulary_from_tagged_data(sentence_list):
    total_vocabulary=set({'^','$'})
    for sentence in sentence_list:
        word_list = re.split("\s+", sentence.rstrip('\n'))
        word_list = list(filter(None, word_list))## drop out empty items
        for word in word_list:
            filtetered_word = return_word_from_tagged_words(word)
            total_vocabulary.add(filtetered_word)
    vocabulary_list = sorted(list(filter(None, total_vocabulary)))
    return dict(zip(vocabulary_list, np.arange(0,len(vocabulary_list),1)))
#sorted(list(total_vocabulary))

def return_unigram_tag_counts(sentence_list, tag_set):
    count_matrix =np.zeros((len(tag_set)))
    count_matrix += len(tag_set.keys()) ######## Add-one smoothing
    ## Set value for sentence start '^' end '$'
    count_matrix[tag_set['^']] += len(sentence_list)
    count_matrix[tag_set['$']] += len(sentence_list)
    for tem_sentence in sentence_list:
        word_list = re.split("\s+", tem_sentence.rstrip('\n'))
        word_list = list(filter(None, word_list))## drop out empty items
        for word in word_list:
            tag = return_tag_from_words(word)
            if(tag in tag_set.keys()):
                tag_index = tag_set[tag]
                count_matrix[tag_index]+=1
    return count_matrix

def return_bigram_tag_counts(sentence_list, tag_set):
    count_matrix=np.zeros((len(tag_set),len(tag_set)))
    count_matrix+=1  ######## Add-one smoothing
    num_word=0
    for tem_sentence in sentence_list:
        word_list = re.split("\s+", tem_sentence.rstrip('\n'))
        word_list = list(filter(None, word_list))## drop out empty items   
        for i in range(0,len(word_list)-1):
            num_word+=1
            if (i==0):
                first_tag = return_tag_from_words(word_list[0])
                if(first_tag in tag_set.keys()):
                    count_matrix[tag_set['^'], tag_set[first_tag]] += 1
            else:
                first_tag=return_tag_from_words(word_list[i])
                second_tag=return_tag_from_words(word_list[i+1])
                if((first_tag in tag_set.keys()) & (second_tag in tag_set.keys())):
                    count_matrix[tag_set[first_tag],tag_set[second_tag]]+=1
    print ("Number of Words: "+ str(num_word))
    return count_matrix

def return_bigram_tag2word_counts(sentence_list, tag_set, vocabulary):
    count_matrix=np.zeros((len(tag_set),len(vocabulary)+1))
    count_matrix+=1/len(vocabulary)  ######## Add-one smoothing
    i=0
    size_sentence=len(sentence_list)
    for tem_sentence in sentence_list:
        i+=1
        if (i%10000==0):
            print ("Traning Data..." + str(str('%.2f' % (i/size_sentence))) + " Completed")
        word_list = re.split("\s+", tem_sentence.rstrip('\n'))
        word_list = list(filter(None, word_list))## drop out empty items        
        for word in word_list:
            tag = return_tag_from_words(word)
            if( tag in tag_set.keys()):
                filtered_word= return_word_from_tagged_words(word)
                if (filtered_word in vocabulary.keys()):
                    ### This process takes most of CPU time
                    count_matrix[tag_set[tag], vocabulary[filtered_word]]+=1
    count_matrix[:,len(vocabulary)]+=1/len(tag_set.keys())
    print ("Traning Data..." + str(1) + " Completed")
    return count_matrix

def return_probability_transition(tag_unigram_counts, bigram_tag_counts):
    size = len(tag_unigram_counts)
    Probability_transition = np.zeros((size, size))
    for i in range(len(tag_unigram_counts)):
        Probability_transition[i] = bigram_tag_counts[i]/tag_unigram_counts[i]
    return Probability_transition
    
def return_probability_emission(tag_unigram_counts, emission_counts):
    matrix_nrow = len(tag_unigram_counts)
    matrix_ncols = emission_counts.shape[1] ## number of columns+ last columns for Out of Vocabulary
    Probability_emission = np.zeros((matrix_nrow, matrix_ncols))
    for i in range(len(tag_unigram_counts)):
        Probability_emission[i] = emission_counts[i]/tag_unigram_counts[i]
    return Probability_emission

def Deconding_viterbi(test_sentence_list, tag_dict, vocabulary_dict, probability_transition, probability_emission):
    best_path_list = []
    tag_list=list(tag_dict)
    print ("")
    print ("Test Data......" + " Start")
    
    percentage = 0
    size_sentence = len(test_sentence_list)
    for sentence in test_sentence_list:
        percentage+=1
        if(percentage%100==0):
            print ("Testing Data..." + str(str('%.2f' % (percentage/size_sentence))) + " Completed")
        word_list = re.split("\s+", sentence.rstrip('\n'))
        word_list = list(filter(None, word_list))## drop out empty items
        
        ## Applying viterbi algorithm
        probability_matrix_viterbi=np.zeros((len(tag_list),len(word_list)))
        tem_probability_matrix=np.zeros(len(tag_list)) ## a tem vector to calculate Viterbi for every time step
        for t in range(len(word_list)):
            for i in range(len(tag_dict)): # for loop over all tags
                filtered_word = return_word_from_tagged_words(word_list[t])
                if (filtered_word in vocabulary_dict.keys()): 
                ## vocabulary_dict.keys() is much faster than list(vocabulary_dict)
                    observation_index = vocabulary_dict[filtered_word]
                else:
                    ### Last Columns of emmision matrix is for <OOV>
                    observation_index = len(vocabulary_dict) 
                    ### Last Columns of emmision matrix is for <OOV>
                if (t==0):
                    ### Initialization Step
                    probability_matrix_viterbi[i,t] = probability_transition[tag_dict['^'],i] * probability_emission[i,observation_index]
                else:
                    ### Recursion step, get all probability to update vt(q)=max()
                    for index_tag in range(len(tag_dict)):
                        tem_probability_matrix[index_tag] = probability_matrix_viterbi[index_tag,t-1] * probability_transition[index_tag,i] * probability_emission[i,observation_index]
                    probability_matrix_viterbi[i,t] = max(tem_probability_matrix)

            max_index_tag = np.argmax(probability_matrix_viterbi[:,t]) ## Return max index from all tag
            #max_prob_tag = probability_matrix_viterbi[max_index_tag,t]
            ## Saving backtrace state
            tag_label   = return_tag_from_words(word_list[t])
            tag_from_HMM= tag_list[max_index_tag]
            best_path_list.append([tag_from_HMM,tag_label])
    return best_path_list

def Deconding_viterbi_with_beam_one(test_sentence_list, tag_dict, vocabulary_dict, probability_transition, probability_emission):
    best_path_list = []
    tag_list=list(tag_dict)
    print ("")
    print ("Using Beam Searching in Viterbi Decoding in Test Data......" + " Start")
    
    percentage = 0
    size_sentence = len(test_sentence_list)
    for sentence in test_sentence_list:
        percentage+=1
        if(percentage%1000==0):
            print ("Testing Data..." + str(str('%.2f' % (percentage/size_sentence))) + " Completed")
        word_list = re.split("\s+", sentence.rstrip('\n'))
        word_list = list(filter(None, word_list))## drop out empty items
        
        ## Applying viterbi algorithm
        probability_matrix_viterbi=np.zeros((len(tag_list),len(word_list)))
        for j in range(len(word_list)):
            for i in range(len(tag_dict)): # for loop over all tags
                filtered_word = return_word_from_tagged_words(word_list[j])
                if (filtered_word in vocabulary_dict.keys()): 
                ## vocabulary_dict.keys() is much faster than list(vocabulary_dict)
                    observation_index = vocabulary_dict[filtered_word]
                else:
                    ### Last Columns of emmision matrix is for <OOV>
                    observation_index = len(vocabulary_dict) 
                    ### Last Columns of emmision matrix is for <OOV>
                if (j==0):
                    ### Initialization Step
                    probability_matrix_viterbi[i,j] = probability_transition[tag_dict['^'],i] * probability_emission[i,observation_index]
                else:
                    ### Recursion step
                    probability_matrix_viterbi[i,j] = max_prob_tag*probability_transition[max_index_tag,i] * probability_emission[i,observation_index]

            max_index_tag = np.argmax(probability_matrix_viterbi[:,j]) ## Return max index from all tag
            max_prob_tag = probability_matrix_viterbi[max_index_tag,j]
            ## Saving backtrace state
            tag_label   = return_tag_from_words(word_list[j])
            tag_from_HMM= tag_list[max_index_tag]
            best_path_list.append([tag_from_HMM,tag_label])
    return best_path_list

def Save_Transition_Matrix(matrix, NAME, tag_set):
    data_prob_output = matrix
    file1 = open('Q2_HMM_'+ NAME + '_Model.txt',"w")

    ## columns name
    file1.writelines(NAME+"\t")
    file1.writelines('\t'.join(map(str,list(tag_set))))
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

def Save_Emission_Matrix(matrix, NAME, tag_set, vocabulary):
    data_prob_output = matrix
    file1 = open('Q2_a_'+ NAME + '_Model.txt',"w")

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

def Save_Results(matrix, NAME):
    data_prob_output = matrix
    file1 = open('Q2_'+ NAME + '_Model.txt',"w+")

    file1.writelines(['predicted_tag', '\t', 'real_tag', '\n'])
    i=1
    for prob_row in data_prob_output:
        file1.writelines([prob_row[0],"\t", prob_row[1], "\n"]) 
        i+=1
    file1.close()
    print("Output Results can be found at the current directory!")
    print("Output Name is: "+ 'Q2_'+ NAME + '_Model.txt')
    return None

def main():
	
	training_data = return_sentence_list_from_file('brown.train.tagged.txt')
	sentence_list = training_data#[0:100]
	tag_dict = return_Tagset_from_train_data(sentence_list)
	vocabulary_dict = return_Vocabulary_from_tagged_data(sentence_list)

	tag_unigram_counts = return_unigram_tag_counts(sentence_list, tag_dict)
	bigram_tag_counts = return_bigram_tag_counts(sentence_list, tag_dict)
	emission_counts = return_bigram_tag2word_counts(sentence_list, tag_dict, vocabulary_dict)

	probability_transition = return_probability_transition(tag_unigram_counts, bigram_tag_counts)

	probability_emission = return_probability_emission(tag_unigram_counts, emission_counts)



	Path_test_data='brown.test.tagged.txt'
	#head_file(Path_training_data, [0,2])
	sentence_list = return_sentence_list_from_file(Path_test_data)
	for_test = sentence_list
	
	
	predicted_tag_sequence = Deconding_viterbi_with_beam_one(for_test, tag_dict, vocabulary_dict, probability_transition, probability_emission)
	#predicted_tag_sequence = Deconding_viterbi(for_test, tag_dict, vocabulary_dict, probability_transition, probability_emission)
	
	
	total_size= len(predicted_tag_sequence)
	correct_count=0
	for x in predicted_tag_sequence:
		if(x[0]==x[1]):
			correct_count+=1

	
	print ("Number of Test Word: " + str(total_size))
	print ("Number of Correct Predicted Tag: " + str(correct_count))
	print ("")
	print ("Overall Accuracy is: " + str(correct_count/total_size))
	
	Name = 'HMM_viterbi_with_Beam_Search_Results'
	Save_Results(predicted_tag_sequence, Name)
	
	return 0
	
print ("Start")
main()
print ("End")
