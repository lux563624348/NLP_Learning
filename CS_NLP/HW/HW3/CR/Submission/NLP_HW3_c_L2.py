#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  NLP_HW3.py
#  
#  Copyright 2019 Who <Who@DESKTOP-5IUT3TE>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


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

def filter_word(word):
    return word.lower()

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


## not needed in HMM
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
    #count_matrix += len(vocabulary) ######## Add-one smoothing
    ## Set value for sentence start <s>
    count_matrix[vocabulary.index('^')] += len(sentence_list)
    count_matrix[vocabulary.index('$')] += len(sentence_list)
    for tem_sentence in sentence_list:
        word_list = re.split("\s+", tem_sentence.rstrip('\n'))
        for word in word_list:
            filtered_word = filter_word(word)
            if(filtered_word in vocabulary):
                word_index = vocabulary.index(filtered_word)
                count_matrix[word_index]+=1
    return count_matrix

def return_bigram_word_counts(sentence_list, vocabulary):
    count_matrix=np.zeros((len(vocabulary),len(vocabulary)))
    #count_matrix+=1  ######## Add-one smoothing
    num_word=0
    for tem_sentence in sentence_list:
        word_list = re.split("\s+", tem_sentence.rstrip('\n'))
        word_list = list(filter(None, word_list))
        
        for i in range(0,len(word_list)-1):
            num_word+=1
            if (i==0):
                first_word = filter_word(word_list[0])
                if(first_word in vocabulary):
                    count_matrix[vocabulary.index('^'), vocabulary.index(first_word)] += 1
            tem_bigram_count_pairs = word_list[i:i+2]
            first_word=filter_word(tem_bigram_count_pairs[0])
            second_word=filter_word(tem_bigram_count_pairs[1])
            
            if((first_word in vocabulary) & (second_word in vocabulary)):
                first_digit=vocabulary.index(first_word)
                second_digit=vocabulary.index(second_word)
                count_matrix[first_digit,second_digit]+=1
    print ("Number of Words: "+ str(num_word))
    return count_matrix

def return_word_from_sentence(sentence):
    list_words=list()
    for word in (re.split("\s+", sentence.rstrip('\n'))):
                    if (word !=''):
                        list_words.append(filter_alphanumeric(word))
    ## if different languague, above line has to be changed
    list_words = list(filter(None, list_words))
    return list_words

def return_prob_of_test_sentence(test_words, P_likelihood_class):
    bag_words = return_word_from_sentence(test_words)
    Log_P_of_sentence=0
    P_likelihood=P_likelihood_class
        
    for i in range(len(bag_words)):
        if bag_words[i] in P_likelihood.keys():
            Log_P_of_sentence+=np.log(P_likelihood[bag_words[i]])

    return Log_P_of_sentence

def test_from_naive_bayes_classifier(test_sentence, P_prior_pos, P_likelihood_pos, P_prior_neg, P_likelihood_neg):
    p_pos = return_prob_of_test_sentence(test_sentence, P_likelihood_pos) + np.log(P_prior_pos) +np.log(P_prior_pos)
    #print (p_pos)
    p_neg = return_prob_of_test_sentence(test_sentence, P_likelihood_neg) + np.log(P_prior_neg) +np.log(P_prior_neg)
    #print (p_neg)
    
    if (p_pos<p_neg):
        classification_result='+'
    else:
        classification_result='-'
    return classification_result

Stop_Words=['the','a', 'and', 'i', 'it', 'is', 'to', 'a', 'of', 'this', 'with', 'for',
                'you', 'that', 'in', 'have', 'my', 'on', 'as', 'but', 'use', 'are', 'phone',
                'has', 'all', 'was', 'so', 'one', 'be', 'at', 'than', 'an']



def main(args):
	
	PATH_Folder='./'

	Set_FileName=['neg.tok', 'pos.tok']

	Total_accuracy=0
	Total_F1=0
	for i in range(10):
		sentence_list_neg   = return_sentence_list_from_file(PATH_Folder+Set_FileName[0])
		X_train_neg, X_test_neg = train_test_split(sentence_list_neg, test_size=0.1, random_state=None)
		vocabulary_list_neg  = return_vocabulary_from_sentence_list(X_train_neg)
		vocabulary_list_neg = list(set(vocabulary_list_neg) - set(Stop_Words))
		
		
		uni_counts_neg = return_unigram_counts(X_train_neg, vocabulary_list_neg)
		bi_counts_neg = return_bigram_word_counts(X_train_neg, vocabulary_list_neg)
		y_neg=list(np.zeros(len(bi_counts_neg))+1)
		
		
		sentence_list_pos  = return_sentence_list_from_file(PATH_Folder+Set_FileName[1])
		X_train, X_test = train_test_split(sentence_list_pos, test_size=0.1, random_state=None)
		vocabulary_list_pos  = return_vocabulary_from_sentence_list(X_train)
		vocabulary_list_pos = list(set(vocabulary_list_pos) - set(Stop_Words))
		uni_counts_pos = return_unigram_counts(X_train, vocabulary_list_pos)
		bi_counts_pos = return_bigram_word_counts(X_train, vocabulary_list_pos)
		
		
		vocabulary_list = vocabulary_list_neg + vocabulary_list_pos
		uni_counts = return_unigram_counts(X_train_neg+X_train, vocabulary_list)
		bi_counts_pos = return_bigram_word_counts(X_train_neg+X_train, vocabulary_list)
		
		
		y_pos=list(np.zeros(len(bi_counts_pos)))
		
		bi_counts= np.zeros(shape=(len(y_pos+y_neg),len(y_pos+y_neg)))
		clf_pos = LogisticRegression(random_state=0).fit(bi_counts, y_pos+y_neg)

		N_doc = uni_counts_neg.sum()+uni_counts_pos.sum()
		P_prior_neg = uni_counts_neg.sum()/N_doc
		P_prior_pos = uni_counts_neg.sum()/N_doc

		size_vocabulary  = len(set(vocabulary_list_neg + vocabulary_list_pos)) -2
		P_likelihood_pos = (uni_counts_pos+1)/(N_doc+size_vocabulary)
		P_likelihood_neg = (uni_counts_neg+1)/(N_doc+size_vocabulary)



		
		
		

		P_likelihood_pos = dict(zip(vocabulary_list_pos, P_likelihood_pos))
		P_likelihood_neg = dict(zip(vocabulary_list_neg, P_likelihood_neg))


		pos_count=0
		for test in sentence_list_pos[:]:
			results = test_from_naive_bayes_classifier(test, P_prior_pos, P_likelihood_pos, P_prior_neg, P_likelihood_neg)
			if (results=='+'):
				pos_count+=1

		neg_count=0
		for test in sentence_list_neg[:]:
			results = test_from_naive_bayes_classifier(test, P_prior_pos, P_likelihood_pos, P_prior_neg, P_likelihood_neg)
			if (results=='-'):
				neg_count+=1

		Total_accuracy+= ((pos_count + neg_count)/( len(sentence_list_neg) + len(sentence_list_pos) ))

		precision = (pos_count)/(len(sentence_list_pos))
		recall = (neg_count)/(len(sentence_list_neg))

		Total_F1+=(2*precision*recall/(recall+precision))
		
	print ("Accuracy: ")
	print (Total_accuracy/10.0)
	print ("F1: ")
	print (Total_F1/10.0)
		
	
	return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
