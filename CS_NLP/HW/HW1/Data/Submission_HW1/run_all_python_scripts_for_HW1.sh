#!/bin/bash
Python_Scripts_Set=(
Letter_Bigram_Model.py
Word_Bigram_Model.py
Word_Bigram_Model_Good_Turing_Smoothing.py
Word_Trigram_Model.py
)

main(){
echo "Run all Python scripts parallel."
for (( i = 0; i <= $(expr ${#Python_Scripts_Set[*]} - 1); i++ ))  ### Loop Operation [Ref.1]
do
	python ${Python_Scripts_Set[i]} & 
done
}
main "$@"
