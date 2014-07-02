#This is also a test file
#test2.py
import nltk
sentence = "I like Tomato and Lettuce"
tokens = nltk.word_tokenize(sentence)
print tokens
tagged_list = nltk.pos_tag(tokens)
print tagged_list

np = []
for x in tagged_list:
	if x[1] == 'NP' or x[1] == 'NNP': #NP -- Noun Phrase
		np.append(x[0])	
print np

