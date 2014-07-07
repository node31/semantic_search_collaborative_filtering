from django.shortcuts import render
from django.http import HttpResponse
from cf.models import *
from django_mongokit import connection,get_database
#from stemming.porter2 import stem
import itertools
import string
import nltk

collName = get_database().cf

my_doc_requirement = u'storing_orignal_doc'
reduced_doc_requirement = u'storing_reduced_doc'
to_reduce_doc_requirement = u'storing_to_be_reduced_doc'
indexed_word_list_requirement = u'storing_indexed_words'

def index(request):
	#return HttpResponse("You are at index page")
	return render(request,'cf/index.html',{})
	
def insert(request):
	connection.register([MyDocs])
	connection.register([ToReduceDocs])
	y = collName.MyDocs()
	y.content = request.POST['f_content']
	y.required_for = my_doc_requirement
	y.save()
	
	z = collName.MyDocs.find_one({'content':y.content,'required_for':my_doc_requirement})
	if z:
		x = collName.ToReduceDocs()
		x.doc_id = z._id
		x.required_for = to_reduce_doc_requirement
		x.save()
		return render(request,'cf/thankYou.html',{})
	return render(request,'cf/error.html',{})

def edit(request):
	return render(request,'cf/edit.html',{})

def edit_object(request):
	connection.register([MyDocs])
	connection.register([ToReduceDocs])
	
	obj_id = ObjectId(request.POST["f_id"])
	x = collName.MyDocs.find_one({"_id":obj_id,'required_for':my_doc_requirement})
	
	if x:	
		x.content = request.POST["f_content"]
		x.save()	
	y = collName.ToReduceDocs.find_one({'doc_id':obj_id,'required_for':to_reduce_doc_requirement})
	if not y:
		z = collName.ToReduceDocs()
		z.doc_id = obj_id
		z.required_for = to_reduce_doc_requirement
		z.save()		
	return render(request,'cf/thankYou.html',{})
	
############################################################################################################################

################################### PRE PROCESSING FOR MAP REDUCE #################################################################	

def pre_process_for_map_reduce(text):
	
	grammar = r"""
	    NBAR:
		{<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
		
	    NP:
		{<NBAR>}
		{<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
	"""
	chunker = nltk.RegexpParser(grammar)	#This is the chunker for nltk.It chunks values accordingly	
	toks = nltk.word_tokenize(text)		#This shall tokenize the words
	postoks = nltk.tag.pos_tag(toks)	#This shall perform tagging of words with their respective parts of speech
	tree = chunker.parse(postoks)	#It makes a tree of the tags and the words which are associated with that particular tag
	
	terms = get_terms(tree) 
	return terms


def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.node=='NP'):
        yield subtree.leaves()

def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it.
       Lemmatization and stemming of the words both are necessary in order to make sure that the words are properly indexed	
    """
    
    lemmatizer = nltk.WordNetLemmatizer()
    stemmer = nltk.stem.porter.PorterStemmer()
    word = word.lower() 	#THE WORD IS CONVERTED INTO LOWER CASE IN THIS STEP
    word = stemmer.stem_word(word)
    word = lemmatizer.lemmatize(word)
    
    return word
    
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword.
       A word is acceptable if 
       		1. It is not a stopword
       		2. The length of the word is less than 40 characters
       		This is because, there is no point in storing a word more than 40chars long.This is because, user is not expected to 
       		type words which are 40 chars long   
    """
    max_word_length = 40
    accepted = bool(2 <= len(word) <= max_word_length and word.lower() not in stopwords)
    return accepted
    
def get_terms(tree):
    result = []	
    for leaf in leaves(tree):
    	for w,t in leaf:
    		if acceptable_word(w):
    			term = normalise(w)
    			result.append(term)        
    return result
############################################################################################################################	

######################################PRE PROCESSING FOR MAP REDUCE ########################################################	
def remove_punctuation(s):
	translate_table = dict((ord(c),None) for c in string.punctuation)
	return s.translate(translate_table)

def mapper(input_value):
	input_value = remove_punctuation(input_value)	
	input_value_l = pre_process_for_map_reduce(input_value)		#This performs pre_processing for map reduce
	#This pre_processing is very important in order to save space
	#This pre_processing function makes the map_reduce function slow
	l = []
	for i in input_value_l:
		l.append([i,1])

	return l
	
	
def reducer(intermediate_key,intermediate_value_list):
	return(intermediate_key,sum(intermediate_value_list))

def map_reduce(x,mapper,reducer):
	groups = {}
	for key,group in itertools.groupby(sorted(mapper(x)),lambda x:x[0]):
		groups[key] = list([y for x,y in group])
	reduced_list = [reducer(intermediate_key,groups[intermediate_key]) for intermediate_key in groups ]
	return reduced_list
	
def perform_map_reduce(request):
	connection.register([MyDocs])
	connection.register([ReducedDocs])
	connection.register([ToReduceDocs])
	
	dltr=list(collName.ToReduceDocs.find({'required_for':to_reduce_doc_requirement}))	#document_list_to_reduce
	
	for doc in dltr:
		doc_id = doc.doc_id
		orignal_doc = collName.MyDocs.find_one({"_id":doc_id,'required_for':my_doc_requirement})
		content_dict = dict(map_reduce(orignal_doc.content,mapper,reducer))
		
		dord = collName.ReducedDocs.find_one({"orignal_id":doc_id,'required_for':reduced_doc_requirement}) #doc of reduced docs
		if dord:
			dord.content=content_dict
			dord.is_indexed = False
			dord.save()
		else:
			new_doc = collName.ReducedDocs()
			new_doc.content = content_dict
			new_doc.orignal_id = doc_id
			new_doc.required_for = reduced_doc_requirement
			new_doc.is_indexed = False
			new_doc.save()
		doc.delete()	
	return render(request,'cf/thankYou.html',{})
############################################################################################################################

################################################## POST PROCESSING FOR MAP REDUCE ############################################
#The code till above was to perform map_reduce
#The code below this will try and perform semantic search
import scipy.sparse
import numpy
import sparsesvd
from math import sqrt


def td_doc():
	"""
	#{'word':{'ObjectId':number_of_occurances,'ObjectId':number_of_occurances}}
	This is the kind of dictionary which is required and will be created on the fly
	Since we have already stored the map reduced documents, this function will be pretty fast.
	The only thing which shall take time in our code is the MapReduce function	
	"""
	
	connection.register([IndexedWordList])
	connection.register([ReducedDocs])
	
	#This is the list of documents which contains the indexed words
	
	lod = collName.IndexedWordList.find({'required_for':indexed_word_list_requirement})	#list_of_documents_cursor
	
	"""
		What does indexing mean?
		In our scenario,indexing simply means to store the number if occurances of a particular word in each and every document.
		
	"""
	mrd = collName.ReducedDocs.find({'required_for':reduced_doc_requirement})	#map_reduced_documents
	mrdl = list(mrd)
	
		
	for pwdl in lod:	
		#particulat_word_list
		start_int = int(pwdl.word_start_id)
		start_char = str(unichr(96+start_int)) 	#This tells what is the starting character of the word
		wod = pwdl.words	#word_object_dictionary		
		
		for pmrd in mrdl:
			#particular_map_reduced_document
			#print pmrd
			if not pmrd.is_indexed:
				wd = pmrd.content
				
				for i in wd:
					if i.startswith(start_char):
						
						if i not in wod:
							wod[i] = {}
						wod[i][str(pmrd.orignal_id)]=wd[i]
		pwdl.words = wod
		#print "WORD OBJECT DICTIONARY AFTER  ---->",wod
		pwdl.save()
	
	for pmrd in mrdl:
		pmrd.is_indexed = True
		pmrd.save()
		
def generate_big_dict():
	#This function will generate a big dictionary i.e. it will simply go and combine all the dictionaries together
	connection.register([IndexedWordList])
	
	lod = collName.IndexedWordList.find({'required_for':indexed_word_list_requirement})
	lodl = list(lod)
	
	prefs = {} #prefs ==> Preferences
	
	for x in lodl:
		if x.words:
			prefs.update(x.words)		
	#print prefs
	return prefs	
	
####
#There are two kinds of similarity functions which we have defined and on whose basis recommendations are given
#If logic for semantic search needs to be changed then the only thing which is to be changed is this similarity function
####
def sim_distance(prefs,d1,d2):
	#This fucntion simply finds the distance between two words. It works very well
	si = {}
	for item in prefs[d1]:	#This item is a dictionary containing book id and rating of that book for a user
		#print prefs[person1]
		if item in prefs[d2]:
			si[item] = 1
			
	if len(si) == 0:
		return 0
		
	#We know add the squares of all the differences
	sum_of_squares = 0
	
	for item in prefs[d1]:
		#print prefs[person1]
		if item in prefs[d2]:
			#print prefs[person2]
			#print "PERSON1 ITEM",item,prefs[d1][item]
			#print "PERSON2 ITEM",item,prefs[d2][item]	
			#print "SUBTRACT",prefs[d1][item] - prefs[d2][item]		
			sum_of_squares += pow(prefs[d1][item] - prefs[d2][item],2)
			#print sum_of_squares
	
	#print "SUM OF SQUARES :):)",sum_of_squares,(1.0/(1+sum_of_squares))
	return (1.0/(1+sum_of_squares))	

	


def sim_pearson(prefs,d1,d2):
	#Theoretically --- The results of pearson similarity should be better, but practically the results are much worse
	#Get the list of mutually rated items
	si = {}
	try:	
		for term in prefs[d1]:			
			if term in prefs[d2]: 
				si[term] = 1
	except KeyError:	
		return 0
	
	#sum calculations
	n = len(si)
	
	#sum of all preferences
	sum1 = sum([prefs[d1][it] for it in si])
	
	sum2 = sum([prefs[d2][it] for it in si])
	

	#Sum of the squares
	sum1Sq = sum([pow(prefs[d1][it],2) for it in si])
	sum2Sq = sum([pow(prefs[d2][it],2) for it in si])

	#Sum of the products
	pSum = sum([prefs[d1][it] * prefs[d2][it] for it in si])
	

	
	num = pSum - (sum1 * sum2/n)
	den = sqrt((sum1Sq - pow(sum1,2)/n) * (sum2Sq - pow(sum2,2)/n))
	
	if den == 0:
		return 0

	r = num/den
		
	return r

def topMatches(prefs,document,n=5,similarity=sim_distance):
	#This function returns the words which are closest to the word which are given to this function
	scores = [(similarity(prefs,document,other),other) for other in prefs if other != document]
	scores.sort()
	scores.reverse()
	return scores[0:n]
	
def recommend(prefs,term,similarity = sim_distance):
	#This function returns the documents which will be closer to the given document
	each_item_total = {}
	similarity_total_for_each_item = {}
	
	for other in prefs:
		if other == term:
			continue
		else:
			sim = similarity(prefs,term,other)
			#print "similarity :):P>>>>>",sim,term,other
			
		if sim==0:
			continue
		
		for single_ObjectId in prefs[other]:
			if single_ObjectId in prefs[term]:				
				if single_ObjectId not in each_item_total:
					each_item_total[single_ObjectId] = 0				
				each_item_total[single_ObjectId] += sim * prefs[other][single_ObjectId]
			
				if single_ObjectId not in similarity_total_for_each_item:
					similarity_total_for_each_item[single_ObjectId] = 0
				similarity_total_for_each_item[single_ObjectId] += sim
		
	
	rankings = []
	
	for single_ObjectId,total_value in each_item_total.items():
		rankings.append((total_value/similarity_total_for_each_item[single_ObjectId],single_ObjectId))
	
	rankings.sort()
	rankings.reverse()
	
	return rankings	
################## FUNCTIONS FOR CALLING/TESTING SEMANTIC SEARCH ########################################

def generate_term_document_matrix(request):
	td_doc()
	return render(request,'cf/thankYou.html',{})	
	
def search_page(request):
	return render(request,'cf/search_home.html',{})

def get_nearby_words(request):
	prefs = generate_big_dict()
	
	search_text = request.POST['f_word']
	search_text_l = search_text.split()
	#print search_text_l
	word_set = set()
	ranking_list = []
	
	stemmer = nltk.stem.porter.PorterStemmer()
		
	for i in search_text_l:
		print i
		score = topMatches(prefs,stemmer.stem_word(i.lower()),n=30,similarity=sim_distance)
		#print "SCORE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n",score	
		for _,word in score:
			word_set.add(word)
		#print word_set
		
		
		#print "NOW I AM PRINTING RECOMMENDATIONS ____________________________________________________________________________"	
		rankings = recommend(prefs,stemmer.stem_word(i.lower()),similarity = sim_distance)
		ranking_list.extend(rankings[0:5])
	
	
	
	#print "THE WORD SET IS PRINTED AS FOLLOWS -------------->>>>>>>>>>>>>\n",word_set	
	
	#print "THE RANKING LIST IS PRINTED AS FOLLOWS ------------->>>>>>>>>>>>>>\n",ranking_list
	
	final_ranking_list = sort_n_avg(ranking_list)
	final_ranking_list.sort()
	final_ranking_list.reverse()
	
	print "THE FINAL RANKING LIST IS PRINTED AS FOLLOWS ------------->>>>>>>>>>>>>>\n",final_ranking_list
	return render(request,'cf/show_search_results.html',{})
	
def sort_n_avg(l):
	"""
		Helper Function for: get_nearby_words()
		Parameters: List containing documents and their ratings
		Return Value:List in which the ratings of the documents have been averaged out
		
		INPUT:l = [(2,'alpha'),(3,'beta'),(1,'alpha'),(4,'alpha'),(5,'gamma'),(1,'alpha'),(2,'beta'),(3,'alpha')]
		OUTPUT:[(2.2, 'alpha'), (2.5, 'beta'), (5.0, 'gamma')]
		
	"""
	visited_list = []
	final_ranking_list = []
	
	for (value,obj_id) in l:
		if obj_id not in visited_list:
			visited_list.append(obj_id)
		
			i = 0
			req_sum = 0
		
			for (val,obj_id_added) in l:
				if obj_id_added == obj_id:
					i = i+1
					req_sum += val			
			if i!=0:
				final_ranking_list.append((float(req_sum)/i,obj_id))
			
	return final_ranking_list
#################################################################################################################	
