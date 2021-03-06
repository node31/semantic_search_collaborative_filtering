from django.shortcuts import render
from django.http import HttpResponse


from cf.models import *

from django_mongokit import connection,get_database
from stemming.porter2 import stem

import itertools
import string

collName = get_database().cf

my_doc_requirement = u'storing_orignal_doc'
reduced_doc_requirement = u'storing_reduced_doc'
to_reduce_doc_requirement = u'storing_to_be_reduced_doc'
indexed_word_list_requirement = u'storing_indexed_words'
id_gen_number = 10000

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
	
	#z = collName.MyDocs.find_one({'content':y.content})
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
	
def removeArticles(text):
	words = text.split()
	articles=['a', 'an', 'and', 'the', 'i', 'is', 'this', 'that', 'there', 'here', 'am', 'on', 'at', 'of','who','why','when','where','what','was','with','which','these','to','as']
	for w in articles:
		if w in words:
			words.remove(w)	
	return words
	
def remove_extra(s):
	#Helper Function for:process_search
	#Pre-condition:A string from which you want to remove unneccesary words
	#Post-Condition:A list of words from without the extra words
	
	#This function works just fine
	s_l = s.split()
	remove_list = ['a', 'an', 'and', 'the', 'i', 'is', 'this', 'that', 'there', 'here', 'am', 'on', 'at', 'of','who','why','when','where','what','was','with','which','these','to','as','by','be','into','for','it','in','he','she','can','it','his','her','has']
	
	for i in remove_list:
		for j in s_l:			
			if i == j:
				s_l.remove(j)
	
	return s_l

	
def remove_punctuation(s):
	translate_table = dict((ord(c),None) for c in string.punctuation)
	return s.translate(translate_table)
	
def mapper(input_value):	
	input_value = remove_punctuation(input_value)		
	input_value = input_value.lower()
	input_value_l = remove_extra(input_value)
	l = []
	for i in input_value_l:
		l.append([stem(i),1])

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
	
def generate_term_document_matrix(request):
	td_doc()
	return render(request,'cf/thankYou.html',{})	
	
def search_page(request):
	return render(request,'cf/search_home.html',{})

def get_nearby_words(request):
	prefs = generate_big_dict()
	
	search_text = request.POST['f_word']
	search_text_l = search_text.split()
	print search_text_l
	word_set = set()
	ranking_set = set()
	
	for i in search_text_l:
		print i
		score = topMatches(prefs,i.lower(),n=100,similarity=sim_pearson)	
		for _,word in score:
			word_set.add(word)
			
		rankings = recommend(prefs,i.lower(),similarity = sim_pearson)
		seq = rankings[0:5]
		for j in seq:
			ranking_set.add(j)
	
	#score = topMatches(prefs,request.POST['f_word'],n=100,similarity=sim_pearson)
	#print score
	#word_list = []
	#for _,word in score:
	#	word_list.append(word)
	#print word_list
	print word_set	
	print ranking_set
	return render(request,'cf/show_search_results.html',{})
	
###################################################################################################################################
#The code till above was to perform map_reduce
#The code below this will try and perform semantic search
import scipy.sparse
import numpy
import sparsesvd
from math import sqrt

#{'word':{'ObjectId':number_of_occurances,'ObjectId':number_of_occurances}}
def td_doc():
	connection.register([IndexedWordList])
	connection.register([ReducedDocs])
	lod = collName.IndexedWordList.find({'required_for':indexed_word_list_requirement})	#list_of_documents_cursor
	mrd = collName.ReducedDocs.find({'required_for':reduced_doc_requirement})	#map_reduced_documents
	mrdl = list(mrd)
	#print "LENGTH OF MAP REDUCED DOCUMENT LIST >>>",len(mrdl)
		
	for pwdl in lod:	
		#particulat_word_list
		start_int = int(pwdl.word_start_id)
		start_char = str(unichr(96+start_int))
		wod = pwdl.words	#word_object_dictionary
		#print "START CHAR---->",start_char
		#print "WORD OBJECT DICTIONARY BEFORE  ---->",wod	
		
		for pmrd in mrdl:
			#particular_map_reduced_document
			#print pmrd
			if not pmrd.is_indexed:
				wd = pmrd.content
				#print "WORD CONTENT OF ",pmrd._id,"\n",wd
				for i in wd:
					if i.startswith(start_char):
						#print i
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
	prefs = {}
	for x in lodl:
		if x.words:
			prefs.update(x.words)		
	#print prefs
	return prefs	
	


def sim_pearson(prefs,d1,d2):
	#Get the list of mutually rated items
	si = {}
	for term in prefs[d1]:
		if term in prefs[d2]: 
			si[term] = 1

	#if they are no rating in common, return 0
	if len(si) == 0:
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

	#Calculate r (Pearson score)
	num = pSum - (sum1 * sum2/n)
	den = sqrt((sum1Sq - pow(sum1,2)/n) * (sum2Sq - pow(sum2,2)/n))
	if den == 0:
		return 0

	r = num/den

	return r

def topMatches(prefs,document,n=5,similarity=sim_pearson):
	scores = [(similarity(prefs,document,other),other)
				for other in prefs if other != document]
	scores.sort()
	scores.reverse()
	return scores[0:n]
	
def recommend(prefs,term,similarity = sim_pearson):
	each_item_total = {}
	similarity_total_for_each_item = {}
	
	for other in prefs:
		if other == term:
			continue
		else:
			sim = similarity(prefs,term,other)
			
		if sim==0:
			continue
		
		for single_ObjectId in prefs[other]:
			if single_ObjectId not in prefs[term] or prefs[term][single_ObjectId] == 0:
				
				each_item_total.setdefault(single_ObjectId,0)
				each_item_total[single_ObjectId] += sim * prefs[other][single_ObjectId]
				
				similarity_total_for_each_item.setdefault(single_ObjectId,0)
				similarity_total_for_each_item[single_ObjectId] += sim
		
	
	rankings = []
	
	for single_ObjectId,total_value in each_item_total.items():
		rankings.append((total_value/similarity_total_for_each_item[single_ObjectId],single_ObjectId))
	
	rankings.sort()
	rankings.reverse()
	return rankings	
	
       
def test_page(request):
	prefs = generate_big_dict()	
	return render(request,"cf/test.html",{})
	

# Create your views here.
