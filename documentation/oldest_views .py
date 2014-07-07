from django.shortcuts import render
from django.http import HttpResponse


from cf.models import *

from django_mongokit import connection,get_database

import itertools

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
	
def mapper(input_value):
	#This mapper function will do one more thing than simply mapping
	#It obtains a string which is split into a list of words
	#After splitting into a list of words it will check whether it has unique-id assigned to it or not.
	#If it has unique-id which has been assigned to it then that id is used otherwise a unique_id is assigned to ir
	
	connection.register([IndexedWordList])
	lod = collName.IndexedWordList.find({'required_for':indexed_word_list_requirement})
	lodl = sorted(list(lod))
	#print lodl

	l=[]
	for i in input_value.split():
		start_char_to_int = ord(i[0]) - 97 #This converts the character to its corresponding ascii value
		if start_char_to_int < 0 or start_char_to_int>26:
			start_char_to_int = 26
		
		#print start_char_to_int
		
		#print "LENGTH OF LIST:::",len(lodl)
		
		pwo = lodl[start_char_to_int] 	#This is the particular word object
		wd = pwo.words		#pw is a dictionary of words
		
		if i not in wd.keys():
			wd[i] = len(wd)
			x = collName.IndexedWordList.find_one({'required_for':indexed_word_list_requirement,'word_start_id':float(start_char_to_int)})
			x.words = wd
			x.save()
		
		#print start_char_to_int
		word_unique_id = start_char_to_int * id_gen_number + wd[i]
		#print word_unique_id," for the word ",i
		l.append([str(word_unique_id),1])
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
			dord.save()
		else:
			new_doc = collName.ReducedDocs()
			new_doc.content = content_dict
			new_doc.orignal_id = doc_id
			new_doc.required_for = reduced_doc_requirement
			new_doc.save()
		doc.delete()
	
	return render(request,'cf/thankYou.html',{})
	
###################################################################################################################################
#The code till above was to perform map_reduce
#The code below this will try and perform semantic search
import scipy.sparse
import numpy
import sparsesvd

def create_td_matrix():
	#This function is responsible for creating a term-document matrix.
	#The way the things will be stored is [(),(),(),......] One Tuple for each document present in the database
	#In each tuple () ---> doc_id,{"word":word_count,"word":word_count} 
	
	connection.register([ReducedDocs])
	
	rdl = list(collName.ReducedDocs.find({'required_for':reduced_doc_requirement})) #ReducedDocList
	tdl = [] #Term-Document List
#	s = set()
	for td in rdl:
		tdl.append((td.orignal_id,td.content))
#		for x in td.content.keys():
#			s.add(x)
#	s_l = list(s)
#	s_l_z = zip(s_l,range(len(s_l)))
	#print "TD-LIST:",tdl	
	#print "SET S:",s
#	print "SET_LIST_ZIPPED:::",s_l_z
#	return tdl,s,s_l_z
	return tdl
	
def find_num_distinct_words():
	
	connection.register([IndexedWordList])
	lod = collName.IndexedWordList.find({'required_for':indexed_word_list_requirement})
	
	lodl = list(lod)
	#print "LENGTH OF LIST FIND_NUM_DISTINCT_WORDS>>>>",len(lodl)
	num_distinct_words = 0
	for i in lodl:
		num_distinct_words+=len(i.words)
	
	#print num_distinct_words
	return num_distinct_words
	

def getCscMatrix(tdl):
	num_nnz,data,indices,indptr = 0,[],[],[0]
	for td in tdl:
	    newIndices = [int(i) for i in td[1].keys()]
            newValues = [v for v in td[1].values()]
            indices.extend(newIndices)
            data.extend(newValues)
            num_nnz += len(newValues)
            indptr.append(num_nnz)
            
        data = numpy.asarray(data)
        indices = numpy.asarray(indices)
        #print "NUMBER OF DISTICT WORDS::::::::::::::::::::::::",find_num_distinct_words()
        cscMatrix = scipy.sparse.csc_matrix((data, indices, indptr),shape=(find_num_distinct_words(),len(tdl) ))
        print "CSC MATRIX::::",cscMatrix
        print "DENSE MATRIX::::",cscMatrix.toDense()
        
def test_page(request):
	#This function is a test function for calling the other functions which we have written
	#tdl,s,s_l_z = create_td_matrix()
	tdl = create_td_matrix()
	print tdl
	getCscMatrix(tdl)
	return render(request,"cf/test.html",{})
	

# Create your views here.
