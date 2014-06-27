from django.db import models

from django_mongokit.document import DjangoDocument
from django_mongokit import connection

from bson import ObjectId



class MyDocs(DjangoDocument):
	structure={
		'content':unicode,
		'required_for':unicode,
	} 
	use_dot_notation = True
connection.register([MyDocs])

class ReducedDocs(DjangoDocument):
	structure={
		'content':dict,
		'orignal_id':ObjectId,
		'required_for':unicode,
		'is_indexed':bool
	}
	use_dot_notation = True

class ToReduceDocs(DjangoDocument):
	structure={
		'doc_id':ObjectId,
		'required_for':unicode,
	}
	use_dot_notation = True

class IndexedWordList(DjangoDocument):
	structure={
		'word_start_id':float,
		'words':dict,
		'required_for':unicode,
	}
	use_dot_notation = True
	#word_start_id = 0 --- a ,1---b,2---c .... 25---z,26--misc.

# Create your models here.
