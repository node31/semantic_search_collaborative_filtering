from django.conf.urls import patterns, include, url

from cf import views

urlpatterns = patterns('',
		url(r'^$',views.index,name = 'index'),
		url(r'^insert/$',views.insert,name = 'insert'),
		url(r'^edit/$',views.edit,name = 'edit'),
		url(r'^edit_object/$',views.edit_object,name = 'edit_object'),
		url(r'^perform_map_reduce/$',views.perform_map_reduce,name = 'perform_map_reduce'),
		url(r'^generate_term_document_matrix/$',views.generate_term_document_matrix,name = 'generate_term_document_matrix'),
		url(r'^search_page/$',views.search_page,name = 'search_page'),
		url(r'^get_nearby_words/$',views.get_nearby_words,name = 'get_nearby_words'),
	)
