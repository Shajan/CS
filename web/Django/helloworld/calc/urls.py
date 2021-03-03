from django.urls import path

from . import views

urlpatterns = [
  path('', views.home, name='home'), # Default page
  path('static', views.home, name='home'),
  path('dynamic', views.dynamic, name='dynamic'),
  path('dynamicUsingBase', views.dynamicUsingBase, name='dynamicUsingBase'),

  path('getDataForm', views.getDataForm, name='getDataForm'),
  path('getDataCompute', views.getDataCompute, name='getDataCompute'),

  path('postDataForm', views.postDataForm, name='postDataForm'),
  path('postDataCompute', views.postDataCompute, name='postDataCompute'),
]
