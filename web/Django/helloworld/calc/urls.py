from django.urls import path

from . import views

urlpatterns = [
  path('', views.home, name='home'), # Default page
  path('static', views.home, name='home'),
  path('dynamic', views.dynamic, name='dynamic')
]
