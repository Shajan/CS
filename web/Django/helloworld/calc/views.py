from django.shortcuts import render

from django.http import HttpResponse

# Create your views here.

def home(request):
  return HttpResponse('<h1>Hello World (calc home)</h1>')

# Use template/dynamic.html
def dynamic(request):
  return render(request, 'dynamic.html', {'name' : 'Shajan'})
