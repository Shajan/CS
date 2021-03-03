from django.shortcuts import render

from django.http import HttpResponse

# Create your views here.

def home(request):
  return HttpResponse('<h1>Hello World (calc home)</h1>')

# Use template/dynamic.html
def dynamic(request):
  return render(request, 'dynamic.html', {'name' : 'Shajan'})

# Use template/dynamicUsingBase.html
def dynamicUsingBase(request):
  return render(request, 'dynamicUsingBase.html', {'name' : 'Shajan'})

def getDataForm(request):
  return render(request, 'getDataForm.html')

# Getting values from http 'GET' request
def getDataCompute(request):
  val1 = request.GET['num1']
  val2 = request.GET['num2']

  try:
    val1 = int(val1)
    val2 = int(val2)
    result = val1 + val2
    result = str(result)
  except:
    result = "Use numbers"

  return render(request, 'result.html', {'result' : result})


def postDataForm(request):
  return render(request, 'postDataForm.html')

# Posting values from http
def postDataCompute(request):
  val1 = request.POST['num1']
  val2 = request.POST['num2']

  try:
    val1 = int(val1)
    val2 = int(val2)
    result = val1 + val2
    result = str(result)
  except:
    result = "Use numbers"

  return render(request, 'result.html', {'result' : result})
