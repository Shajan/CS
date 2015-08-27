"""
Sample to show different python language features
This is a multiline comment
"""

# Variables
int_val = 3
float_val = 1.5
bool_val = True

print int_val

# function definition. Python uses indendation to specify scope
def foo():
  return 12

print foo()

# exponential operator
a = 2 ** 5 
print a # 32

# print
print("%.2f, %d, %.0f" % (123.657, 25.7, 25.7)) # 123.66, 25, 26

# Strings
print "FOOBAR"[0] 	# F
print len("foobar") 	# 6
print "FOOBAR".lower() 	# foobar
print "foo".upper() 	# FOO
a = 2.4
print str(a)		# converts to string type
print "ab" + str(2)	# ab2 + requires string type

# get input from console
""" uncomment to run
name = raw_input("What is your name? ")
print "Hello " + name
"""

