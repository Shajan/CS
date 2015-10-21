CLASS=${1:-HelloWorld}
javac $CLASS.java
java -ea $CLASS
#-ea for enable assertions
