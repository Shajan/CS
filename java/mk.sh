CLASS=${1:-HelloWorld}
shift

javac $CLASS.java
java $CLASS $@
