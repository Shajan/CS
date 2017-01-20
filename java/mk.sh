CLASS=${1:-Test}
shift

javac $CLASS.java
java $CLASS $@
