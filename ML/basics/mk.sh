CLASS=${1:-Simple}
shift

javac $CLASS.java
java $CLASS $@
