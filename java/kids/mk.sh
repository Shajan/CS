CLASS=${1:-MinMax}
shift

javac $CLASS.java
java $CLASS $@
