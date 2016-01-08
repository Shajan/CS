CLASS=${1:-Graph}
shift

javac $CLASS.java
java $CLASS $@
