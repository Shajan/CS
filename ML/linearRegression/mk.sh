CLASS=${1:-GradientDescent}
shift

javac $CLASS.java
java $CLASS $@
