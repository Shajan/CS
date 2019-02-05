set -e
CLASS=${1:-ConcurrentHashPerf}

JARS=~/.m2/repository/com/google/guava/guava/19.0/guava-19.0.jar
javac -cp $JARS $CLASS.java
java -cp $JARS:. $CLASS

