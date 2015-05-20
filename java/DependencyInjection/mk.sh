#JAR_ROOT=/System/Library/Java/JavaVirtualMachines/1.6.0.jdk/Contents/Home/lib
JAR_ROOT=/Users/sdasan/tmp/jars/
JARS="$JAR_ROOT/javax.inject-1.jar"
javac -cp $JARS DPI.java
java -cp $JARS DPI
