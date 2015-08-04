JARS=~/.m2/repository/com/google/guava/guava/17.0/guava-17.0.jar
javac -cp $JARS CustomRateLimiter.java
java -cp $JARS:. CustomRateLimiter
