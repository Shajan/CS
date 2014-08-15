Setup:
	Install Thrift
		download, build and install http://thrift.apache.org/
	Install slf4j required for java
		http://www.slf4j.org/download.html slf4j-api-1.5.8.jar,slf4j-log4j12-1.5.8.jar,log4j-1.2.14.jar
	Install scrooge required for scala
		'git clone https://github.com/twitter/scrooge && cd scroonge && ./sbt +publish-local'

To compile: 'mk.sh'
	Avoiding a makefile on purpose to keep things simple and transparent.
	'./mk.sh clean' to clean up
	Thrift generated files go to ./gen-cpp/*, binary is './sample'

Cross Language: First compile using ./mk.sh, then..
	./mk.sh c2j [Writes using C++, reads using java]
	./mk.sh j2c [Writes using java, reads using c++]

To create serialized data using C++ run: ./sample write
	Data file is created in ./data.bin
To read serialized data using C++ run: ./sample
	Reads data from ./data.bin

Thrift:
	To play with thrift, change sample.thrift then run ./mk.sh to regenerate ./sample

Serialization:
	The serilized data can be written/read using any thirft supported language, as long as the same thrift definition (sample.thrift) is used to generate the serialization/deserialization code.

Versioning:
	To understand versioning, change sample.thrift with optional fields. Read and write using code generated from different versions of sample.thrift (ones with optional fields defined and ones without). Try both cases ('old client reading new data' as well as 'new client reading old data')

Scala:
