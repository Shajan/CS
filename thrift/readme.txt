To compile, execute 'mk.sh'
	Avoiding a makefile on purpose to keep things simple and transparent.
	'./mk.sh clean' to celean up
	Thrift generated files go to ./gen-cpp/*, binary is './sample'
To create serialized data run: ./sample write
	Data file is created in /tmp/data.bin
To read serialized data run: ./sample
	Reads data from /tmp/data.bin

Thrift:
	To play with thrift, change sample.thrift then run ./mk.sh to regenerate ./sample

Serialization:
	The serilized data can be written/read using any thirft supported language, as long as the same thrift definition (sample.thrift) is used to generate the serialization/deserialization code.

Versioning:
	To understand versioning, change sample.thrift with optional fields. Read and write using code generated from different versions of sample.thrift (ones with optional fields defined and ones without). Try both cases ('old client reading new data' as well as 'new client reading old data')
