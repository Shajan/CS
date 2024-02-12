# Compiler and flags
CC=gcc
CFLAGS=-Wall -g

# Target executable name
TARGET=main

# Source and object files
SRCS=$(wildcard *.c)
OBJS=$(SRCS:.c=.o)

# Default target
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

# Pattern rule for object files
%.o: %.c
	$(CC) $(CFLAGS) -c $<

# Include dependencies
DEPS=$(OBJS:.o=.d)
-include $(DEPS)

# Generate dependencies
%.d: %.c
	$(CC) $(CFLAGS) -MM -MT $@ -MT $(@:.d=.o) -MF $@ $<

# Clean up
clean:
	rm -f $(TARGET) $(OBJS) $(DEPS)

