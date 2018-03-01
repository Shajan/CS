MAX_PARALLEL=10

seq 1 9 | xargs -P --replace ${MAX_PARALLEL} echo {}
