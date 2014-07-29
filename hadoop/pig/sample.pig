data = LOAD 'data.csv' USING PigStorage(',') AS (name:chararray, age:int, status:chararray, state:chararray);

-- Find the number of records from each state
stats = FOREACH (GROUP data BY state) GENERATE $0, COUNT($1);
DUMP stats;

-- Find the records with age less than 30
young = FILTER data BY age < 30;
DUMP young;

-- Join to Expand state names
state_name = LOAD 'state.csv' USING PigStorage(',') AS (state:chararray, stateName:chararray);
expanded = JOIN data by state, state_name by state; -- expanded will have all cols or A and F
reduced = FOREACH expanded GENERATE name, age, status, stateName;
DUMP reduced;

