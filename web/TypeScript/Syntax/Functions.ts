// Regular function
function multiply(a: number, b: number): number {
  return a * b;
}

// Inline annotation
const add = (a: number, b: number): number => {
  return a + b;
};

// Function signature (aka call signature) in a type
type Adder = (a: number, b: number) => number;

const plus: Adder = (x, y) => x + y;

// Optional parameter
function greet(name?: string) {
  if (name) console.log("Hello " + name);
  else console.log("Hello Stranger");
}

greet();
greet("Alice");

// Readonly paramter
function printId(readonly id: string) {
  // id = "new"; // Compile time Error!
  console.log(id);
}

// Function overloads
function format(input: number): string;
function format(input: string): string;

// Actual implementation must handle all variants
function format(input: number | string): string {
  return `Value: ${input}`;
}

format(10);     // OK
format("hi");   // OK

// Predicates
function filterStrings(
  arr: string[],
  predicate: (s: string) => boolean
): string[] {
  return arr.filter(predicate);
}

const result = filterStrings(["a", "bb", "ccc"], s => s.length > 1);

// Generics
function wrap<T>(value: T): T[] {
  return [value];
}

const n = wrap(5);       // T = number → number[]
const s = wrap("hi");    // T = string → string[]

function getLength<T extends { length: number }>(item: T): number {
  return item.length;
}

getLength("hello");    // OK (string has length)
getLength([1, 2, 3]);  // OK (array has length)

function pair<A, B>(a: A, b: B): [A, B] {
  return [a, b];
}

const p = pair("age", 42); // ["age", 42]

