/************************************************************
 * 1. Structural Typing
 ************************************************************/

// In TypeScript, if two types have the same shape, they are compatible.
interface Point { x: number; y: number; }

class Pixel {
  constructor(public x: number, public y: number) {}
}

let p: Point = new Pixel(10, 20);   // OK because structure matches


/************************************************************
 * 2. Basic Types
 ************************************************************/

let count: number = 42;
let myname: string = "Shajan";
let isReady: boolean = true;

// any: opt out of type checking (not safe)
let anything: any = 123;
// anything.toUpperCase() // Runtime error!
anything = "now a string"; // allowed

// unknown: safer alternative to `any`
let maybe: unknown = "hello";
// console.log(maybe.toUpperCase()); // Compile time Error (must narrow first)
if (typeof maybe === "string") {
  console.log(maybe.toUpperCase()); // OK after narrowing
}

// never: for impossible code paths
function fail(msg: string): never {
  throw new Error(msg);
}

/************************************************************
 * 3. Type Inference
 ************************************************************/

// TS infers number
let inferred = 10;
// inferred = "oops"; // Compile time  error

// Function return type inferred
function add(a: number, b: number) {
  return a + b;
}


/************************************************************
 * 4. Union Types
 ************************************************************/

// A value can be one of several types
let id: number | string = 10;
id = "user-123"; // OK

// Using unions in functions
function formatId(id: number | string): string {
  if (typeof id === "number") {
    return `#${id}`; // Template substitution, returns #123 (when id is 123)
  }
  return id.toUpperCase();
}


/************************************************************
 * 5. Intersection Types
 ************************************************************/

// Combine multiple types into one
type HasName = { name: string };
type HasId = { id: number };

type User = HasName & HasId;

const user: User = { name: "Alice", id: 42 };


/************************************************************
 * 6. Type Narrowing
 ************************************************************/

function printValue(value: number | string | null) {
  if (value === null) {
    console.log("Nothing here!");
  } else if (typeof value === "string") {
    console.log("String length:", value.length);
  } else {
    console.log("Number squared:", value * value);
  }
}


/************************************************************
 * 7. Literal Types
 ************************************************************/

// Useful for finite options (like enums, but simpler)
let direction: "north" | "south" | "east" | "west";
direction = "north";
// direction = "up"; // Compile time error


/************************************************************
 * 8. Type Aliases
 ************************************************************/

// A readable name for a type, union, or complex shape
type Coordinate = { x: number; y: number };

const c: Coordinate = { x: 1, y: 2 };

// Type assertion (casting)
let x: unknown = "hello";
let y = x as string;  // trust me, it's a string
let z = <string>x;

