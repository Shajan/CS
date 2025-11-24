// Interface ---------------------------------------------

interface Person {
  name: string;
  age: number;
}

const p: Person = {
  name: "Alice",
  age: 30,
};

// Student has everything Person has, plus extra fields
interface IStudent extends Person {
  grade: number;
}

const mystudent: IStudent = {
  name: "Bob",
  age: 20,
  grade: 3,
};

// Interfaces with same name merges
interface IUser {
  id: string;
}

interface IUser {
  name: string;
}

const myuser: IUser = {
  id: "1",
  name: "Alice",   // OK because both interfaces merged
};


// Type aliasing ---------------------------------------------

type PersonType = {
  name: string;
  age: number;
};

const pt: PersonType = {
  name: "Carol",
  age: 40,
};

// A value can be one of these strings
type Direction = "north" | "south" | "east" | "west";

let d: Direction;
d = "north";   // OK
// d = "up";   // Compile time Error


type HasId = { id: number };
type HasName = { name: string };

// Intersection: must have both id and name
type Identified = HasId & HasName;

const x: Identified = {
  id: 1,
  name: "Thing",
};


// Types can have methods
type Greeter = {
  name: string;
  greet(): void;
};

const g: Greeter = {
  name: "Alice",
  greet() {
    console.log("Hello from", this.name);
  }
};

// Type alias for a function
type Adder = (a: number, b: number) => number;

const add: Adder = (x, y) => x + y;


// Class ---------------------------------------------

class Counter {
  private value: number;

  constructor(initial: number = 0) {
    this.value = initial;
  }

  increment() {
    this.value++;
  }

  getValue(): number {
    return this.value;
  }
}

const c = new Counter(5);
c.increment();
console.log(c.getValue()); // 6

// Class implementing interface
interface Logger {
  log(message: string): void;
}

class ConsoleLogger implements Logger {
  log(message: string): void {
    console.log("LOG:", message);
  }
}

const logger: Logger = new ConsoleLogger();
logger.log("Hello");

// Class implementing type
type PointLike = {
  x: number;
  y: number;
};

class Point2D implements PointLike {
  constructor(public x: number, public y: number) {}
}

const pt2: PointLike = new Point2D(10, 20);


// Class implementing interface with storage
interface IPerson { // No storage, just declaration
  readonly id: string;   // must be set once, cannot change
  name: string;
  age: number;
}

class Student implements IPerson {
  readonly id: string;   // must satisfy the interface
  name: string;
  age: number;
  grade: number;

  constructor(id: string, name: string, age: number, grade: number) {
    this.id = id;     // ✔️ allowed — set once here
    this.name = name;
    this.age = age;
    this.grade = grade;
  }
}

const s = new Student("stu-1", "Alice", 20, 3);

// s.id = "new-id";  // Compile time error

// Advanced ----------------------------------------------------------------

// Two types are considered compatible if their structure matches
interface Point { x: number; y: number; }

class Pixel {
  constructor(public x: number, public y: number) {}
}

let p: Point = new Pixel(1, 2); //  OK: structure matches

// Exhaustive checking, good coding practice

type Shape = { kind: "circle" } | { kind: "square" };

function area(s: Shape) {
  switch (s.kind) {
    case "circle": return 1;
    case "square": return 4;
    default:
      const _exhaustive: never = s; // Error if a new kind is added
  }
}

// Type predicate
type Dog = { kind: "dog", bark: () => void };
type Cat = { kind: "cat", meow: () => void };
type Pet = Dog | Cat;

function isDog(p: Pet): p is Dog {
  // `p is Dog`
  // If this function returns true, then p should be treated as type Dog from that point onward.
  return p.kind === "dog";
}

function speak(p: Pet) {
  if (isDog(p)) {
    p.bark();   //  narrowed to Dog
  } else {
    p.meow();   //  narrowed to Cat
  }
}

