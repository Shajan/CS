/************************************************************
 * Very basic mapped type examples
 * (All of this runs ONLY in the TypeScript type system,
 *  nothing here exists at runtime.)
 ************************************************************/

/***********************
 * 1. A simple base type
 ***********************/
type Person = {
  name: string;
  age: number;
};

/************************************************************
 * 2. keyof Person
 *
 * - keyof Person is a TYPE, not a value.
 * - It becomes: "name" | "age"
 * - You cannot log it at runtime.
 ************************************************************/
type PersonKeys = keyof Person;
// PersonKeys is the union type: "name" | "age"


// Lokup Type of different fields
type PersonName = Person["name"];   // string
type PersonAge  = Person["age"];    // number

/************************************************************
 * 3. Make all properties readonly
 *
 * - [K in keyof T] is a MAPPED TYPE.
 * - Think: “for each property K in T, create a new property K…”
 * - This is purely a TYPE TRANSFORMATION, no runtime loop.
 ************************************************************/
type ReadonlyPerson<T> = {
  readonly [K in keyof T]: T[K]; // type-system “for-each key”
  // Syntax: 'T[K]' look up the type of property K inside the type T.
};

type PersonReadonly = ReadonlyPerson<Person>;

/*
   PersonReadonly becomes:
   {
     readonly name: string;
     readonly age: number;
   }
*/

const p1: PersonReadonly = { name: "Alice", age: 30 };

// p1.name = "Bob"; // Error: cannot assign to readonly property


/************************************************************
 * 4. Make all properties optional
 *
 * - Same pattern, but adding "?" to each property.
 ************************************************************/
type Optional<T> = {
  [K in keyof T]?: T[K];
};

type PersonOptional = Optional<Person>;

/*
   PersonOptional becomes:
   {
     name?: string;
     age?: number;
   }
*/

const p2: PersonOptional = { name: "Charlie" }; // age is optional


/************************************************************
 * 5. Make all properties nullable
 *
 * - Add "| null" to each property’s type.
 ************************************************************/
type Nullable<T> = {
  [K in keyof T]: T[K] | null;
};

type PersonNullable = Nullable<Person>;

/*
   PersonNullable becomes:
   {
     name: string | null;
     age: number | null;
   }
*/

const p3: PersonNullable = {
  name: null,
  age: 42,
};


/************************************************************
 * 6. IMPORTANT: None of this exists at runtime
 *
 * - "type", "keyof", and "[K in keyof T]" are erased when
 *   TypeScript compiles to JavaScript.
 * - They only help the COMPILER understand and check your code.
 *
 * - You cannot do:
 *     const x = keyof Person;   // ❌ not allowed
 *     console.log(PersonKeys);  // ❌ type is not a value
 *
 * - If you want runtime keys, use real values:
 ************************************************************/
const personValue = { name: "Dana", age: 50 };
const runtimeKeys = Object.keys(personValue); // ["name", "age"]
console.log(runtimeKeys);

/*
   Here:
   - "personValue" and "runtimeKeys" are REAL values (runtime).
   - "keyof Person" and "[K in keyof T]" are TYPE-LEVEL only.
*/


// Type manipulation ----------------------------------------------------

type SuperPerson = {
  name: string;
  age: number;
  email: string;
};

/****************************************************************
 * Pick<T, K>
 * ----------
 * Creates a NEW type with ONLY the selected keys.
 *
 * Here:
 *   JustName becomes:
 *     { name: string }
 *
 * This is like choosing a subset of fields.
 ****************************************************************/
type JustName = Pick<SuperPerson, "name">;


/****************************************************************
 * Partial<T>
 * ----------
 * Makes ALL properties of T optional.
 *
 * Here:
 *   PartialPerson becomes:
 *     {
 *       name?: string;
 *       age?: number;
 *       email?: string;
 *     }
 *
 * Often used for "patch" updates or constructing objects gradually.
 ****************************************************************/
type PartialPerson = Partial<SuperPerson>;


/****************************************************************
 * Required<T>
 * -----------
 * Makes ALL properties of T required.
 *
 * This is mainly useful when a type has optional fields and
 * you want a version where EVERYTHING must be provided.
 *
 * Here SuperPerson already has no optional fields, so the result
 * is the same:
 *     {
 *       name: string;
 *       age: number;
 *       email: string;
 *     }
 ****************************************************************/
type RequiredPerson = Required<SuperPerson>;

