# Understanding React by Following What Actually Happens

This document explains React by following what happens at runtime.
No prior React or UI framework knowledge is assumed.

---

## What React Is Trying to Do

React exists to solve a simple problem:

You have some data.
You want the screen to reflect that data.
When the data changes, the screen should update automatically.

Instead of manually updating the screen, React asks you to **describe what the
UI should look like**, and React takes responsibility for updating the screen.

---

## What the Developer Writes

As a developer, you write code like this:

```jsx
<App />
```

This looks like HTML, but it is not.
This line does **not** display anything.
It does **not** call a function.

It is only a description.

Before the program runs, this syntax is converted into plain JavaScript.
When that JavaScript runs, it produces an object like this:

```js
{
  type: App,
  props: {}
}
```

This object means:

“There is something called `App`, and it should be used with these inputs.”

At this moment:
- No UI exists
- No function has been executed
- Nothing is on the screen

---

## What React Does With That Description

React receives this object and inspects it.

React asks:
- Is `type` a function?
- Or is it a built-in UI element like `"div"` or `"h1"`?

In this case, `type` is a function (`App`).

So **React** — not the developer — decides to call it:

```js
App(props)
```

The developer does **not** call `App()` directly.
React controls when and how this happens.

---

## What the Function Returns

The developer might have written:

```jsx
function App() {
  return <Hello name="Alice" />;
}
```

When React calls `App()`, it returns another description.

That returned description looks like:

```js
{
  type: Hello,
  props: { name: "Alice" }
}
```

Important details:
- This is a **new object**
- The earlier object is unchanged
- React never modifies descriptions in place

---

## React Repeats the Same Process

React now looks at this new object.

Again:
- `type` is a function (`Hello`)
- So React calls it

```js
Hello({ name: "Alice" })
```

The developer might have written:

```jsx
function Hello({ name }) {
  return <h4>Hello {name}</h4>;
}
```

That call produces another description:

```js
{
  type: "h4",
  props: { children: "Hello Alice" }
}
```

---

## When Does Anything Appear on Screen?

Only when React reaches descriptions whose `type` refers to built-in UI elements
(like `"div"`, `"h4"`, `"button"`) does React know how to map them to real UI.

Until then:
- Everything is just JavaScript objects
- Nothing is displayed
- No UI is created

The entire UI is built by **repeatedly turning descriptions into new descriptions**
until only real UI elements remain.

---

## Passing Data Into Functions

Sometimes a function needs data to decide what to describe.

The developer writes:

```jsx
<Hello name={inputName} />
```

This produces:

```js
{
  type: Hello,
  props: { name: "Alice" }
}
```

React passes this `props` object when calling the function:

```js
Hello({ name: "Alice" })
```

The function can **read** this data.
It cannot change it.

---

## Data That Changes Over Time

So far, all descriptions are based on fixed values.

Real applications need values that change:
- counters
- form input
- toggles
- fetched data

Functions normally forget everything after they run.
If React simply called functions repeatedly, nothing could persist.

React solves this by **remembering certain values between calls**.

---

## Remembered Values and Re-Running Functions

Inside a function, the developer can ask React to remember a value:

```jsx
const [count, setCount] = rememberValue(0);
```

Conceptually:
- `count` is the remembered value
- `setCount` tells React to update it

When `setCount` is called:
- React updates the remembered value
- React schedules another call to the function
- The function runs again
- New descriptions are produced
- React updates the screen if needed

The developer never updates the screen directly.

---

## Why the Order of These Requests Matters

React remembers values based on **the order in which they are requested**.

Because of this:
- These requests must happen in the same order every time
- They cannot be inside conditions or loops

This rule exists because of how React stores remembered values internally,
not because of arbitrary design.

---

## Introducing the Names (After the Behavior Is Clear)

Now that the behavior is understood, the standard terms make sense.

### UI Description Objects → Elements

Objects like:

```js
{
  type: "h4",
  props: { children: "Hello Alice" }
}
```

are called **elements**.

An element:
- is immutable
- is plain data
- describes what should exist in the UI

---

### Functions React Calls → Components

Functions like `App` and `Hello` are called **components**.

A component:
- is written by the developer
- is called by React
- returns UI descriptions

---

### Shorthand Syntax → JSX

The HTML-like syntax is called **JSX**.

JSX:
- is converted before execution
- never exists at runtime
- is only a convenient way to write descriptions

---

### Passed-In Data → Props

The data React passes into components is called **props**.

Props:
- flow from parent to child
- are read-only
- control what the component describes

---

### Remembered Values → State

The values React remembers between function calls are called **state**.

State:
- lives inside React
- triggers re-execution when updated
- causes new descriptions to be created

---

### Memory Mechanism → Hooks

The mechanism React uses to remember values and behavior is called **hooks**.

Hooks:
- rely on call order
- attach memory to functions
- only work when React controls execution

---

## Final Mental Model

The developer writes functions and UI descriptions.
React turns descriptions into function calls.
Functions return new descriptions.
React remembers selected values between calls.
React updates the screen to match the latest descriptions.

---

## One-Sentence Summary

React lets developers describe UI as JavaScript data and functions, and React controls execution and screen updates to keep everything in sync.

