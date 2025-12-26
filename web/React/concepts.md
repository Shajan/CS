# React concepts

This document explains React by following what actually happens when a program runs.
It clearly distinguishes between what the developer writes and what React does.

No prior React or UI framework knowledge is assumed.

---

## What React Is For

React is designed to solve one problem:

You have data.
You want a screen that reflects that data.
When the data changes, the screen should update automatically.

Instead of manually updating the screen, React asks you to describe what the UI
should look like. React then takes responsibility for keeping the screen in sync
with the data.

---

## What the Developer Writes

As a developer, you write code like this:

```jsx
<App />
```

This looks like HTML, but it is not.
This line does not display anything.
It does not call a function.

It is only a description.

Before the program runs, this syntax is converted into plain JavaScript.
When that JavaScript executes, it produces an object like this:

```js
{
  type: App,
  props: {}
}
```

This object means:

“There is something called `App`, and it should be used with these inputs.”

At this point:
- No UI exists
- No function has been executed
- Nothing is on the screen

---

## What React Does With the Description

React receives the object and inspects it.

React checks what `type` refers to.
If `type` is a function, React decides to call it.

So React — not the developer — calls:

```js
App(props)
```

The developer does not call `App()` directly.
React controls when and how functions are executed.

---

## What the Function Produces

The developer might have written:

```jsx
function App() {
  return <Hello name="Alice" />;
}
```

When React calls `App()`, the function returns another description.
That description looks like:

```js
{
  type: Hello,
  props: { name: "Alice" }
}
```

Important details:
- This is a new object
- The previous object is unchanged
- React never modifies descriptions in place

---

## How React Continues

React now examines this new object.

Again:
- `type` refers to a function (`Hello`)
- React calls it

```js
Hello({ name: "Alice" })
```

The developer may have written:

```jsx
function Hello({ name }) {
  return <h4>Hello {name}</h4>;
}
```

That call returns another description:

```js
{
  type: "h4",
  props: { children: "Hello Alice" }
}
```

---

## When the Screen Is Actually Updated

Only when React encounters descriptions whose `type` refers to built-in UI elements
such as `"div"`, `"h4"`, or `"button"` does React know how to map them to real UI.

Until that point:
- Everything is just JavaScript objects
- No UI exists
- Nothing is rendered

The entire interface is produced by repeatedly turning descriptions into new
descriptions until only built-in UI elements remain.

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

React passes this object when calling the function:

```js
Hello({ name: "Alice" })
```

The function can read this data.
It cannot change it.

---

## Data That Changes Over Time

So far, all descriptions depend on fixed values.

Real applications need values that change:
- counters
- form input
- toggles
- fetched data

Functions normally forget everything after they run.
If React simply called functions repeatedly, nothing could persist.

React solves this by remembering certain values between calls.

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

## Why Call Order Matters

React remembers values based on the order in which they are requested.

Because of this:
- These requests must occur in the same order every time
- They cannot be inside conditions or loops

This rule exists because of how React stores remembered values internally.

---

## Terminology

The concepts described above have standard names.

### Elements

Objects like:

```js
{
  type: "h4",
  props: { children: "Hello Alice" }
}
```

are called elements.

An element:
- is immutable
- is plain data
- describes what should exist in the UI

---

### Components

Functions like `App` and `Hello` are called components.

A component:
- is written by the developer
- is called by React
- returns UI descriptions

---

### JSX

The HTML-like syntax used to write descriptions is called JSX.

JSX:
- is converted before execution
- never exists at runtime
- is only a convenient way to write descriptions

---

### Props

The data passed into components by React is called props.

Props:
- flow from parent to child
- are read-only
- determine what the component describes

---

### State

Values remembered by React between function calls are called state.

State:
- lives inside React
- triggers re-execution when updated
- causes new descriptions to be created

---

### Hooks

The mechanism React uses to remember values and behavior is called hooks.

Hooks:
- attach memory to functions
- rely on consistent call order
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

React lets developers describe UI as JavaScript data and functions, and React controls execution and screen updates to keep the UI in sync with changing data.
