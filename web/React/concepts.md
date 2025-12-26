# React: A Beginner’s Mental Model

This document explains React from first principles.
You do NOT need to know React, UI frameworks, or web development concepts beforehand.

---

## The Big Idea

React answers one question:

“How do I describe what my user interface should look like, based on data that can change over time?”

React’s approach:
- You describe the UI as data
- React figures out how to update the screen

You do NOT manually update the screen.

---

## Step 1: Describing UI with JSX

Normally, JavaScript has no built-in way to describe UI structure.
React introduces **JSX**, a special syntax that *looks like HTML*.

```jsx
<h1>Hello</h1>
```

Important:
- JSX is **not HTML**
- JSX is **not JavaScript**
- JSX is only a **writing convenience**

Before the program runs, JSX is converted into normal JavaScript.

```jsx
<h1>Hello</h1>
```

becomes something like:

```js
jsx("h1", { children: "Hello" })
```

This conversion happens before execution.
Browsers never see JSX.

Think of JSX as shorthand for writing UI descriptions.

---

## Step 2: What JSX Produces — Elements

When JSX is converted and executed, it produces a **React element**.

A React element is:
- a plain JavaScript object
- immutable (it never changes)
- a description of UI, not actual UI

Example:

```jsx
const el = <h1>Hello</h1>;
```

creates this object:

```js
{
  type: "h1",
  props: { children: "Hello" }
}
```

This object:
- is NOT a DOM element
- does NOT appear on screen
- does NOT do anything by itself

It is only a **description**.

Think of a React element as a blueprint, not a building.

---

## Step 3: Components — Functions That Produce UI Descriptions

Writing individual elements is not enough for real apps.
We need reusable pieces.

A **component** is just a JavaScript function.

```jsx
function Hello() {
  return <h1>Hello</h1>;
}
```

What this function does:
- It returns a React element (a UI description)

Using the component:

```jsx
<Hello />
```

does NOT call the function yet.

It creates another element:

```js
{
  type: Hello,
  props: {}
}
```

Later, React:
- sees that `type` is a function
- calls the function itself
- receives the returned element

Important rule:
You do not control when components run.
React does.

---

## Step 4: Props — Passing Data Into Components

Components often need data.

**Props** are how data is passed into components.

```jsx
function Greeting({ name }) {
  return <h1>Hello {name}</h1>;
}
```

Using it:

```jsx
<Greeting name="Alice" />
```

Props:
- are inputs to a component
- are read-only
- flow from parent to child

A component cannot change its props.
It can only use them to decide what UI to describe.

---

## Step 5: Why State Exists

Props alone are not enough.
Some data changes over time.

Examples:
- counters
- form input
- toggles
- fetched data

This is where **state** comes in.

State is:
- data remembered by React
- tied to a component
- preserved across re-runs of the component

Example:

```jsx
const [count, setCount] = useState(0);
```

Meaning:
- `count` is the current value
- `setCount` tells React to update it

When state changes:
- React re-runs the component
- new UI descriptions are produced
- React updates the screen if needed

You never update the screen directly.

---

## Step 6: Why Components Run Again

A key idea:

React components are just functions.
React runs them again when:
- state changes
- props change
- a parent component updates

Running again does NOT mean recreating everything.
React compares old descriptions to new ones and updates only what changed.

---

## Step 7: Hooks — How React Remembers Things

If components are just functions, how does React remember state?

Answer: **Hooks**

Hooks are special function calls that:
- only work inside components
- must be called in the same order every time
- let React store information between renders

Example:

```jsx
useState(0)
```

React internally remembers:
- “this is the first hook call”
- “this value belongs here”

Because of this:
- hooks must not be inside loops or conditions
- call order must stay consistent

Hooks are how React attaches memory and behavior to functions.

---

## Final Mental Model

React works like this:

- JSX is converted into JavaScript
- JavaScript creates element objects
- Elements describe what the UI should be
- Components are functions that return elements
- Props are inputs to components
- State is remembered data that can change
- Hooks let React remember state between renders
- React controls when functions run
- You only describe what the UI should look like

You describe.
React updates.

---

## One-Sentence Summary

React lets you describe your UI as data, and it automatically keeps the screen in sync when that data changes.
