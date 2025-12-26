import { useState } from "react";

function Counter() {
  const [count, setCount] = useState(0);
  // Do not change count directly, react will not know the state changed and will not re-render
  // Call setCount() instead to change count
  return <h1>{count}</h1>;
}
