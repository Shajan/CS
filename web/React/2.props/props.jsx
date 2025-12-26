// Props are read-only inputs
// Components are pure functions
// Parents own the <datalist>

// Parents pass immutable properties to children
function App() {
  return <Greeting name="Alice" />;
}

function Greeting({ name }) {
  return <h1>{name}</h1>;
}

// Parents can provide children with callbacks (as part of props)
function Parent() {
  const handleClick = () => {
    console.log("Child asked parent to do something");
  };
  return <Child onClick={handleClick} />;
}

function Child({ onClick }) {
  return <button onClick={onClick}>Click</button>;
}
