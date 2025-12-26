function Counter() {
  const [count, setCount] = useState(0);
  const [name, setName] = useState("Alice");
}
// React does not know variables by name. It knows them by useState() call order. 
//   Implies: Hooks must be called unconditionally and in the same order every render
//   WRONG! if (x) { useState(0) }


useEffect(() => {
  // side effect
  return () => {
    // cleanup
  };
}, [dependencies]);
