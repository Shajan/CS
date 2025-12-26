console.log("React version:", React.version); // 18.3.1

// 1) Plain element (JSX -> React.createElement)
const el1 = <h1>Hello from JSX</h1>;
console.log("el1 (JSX element):", el1); // props: {children : 'Hello from JSX'}, type: 'h1'

// 2) Component definition
function Title(props) {
  return <h2 className="title">Hello, {props.name}</h2>;
}

console.log("Title (component function):", Title);
// Title (component function): ƒ Title(props) {
//   return /*#__PURE__*/
//   React.createElement("h2", { className: "title" }, "Hello, ", props.name);
// }

// 3) Element whose type is a component (NOT executed yet)
const el2 = <Title name="Shajan" />;
console.log("el2 (<Title /> element, not executed):", el2);
// props : name : "Shajan"; type: Title

console.log("el2.type === Title ?", el2.type === Title); // true

// 4) Execute the component yourself (this returns an element)
const el3 = Title({ name: "Shajan" });
console.log("el3 (Title(...) executed result):", el3);
// props : children : ["Hello, ", "Shajan"]
// type : 'h2'

// 5) A small tree
const tree = (
  <div id="root">
    <Title name="A" />
    <Title name="B" />
    <p>Done</p>
  </div>
);
console.log("tree (element tree):", tree);
// props: {id: 'root', children: Array[3]}, type 'div'

// 6) Inspect children shapes
console.log("tree.props.children:", tree.props.children);
// children[0] : props: {name: 'A'}, type: Title(props)
// children[1] : props: {name: 'B'}, type: Title(props)
// children[2] : props: {children: 'Done'}, type: 'p'
