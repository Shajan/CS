// React element is data that get transformed to a map
const element1 = <h1>I am an element!</h1>;
console.log(element1);

//{
//  '$$typeof': Symbol(react.transitional.element),
//  type: 'h1',
//  key: null,
//  props: { children: 'I am an element!' },
//  _owner: null,
//  _store: {}
//}


// React component returns elements
function MyComponent() {
  return <h1>I am an element and returned by a component</h1>;
}

// creates an element with type MyComponent
const element2 = <MyComponent />;

console.log(element2);
//{
//  '$$typeof': Symbol(react.transitional.element),
//  type: [Function: MyComponent],
//  key: null,
//  props: {},
//  _owner: null,
//  _store: {}
//}


// calls the function and returns its element
const result = MyComponent()

console.log(result)
//{
//  '$$typeof': Symbol(react.transitional.element),
//  type: 'h1',
//  key: null,
//  props: { children: 'I am an element and returned by a component' },
//  _owner: null,
//  _store: {}
//}
