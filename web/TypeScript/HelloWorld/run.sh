# Generate hello.js
SRC=HelloWorld

compile() {
  tsc --outDir ./dist ./${SRC}.ts
}

run() {
  node ./dist/${SRC}.js
  #npx tsc ./${SRC}.js
}

cleanup() {
  rm -rf ./dist
}

compile
run
cleanup
