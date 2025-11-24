SRC=${1:-Sample}

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
