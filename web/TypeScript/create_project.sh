#!/bin/bash

# if necessary, install the following on mac
#
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# brew install node
# brew install pnpm
# pnpm add -g typescript ts-node

# --------------------------------------------------
# 1. Create Project Structure
# --------------------------------------------------
echo "Creating TypeScript project..."

mkdir -p ./hello-ts/src
cd ./hello-ts

pnpm init

# --------------------------------------------------
# 2. Create HelloWorld.ts
# --------------------------------------------------
echo "Creating src/HelloWorld.ts..."

cat <<EOF > src/HelloWorld.ts
const message: string = "Hello, TypeScript World!";
console.log(message);
EOF

# --------------------------------------------------
# 3. Create tsconfig.json
# --------------------------------------------------
echo "Creating tsconfig.json..."

cat <<EOF > tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "CommonJS",
    "strict": true,
    "outDir": "dist",
    "rootDir": "src"
  },
  "include": ["src/**/*.ts"]
}
EOF

# --------------------------------------------------
# 4. Create .gitignore
# --------------------------------------------------
echo "Creating .gitignore..."

cat <<EOF > .gitignore
# Node
node_modules/

# TypeScript build output
dist/
out/

# Logs
npm-debug.log
yarn-error.log
pnpm-debug.log

# Optional IDE folders
.vscode/
.idea/

# Environment files
.env
EOF

# --------------------------------------------------
# 5. Create cleanup script
# --------------------------------------------------
echo "Creating cleanup script..."

cat <<EOF > cleanup.sh
#!/bin/bash
echo "Removing compiled artifacts..."
rm -rf dist
echo "Cleanup complete."
EOF

chmod +x cleanup.sh

# --------------------------------------------------
# 6. Compile and run
# --------------------------------------------------
echo "Compiling TypeScript..."
tsc

echo "Running output:"
node dist/HelloWorld.js

echo "Setup complete."

