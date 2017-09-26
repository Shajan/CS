#!/bin/bash

read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]
then
  echo "Continuing.."
else
  echo "Aborting.."
fi
