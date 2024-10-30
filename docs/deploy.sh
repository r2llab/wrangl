#!/usr/bin/env bash
set -x
set -e
env WRANGL_DOCS_HOST=https://github.com/r2llab/wrangl/blob/main/wrangl/ wdocs
rm -rf docs/deploy
git clone -b gh-pages git@github.com:r2llab/wrangl docs/deploy

rm -rf docs/deploy/*
cp -r docs/build/* docs/deploy
git -C docs/deploy add -f .
git -C docs/deploy commit -m "auto push docs"
git -C docs/deploy push -f
