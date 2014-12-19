#!/bin/bash

cp -r $RECIPE_DIR/../.. $SRC_DIR
rm -rf $SRC_DIR/build
$PYTHON setup.py clean
$PYTHON setup.py install
