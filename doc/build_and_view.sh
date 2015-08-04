#!/bin/sh
set -e

export PATH=/Users/joschu/Src/anaconda/bin:$PATH # In case this is being run by sublime
make html

index=_build/html/index.html
if [ `uname` = Linux ]
then
    google-chrome $index
else
    open -a Google\ Chrome  $index
fi