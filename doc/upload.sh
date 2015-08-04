#!/bin/bash
rsync -azvu --delete --progress  _build/html/ pabbeel@rll.berkeley.edu:/var/www/cgt
