#!/bin/bash
for i in {1...484}
do
   python quantitative.py --idx $i 60
done
