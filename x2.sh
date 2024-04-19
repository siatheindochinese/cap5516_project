#!/bin/bash
for (( i=1; i<=484; i++ ))
do
   python quantitative.py --idx $i --res 120
done
