#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <input_file>"
  exit 1
fi

input_file="$1"
echo "Input file: $input_file"
# Use echo "$OUTPUT" | sed to convert the pairs to the specified format
OUTPUT=$(sed 's/.*\[\(.*\)|/\1/' "$input_file")
OUTPUT=$(echo "$OUTPUT" | sed 's/ \].*/;/g' )
OUTPUT=$(echo "$OUTPUT" | sed 's/ /,/g' )
OUTPUT=$(echo "$OUTPUT" | sed ':a;N;$!ba;s/\n//g' )

echo "$OUTPUT"
echo "Script executed successfully!"
exit 0