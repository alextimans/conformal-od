#!/bin/bash

echo "++++++++++++++++++++++++++++++++++++"
echo "+++ start experiments run script +++"
echo "++++++++++++++++++++++++++++++++++++"
echo ""

# Execute a Python script to generate commands.txt
# echo "+++ generating commands file"
# python conformalbb/commands.py
# echo ""

# Remove existing log file
rm -f run_error_log.txt

# Read commands from commands.txt and execute them
echo "+++ executing commands in series"
echo ""

while read -r command; do

    echo "executing command: $command"
    echo ""

    # Run the Python command and display the output in real time,
    # also capture the output
    output=$(mktemp)
    bash -c "$command" | tee "$output"

    # Check if command was successful
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        # Command failed. Log to error file
        echo "FAILED command: $command" >> run_error_log.txt
        cat "$output" >> run_error_log.txt
        echo "" >> run_error_log.txt
        echo "" >> run_error_log.txt
    fi
    rm "$output"

done < commands.txt

echo "++++++++++++++++++++++++++++++++++"
echo "+++ end experiments run script +++"
echo "++++++++++++++++++++++++++++++++++"