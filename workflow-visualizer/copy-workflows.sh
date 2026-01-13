#!/bin/bash

# Script to copy workflow YAML files from the main project to the visualizer's public folder
# This makes them easily accessible for testing

SOURCE_DIR="../csv_analyzer/workflows/definitions"
DEST_DIR="./public/workflows"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Copy all YAML files
echo "Copying workflow files..."
cp "$SOURCE_DIR"/*.yaml "$DEST_DIR/" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ Workflow files copied successfully to $DEST_DIR"
    echo ""
    echo "Available workflows:"
    ls -1 "$DEST_DIR"/*.yaml 2>/dev/null | xargs -n 1 basename
else
    echo "✗ Error: Could not find workflow files at $SOURCE_DIR"
    echo "Please ensure the csv_analyzer project is in the parent directory"
    exit 1
fi

echo ""
echo "You can now drag and drop these files into the visualizer!"

