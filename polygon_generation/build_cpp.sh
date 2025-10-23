#!/bin/bash

# Build script for C++ polygon solver
echo "Building C++ maximum area polygon solver..."

# Check if pybind11 is available
python3 -c "import pybind11" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing pybind11..."
    pip install pybind11
fi

# Clean previous builds
rm -rf build/ *.so *.egg-info/

# Build the extension
python3 setup.py build_ext --inplace

# Check if build was successful
if [ -f "max_area_polygon_cpp*.so" ]; then
    echo "Build successful! C++ extension is ready."
    ls -la max_area_polygon_cpp*.so
else
    echo "Build failed!"
    exit 1
fi

echo "Testing the C++ module..."
python3 -c "
import numpy as np
try:
    import max_area_polygon_cpp
    print('✓ C++ module imported successfully')
    
    # Test with a simple square
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    order, area = max_area_polygon_cpp.max_area_polygon(points)
    print(f'✓ Test successful: area = {area:.6f}')
    
except ImportError as e:
    print('✗ Failed to import C++ module:', e)
    exit(1)
except Exception as e:
    print('✗ Test failed:', e)
    exit(1)
"

echo "C++ solver is ready to use!"