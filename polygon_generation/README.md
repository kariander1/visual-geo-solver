### **Polygon Generation**
Compile the polygon generation tool:
```bash
cd polygon_generation/
chmod +x build_cpp.sh
./build_cpp.sh
```

Then generate polygons:
```bash
python polygon_generation/generate_polygons_data.py --skip-images
```