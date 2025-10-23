### **Steiner Tree Generation**
Build GeoSteiner 5.3 for optimal Steiner tree data generation:
```bash
cd geosteiner-5.3/
chmod +x configure
./configure
make
```

Then generate steiner trees:
```bash
python steiner_generation/generate_steiner_data.py --skip-images
```