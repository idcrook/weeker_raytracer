# NOTES



## 3D scatter plots

Install `gnuplot`

```
sudo apt-get install -y gnuplot
```

### plotting the weighted random spherical mapping


```
g++ src/xx7.cpp
./a.out > output/data.txt
```

now in gnuplot

```
splot 'output/data.txt' using 1:2:3 with points palette pointsize 3 pointtype 7
```

### plotting the importance sampling hemispherical (cosine) mapping


```
g++ src/xx7b.cpp
./a.out > output/data2.txt
```

now in gnuplot

```
splot 'output/data2.txt' using 1:2:3 with points palette pointsize 3 pointtype 7
```


## using clang

```
sudo apt install clang
make CXX=clang
```
