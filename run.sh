#!/bin/bash

echo "Running Assignment 2 template of Amy Louca (1687077)"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist create it!"
  mkdir plots
fi

echo "Creating the textfiles directory if it does not exist"
if [ ! -d "textfiles" ]; then
  echo "Directory does not exist create it!"
  mkdir textfiles
fi

echo "Creating the movie directories if it does not exist"
if [ ! -d "movie_2d" ]; then
  echo "Directory does not exist create it!"
  mkdir movie_2d
fi

if [ ! -d "movies" ]; then
  echo "Directory does not exist create it!"
  mkdir movies
fi

if [ ! -d "movie_3d" ]; then
  mkdir movie_3d
fi

wget https://home.strw.leidenuniv.nl/~nobels/coursedata/randomnumbers.txt
wget strw.leidenuniv.nl/~nobels/coursedata/GRBs.txt
wget strw.leidenuniv.nl/~nobels/coursedata/colliding.hdf5

echo "Running script Q1.py ..."
python3 Q1.py 

echo "Running script Q2.py ..."
python3 Q2.py 

echo "Running script Q3.py ..."
python3 Q3.py 

echo "Running script Q4.py ..."
python3 Q4.py > textfiles/int_div.txt

echo "Running script Q5.py ..."
python3 Q5.py

echo "Running Script Q6.py ..."
python3 Q6.py

echo "Running Script Q7.py ..."
python3 Q7.py > textfiles/tree.txt

python3 plot.py

ffmpeg -r 30 -f image2 -s 64x64 -i movie_2d/%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p movies/movie_2d.mp4
ffmpeg -r 30 -f image2 -s 64x64 -i movie_3d/XY_%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p movies/movie_3d_XY.mp4
ffmpeg -r 30 -f image2 -s 64x64 -i movie_3d/XZ_%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p movies/movie_3d_XZ.mp4
ffmpeg -r 30 -f image2 -s 64x64 -i movie_3d/YZ_%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p movies/movie_3d_YZ.mp4

#echo "Running Script Q8.py ..."
#python3 Q8.py 

echo "Generating the pdf"

pdflatex assignment2.tex
