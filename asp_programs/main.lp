#const w=3.
#const h=3.

width(0..w-1).
heigth(0..h-1).

form(triangle;circle;cross;square;diamond).
color(yellow;blue;green;red;purple).
size(small;medium;big).

{obj(X,Y,F,C,S) : form(F), color(C), size(S)} = 1 :- width(X), heigth(Y).

#show obj/5.