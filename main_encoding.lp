#const w=3.
#const h=3.

width(0..w-1).
heigth(0..h-1).

shape(triangle;circle;cross;square).
color(pink;blue;green;red).

{obj(X,Y,S,C) : shape(S), color(C)} = 1 :- width(X), heigth(Y).

#show obj/4.