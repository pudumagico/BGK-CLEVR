#const w=5.
#const h=5.

width(0..w-1).
heigth(0..h-1).

shape(triangle;circle;cross;square;diamond).
color(yellow;blue;green;red;purple).

{obj(X,Y,S,C) : shape(S), color(C)} = 1 :- width(X), heigth(Y).


% Example 2: All objects in the grid must be of different shapes
% #const c=1.
% :- c=1, obj(X1, Y1, S, _), obj(X2, Y2, S, _), (X1, Y1) != (X2, Y2).

% % Example 2 Converse: At least one pair of objects in the grid must have the same shape
% some_same_shape :- obj(X1, Y1, S, _), obj(X2, Y2, S, _), (X1, Y1) != (X2, Y2).
% :- c=0, not some_same_shape.

#show obj/4.