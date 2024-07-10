% Example 4: All objects in the same row must have different colors
#const c=1.
:- c=1, obj(X1, Y, _, C1), obj(X2, Y, _, C2), X1 != X2, C1 = C2.

% Example 4 Converse: At least one pair of objects in the same row must have the same color
row_same_color(Y) :- heigth(Y), width(X1), width(X2), X1 != X2, obj(X1, Y, _, C1), obj(X2, Y, _, C2), C1 = C2.
:- c=0, heigth(Y), not row_same_color(Y).
