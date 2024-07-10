% Example 1: All objects in a column must have the same color
#const c=1.
:- c=1, obj(X, Y1, _, C1), obj(X, Y2, _, C2), Y1 != Y2, C1 != C2.

% Example 1 Converse: At least one pair of objects in a column must have different colors
col_diff_color(X) :- width(X), heigth(Y1), heigth(Y2), Y1 != Y2, obj(X, Y1, _, C1), obj(X, Y2, _, C2), C1 != C2.
:- c=0, width(X), not col_diff_color(X).