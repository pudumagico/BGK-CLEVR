% Example 8: All objects in the grid must be of the same color
#const c=1.
:- c=1, obj(X1, Y1, _, C1), obj(X2, Y2, _, C2), (X1, Y1) != (X2, Y2), C1 != C2.

% Example 8 Converse: At least one pair of objects in the grid must have different colors
some_diff_color :- obj(X1, Y1, _, C1), obj(X2, Y2, _, C2), (X1, Y1) != (X2, Y2), C1 != C2.
:- c=0, not some_diff_color.
