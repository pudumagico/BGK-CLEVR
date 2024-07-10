% Example 10: All objects in the first row must have the same color
#const c=1.
:- c=1, obj(X1, 0, _, C1), obj(X2, 0, _, C2), X1 != X2, C1 != C2.

% Example 10 Converse: At least one pair of objects in the first row must have different colors
first_row_diff_color :- width(X1), width(X2), X1 != X2, obj(X1, 0, _, C1), obj(X2, 0, _, C2), C1 != C2.
:- c=0, not first_row_diff_color.