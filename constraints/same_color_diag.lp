% Example 6: All objects in the diagonal must be of the same color
#const c=1.
:- c=1, obj(X, X, _, C1), obj(X1, X1, _, C2), X != X1, C1 != C2.

% Example 6 Converse: At least one pair of objects in the diagonal must have different colors
diag_diff_color :- width(X), width(X1), X != X1, obj(X, X, _, C1), obj(X1, X1, _, C2), C1 != C2.
:- c=0, not diag_diff_color.
