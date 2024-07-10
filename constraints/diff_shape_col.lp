% Example 5: No two objects in the same column can have the same shape
#const c=1.
:- c=1, obj(X, Y1, S1, _), obj(X, Y2, S2, _), Y1 != Y2, S1 = S2.

% Example 5 Converse: At least one pair of objects in the same column must have the same shape
col_same_shape(X) :- width(X), heigth(Y1), heigth(Y2), Y1 != Y2, obj(X, Y1, S1, _), obj(X, Y2, S2, _), S1 = S2.
:- c=0, width(X), not col_same_shape(X).