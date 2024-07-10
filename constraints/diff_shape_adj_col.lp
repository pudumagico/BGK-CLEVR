% Example 7: No objects in adjacent columns can have the same shape
#const c=1.
:- c=1, obj(X, Y, S1, _), obj(X+1, Y, S2, _), S1 = S2.

% Example 7 Converse: At least one pair of objects in adjacent columns must have the same shape
adj_col_same_shape :- obj(X, Y, S1, _), obj(X+1, Y, S2, _), S1 = S2.
:- c=0, not adj_col_same_shape.
