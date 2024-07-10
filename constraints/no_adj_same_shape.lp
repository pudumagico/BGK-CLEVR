% Example 9: No two adjacent objects can have the same shape
#const c=1.
:- c=1, obj(X, Y, S, _), obj(X1, Y, S, _), X1=X+1.
:- c=1, obj(X, Y, S, _), obj(X, Y1, S, _), Y1=Y+1.

% Example 9 Converse: At least one pair of adjacent objects must have the same shape
adj_same_shape :- obj(X, Y, S, _), obj(X1, Y, S, _), X1=X+1.
adj_same_shape :- obj(X, Y, S, _), obj(X, Y1, S, _), Y1=Y+1.
:- c=0, not adj_same_shape.
