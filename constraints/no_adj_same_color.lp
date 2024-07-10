% Example 3: No two adjacent objects can have the same color
#const c=1.
:- c=1, obj(X, Y, _, C), obj(X1, Y, _, C), X1=X+1.
:- c=1, obj(X, Y, _, C), obj(X, Y1, _, C), Y1=Y+1.

% Example 3 Converse: At least one pair of adjacent objects must have the same color
adj_same_color :- obj(X, Y, _, C), obj(X1, Y, _, C), X1=X+1.
adj_same_color :- obj(X, Y, _, C), obj(X, Y1, _, C), Y1=Y+1.
:- c=0, not adj_same_color.
