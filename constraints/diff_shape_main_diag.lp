% Example 11: All objects in the main diagonal must have different shapes
#const c=1.
:- c=1, obj(X, X, S1, _), obj(Y, Y, S2, _), X != Y, S1 = S2.

% Example 11 Converse: At least one pair of objects in the main diagonal must have the same shape
main_diag_same_shape :- width(X), width(Y), X != Y, obj(X, X, S1, _), obj(Y, Y, S2, _), S1 = S2.
:- c=0, not main_diag_same_shape.