% Example 12: Objects in the corners must all be the same shape
#const c=1.
:- c=1, obj(0, 0, S1, _), obj(w-1, 0, S2, _), S1 != S2.
:- c=1, obj(0, 0, S1, _), obj(0, h-1, S3, _), S1 != S3.
:- c=1, obj(0, 0, S1, _), obj(w-1, h-1, S4, _), S1 != S4.

% Example 12 Converse: At least one pair of objects in the corners must have different shapes
corner_diff_shape :- obj(0, 0, S1, _), obj(w-1, 0, S2, _), S1 != S2.
corner_diff_shape :- obj(0, 0, S1, _), obj(0, h-1, S3, _), S1 != S3.
corner_diff_shape :- obj(0, 0, S1, _), obj(w-1, h-1, S4, _), S1 != S4.
:- c=0, not corner_diff_shape.