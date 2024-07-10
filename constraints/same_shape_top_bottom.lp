
% Example 14: The top row and bottom row must have the same shapes
#const c=1.
:- c=1, obj(X, 0, S1, _), obj(X, h-1, S2, _), S1 != S2.

% Example 14 Converse: The top row and bottom row must have at least one different shape
top_bottom_diff_shape :- width(X), obj(X, 0, S1, _), obj(X, h-1, S2, _), S1 != S2.
:- c=0, not top_bottom_diff_shape.