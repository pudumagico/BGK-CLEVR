% Example 15: All objects in the grid must be either triangle or square
#const c=1.
:- c=1, obj(X, Y, S, _), S != triangle, S != square.

% Example 15 Converse: There must be at least one object that is neither a triangle nor a square
non_triangle_square :- obj(X, Y, S, _), S != triangle, S != square.
:- c=0, not non_triangle_square.