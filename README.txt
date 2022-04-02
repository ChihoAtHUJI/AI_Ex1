777713736
327742151
*****
Comments:
We used the manhattan max distance in order to evaluate the heuristic functions.
Each board we evaluate every tile distance between target and the tile. If that tile is the minimum distance
of all tiles then we save it and if we find the minimum distance for every target then we sum up the distances.

But this might make non-admissible heuristic function, so we added the bound in order to avoid overestimating.
For example, on blokus_corner_problem, we set the bound as board_w + board_h, in order to get to the 3 corners,
there must be at least board_w + board_h tiles to approach them.

On blokus_cover_problem, we set the bound as the maximum manhattan max distance between (targets + starting_points)
In order to cover all the targets, blokus need to use at least number of bound that we set.

So, until our heuristic hits the bound, we use uniform function which never overestimate the distance and afterward we
use the minimum of all the distances, so it couldn't overestimate the distance. Also, each step the number of tiles
are more than our evaluated heuristic to get there, so it is consistent
