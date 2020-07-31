# Neil and Bob Halite Bot

### New Features
### Game Visualizer
Now you can highlight cells in yellow!
```python
def draw_game(boards: Board[], highlights: Point[][])
```
It also displays board coordinates along the outside edges as well as the player score on the right hand side.
![image](https://user-images.githubusercontent.com/2446659/88866668-fd391200-d1d0-11ea-89ef-832ca3be28c2.png)


# Strategies

## Behaviors
Our theory is that ships will have a behavior assigned to them by some higher level strategy engine. The behaviors will probably be things like

* Mine
* Return
* Harrass
* Border Patrol

The most important one to begin with is mining.

## Mining Strategies

### Greedy Algorithms

There are three greedy objectives that need to be balanced

1. Should I stay on my current cell?
2. Should I move toward the best cell (closest distance, most halite)?
3. Should I move toward the biggest cluster of halite?

An optimal path probably involves walking toward the largest cluster and ocasionally mining a bit on the way. When you hit a particularly large halite cell then you may need to mine it for several turns before moving on.

My proposal for greedy mining is to use best cell for short range planning and biggest cluster for long range planning. Basically I propose finding the next best cell of halite to move toward but weighting cells that are in the same direction as the biggest cluster of halite more heavily. The cluster algorithm is kinda like gravity we find the sum vector of all forces exerted on the ship by each halite cell. The force is proportionate to the amount of halite and reduces with distance. We can also add other forces like repelling away from other miners or enemy ships. These functions need not be continuous, they can have step functions at thresholds (ie never mine a cell with less than X halite, don't bother considering cells more than X distance away).

### Planning algorithm

There may be some decisions that will be helpful to have some kind of [local search](https://en.wikipedia.org/wiki/Local_search_(optimization)) engine. In some cases, searching around the search space may have some randomization so it may look like [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing).

For example, suppose we want to find out what is the maximum number of halite we can get by some time t_max. A solution would be some Z(t) where t goes from 0 to t_max. Z(0) is our initial location and Z(t_max) is our dropoff location. Z(t) must be adjacent to Z(t-1). The distance between Z(t) and Z(0) must be <= t and the distance between Z(t_max) and Z(t_max - r) must be <= r. No solution probably involves doubling back so If Z(t) == Z(r), |t-r| must be one or less. If this is still too expensive, it may be safe to assume that a solution involves a cell with a large amount of halite so we can try to find solutions that include the cells with the largest amount of halite nearby. That should be enough to constrain the solution space so that we can quickly find a good solution.

## Oracles
Oracles are functions that can answer questions like

* what is the most likely next N moves that my opponents will make?
* If I generate a miner at this point in the game, am I likely to make a positive return on the investment?
* What is the path for my miner that will harvest the most halite?

## Board Control

We have a theory that board control is advantageous. The theory is that it is much easier to herd enemy ships than it is to capture them (similar to two kings on a chess board). Effective herding may be able to push enemy miners away from the high value mining locations and secure them for our own miners.

My theory is that board control is trying to maximize the amount of the board where the closest empty ship is owned by us. The larger area that is, the more area our miners will have to safely mine. Territory with large amounts of halite is especially valuable. The border patrol behavior is to make moves that will herd enemy ships away from our territory, move toward vulnerable edges of our perimeter, while expanding our territory into valuable areas of the map.

The other part of board control is harrassment. Which is to send empty ships behind enemy lines to disrupt the area of control of our opponents. Harrassment behavior most likely involves chasing around enemy miners to prevent them from mining high value halite and returning their cargo. Because we have three opponents, investing in harrassment is probably only useful against opponents that are close to us in rank (sabotaging the opponent that is just above us in rank to take his place or just below us to prevent them from taking our place). 

### Herding

The two kings theory of herding is that rather than pursing enemy ships around the board, we establish a boundary and prevent enemy ships from crossing it. 

For instance in the below image, suppose B wants to prevent A from reaching X. B should not move directly onto A. It should stay where it is. 

![image](https://user-images.githubusercontent.com/2446659/88663039-83097000-d0a0-11ea-8c47-a15e45951ea0.png)

If A goes left to B4, B should go to B5

![image](https://user-images.githubusercontent.com/2446659/88663348-180c6900-d0a1-11ea-83cc-a23dcde4b9d3.png)

B can prevent A from ever getting to X, by simply guarding row 5. Its obviously more complicated when there are more ships and halite involved, but the overall concept is that it is easier to block enemy ships from crossing boundaries than it is to chase them down and destroy them.

## Useful Utilities

precomputing things like distance for every point to every point will be a bit large. 21 x 21 x 21 x 21 x 4 = 760kB. Manhattan distance is pretty cheap anyways so it may not be worth it.

### Vector Distance
Vector distance gives the shortest distance to a point as a vector (dx, dy). Positive dx means EAST. Positive dy means SOUTH (for some reason). 
```python
center = Point(board_size // 2, board_size // 2)
def vector_distance(origin: Point, dest: Point, board_size: int):
       return (dest - origin + center) % board_size - center
```
### Manhattan Distance
I think this is the cheapest way to do a Manhattan distance.
```python
def manhattan_distance(p1: Point, p2: Point, board_size: int):
    dp = abs(vector_distance)
    return dp.x + dp.y
```
