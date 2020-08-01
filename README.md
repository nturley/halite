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

## Parameters. 
Should I mine my current cell or move to a bigger cell?
Should I continue mining or return my cargo to the nearest shipyard?
Should I mine to a far away big cell or a close by small cell?
Should I mine toward a large cluster of halite or a single cell with more halite?
Should I mine close to a shipyard or to close to big halite cells?
From how far away should I chase after vulnerable enemy ships?
Should I build a new shipyard or return to an existing shipyard?

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

## Floyd Marshall

I think this one is the winner. It simultaneously solves the best mining route for all spaces simultaneously. Instead starting at a ship and searching for halite. We start at halite and search for ships. One of the cool parts of this algorithm is that it can find mining routes for multiple ships at the same time!

The way it works is it calculates how much halite can be mined in one turn from each square. That represents the best move for each position for t_max = 1. Then we expand each of those values outward one square. We compare mining from the same square twice and moving to a neighbor and mining it once. This represents the best move for t_max = 2. We keep iterating over the matrix until we reach all of our ships at our target t_max.

This finds the best move for one ship and the longer it runs, the longer t_max we solve for. This exploits the fact that the best solution for t_max=N at point P is related to the best solution for t_max = N + 1 at point P's neighbors.

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

# Bot Analysis

## Optimus

* Optimus gives each ship a collection of action options, Eliminates options that are unacceptable, and adds options that automatically will fire
  * always move to an adjacent position with a vulnerable enemy ship.
  * Unacceptable targets seem to be positions that are adjacent to enemy ships with lower halite
* Optimus assigns each ship a target.
* The target will add one or more actions to the action collection of the ship.
* All possible directions are at the back of the list of the action collection.

### Mining Strategy
Optimus rates each possible destination with a score and sets that max score as the destination as the target for that ship.
the score for mining is based on the expected halite per turn assuming a path that goes directly to the
target and then directly to the nearest shipyard. This means that we mine cells that are

* close to us
* close to a shipyard
* have a lot of halite

I'm still trying to understand the equation for number of turns to mine. something about logarithm of carried/remaining ratio and then a lookup table.

All of this seems like a lot of work that doesn't take into account the possibility of simply __not__ returning the halite back to the shipyard.
You can ignore the travel cost to the shipyard if you don't need the money right now to buy more ships. I think this would ignore the value
in mining two medium cells that are adjacent.

This will miss optimal solutions that involve mining clusters of medium halite especially clusters that are far away.

The score for returning the halite is carried_halite/distance. This is logical because a ship should return home if it is very close
(the cost for returning is low), or if is very full (the cost of losing the cargo is high).

I'm guessing scipy.optimize tries to find a unique target for each ship.

### Shipyard Conversion
Only creates a shipyard if
 * There are no shipyards. Chooses the ship with the most halite.
 * danger without escape
 * last turn and we need to return our halite

Doesn't seem to be any planning about where to place the shipyard besides not using a ship that is in danger.

### Combat
* Vulnerable ships present a better destination target (adds on top of halite)
* Do not choose a lower halite ship as a destination
  * This prevents us from trying to mine a square that an enemy ship is already mining.

Enemy ships that are on less than 50 halite seem to be invisible as far as selecting destinations but they do contribute to attack and avoid matrix.

### Ship Production
Always tries to maintain a fleet of 20 ships, near end of game only tries to maintain 15. Why would you __ever__ create a ship after turn 375?

### Vulnerabilities

1. Mining isn't optimal. Doesn't try to mine near clusters. Doesn't try to mine too far from shipyards.
2. If it experiences significant ship loss near the end of game, it will waste tremendous resources to replace ships that will never pay themselves off.
3. Even if there is a giant crowd of enemy ships two squares away, it will happy run toward it until it comes right up to them.
4. No hazard path planning. It just beelines to the target until it notices a hazard. Then it will just move back and forth until it's path clears. It won't even try to choose a different target.
5. No defensive moves to clear hazards or offensive moves to harrass opponents. It pretty much just mines.
6. Prone to distraction. A heavy halite ship could be chased for eternity.

## Swarm

### Mining

This mining algorithm is much greedier. It only considers neighboring cells and if it can't find any with a non-neglible amount of halite, then it just patrols around the map.

I can't even find where it returns the halite to the shipyard.

### Shipyard Conversion

If there are no shipyards and the ship can afford it or enemy forces us into a position with no escape.

### Ship Production
maintain a fleet of 9 or 3 depending on how close to the end of the game we are

### Combat
Avoid hostile ships always

### Vulnerabilities

This is much more primitive than Optimus

1. Mining is nowhere near optimal.
2. Each ship only ever considers neighboring cells
3. No defensive moves or even opportunistic attacks.


## Replay Analysis

We are blue and win in this game.
![image](https://user-images.githubusercontent.com/2446659/89079465-2d58f000-d34c-11ea-9f03-f21822d6517d.png)

We took an early lead because of our second shipyard which made our mining more efficient and allowed us to increase
our fleet size faster than our opponents. The third shipyard and additional ships pushed us even further to the top.

----

We are green and lose this game.
![image](https://user-images.githubusercontent.com/2446659/89079479-38ac1b80-d34c-11ea-9250-96ae17f1d3ed.png)

This game we lost is particularly instructive. The map cannot support more than about 40 ships simultaneously mining.
None of the players realize this and continue to invest in ships for most of the game. The halite starts to deplete and income starts dropping
for all players. Halfway through the game, our average harvest per turn drops to about 40 while we have 25 ships mining so each
ship is averaging about 1.6 halite per turn which means that it takes 312 turns to get a positive return on investment for a ship.
So every single ship built at this point is a loss. The winner is whoever buys the fewest ships at this point. Orange builds the fewest ships
and wins.

The amount of money wasted on buying ships was really tragic. We bought 4 ships after turn 250. We lost by about 2000 halite.
