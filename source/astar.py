
import heapq

#spaces are now point objects, actions are now actions
def a_star(get_neighbors, start_position, goal_test, heuristic):
    """ Generic A-Star algorithm.

        get_neighbors: function to explore a point's neighbors
        start_position: Point
        goal_test: function(Point) determines when to stop
        heuristic: function(Points) determines how close it is (distance)

        return: shortest path length to goal or -1 if impossible
    """

    # (path length + h + heurtistic, Point, path length)
    pq = [(0, start_position, 0)]

    explored = set()
    while len(pq) > 0:
        # pop is based off tuple[0]
        search_state = heapq.heappop(pq)
        explored.add(search_state[1])
        
        if goal_test(search_state[1]):
            # reuturn the path length, which is at [2]
            return search_state[2]
        else:
            for point in get_neighbors(search_state[1]):
                if point not in explored:
                    heapq.heappush(pq, (search_state[2]+1 + heuristic(point), point, search_state[2]+1))
    return -1
