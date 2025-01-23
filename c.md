# Assignment #C: 五味杂陈
Updated 1148 GMT+8 Dec 10, 2024
2024 fall, Complied by 同学的姓名、院系
说明：
1）请把每个题目解题思路（可选），源码 Python, 或者 C++（已经在 Codeforces/Openjudge 上
AC），截图（包含 Accepted ），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用
word）。AC 或者没有 AC，都请标上每个题目大致花费时间。
2）提交时候先提交 pdf 文件，再把 md 或者 doc “ ”文件上传到右侧 作业评论 。Canvas 需要有同学清
晰头像、提交文件有 pdf、"作业评论"区有上传的 md 或者 doc 附件。
3）如果不能在截止前提交作业，请写明原因。

## 1. 题目
### 1115. 取石子游戏
dfs, https://www.acwing.com/problem/content/description/1117/
思路：不妨设α > b。如果a/6 < 2，则第一步走法是唯一的。如果Ыa，则先手必胜(先手直接将a取完即可)。下面证明，如果a/6 >2则先手必胜。考虑(a mod b,b)和(b+a mod b,b)这两个状态可以证明这两个状态一定不能都是先手必胜态，因为如果(b+a mod b,b)是必胜态，但是下一步的走法唯一，所以(a mod b,b)是必败态。所以两个状态必然其一是必败态，从而a> 2b的时候先手必胜。余下的情况直接递归只解决即可。
代码：

```c++
#include <iostream>
using namespace std;

bool dfs(int a, int b) {
  if (a / b >= 2 || a == b) return true;
  return !dfs(b, a - b);
}

int main() {
  int a, b;
  while (cin >> a >> b, a | b) {
    if (a < b) swap(a, b);
    dfs(a, b) ? puts("win") : puts("lose");
  }
}
```
代码运行截图 （至少包含有"Accepted"）

![image-20250122231308784](https://i.postimg.cc/mDDRxYPp/c1.png)

### 25570: 洋葱
Matrices, http://cs101.openjudge.cn/practice/25570
思路：将Matrix中的每个数归到第某圈洋葱的和即可
代码：

```python
from math import ceil
n = int(input()) 
matrix = [0 for _ in range(n)] 
for i in range(n): 
    matrix[i] = [int(_) for _ in input().split()] 
ans = [0] * ceil(n/2) 
for i in range(n): 
    for j in range(n): 
        ans[min(i, j, n-1-i, n-1-j)] += matrix[i][j] 
print(max(ans))
```
代码运行截图 （至少包含有"Accepted"）

![image-20250122232341263](https://i.postimg.cc/5tfbgdbV/c2.png)

### 1526C1. Potions(Easy Version)
greedy, dp, data structures, brute force, *1500, https://codeforces.com/problemset/problem/1526/C1
思路：

1.贪心 + 小顶堆策略:
从左到右模拟喝药水的过程，尽量多喝药水，但确保健康值(health)始终非负。使用**小顶堆(min-heap)**记录喝过的负值药水，方便在健康值不足时移除最小的负值药水(即最“伤害”的药水)如果当前药水会导致健康值变为负值，移除堆中对健康值影响最大的负值药水，这相当于撤销之前的选择。
2、实现步骤:遍历每个药水:如果喝当前药水后健康值仍非负，则直接喝。如果健康值变负，则从堆中移除影响最大的负值药水(堆顶)记录喝下的药水总数，最终输出。
3、时间复杂度:
堆的插入和删除操作是 O(logn)。
总共有 几 个药水，整体复杂度为 O(n logn)。

代码：

```python
import heapq

def max_potions(n, a):
    health = 0  # 当前健康值
    count = 0   # 喝下的药水数量
    min_heap = []  # 小顶堆，用于存储负值药水

    for potion in a:
        health += potion  # 假设喝下当前药水
        heapq.heappush(min_heap, potion)  # 将药水加入堆中
        count += 1  # 喝下药水数量加1

        # 如果健康值变为负数，则移除最伤害的药水
        if health < 0:
            health -= heapq.heappop(min_heap)  # 移除堆顶药水
            count -= 1  # 喝下药水数量减1
    
    return count

n = int(input())
a = list(map(int, input().split()))

print(max_potions(n, a))
```
代码运行截图 （至少包含有"Accepted"）

![image-20250122233144683](https://i.postimg.cc/SKVqtTxz/c3.png)

### 22067: 快速堆猪
辅助栈，http://cs101.openjudge.cn/practice/22067/
思路：可以创建两个栈:一个用于存储所有的元素，另一个用于追踪最小值。当向主栈中 push 元素时，同时检查是否需要更新最小值栈;当从主栈中 pop元素时，也检査是否需要从最小值栈中移除相应的最小值。
代码：

```python
class MinStack:
    def __init__(self):
        self.stack = []          # 主栈
        self.min_stack = []      # 辅助栈，用来保存每个状态下的最小值

    def push(self, x):
        self.stack.append(x)
        if not self.min_stack or x <= self.min_stack[-1]:
            self.min_stack.append(x)

    def pop(self):
        if self.stack:  # 如果主栈非空才执行pop
            top = self.stack.pop()
            if top == self.min_stack[-1]:
                self.min_stack.pop()

    def min(self):
        if self.min_stack:
            return self.min_stack[-1]
        else:
            return None  # 栈为空时返回None

# 使用 MinStack 类
min_stack = MinStack()

while True:
    try:
        command = input().strip()
        if command.startswith('push'):
            value = int(command.split()[1])
            min_stack.push(value)
        elif command.startswith('pop'):
            min_stack.pop()  # 空栈时不进行任何操作
        elif command.startswith('min'):
            min_value = min_stack.min()
            if min_value is not None:
                print(min_value)  # 只有在栈非空时才打印最小值
    except EOFError:
        break
```
代码运行截图 （至少包含有"Accepted"）

![image-20250122232555404](https://i.postimg.cc/sXNz3MSj/c4.png)

### 20106: 走山路
Dijkstra, http://cs101.openjudge.cn/practice/20106/
思路：

**图的表示**：

- 地图是一个 m×n 的二维矩阵，其中高度是权重，`"#"` 表示无法通过。

**最短路径问题**：

- 每次移动时，体力消耗是当前高度和目标高度的差的绝对值。
- 我们需要找到起点到终点的最短路径。

**处理特殊情况**：

- 如果起点或终点是 `"#"`，直接输出 `NO`。
- 如果起点和终点相同，输出 `0`。

**使用 Dijkstra 算法**：

- 用优先队列（小顶堆）实现。
- 初始化一个二维数组 `dist`，记录从起点到每个点的最小体力消耗。
- 起点的 `dist` 初始化为 `0`，其他点初始化为无穷大。
- 按照体力消耗的最小值依次扩展节点。

**边界条件**：

- 检查上下左右的移动是否超出地图边界或移动到 `"#"` 位置。

代码：

```python
import heapq

def min_effort(m, n, terrain, tests):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右方向
    results = []

    def dijkstra(start, end):
        if terrain[start[0]][start[1]] == "#" or terrain[end[0]][end[1]] == "#":
            return "NO"

        dist = [[float('inf')] * n for _ in range(m)]  # 最短路径表
        dist[start[0]][start[1]] = 0
        pq = [(0, start[0], start[1])]  # (当前体力消耗, 当前行, 当前列)

        while pq:
            cost, x, y = heapq.heappop(pq)

            # 如果到达终点
            if (x, y) == end:
                return cost

            # 如果当前点的体力消耗已经超过记录，跳过
            if cost > dist[x][y]:
                continue

            # 扩展上下左右的邻居节点
            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if 0 <= nx < m and 0 <= ny < n and terrain[nx][ny] != "#":
                    new_cost = cost + abs(int(terrain[nx][ny]) - int(terrain[x][y]))
                    if new_cost < dist[nx][ny]:
                        dist[nx][ny] = new_cost
                        heapq.heappush(pq, (new_cost, nx, ny))

        return "NO"  # 如果无法到达终点

    for start_row, start_col, end_row, end_col in tests:
        result = dijkstra((start_row, start_col), (end_row, end_col))
        results.append(result)

    return results

m, n, p = map(int, input().split())
terrain = [input().split() for _ in range(m)]
tests = [tuple(map(int, input().split())) for _ in range(p)]

results = min_effort(m, n, terrain, tests)

for res in results:
    print(res)
```
代码运行截图 （至少包含有"Accepted"）

![image-20250122233413329](https://i.postimg.cc/nVXjB731/c5.png)

### 04129: 变换的迷宫
bfs, http://cs101.openjudge.cn/practice/04129/
思路：

**多维状态表示**：

- 每个状态需要表示当前位置 (x,y)(x, y)(x,y) 和当前时间 ttt。用三元组 (x,y,t)(x, y, t)(x,y,t) 表示 BFS 的节点状态。

**能否走过石头**：

- 当时间 ttt 是 KKK 的倍数时，石头消失，可以走到石头所在的位置。
- 在其他时间，石头无法通过。

**BFS 实现**：

- 从起点开始，初始化队列。
- 每次扩展时，将上下左右四个方向的可达位置加入队列。
- 用一个三维数组 `visited[x][y][t % K]` 表示在特定时间状态下是否访问过该位置，避免重复搜索。

**终止条件**：

- 如果某个时刻到达终点，输出当前花费的时间。
- 如果 BFS 遍历完所有可能状态仍未找到路径，输出 `"Oop!"`。

**特殊情况**：

- 如果起点和终点位置是石头，则需要检查初始时间是否可以直接通过。

代码：

```python
import heapq
from math import inf

# 四个基本方向：右、下、左、上
DIRECTIONS = [(0, 1), (1, 0), (-1, 0), (0, -1)]

def find_shortest_path(grid, start, end, dimensions, cycle_length):
    """
    寻找从起点到终点的最短路径。
    
    :param grid: 地图信息（0为空地，1为石头）
    :param start: 起点坐标 (x, y)
    :param end: 终点坐标 (x, y)
    :param dimensions: 地图尺寸 (rows, cols)
    :param cycle_length: 穿过石头的时间周期
    :return: 最短时间或 "Oop!" 表示无法到达
    """
    rows, cols = dimensions
    visited = [[[False] * cols for _ in range(rows)] for _ in range(cycle_length)]
    priority_queue = [(0,) + start]  # 初始时间为0加上起点坐标
    
    while priority_queue:
        time, x, y = heapq.heappop(priority_queue)
        if (x, y) == end:  # 到达终点
            return time
        
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            new_time = time + 1
            
            if not (0 <= nx < rows and 0 <= ny < cols):  # 检查是否在地图内
                continue
                
            if grid[nx][ny] == 1 and new_time % cycle_length == 0 and not visited[new_time % cycle_length][nx][ny]:  # 穿石头
                visited[new_time % cycle_length][nx][ny] = True
                heapq.heappush(priority_queue, (new_time, nx, ny))
            elif grid[nx][ny] == 0 and not visited[new_time % cycle_length][nx][ny]:  # 普通空地
                visited[new_time % cycle_length][nx][ny] = True
                heapq.heappush(priority_queue, (new_time, nx, ny))
                
    return "Oop!"

def main():
    test_cases = int(input())
    results = []
    
    for _ in range(test_cases):
        rows, cols, cycle_length = map(int, input().split())
        grid = []
        start = None
        end = None
        
        for i in range(rows):
            line = input()
            row = []
            for j, char in enumerate(line):
                if char == "S":
                    start = (i, j)
                    row.append(0)  # 起点当空地
                elif char == "E":
                    end = (i, j)
                    row.append(0)  # 终点当空地
                elif char == "#":
                    row.append(1)  # 石头
                else:
                    row.append(0)  # 空地
            grid.append(row)
            
        result = find_shortest_path(grid, start, end, (rows, cols), cycle_length)
        results.append(result)

    for result in results:
        print(result)

if __name__ == "__main__":
    main()
```
代码运行截图 （至少包含有"Accepted"）

![image-20250123001605957](https://i.postimg.cc/6TsNGFYF/c6.png)

## 2. 学习总结和收获
