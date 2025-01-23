# Assignment #D: 十全十美
Updated 1254 GMT+8 Dec 17, 2024
2024 fall, Complied by 同学的姓名、院系
说明：
1）请把每个题目解题思路（可选），源码 Python, 或者 C++（已经在 Codeforces/Openjudge 上
AC），截图（包含 Accepted ），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用
word）。AC 或者没有 AC，都请标上每个题目大致花费时间。
2）提交时候先提交 pdf 文件，再把 md 或者 doc “ ”文件上传到右侧 作业评论 。Canvas 需要有同学清
晰头像、提交文件有 pdf、"作业评论"区有上传的 md 或者 doc 附件。
3）如果不能在截止前提交作业，请写明原因。
## 1. 题目
### 02692: 假币问题
brute force, http://cs101.openjudge.cn/practice/02692
思路：只有一个是假币，因此我们可以考虑让每个都当一次假币，然后假设他是重的还是轻的，验证就可以了。
代码：

```python
n = int(input())

def check(coins, case):
    for item in case:
        left, right, res = item.split()

        left_total = sum(coins[i] for i in left)
        right_total = sum(coins[i] for i in right)

        if left_total == right_total and res != 'even':
            return False
        elif left_total < right_total and res != 'down':
            return False
        elif left_total > right_total and res != 'up':
            return False

    return True

for _ in range(n):
    case = [input().strip() for _ in range(3)]

    for counterfeit in 'ABCDEFGHIJKL':
        found = False
        for weight in [-1, 1]:
            coins = {coin: 0 for coin in 'ABCDEFGHIJKL'}
            coins[counterfeit] = weight

            if check(coins, case):
                found = True
                tag = "light" if weight == -1 else "heavy"
                print(f'{counterfeit} is the counterfeit coin and it is {tag}.')
                break
        if found:
            break
```
代码运行截图 （至少包含有"Accepted"）

![image-20250123002001714](https://i.postimg.cc/sxGrx39p/d1.png)

### 01088: 滑雪
dp, dfs similar, http://cs101.openjudge.cn/practice/01088
思路：可以用 sort 代替 heapq，因为 heapq 的主要优势是动态获取最小最大值，而这里并不需要动态插入或删除元素。我们只需按照高度从小到大处理所有点，使用排序即可实现相同的逻辑。
代码：

```python
rows, cols = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(rows)]

# 将所有点按高度从小到大排序
points = sorted([(matrix[i][j], i, j) for i in range(rows) for j in range(cols)])

# 每个点的L值初始化为1
dp = [[1] * cols for _ in range(rows)]

# 定义方向数组，用于遍历上下左右
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# 记录最长递增路径长度
longest_path = 1

# 从低到高，前面的不会对后面造成影响！
for height, x, y in points:
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and matrix[nx][ny] < height:
            dp[x][y] = max(dp[x][y], dp[nx][ny] + 1)
    longest_path = max(longest_path, dp[x][y])

print(longest_path)
```
代码运行截图 （至少包含有"Accepted"）

![image-20250123002259195](https://i.postimg.cc/zXWyPNZb/d2.png)

### 25572: 螃蟹采蘑菇
bfs, dfs, http://cs101.openjudge.cn/practice/25572/
思路：bfs,先遍历矩阵找到两个起点，再找出两者关系，之后只对其中一个进行重点关注
代码：

```python
from collections import deque

# 定义四个方向：右、下、左、上
dire = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def bfs(a, x1, y1, x2, y2):
    visit = set()  # 使用集合来避免重复访问
    queue = deque([(x1, y1, x2, y2)])
    visit.add((x1, y1, x2, y2))  # 初始点加入访问集合

    while queue:
        xa, ya, xb, yb = queue.popleft()
        # 遍历四个方向
        for xi, yi in dire:
            # 计算新位置
            nx1, ny1 = xa + xi, ya + yi
            nx2, ny2 = xb + xi, yb + yi

            # 判断新位置是否合法
            if 0 <= nx1 < a and 0 <= ny1 < a and 0 <= nx2 < a and 0 <= ny2 < a:
                if (nx1, ny1, nx2, ny2) not in visit and Matrix[nx1][ny1] != 1 and Matrix[nx2][ny2] != 1:
                    # 加入队列并标记访问
                    queue.append((nx1, ny1, nx2, ny2))
                    visit.add((nx1, ny1, nx2, ny2))
                    # 检查是否到达目标
                    if Matrix[nx1][ny1] == 9 or Matrix[nx2][ny2] == 9:
                        return True
    return False

# 读取输入
a = int(input())
Matrix = [list(map(int, input().split())) for _ in range(a)]

# 找到第一个和第二个 '5' 的位置
x1, y1, x2, y2 = -1, -1, -1, -1
found_first = False

for i in range(a):
    for j in range(a):
        if Matrix[i][j] == 5:
            if not found_first:
                x1, y1 = i, j
                Matrix[i][j] = 0  # 标记为已访问
                found_first = True
            else:
                x2, y2 = i, j
                Matrix[i][j] = 0  # 标记为已访问
                break
    if x2 != -1:  # 如果第二个 5 已经找到
        break

# 运行 BFS 检查是否可以从 (x1, y1) 到 (x2, y2)
check = bfs(a, x1, y1, x2, y2)
print('yes' if check else 'no')
```
代码运行截图 （至少包含有"Accepted"）

![image-20250123002345742](https://i.postimg.cc/RZBNkY4P/d3.png)

### 27373: 最大整数
dp, http://cs101.openjudge.cn/practice/27373/
思路：先对数据进行预处理，再进行dp，由于一个数只能用一次，所以添加了一个temp来缓存。可以用滚动数组反向更新优化temp。
代码：

```python
# 谭宇睿 工学院
limit = int(input())
n = int(input())
lst = list(map(str, input().split()))


def sorting(a, b):
    if int(a + b) > int(b + a):
        return True
    else:
        return False


for i in range(n):
    for j in range(i, n):
        if not sorting(lst[i], lst[j]):
            lst[i], lst[j] = lst[j], lst[i]

dp = [0] * (limit + 1)  # dp[i]表示i位数时最大值
temp = [0] * (limit + 1)
for j in lst:
    for i in range(len(j), limit + 1):
        dp[i] = max(dp[i], int(str(temp[i - len(j)]) + j))
    temp[:] = dp
print(dp[limit])
```
代码运行截图 （至少包含有"Accepted"）

![image-20250123002625139](https://i.postimg.cc/v8C1zBjw/d4.png)

### 02811: 熄灯问题
brute force, http://cs101.openjudge.cn/practice/02811
思路：A 记开关实时亮暗，B记开关操作与否。一旦确定第1行的操作，剩 余第2、3、4、5 行的操作就确定;枚举第1行(共2^6^=64 种可能)小知识:
代码：

```python
X = [[0,0,0,0,0,0,0,0]]
Y = [[0,0,0,0,0,0,0,0]]
for _ in range(5):
    X.append([0] + [int(x) for x in input().split()] + [0])
    Y.append([0 for x in range(8)])    
X.append([0,0,0,0,0,0,0,0])
Y.append([0,0,0,0,0,0,0,0])

import copy
for a in range(2):
    Y[1][1] = a
    for b in range(2):
        Y[1][2] = b
        for c in range(2):
            Y[1][3] = c
            for d in range(2):
                Y[1][4] = d
                for e in range(2):
                    Y[1][5] = e
                    for f in range(2):
                        Y[1][6] = f
                        
                        A = copy.deepcopy(X)
                        B = copy.deepcopy(Y)
                        for i in range(1, 7):
                            if B[1][i] == 1:
                                A[1][i] = abs(A[1][i] - 1)
                                A[1][i-1] = abs(A[1][i-1] - 1)
                                A[1][i+1] = abs(A[1][i+1] - 1)
                                A[2][i] = abs(A[2][i] - 1)
                        for i in range(2, 6):
                            for j in range(1, 7):
                                if A[i-1][j] == 1:
                                    B[i][j] = 1
                                    A[i][j] = abs(A[i][j] - 1)
                                    A[i-1][j] = abs(A[i-1][j] - 1)
                                    A[i+1][j] = abs(A[i+1][j] - 1)
                                    A[i][j-1] = abs(A[i][j-1] - 1)
                                    A[i][j+1] = abs(A[i][j+1] - 1)
                        if A[5][1]==0 and A[5][2]==0 and A[5][3]==0 and A[5][4]==0 and A[5][5]==0 and A[5][6]==0:
                            for i in range(1, 6):
                                print(" ".join(repr(y) for y in [B[i][1],B[i][2],B[i][3],B[i][4],B[i][5],B[i][6] ]))
```
代码运行截图 （至少包含有"Accepted"）

![image-20250123002643818](https://i.postimg.cc/pLJpRBdP/d5.png)

### 08210: 河中跳房子
binary search, greedy, http://cs101.openjudge.cn/practice/08210/
思路：

1、问题本质:
通过移除不超过 M 个岩石，调整从起点到终点的跳跃间距，使得所有跳跃中的最短跳跃距离最大化。
2、二分答案:

- 我们将最短跳跃距离记为 D，然后通过二分法查找最大的 D
- 二分范围:从1到 工。
- 判定条件:给定一个 D，判断是否可以移除最多 M 个岩石，使得所有相邻岩石的间距都不小于 D。

3、判定逻辑:

- 从起点开始遍历岩石，尝试保证每一段距离都至少为 D
- 如果当前岩石与上一个被保留的岩石的距离小于 D，移除该岩石。
- 统计移除的岩石数量，，若不超过 M，则当前 D 是可行的。

4、实现细节:

- 按距离排序的岩石数组，包括起点0和终点L。
- 使用二分查找，逐步缩小范围，找到最大的 D

代码：

```python
def max_min_jump(L, N, M, rocks):
    # 加入起点和终点，并排序
    rocks = [0] + sorted(rocks) + [L]
    
    def can_achieve(D):
        """
        判断是否可以在移除最多 M 个岩石的情况下，保证最小跳跃距离不小于 D。
        """
        removed = 0  # 移除的岩石数
        last_position = 0  # 上一个被保留的岩石位置
        
        for i in range(1, len(rocks)):
            # 当前岩石与上一个保留岩石的距离
            if rocks[i] - rocks[last_position] < D:
                # 如果距离小于 D，则移除当前岩石
                removed += 1
                if removed > M:
                    return False
            else:
                # 保留当前岩石，更新上一个保留的位置
                last_position = i
        
        return True

    # 二分查找
    left, right = 1, L
    answer = 0

    while left <= right:
        mid = (left + right) // 2
        if can_achieve(mid):
            answer = mid  # 记录可行的最小跳跃距离
            left = mid + 1  # 尝试更大的距离
        else:
            right = mid - 1  # 尝试更小的距离

    return answer


# 输入处理
L, N, M = map(int, input().split())
rocks = [int(input()) for _ in range(N)]

# 输出结果
print(max_min_jump(L, N, M, rocks))
```
代码运行截图 （至少包含有"Accepted"）

![image-20250123003140872](https://i.postimg.cc/0N5rQJ6D/d6.png)

## 2. 学习总结和收获
