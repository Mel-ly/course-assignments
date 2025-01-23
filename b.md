# Assignment #B: Dec Mock Exam 大雪前一天
Updated 1649 GMT+8 Dec 5, 2024
2024 fall, Complied by 同学的姓名、院系
说明：
1 ）月考： AC6（请改为同学的通过数） “ ”。考试题目都在 题库（包括计概、数算题目） 里面，
按照数字题号能找到，可以重新提交。作业中提交自己最满意版本的代码和截图。
2）请把每个题目解题思路（可选），源码 Python, 或者 C++（已经在 Codeforces/Openjudge 上
AC），截图（包含 Accepted ），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用
word）。AC 或者没有 AC，都请标上每个题目大致花费时间。
3）提交时候先提交 pdf 文件，再把 md 或者 doc “ ”文件上传到右侧 作业评论 。Canvas 需要有同学清
晰头像、提交文件有 pdf、"作业评论"区有上传的 md 或者 doc 附件。
4）如果不能在截止前提交作业，请写明原因。
## 1. 题目
### E22548: 机智的股民老张
http://cs101.openjudge.cn/practice/22548/
思路：

**初始化**：

- 设定一个变量 `min_price` 来记录到当前为止的最低价格，初始值设为无穷大。
- 设定一个变量 `max_profit` 来记录当前可以获得的最大利润，初始值为 0。

**遍历数组**：

- 对于每个价格：
  - 更新 `min_price`，记录当前为止的最低价格。
  - 计算当前价格与最低价格的差值，作为可能的利润。
  - 更新 `max_profit`，记录到当前为止的最大利润。

**返回结果**：

- 遍历结束后，`max_profit` 即为最大利润。
- 如果价格一直下跌，`max_profit` 会保持为 0。

代码：

```python
def max_profit(prices):
    """
    计算股票的最大利润
    :param prices: List[int]，股票价格列表
    :return: int，最大利润
    """
    min_price = float('inf')  # 初始化最小价格为无穷大
    max_profit = 0  # 初始化最大利润为 0

    for price in prices:
        # 更新最低价格
        if price < min_price:
            min_price = price
        # 更新最大利润
        max_profit = max(max_profit, price - min_price)

    return max_profit


# 输入
prices = list(map(int, input().split()))
# 输出最大利润
print(max_profit(prices))
```
代码运行截图 （至少包含有"Accepted"）

![image-20250122205438953](https://i.postimg.cc/gjxYhVdL/b1.png)

### M28701: 炸鸡排
greedy, http://cs101.openjudge.cn/practice/28701/
思路：

**持续时间的定义**：

- 炸锅最多可以持续的时间取决于炸锅内最慢的一块鸡排的炸熟时间。
- 如果有多块鸡排可以同时放入炸锅，则每次选择剩余炸熟时间最长的 kkk 块鸡排。

**贪心策略**：

- 对鸡排的炸熟时间 t[i] 进行降序排序。
- 模拟炸锅工作，每次从剩余鸡排中选出前 k 块鸡排炸熟。

**二分答案**：

- 通过二分法判断当前假设的最大持续时间是否可行。
- 使用贪心检查是否能将所有鸡排在假设的持续时间内炸熟。

**判断可行性**：

- 累积每块鸡排在当前假设时间下可分配的炸锅次数。
- 判断这些次数能否满足鸡排炸熟的需求。

代码：

```python
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    k = int(data[1])
    times = [int(data[2 + i]) for i in range(n)]
    
    # 计算总炸制时间
    total_time = sum(times)
    
    # 对炸制时间进行排序
    times.sort()
    
    # 初始最大持续时间为总炸制时间除以 k
    max_time = total_time / k
    
    # 如果最长的炸制时间大于或等于 max_time，则需要调整 k 的值
    if times[-1] > max_time:
        for i in range(n - 1, -1, -1):
            if times[i] <= max_time:
                break
            total_time -= times[i]
            k -= 1
            max_time = total_time / k
    
    # 输出结果，保留三位小数
    print(f"{max_time:.3f}")

if __name__ == "__main__":
    main()
```
代码运行截图 （至少包含有"Accepted"）

![image-20250123010544420](https://i.postimg.cc/pr3PJmXG/b2.png)

### M20744: 土豪购物
dp, http://cs101.openjudge.cn/practice/20744/
思路：

使用两个 DP 数组分别处理上述两种情况：

表示dp[i]以第 i 个商品结尾的连续子数组最大和

，不考虑放回商品。

- 状态转移公式：`dp1[i] = max(dp1[i - 1] + a[i], a[i])`

- 解释：当前商品 `a[i]` 要么加入之前的子数组，要么单独成为一个新的子数组。

  dp2[i]表示以第 i 个商品结尾的连续子数组最大和，允许放回其中一个商品。

- 状态转移公式：`dp2[i] = max(dp1[i - 1], dp2[i - 1] + a[i], a[i])`
- 解释：
  - `dp1[i - 1]` 表示选择前面的子数组，但不加当前商品。
  - `dp2[i - 1] + a[i]` 表示当前商品加入到之前可能已经放回一个商品的子数组。
  - `a[i]` 表示单独选择当前商品。

最终答案是 `dp2` 数组中的最大值。因为 `dp2` 考虑了放回一个商品的情况，所以它能提供最大的可能价值

代码：

```python
def kadane(nums):
    max_ending_here = max_so_far = nums[0]
    for x in nums[1:]:
        max_ending_here = max(x, max_ending_here + x)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far

def max_sum_shopping(values):
    # 不放回商品的情况下的最大价值总和
    max_without_deletion = kadane(values)

    # 如果整个数列的和都是负的，则土豪只能选择一个价值最大的商品
    if max_without_deletion < 0:
        return max(values)

    # 准备两个数组来存储从左到右和从右到左的最大子数组和
    left_max_sums = [0] * len(values)
    right_max_sums = [0] * len(values)

    # 从左到右的最大子数组和
    current = 0
    for i in range(len(values)):
        current = max(0, current + values[i])
        left_max_sums[i] = current

    # 从右到左的最大子数组和
    current = 0
    for i in range(len(values) - 1, -1, -1):
        current = max(0, current + values[i])
        right_max_sums[i] = current

    # 放回一个商品时的最大价值总和
    max_with_deletion = 0
    for i in range(1, len(values) - 1):
        max_with_deletion = max(max_with_deletion, left_max_sums[i - 1] + right_max_sums[i + 1])

    # 返回放回一个商品和不放回一个商品两种情况下的最大价值
    return max(max_with_deletion, max_without_deletion)

# 读取输入并处理
values_str = input().strip()
values = list(map(int, values_str.split(',')))
print(max_sum_shopping(values))
```
代码运行截图 （至少包含有"Accepted"）

![image-20250122225632896](https://i.postimg.cc/xjZj24yv/b3.png)

### T25561: 2022 决战双十一
brute force, dfs, http://cs101.openjudge.cn/practice/25561/
思路：分别计算出各家店的金额数再来比较能否进行满减即可,
代码：

```python
n,m=map(int,input().split())#商品数、店铺数

values=[[-1]*n for _ in range(m)]#values[i-1][j-1]表示第i个店铺的第j件商品售价
for good in range(n):
    l=[[int(y) for y in z.split(':')] for z in input().split()]
    for L in l:
        values[L[0]-1][good]=L[1]

coupons=[0]*m#coupons[i]表示第i+1个店铺的优惠卷情况,其实是二维数组
for j in range(m):
    coupons[j]=[[int(y) for y in z.split('-')] for z in input().split()]

strategy=[0]*m#每种方案中每个店铺(0~m-1)应付的cost
output=[]#统计每种可能的购买方案的最终花费
def dfs(i):#考虑购买第i个物品 0~n-1
    global strategy
    if i == n:  # 0~n-1的物品全部购买完毕,统计,输出结果
        result = sum(strategy)
        #在此时使用跨店满减卷，再结算店内满减
        result-=(result//300)*50
        for shop_index in range(m):
            cost = strategy[shop_index]
            max_discount = 0  # 在商店的优惠卷中搜索最大的优惠额度
            for coupon in coupons[shop_index]:
                if coupon[0] <= cost and coupon[1] >= max_discount:
                    max_discount = coupon[1]
            result -= max_discount
        output.append(result)
        return
    for shops in range(m):
        if values[shops][i]>=0:
            strategy[shops]+=values[shops][i]
            dfs(i+1)
            strategy[shops]-=values[shops][i]

dfs(0)
print(min(output))
```
代码运行截图 （至少包含有"Accepted"）

![image-20250122230238590](https://i.postimg.cc/02Jk20cL/b4.png)

### T20741: 两座孤岛最短距离
dfs, bfs, http://cs101.openjudge.cn/practice/20741/
思路：先通过一次DFS遍历第一座孤岛，存储其所有点的坐标，随后使之“沉没”。在扫描到陆地（必属于第二座孤岛）后，计算其与之前存储的所有点间最短路径的长度，取最小值即可。
代码：

```python
from collections import deque


class Solution:
    def shortestBridge(self, grid) -> int:
        m, n = len(grid), len(grid[0])
        points = deque()

        def dfs(points, grid, m, n, i, j):
            if i < 0 or i == m or j < 0 or j == n or grid[i][j] == 2:
                return
            if grid[i][j] == 0:
                points.append((i, j))
                return

            grid[i][j] = 2
            dfs(points, grid, m, n, i - 1, j)
            dfs(points, grid, m, n, i + 1, j)
            dfs(points, grid, m, n, i, j - 1)
            dfs(points, grid, m, n, i, j + 1)

        flag = False
        for i in range(m):
            if flag:
                break
            for j in range(n):
                if grid[i][j] == 1:
                    dfs(points, grid, m, n, i, j)
                    flag = True
                    break

        x, y, count = 0, 0, 0
        while points:
            count += 1
            n_points = len(points)
            while n_points > 0:
                point = points.popleft()
                r, c = point[0], point[1]
                for k in range(4):
                    x, y = r + direction[k], c + direction[k + 1]
                    if x >= 0 and y >= 0 and x < m and y < n:
                        if grid[x][y] == 2:
                            continue
                        if grid[x][y] == 1:
                            return count
                        points.append((x, y))
                        grid[x][y] = 2
                n_points -= 1

        return 0


direction = [-1, 0, 1, 0, -1]

n = int(input())
grid = []
for i in range(n):
    row = list(map(int, list(input())))
    grid.append(row)

print(Solution().shortestBridge(grid))
```
代码运行截图 （至少包含有"Accepted"）

![image-20250122230627220](https://i.postimg.cc/y84VJ3GW/b5.png)

### T28776: 国王游戏
greedy, http://cs101.openjudge.cn/practice/28776
思路：

考虑最后一个大臣，其获得的钱数是所有人左手上的数的乘积（一个定值），除以自己左手和右手的乘积，所以要让他左手右手的乘积最大，所以就有了排序的key。不过这题很奇怪，用floor来取整会wa，用整除就可以。

大臣获得的金币是自己以及前面所有左手乘积除以自己的左右手乘积，从这个思路出发按左右手乘积升序排列即可获得最大值的最小值。不是最后一个是maxmin，反例：国王1 1，三个大臣，6 1;4 2;1 10。

代码：

```python
from typing import List
def Solution(n:int, a:int, b:int, lst:List[List]) -> int:
    lst.sort(key=lambda x: (x[0] * x[1]))
    ans = 0
    for i in range(n):
        ans = max(ans, a // lst[i][1])
        a *= lst[i][0]
    return ans
if __name__ == "__main__":
    n = int(input())
    a, b = map(int, input().split())
    lst = []
    for i in range(n):
        lst.append([int(_) for _ in input().split()])
    # 时间复杂度O(nlogn)，空间复杂度O(n)

    print(Solution(n, a, b, lst))
```
代码运行截图 （至少包含有"Accepted"）

![image-20250122230826638](https://i.postimg.cc/8zRTLP84/b6.png)

## 2. 学习总结和收获
