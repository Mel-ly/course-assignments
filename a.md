# Assignment #A: dp & bfs
Updated 2 GMT+8 Nov 25, 2024
2024 fall, Complied by 同学的姓名、院系
说明：
1）请把每个题目解题思路（可选），源码 Python, 或者 C++（已经在 Codeforces/Openjudge 上
AC），截图（包含 Accepted ），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用
word）。AC 或者没有 AC，都请标上每个题目大致花费时间。
2）提交时候先提交 pdf 文件，再把 md 或者 doc “ ”文件上传到右侧 作业评论 。Canvas 需要有同学清
晰头像、提交文件有 pdf、"作业评论"区有上传的 md 或者 doc 附件。
3）如果不能在截止前提交作业，请写明原因。
## 1. 题目
### LuoguP1255 数楼梯
dp, bfs, https://www.luogu.com.cn/problem/P1255
思路：

如果楼梯有 N 阶，可以有以下两种选择：

- 第一步上 1 阶，然后剩下 N−1 阶。
- 第一步上 2 阶，然后剩下 N−2 阶。

因此，问题可以分解为： f(N)=f(N−1)+f(N−2)

代码：

```python
def count_ways(n):
    if n == 1:
        return 1
    if n == 2:
        return 2

    prev2, prev1 = 1, 2  # f(1) = 1, f(2) = 2
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current

    return prev1

# 输入楼梯数
n = int(input())
# 输出走法总数
print(count_ways(n))
```
代码运行截图 （至少包含有"Accepted"）

![image-20250122194539449](https://i.postimg.cc/nrWhSdT1/a1.png)

### 27528: 跳台阶
dp, http://cs101.openjudge.cn/practice/27528/
思路：

**台阶走法描述**：

- 狄贵可以一次走 1 到 N 级台阶的任意步数。

- 因此，问题可以分解为：

  f(N)=f(N−1)+f(N−2)+⋯+f(1)+1

  - f(N−1),f(N−2),…表示走 N−1,N−2,…,后再一步到顶。
  - +1：直接一步走上 N 级。

**递推公式**：

- 我们可以将递归公式进一步简化： f(N)=2×f(N−1)

**初始条件**：

- f(1)=1，表示只有一种走法。

代码：

```python
def count_ways(n):
    # 初始条件
    if n == 1:
        return 1

    # 动态规划数组
    dp = [0] * (n + 1)
    dp[1] = 1

    # 递推计算
    for i in range(2, n + 1):
        dp[i] = 2 * dp[i - 1]

    return dp[n]

n = int(input())
print(count_ways(n))
```
代码运行截图 （至少包含有"Accepted"）

![image-20250122194852902](https://i.postimg.cc/FH3HGVSj/a2.png)

### 474D. Flowers
dp, https://codeforces.com/problemset/problem/474/D
思路：

定义 f(n) 为长度为 nnn 的有效排列数。

转移方程： f(n)=f(n−1)+f(n−k)

- 如果最后一个是 `R`，则前面为 f(n−1)
- 如果最后 k 个是 `W`，则前面为 f(n−k)

代码：

```python
MOD = 1000000007  # 模数，用于取模避免溢出

def solve(t, k, queries):
    # 预计算 f(n)，表示长度为 n 的有效排列数
    max_n = 100000
    f = [0] * (max_n + 1)  # 初始化 f 数组
    f[0] = 1  # 基础情况：空序列有 1 种方式

    for n in range(1, max_n + 1):
        # 如果当前序列以 R 结尾，则方案数为 f(n-1)
        f[n] = f[n - 1]
        # 如果当前序列以 k 个连续 W 结尾，则加上 f(n-k)
        if n >= k:
            f[n] += f[n - k]
        f[n] %= MOD  # 对结果取模

    # 预计算前缀和 s(n)，便于快速计算区间和
    s = [0] * (max_n + 1)
    for n in range(1, max_n + 1):
        s[n] = (s[n - 1] + f[n]) % MOD

    # 处理每个查询
    results = []
    for a, b in queries:
        # 区间 [a, b] 的结果为 s(b) - s(a-1)
        results.append((s[b] - s[a - 1] + MOD) % MOD)
    return results

# 输入处理
import sys
input = sys.stdin.read
data = input().splitlines()

# 第一行包含 t 和 k
t, k = map(int, data[0].split())
# 接下来的 t 行表示查询
queries = [tuple(map(int, line.split())) for line in data[1:]]

# 计算结果
results = solve(t, k, queries)
# 输出每个查询的结果
print('\n'.join(map(str, results)))
```
代码运行截图 （至少包含有"Accepted"）

![image-20250122200141007](https://i.postimg.cc/k5CXHRNL/a3.png)

### LeetCode5.最长回文子串
dp, two pointers, string, https://leetcode.cn/problems/longest-palindromic/substring/
思路：

- 回文子串以中心为对称轴，可分为两种情况：
- - 回文长度为奇数：中心为一个字符。
  - 回文长度为偶数：中心为两个字符。
- 从每个字符为中心扩展两侧，记录最长回文子串。

代码：

```python
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        def expand_around_center(left, right):
            """
            扩展中心，寻找回文
            :param left: 左指针
            :param right: 右指针
            :return: 当前最长回文的起始和结束索引
            """
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return left + 1, right - 1

        start, end = 0, 0  # 最长回文子串的起始和结束索引
        for i in range(len(s)):
            # 奇数长度回文
            l1, r1 = expand_around_center(i, i)
            # 偶数长度回文
            l2, r2 = expand_around_center(i, i + 1)

            # 更新最长回文子串
            if r1 - l1 > end - start:
                start, end = l1, r1
            if r2 - l2 > end - start:
                start, end = l2, r2

        return s[start:end + 1]
```
代码运行截图 （至少包含有"Accepted"）

![image-20250122195836152](https://i.postimg.cc/25x8Nj2y/a4.png)

### 12029: 水淹七军
bfs, dfs, http://cs101.openjudge.cn/practice/12029/
思路：

- 从所有放水点出发，利用 BFS 或 DFS 标记所有可以被淹没的区域。
- 检查 A 国司令部的位置是否在被淹没的区域内。
- 如果是，输出 `Yes`，否则输出 `No`。

代码：

```c++
#include <bits/stdc++.h>
using namespace std;

const int MAXN = 202;

// 全局变量
int n, m;              // 地形矩阵大小
int vis[MAXN][MAXN];   // 访问标记数组
int a[MAXN][MAXN];     // 高度数组
int rp, cp;            // 司令部位置
int p;                 // 放水点个数

// 上下左右四个方向
const int dr[] = {1, 0, -1, 0};
const int dc[] = {0, 1, 0, -1};

// 深度优先搜索函数
void dfs(int row, int col) {
    vis[row][col] = 1;  // 当前点标记为已访问
    for (int i = 0; i < 4; ++i) {
        int r = row + dr[i];
        int c = col + dc[i];
        // 检查是否可以流向目标点
        if (r >= 1 && r <= m && c >= 1 && c <= n && !vis[r][c] && a[row][col] > a[r][c]) {
            a[r][c] = a[row][col];  // 更新目标点的水面高度
            dfs(r, c);  // 递归搜索
        }
    }
}

int main() {
    int K;  // 数据组数
    cin >> K;

    while (K--) {
        memset(vis, 0, sizeof(vis));  // 重置访问标记数组

        // 输入矩阵大小
        cin >> m >> n;

        // 输入高度数组
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                cin >> a[i][j];
            }
        }

        // 输入司令部位置
        cin >> rp >> cp;

        // 输入放水点个数
        cin >> p;

        // 从每个放水点开始执行 DFS
        while (p--) {
            int t1, t2;
            cin >> t1 >> t2;
            dfs(t1, t2);
        }

        // 判断司令部位置是否被淹
        cout << (vis[rp][cp] ? "Yes" : "No") << endl;
    }

    return 0;
}
```
代码运行截图 （至少包含有"Accepted"）

![image-20250122200716047](https://i.postimg.cc/zBXDzmcj/a5.png)

### 02802: 小游戏
bfs, http://cs101.openjudge.cn/practice/02802/
思路：本问题是路径搜索问题，要求在一个矩形棋盘上，判断两点是否可以通过水平和竖直的线段相连，同时满足路径不能穿过其他卡片。需要找到路径中的最小线段数。

代码：

```c++
#include<iostream>
#include<cstring>
using namespace std;

#define MAXIN 75
char board[MAXIN+2][MAXIN+2];//使用数组定义矩形板
bool mark[MAXIN+2][MAXIN+2];//标记数组，标记走过的格子

int minStep,w,h;
int to[4][2] = {{0,1},{1,0},{0,-1},{-1,0}};//定义四个方向

//(startx,starty)起始位置，(endx,endy)终点位置
//(currentX,currentY)当前位置，step为截止到上一步的路径长度
//f为上一步的方向
void Search(int currentX,int currentY,int endX,int endY,int step,int f){
	if(step > minStep)return;
	
	//递归的终止条件：到达终点
	if(currentX == endX && currentY == endY){
		if(step < minStep)
			minStep = step;//更新最小路径
		return;
	}
	
	//枚举下一步的方向
	for(int i = 0;i < 4;i++){
		//枚举得到新的位置
		int x = currentX + to[i][0];
		int y = currentY + to[i][1];
		
		//如果新位置有效
		if(((x > -1) && (x < w+2) && (y > -1) && (y < h+2)) && (((board[y][x] == ' ') && (mark[y][x] == false)) || (((x==endX) && (y==endY)) && (board[y][x] == 'X')))) {
			mark[y][x] = true;//标记该位置走过
			
			if(f == i)//如果上一步的方向==这一步的方向
				Search(x,y,endX,endY,step,i);
			else
				Search(x,y,endX,endY,step+1,i);
			
			//找完一条路再将其还原
			mark[y][x] = false;
		}	
	}
}

int main(){    
	
	int boardNum = 0;
	while(cin >> w >> h){//输入矩形板宽*长
		
		if(w == 0 && h == 0)break;//矩形板为空跳出
		boardNum++;
		cout << "Board #" << boardNum << ":" << endl;
		
		int i,j;
		
		//将左和上的边缘填空
		for(i=0;i<MAXIN+2;i++)
			board[0][i] = board[i][0] = ' ';
		
		//读入矩形板的布局
		for(i=1;i<=h;i++){
			getchar();//清空缓存区
			for(j=1;j<=w;j++)board[i][j] = getchar();
		}
		
		//将右和下的边缘填空
		for(i=0;i<w;i++)
			board[h+1][i+1] = ' ';
		for(i=0;i<=h;i++)
			board[i+1][w+1] = ' ';
		
		int x1,y1,x2,y2,count = 0;
		while(cin >> x1 >> y1 >> x2 >> y2){
			if(x1 == 0 && y1 == 0 && x2 == 0 && y2 == 0)break;
			count++;
			
			minStep = 100000;//初始化
			memset(mark,false,sizeof(mark));//将maark数组全部设为未被走过
			
			//寻找最短路径
			Search(x1,y1,x2,y2,0,-1);
			
			if(minStep < 100000)
				cout << "Pair " << count << ": " << minStep << " segments."<< endl;
			else
				cout << "Pair " << count << ": impossible."<<endl;
		}
		cout<<endl;
	}
	return 0;
}
```
代码运行截图 （至少包含有"Accepted"）

![image-20250122202130982](https://i.postimg.cc/j5gxCgVr/a6.png)

## 2. 学习总结和收获
