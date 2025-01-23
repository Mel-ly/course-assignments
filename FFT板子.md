存个偷来的多项式板子，用过的都说好（赞）

### C++版本

```c++
constexpr int P(998244353), G(3), L(1 << 18);  // 定义常量 P（模数），G（原根），L（FFT长度）

// 增加操作，避免超出 P 的范围
inline void inc(int &x, int y) {
  x += y;
  if (x >= P) x -= P;  // 如果结果大于等于 P，则减去 P
}

// 减少操作，避免小于 0
inline void dec(int &x, int y) {
  x -= y;
  if (x < 0) x += P;  // 如果结果小于 0，则加上 P
}

// 模运算
inline int mod(LL x) { return x % P; }

// 快速幂计算 (x^k % P)，默认为 k = P-2（模逆）
int fpow(int x, int k = P - 2) {
  int r = 1;
  // 快速幂迭代
  for (; k; k >>= 1, x = 1LL * x * x % P) {
    if (k & 1) r = 1LL * r * x % P;  // 如果 k 为奇数，则乘上当前的 x
  }
  return r;
}

// 预处理计算 w, fac, ifac, inv 数组
int w[L], fac[L], ifac[L], inv[L], _ = [] {
  w[L / 2] = 1;
  // 计算 w 数组的值，w 是用于快速傅里叶变换的旋转因子
  for (int i = L / 2 + 1, x = fpow(G, (P - 1) / L); i < L; i++) 
    w[i] = 1LL * w[i - 1] * x % P;
  
  // 对称地填充 w 数组
  for (int i = L / 2 - 1; i >= 0; i--) 
    w[i] = w[i << 1];
  
  // 计算阶乘数组 fac
  fac[0] = 1;
  for (int i = 1; i < L; i++) 
    fac[i] = 1LL * fac[i - 1] * i % P;
  
  // 计算阶乘的逆元数组 ifac
  ifac[L - 1] = fpow(fac[L - 1]);
  for (int i = L - 1; i; i--) {
    ifac[i - 1] = 1LL * ifac[i] * i % P;
    inv[i] = 1LL * ifac[i] * fac[i - 1] % P;
  }
  return 0;
}();

// 进行快速傅里叶变换（DFT）
void dft(int *a, int n) {
  assert((n & n - 1) == 0);  // 确保 n 是 2 的幂
  for (int k = n >> 1; k; k >>= 1) {  // 从 k = n/2 开始，逐步减半
    for (int i = 0; i < n; i += k << 1) {  // i 为每个段的起始点
      for (int j = 0; j < k; j++) {
        int &x = a[i + j], y = a[i + j + k];
        // 计算并更新当前段的元素
        a[i + j + k] = 1LL * (x - y + P) * w[k + j] % P;
        inc(x, y);  // 更新 x 的值
      }
    }
  }
}

// 进行逆快速傅里叶变换（IDFT）
void idft(int *a, int n) {
  assert((n & n - 1) == 0);  // 确保 n 是 2 的幂
  for (int k = 1; k < n; k <<= 1) {  // 从 k = 1 开始，逐步扩大
    for (int i = 0; i < n; i += k << 1) {  // i 为每个段的起始点
      for (int j = 0; j < k; j++) {
        int x = a[i + j], y = 1LL * a[i + j + k] * w[k + j] % P;
        // 计算并更新当前段的元素
        a[i + j + k] = x - y < 0 ? x - y + P : x - y;
        inc(a[i + j], y);  // 更新 a[i + j] 的值
      }
    }
  }
  for (int i = 0, inv = P - (P - 1) / n; i < n; i++)
    a[i] = 1LL * a[i] * inv % P;  // 乘上逆元，完成归一化
  std::reverse(a + 1, a + n);  // 反转数组，完成逆变换
}

// 获取规范化的长度
inline int norm(int n) { return 1 << std::__lg(n * 2 - 1); }

// 定义多项式类 Poly，继承自 std::vector<int>
struct Poly : public std::vector<int> {
#define T (*this)  
  using std::vector<int>::vector;

  // 多项式连接操作
  void append(const Poly &r) {
    insert(end(), r.begin(), r.end());
  }

  // 获取多项式长度
  int len() const { return size(); }

  // 多项式取反操作
  Poly operator-() const {
    Poly r(T);
    for (auto &x : r) x = x ? P - x : 0;  // 如果系数不为 0，取模 P 后取反
    return r;
  }

  // 多项式加法
  Poly &operator+=(const Poly &r) {
    if (r.len() > len()) resize(r.len());  // 调整大小
    for (int i = 0; i < r.len(); i++) inc(T[i], r[i]);  // 对应项相加
    return T;
  }

  // 多项式减法
  Poly &operator-=(const Poly &r) {
    if (r.len() > len()) resize(r.len());  // 调整大小
    for (int i = 0; i < r.len(); i++) dec(T[i], r[i]);  // 对应项相减
    return T;
  }

  // 多项式乘法
  Poly &operator^=(const Poly &r) {
    if (r.len() < len()) resize(r.len());  // 调整大小
    for (int i = 0; i < len(); i++) T[i] = 1LL * T[i] * r[i] % P;  // 对应项相乘
    return T;
  }

  // 多项式常数乘法
  Poly &operator*=(int r) {
    for (int &x : T) x = 1LL * x * r % P;  // 每一项乘以常数
    return T;
  }

  // 多项式加法（返回新对象）
  Poly operator+(const Poly &r) const { return Poly(T) += r; }

  // 多项式减法（返回新对象）
  Poly operator-(const Poly &r) const { return Poly(T) -= r; }

  // 多项式乘法（返回新对象）
  Poly operator^(const Poly &r) const { return Poly(T) ^= r; }

  // 多项式常数乘法（返回新对象）
  Poly operator*(int r) const { return Poly(T) *= r; }

  // 多项式左移
  Poly &operator<<=(int k) { return insert(begin(), k, 0), T; }

  // 多项式左移（返回新对象）
  Poly operator<<(int r) const { return Poly(T) <<= r; }

  // 多项式右移
  Poly operator>>(int r) const { return r >= len() ? Poly() : Poly(begin() + r, end()); }

  // 多项式右移（修改原对象）
  Poly &operator>>=(int r) { return T = T >> r; }

  // 截取多项式的前 k 项
  Poly pre(int k) const { return k < len() ? Poly(begin(), begin() + k) : T; }

  // 对多项式进行 DFT 变换
  friend void dft(Poly &a) { dft(a.data(), a.len()); }

  // 对多项式进行 IDFT 变换
  friend void idft(Poly &a) { idft(a.data(), a.len()); }

  // 多项式卷积（使用 DFT 和 IDFT）
  friend Poly conv(const Poly &a, const Poly &b, int n) {
    Poly p(a), q;
    p.resize(n), dft(p);
    p ^= &a == &b ? p : (q = b, q.resize(n), dft(q), q);
    idft(p);
    return p;
  }

  // 多项式乘法（直接使用 DFT 和 IDFT）
  friend Poly operator*(const Poly &a, const Poly &b) {
    int len = a.len() + b.len() - 1;
    if (a.len() <= 16 || b.len() <= 16) {  // 如果多项式较小，直接使用暴力方法
      Poly c(len);
      for (int i = 0; i < a.len(); i++)
        for (int j = 0; j < b.len(); j++)
          c[i + j] = (c[i + j] + 1LL * a[i] * b[j]) % P;
      return c;
    }
    return conv(a, b, norm(len)).pre(len);  // 否则使用 DFT 和 IDFT
  }
#undef T
};
```

### Python版本

```python
# 定义常量 P、G、L
P = 998244353
G = 3
L = 1 << 18

# 增加操作，避免超出 P 的范围
def inc(x, y):
    x += y
    if x >= P:
        x -= P
    return x

# 减少操作，避免小于 0
def dec(x, y):
    x -= y
    if x < 0:
        x += P
    return x

# 取模操作
def mod(x):
    return x % P

# 快速幂计算 (x^k % P)，默认为 k = P-2（模逆）
def fpow(x, k=P-2):
    r = 1
    while k:
        if k & 1:
            r = (r * x) % P
        x = (x * x) % P
        k >>= 1
    return r

# 初始化常量 w, fac, ifac, inv 等
w = [0] * L
fac = [0] * L
ifac = [0] * L
inv = [0] * L

# 预处理 w, fac, ifac, inv 数组
def init():
    w[L // 2] = 1
    # 计算 w 数组的值
    x = fpow(G, (P - 1) // L)
    for i in range(L // 2 + 1, L):
        w[i] = (w[i - 1] * x) % P
    for i in range(L // 2 - 1, -1, -1):
        w[i] = w[i * 2]

    fac[0] = 1
    # 计算阶乘数组 fac
    for i in range(1, L):
        fac[i] = (fac[i - 1] * i) % P

    # 计算阶乘的逆元数组 ifac
    ifac[L - 1] = fpow(fac[L - 1])
    for i in range(L - 1, 0, -1):
        ifac[i - 1] = (ifac[i] * i) % P

    # 计算逆元数组 inv
    for i in range(1, L):
        inv[i] = (ifac[i] * fac[i - 1]) % P

# 进行快速傅里叶变换（DFT）
def dft(a, n):
    assert (n & (n - 1)) == 0  # 确保 n 是 2 的幂
    for k in range(n >> 1, 0, k >> 1):
        for i in range(0, n, k << 1):
            for j in range(k):
                x = a[i + j]
                y = a[i + j + k]
                a[i + j + k] = ((x - y + P) * w[k + j]) % P
                a[i + j] = (x + y) % P

# 进行逆快速傅里叶变换（IDFT）
def idft(a, n):
    assert (n & (n - 1)) == 0  # 确保 n 是 2 的幂
    for k in range(1, n, k << 1):
        for i in range(0, n, k << 1):
            for j in range(k):
                x = a[i + j]
                y = (a[i + j + k] * w[k + j]) % P
                a[i + j + k] = (x - y + P) % P
                a[i + j] = (x + y) % P

    inv_n = fpow(n)  # 求 n 的模逆
    for i in range(n):
        a[i] = (a[i] * inv_n) % P
    a.reverse()

# 获取规范化的长度
def norm(n):
    return 1 << (n * 2 - 1).bit_length()

# 定义多项式类 Poly
class Poly(list):
    def append(self, r):
        self.extend(r)

    def len(self):
        return len(self)

    def __neg__(self):
        return Poly([P - x if x != 0 else 0 for x in self])

    def __iadd__(self, r):
        if r.len() > self.len():
            self.extend([0] * (r.len() - self.len()))
        for i in range(r.len()):
            self[i] = inc(self[i], r[i])
        return self

    def __isub__(self, r):
        if r.len() > self.len():
            self.extend([0] * (r.len() - self.len()))
        for i in range(r.len()):
            self[i] = dec(self[i], r[i])
        return self

    def __ixor__(self, r):
        if r.len() < self.len():
            self.extend([0] * (self.len() - r.len()))
        for i in range(len(self)):
            self[i] = (self[i] * r[i]) % P
        return self

    def __imul__(self, r):
        for i in range(len(self)):
            self[i] = (self[i] * r) % P
        return self

    def __add__(self, r):
        return Poly(self) + r

    def __sub__(self, r):
        return Poly(self) - r

    def __xor__(self, r):
        return Poly(self) ^ r

    def __mul__(self, r):
        return Poly(self) * r

    def __lshift__(self, k):
        self = [0] * k + self
        return self

    def __rshift__(self, r):
        if r >= len(self):
            return Poly()
        return Poly(self[r:])

    def pre(self, k):
        return self[:k]

    def deriv(self):
        if not self:
            return Poly()
        return Poly([(i * self[i]) % P for i in range(1, len(self))])

    def integ(self):
        if not self:
            return Poly()
        return Poly([0] + [(fpow(i + 1) * self[i]) % P for i in range(len(self))])

    def inv(self, m):
        x = [fpow(self[0])]
        for k in range(1, m, k * 2):
            x.append((-conv(self[:k * 2], x, k * 2) >> k) * x).pre(k)
        return x.pre(m)

    def log(self, m):
        return (self.deriv() * self.inv(m)).integ().pre(m)

# 多项式乘法（直接使用傅里叶变换）
def conv(a, b, n):
    p = Poly(a)
    p.resize(n)
    dft(p)
    q = Poly(b)
    q.resize(n)
    dft(q)
    q ^= p
    idft(p)
    return p

# 测试例子
init()
```

