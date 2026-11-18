// 固定核 GP 的全共形候选反演：热循环加速器
//
// 为什么要 C++
// ------------
// 单次 GP 拟合（n<=24，Matern 5/2 + 白噪声）在 numpy 里约 16.6 µs，其中大半是
// scipy 的调用开销，用 C++ 改写收益极小。真正贵的是全共形区间：每次要在候选
// 标签网格上反复重建 (n+1)x(n+1) 协方差并做 Cholesky，典型 65 点扫描 + 每侧
// 约 50 步二分，合计约 420 次小规模分解，实测 6.9 ms/区间。
//
// 报告需要覆盖率的置信区间，那意味着几十万次区间调用（2000 次自助抽样 x 264
// 个单元格）。纯 Python 路径要一小时以上，因此这一个热循环值得用 C++ 重写。
// 其余部分保持在 Python 里，便于阅读与修改。
//
// 数学约定（必须与 _tmp/algo_explore/results/gp2d_core.py 完全一致）
// ------------------------------------------------------------------
//   核：   k(x,x') = signal * f(s),  s = sqrt(5) * ||(x-x')/tau_vec||
//          f(s) = (1 + s + s^2/3) * exp(-s)              （Matern 5/2）
//   训练协方差：K = k(Z,Z) + (noise + jitter) * I        （对角线含观测噪声）
//   预测行：用无噪声的 k(Z_test, Z)                       （不能用 K，否则 K@K^-1 z = z）
//   共形象：rank k = ceil((n+1)(1-alpha))，候选与其它 n 个分数比较
//   接受条件：|z_cand - fit_cand| <= 第 k 小的 |z_i - fit_i|
//
// 输入输出都在"已标准化"的坐标里（X 与 y 的标准化由 Python 侧的冻结 scaler
// 负责），本文件只做核与反演，不做标准化，以免两处口径漂移。
//
// 编译见 code/tools/fast_conformal.py；未编译时 Python 自动回退到 numpy 路径。

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

namespace {

constexpr double kSqrt5 = 2.2360679774997896;

// Matern 5/2 相关函数，参数 s = sqrt(5) * 归一化距离
inline double matern52(double s) {
    const double s2 = s * s;
    return (1.0 + s + s2 / 3.0) * std::exp(-s);
}

// 原地 Cholesky（下三角），返回 false 表示矩阵非正定
bool cholesky(std::vector<double>& a, int n) {
    for (int i = 0; i < n; ++i) {
        double* row_i = &a[(size_t)i * n];
        for (int j = 0; j <= i; ++j) {
            double* row_j = &a[(size_t)j * n];
            double sum = row_i[j];
            for (int k = 0; k < j; ++k) sum -= row_i[k] * row_j[k];
            if (i == j) {
                if (sum <= 0.0) return false;
                row_i[i] = std::sqrt(sum);
            } else {
                row_i[j] = sum / row_j[j];
            }
        }
    }
    return true;
}

// 用 Cholesky 因子解 L L^T x = b（原地返回 x）
void cholesky_solve(const std::vector<double>& l, int n, std::vector<double>& b) {
    for (int i = 0; i < n; ++i) {
        double sum = b[i];
        const double* row_i = &l[(size_t)i * n];
        for (int k = 0; k < i; ++k) sum -= row_i[k] * b[k];
        b[i] = sum / row_i[i];
    }
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int k = i + 1; k < n; ++k) sum -= l[(size_t)k * n + i] * b[k];
        b[i] = sum / l[(size_t)i * n + i];
    }
}

// 两点之间的归一化距离（按每维长度尺度）
inline double scaled_distance(const double* x, const double* y, const double* tau, int d) {
    double acc = 0.0;
    for (int k = 0; k < d; ++k) {
        const double delta = (x[k] - y[k]) / tau[k];
        acc += delta * delta;
    }
    return std::sqrt(acc);
}

// 给定候选标签 z_cand，返回 (候选残差 - 第 k 小训练残差)
// z_in 为 n*d 输入坐标，z_y 为 n 个标准化的训练响应，两者必须分开传，
// 混用会把响应值当坐标（本文件曾是此错误的来源，对拍即暴露）。
// 返回 false 表示协方差非正定
bool margin(const std::vector<double>& z_in, const std::vector<double>& z_y,
            const double* z_test, int n, int d,
            const double* tau, double signal, double noise, double z_cand,
            int rank_k, std::vector<double>& scratch_k, std::vector<double>& scratch_z,
            std::vector<double>& scratch_fit, double* out_margin) {
    const int m = n + 1;
    // 组装增广协方差：前 n 个为训练点，第 n 个为测试点
    scratch_k.assign((size_t)m * m, 0.0);
    for (int i = 0; i < n; ++i) {
        const double* xi = &z_in[(size_t)i * d];
        for (int j = 0; j <= i; ++j) {
            const double* xj = &z_in[(size_t)j * d];
            const double val = signal * matern52(kSqrt5 * scaled_distance(xi, xj, tau, d));
            scratch_k[(size_t)i * m + j] = val;
            scratch_k[(size_t)j * m + i] = val;
        }
        scratch_k[(size_t)i * m + i] += noise;   // 观测噪声只加在对角线
    }
    for (int i = 0; i < n; ++i) {
        const double* xi = &z_in[(size_t)i * d];
        const double val = signal * matern52(kSqrt5 * scaled_distance(xi, z_test, tau, d));
        scratch_k[(size_t)n * m + i] = val;
        scratch_k[(size_t)i * m + n] = val;
    }
    scratch_k[(size_t)n * m + n] = signal + noise;

    // 预测要用无噪声的格拉姆行，先保存对角线上的噪声量以便之后扣除
    const double noise_total = noise;

    // 解 K alpha = z_aug
    scratch_z.assign((size_t)m, 0.0);
    std::memcpy(scratch_z.data(), z_y.data(), sizeof(double) * n);
    scratch_z[n] = z_cand;
    std::vector<double> chol = scratch_k;
    if (!cholesky(chol, m)) return false;
    cholesky_solve(chol, m, scratch_z);

    // 拟合值 = 无噪声格拉姆行 @ alpha。K 的对角线含观测噪声，因此每个分量的
    // K[i][:]@alpha 都要扣掉 noise*alpha_i；测试点也加过噪声（gram_train 对
    // 增广后的全部对角线一律加），故这里对 m 个分量统一处理。
    scratch_fit.assign((size_t)m, 0.0);
    for (int i = 0; i < m; ++i) {
        double acc = 0.0;
        const double* row = &scratch_k[(size_t)i * m];
        for (int j = 0; j < m; ++j) acc += row[j] * scratch_z[j];
        scratch_fit[i] = acc - noise_total * scratch_z[i];
    }

    // 残差绝对值
    for (int i = 0; i < m; ++i) {
        const double target = (i < n) ? z_y[i] : z_cand;
        scratch_z[i] = std::fabs(target - scratch_fit[i]);
    }

    if (rank_k > n) {
        // 有限样本下 1-alpha 不可达：阈值退化为无穷，直接拒绝
        *out_margin = 1e300;
        return true;
    }
    // 第 k 小的训练残差（1-indexed -> 索引 k-1）
    std::vector<double> train_resid(scratch_z.begin(), scratch_z.begin() + n);
    std::nth_element(train_resid.begin(), train_resid.begin() + (rank_k - 1),
                     train_resid.end());
    *out_margin = scratch_z[n] - train_resid[rank_k - 1];
    return true;
}

}  // namespace

extern "C" {

// 计算一个全共形区间（标准化坐标）。
// 返回 1 表示找到区间，0 表示未找到（全拒绝）或协方差非正定。
int full_conformal_interval_std(int n, int d,
                                const double* z_train,   // n*d
                                const double* y_train_std,  // n
                                const double* z_test,    // d
                                const double* tau,       // d
                                double signal, double noise,
                                double center, double spread,
                                int grid, double span,
                                double alpha, int max_expand,
                                double* out_left, double* out_right) {
    if (n < 2 || d < 1 || grid < 3) return 0;
    const int rank_k = (int)std::ceil((double)(n + 1) * (1.0 - alpha));

    std::vector<double> z_tr(z_train, z_train + (size_t)n * d);
    std::vector<double> z_y(y_train_std, y_train_std + n);
    std::vector<double> scratch_k, scratch_z, scratch_fit;
    double scale = (spread > 0.0) ? spread : 1.0;

    std::vector<double> cand((size_t)grid), marg((size_t)grid);
    std::vector<char> accepted((size_t)grid);
    int first = -1, last = -1;
    bool bracketed = false;

    for (int attempt = 0; attempt < max_expand && !bracketed; ++attempt) {
        for (int i = 0; i < grid; ++i) {
            cand[i] = center + (2.0 * span * i / (grid - 1) - span) * scale;
        }
        first = -1;
        last = -1;
        bool any_accept = false;
        for (int i = 0; i < grid; ++i) {
            double mg = 0.0;
            if (!margin(z_tr, z_y, z_test, n, d, tau, signal, noise, cand[i], rank_k,
                        scratch_k, scratch_z, scratch_fit, &mg)) {
                return 0;  // 非正定：向上暴露，不在 C 层静默修补
            }
            accepted[i] = (mg <= 0.0) ? 1 : 0;
            if (accepted[i]) {
                any_accept = true;
                if (first < 0) first = i;
                last = i;
            }
        }
        if (any_accept && first > 0 && last < grid - 1) bracketed = true;
        else scale *= 2.0;
    }
    if (first < 0) return 0;

    auto bisect = [&](double inside, double outside) -> double {
        double mg_inside = 0.0;
        margin(z_tr, z_y, z_test, n, d, tau, signal, noise, inside, rank_k,
               scratch_k, scratch_z, scratch_fit, &mg_inside);
        const bool inside_ok = (mg_inside <= 0.0);
        for (int it = 0; it < 50; ++it) {
            const double mid = 0.5 * (inside + outside);
            double mg = 0.0;
            margin(z_tr, z_y, z_test, n, d, tau, signal, noise, mid, rank_k,
                   scratch_k, scratch_z, scratch_fit, &mg);
            if ((mg <= 0.0) == inside_ok) inside = mid;
            else outside = mid;
        }
        return inside_ok ? inside : outside;
    };

    const double left = (first > 0) ? bisect(cand[first], cand[first - 1]) : cand[first];
    const double right = (last < grid - 1) ? bisect(cand[last], cand[last + 1]) : cand[last];
    *out_left = left;
    *out_right = right;
    return 1;
}

// 批量：对 n_cases 个 (训练集, 测试点, 目标) 组合求区间，减少调用开销
int full_conformal_batch_std(int n_cases, int n, int d,
                             const double* z_train, const double* y_train_std,
                             const double* z_test, const double* tau,
                             double signal, double noise,
                             const double* center, const double* spread,
                             int grid, double span, double alpha, int max_expand,
                             double* out_left, double* out_right) {
    int ok = 0;
    for (int c = 0; c < n_cases; ++c) {
        const int status = full_conformal_interval_std(
            n, d,
            z_train + (size_t)c * n * d,
            y_train_std + (size_t)c * n,
            z_test + (size_t)c * d,
            tau, signal, noise, center[c], spread[c],
            grid, span, alpha, max_expand,
            out_left + c, out_right + c);
        ok += status;
    }
    return ok;
}

}  // extern "C"
