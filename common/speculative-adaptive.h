#pragma once

// adaptive draft length for the drafters that score each draft token (MTP, DSpark)
//
// learns while generating:
//  - how likely a draft token is to be accepted, given the drafter's confidence in it and that the tokens before it
//    were accepted: binned, decayed, starting from acceptance = confidence and kept non-decreasing in the confidence
//  - how long a step takes for a draft of L tokens: exponentially weighted least squares T(L) = a + b*L, plus a
//    per-length correction for what a line does not capture (e.g. the checkpoint replay of drafts longer than n_rs_seq)
// and drafts the length L that maximizes the expected tokens per second, (1 + P_1 + ... + P_L) / T(L), with P_i the
// probability that the first i draft tokens are all accepted
//
// a step is measured from the start of a draft to the start of the next draft of the same sequence: its time, and from
// the positions the number of tokens it produced, so the verification, rollbacks and replays are all included
//
// until it has measured enough steps (and steps of different lengths), it drafts while the tokens so far are all
// accepted with at least p_warm (by the acceptance learned so far, the confidence until then) and submits them all;
// the lengths then vary with the confidences, and only if they do not, some drafts are cut short at random

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

struct common_spec_adaptive {
    // parameters
    int32_t n_max     = 0;     // longest draft
    double  explore   = 0.05;  // probability of drafting one token more or less than the best length
    double  p_floor   = 0.1;   // stop drafting when the probability that all the tokens so far are accepted gets lower
    double  p_warm    = 0.5;   // the same while warming up, the whole draft is submitted
    int32_t n_warmup  = 32;    // steps measured before the learned rule is used
    double  lambda    = 0.97;  // forgetting of the step times, per step
    double  gamma     = 0.998; // forgetting of the acceptance statistics, per draft token observed
    double  prior_w   = 4.0;   // weight of the prior (acceptance = confidence), in tokens per bin
    double  max_step  = 60.0;  // longer steps (s) are not measured
    int32_t log_every = 128;   // steps between summaries, 0 = none

    static constexpr int NB = 20;

    // acceptance per confidence bin
    double cal_n[NB] = {};
    double cal_a[NB] = {};
    double cal_v[NB] = {};
    bool   cal_dirty = true;

    // step time: weighted sums of the line fit, and per length the mean residual
    double sw = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    double b_last = -1.0; // the last cost of a token that the lengths could tell, < 0 = none yet
    std::vector<double> res_m;
    std::vector<double> res_w;
    int64_t n_steps = 0;

    // per sequence: the step in progress
    struct seq_state {
        bool    valid = false;
        int64_t t0    = 0;
        int64_t pos0  = 0;
        std::vector<float> conf; // the confidences of the tokens of the submitted draft
    };
    std::vector<seq_state> seqs;

    std::mt19937 rng { 0x5eed };

    // statistics since the last summary
    int64_t st_steps  = 0;
    int64_t st_tokens = 0;
    int64_t st_draft  = 0;
    double  st_time   = 0.0;
    bool    was_warm  = false;

    void init(int32_t n_seq, int32_t n_max_) {
        n_max = std::max(0, n_max_);
        seqs.assign(std::max(1, n_seq), seq_state());
        res_m.assign(n_max + 1, 0.0);
        res_w.assign(n_max + 1, 0.0);
    }

    //
    // acceptance
    //

    static int   bin(double c) { return std::clamp((int) (c * NB), 0, NB - 1); }
    static double mid(int b)   { return (b + 0.5) / NB; }

    void observe(double c, bool accepted) {
        for (int b = 0; b < NB; ++b) {
            cal_n[b] *= gamma;
            cal_a[b] *= gamma;
        }
        const int b = bin(c);
        cal_n[b] += 1.0;
        cal_a[b] += accepted ? 1.0 : 0.0;
        cal_dirty = true;
    }

    // per bin the observed rate shrunk towards the confidence, then pooled where it would decrease
    void refit() {
        double v[NB], w[NB];
        int    len[NB];
        int    m = 0;
        for (int b = 0; b < NB; ++b) {
            double vb = (cal_a[b] + prior_w*mid(b)) / (cal_n[b] + prior_w);
            double wb = cal_n[b] + prior_w;
            int    lb = 1;
            while (m > 0 && v[m - 1] > vb) {
                vb = (v[m - 1]*w[m - 1] + vb*wb) / (w[m - 1] + wb);
                wb += w[m - 1];
                lb += len[m - 1];
                m--;
            }
            v[m] = vb; w[m] = wb; len[m] = lb; m++;
        }
        int b = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < len[i]; ++j) {
                cal_v[b++] = std::clamp(v[i], 0.0, 1.0);
            }
        }
        cal_dirty = false;
    }

    // the probability that a draft token of confidence c is accepted, if the tokens before it are
    double acc(double c) {
        if (cal_dirty) {
            refit();
        }
        return cal_v[bin(c)];
    }

    //
    // step time
    //

    bool line(double & a, double & b) const {
        if (sw <= 0.0) {
            return false;
        }
        const double mx = sx/sw;
        const double my = sy/sw;
        const double vx = sxx/sw - mx*mx;
        if (vx < 0.05) {
            // the recent lengths do not vary enough to tell the cost of a token (e.g. the best length stayed the
            // same for a while): keep the last cost of a token that they could tell, at the current level
            if (b_last < 0.0) {
                return false;
            }
            b = b_last;
            a = my - b*mx;
            return true;
        }
        b = std::max(0.0, (sxy/sw - mx*my) / vx);
        a = my - b*mx;
        return true;
    }

    // the expected time of a step with a draft of L tokens, in seconds
    double cost(int L) const {
        double a = 0.0, b = 0.0;
        if (!line(a, b)) {
            return 1.0;
        }
        double t = a + b*L;
        if (L >= 0 && L < (int) res_w.size() && res_w[L] > 0.0) {
            t += res_m[L] * res_w[L] / (res_w[L] + 3.0); // shrunk until the length was seen a few times
        }
        return std::max(t, 1e-4);
    }

    void observe_step(int L, double t) {
        if (L < 0 || L > n_max) {
            return;
        }
        // a step much slower than expected (e.g. the batch had other work) counts as at most 4x the expectation
        if (warm()) {
            t = std::min(t, 4.0*cost(L));
        }
        sw  = lambda*sw  + 1.0;
        sx  = lambda*sx  + L;
        sy  = lambda*sy  + t;
        sxx = lambda*sxx + (double) L*L;
        sxy = lambda*sxy + L*t;

        if (sxx/sw - (sx/sw)*(sx/sw) >= 0.05) {
            b_last = std::max(0.0, (sxy/sw - (sx/sw)*(sy/sw)) / (sxx/sw - (sx/sw)*(sx/sw)));
        }

        double a = 0.0, b = 0.0;
        if (line(a, b)) {
            // the mean over the last ~10 steps of this length
            res_w[L] = std::min(res_w[L] + 1.0, 10.0);
            res_m[L] += (t - (a + b*L) - res_m[L]) / res_w[L];
        }
        n_steps++;
    }

    bool warm() const {
        double a, b;
        return n_steps >= n_warmup && line(a, b);
    }

    //
    // drafting
    //

    // a new request: the time since the last draft is not a step
    void reset(int s) {
        if (s >= 0 && s < (int) seqs.size()) {
            seqs[s].valid = false;
        }
    }

    // a draft of sequence s starts at position pos0 (of the last sampled token): the previous step ends here
    void start(int s, int64_t pos0, int64_t t_us) {
        if (s < 0 || s >= (int) seqs.size()) {
            return;
        }
        auto & q = seqs[s];
        if (q.valid) {
            const int64_t tokens = pos0 - q.pos0;
            const int     L      = (int) q.conf.size();
            const double  t      = (t_us - q.t0) * 1e-6;
            // otherwise not a step of this draft (no draft in between, a context shift, ...)
            if (tokens >= 1 && tokens - 1 <= L && t > 0.0 && t < max_step) {
                const int A = (int) tokens - 1;
                for (int j = 0; j < A; ++j) {
                    observe(q.conf[j], true);
                }
                if (A < L) {
                    observe(q.conf[A], false);
                }
                observe_step(L, t);

                st_steps++;
                st_tokens += tokens;
                st_draft  += L;
                st_time   += t;
            }
        }
        q.valid = true;
        q.t0    = t_us;
        q.pos0  = pos0;
        q.conf.clear();
    }

    // keep drafting while the first tokens are all accepted with at least p_floor (learned, the length is chosen
    // afterwards) or p_warm (warming up, the whole draft is submitted)
    bool more(double p_all) const {
        return p_all >= (warm() ? p_floor : p_warm);
    }

    // the length to submit, of a draft whose tokens have the confidences conf
    // warm: the best length, sometimes one more or less. otherwise all of them, and once the warmup steps are done
    // without the lengths varying enough to tell the cost of a token, sometimes fewer
    int choose(const std::vector<float> & conf) {
        const int n = std::min<int>((int) conf.size(), n_max);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        if (!warm()) {
            if (n > 0 && n_steps >= n_warmup && u(rng) < 0.3) {
                return std::uniform_int_distribution<int>(0, n - 1)(rng);
            }
            return n;
        }
        int    best_L = 0;
        double best   = 1.0 / cost(0);
        double p_all  = 1.0;
        double sum    = 0.0;
        for (int i = 0; i < n; ++i) {
            p_all *= acc(conf[i]);
            sum   += p_all;
            const double v = (1.0 + sum) / cost(i + 1);
            if (v > best) {
                best   = v;
                best_L = i + 1;
            }
        }
        const double r = u(rng);
        if (r < explore/2 && best_L < n) {
            best_L++;
        } else if (r < explore && best_L > 0) {
            best_L--;
        }
        return best_L;
    }

    // the draft of sequence s that is submitted for verification: its first L tokens
    void submitted(int s, const std::vector<float> & conf, int L) {
        if (s < 0 || s >= (int) seqs.size()) {
            return;
        }
        L = std::clamp(L, 0, (int) conf.size());
        seqs[s].conf.assign(conf.begin(), conf.begin() + L);
    }

    //
    // reporting
    //

    // true once when the learned rule takes over
    bool just_warmed() {
        if (!was_warm && warm()) {
            was_warm = true;
            return true;
        }
        return false;
    }

    bool should_log() const {
        return log_every > 0 && st_steps >= log_every;
    }

    std::string summary() {
        char buf[512];
        double a = 0.0, b = 0.0;
        const bool ok = line(a, b);
        const double ts = st_steps > 0 ? (double) st_tokens / st_steps : 0.0;
        const double ms = st_steps > 0 ? 1e3 * st_time / st_steps : 0.0;
        int n = snprintf(buf, sizeof(buf),
                "adaptive draft: %lld steps, %.2f tokens/step, %.0f ms/step (%.2f t/s), drafted %.2f/step, %s",
                (long long) st_steps, ts, ms, st_time > 0.0 ? st_tokens / st_time : 0.0,
                st_steps > 0 ? (double) st_draft / st_steps : 0.0, warm() ? "learned" : "warming up");
        if (ok && n > 0 && n < (int) sizeof(buf)) {
            n += snprintf(buf + n, sizeof(buf) - n, ", step %.0f ms + %.0f ms/token, T(L) =", 1e3*a, 1e3*b);
            for (int L = 0; L <= n_max && n > 0 && n < (int) sizeof(buf); ++L) {
                n += snprintf(buf + n, sizeof(buf) - n, " %.0f", 1e3*cost(L));
            }
        }
        if (n > 0 && n < (int) sizeof(buf)) {
            snprintf(buf + n, sizeof(buf) - n, ", accepted at confidence 0.5/0.7/0.9: %.2f/%.2f/%.2f",
                    acc(0.5), acc(0.7), acc(0.9));
        }
        st_steps = st_tokens = st_draft = 0;
        st_time  = 0.0;
        return buf;
    }
};
