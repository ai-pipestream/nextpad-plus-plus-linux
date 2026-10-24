#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace SemanticHeatmap {

/* -1 stricter, 0 standard, 1 broader. Scores themselves are not changed.
 *
 * The stops sit on the cosine range these sentence models actually
 * produce. An unrelated sentence is near 0. A paraphrase that shares
 * little wording is near 0.5. A near copy is near 0.8, and only an
 * identical sentence reaches 1. A ramp that stayed red through 0.35
 * and turned green only past 0.90 painted that whole spread as one red.
 */
inline uint32_t colorBGR(double score, int sensitivity = 0) {
    static constexpr struct { double score; int r, g, b; } stops[] = {
        {0.00, 0xD6, 0x45, 0x41},
        {0.10, 0xD6, 0x45, 0x41},
        {0.18, 0xB2, 0x69, 0x64},
        {0.28, 0x8E, 0x8E, 0x8E},
        {0.34, 0x8E, 0x8E, 0x8E},
        {0.44, 0x2E, 0xCC, 0x71},
        {0.56, 0x0B, 0x8A, 0x45},
        {1.00, 0x0B, 0x8A, 0x45},
    };
    if (!std::isfinite(score)) return 0x8E8E8E;
    double t = std::clamp(score, 0.0, 1.0);
    t = std::pow(t, sensitivity < 0 ? 1.4 : sensitivity > 0 ? 0.7 : 1.0);
    for (unsigned i = 1; i < sizeof(stops) / sizeof(stops[0]); ++i) {
        if (t > stops[i].score) continue;
        const auto &a = stops[i - 1];
        const auto &b = stops[i];
        const double f = (t - a.score) / (b.score - a.score);
        const int r = static_cast<int>(a.r + (b.r - a.r) * f);
        const int g = static_cast<int>(a.g + (b.g - a.g) * f);
        const int blue = static_cast<int>(a.b + (b.b - a.b) * f);
        return static_cast<uint32_t>((blue << 16) | (g << 8) | r);
    }
    return 0x458A0B;
}

/* Six legend samples. Band membership is nearest sample in raw score
 * space, so it stays put when sensitivity changes the painted colors. */
inline constexpr double kLegendScores[6] = {0.10, 0.18, 0.28, 0.36, 0.46, 0.58};

inline int bandForScore(double score) {
    if (!std::isfinite(score)) return -1;
    int best = 0;
    double best_dist = 2.0;
    for (int i = 0; i < 6; ++i) {
        double d = std::abs(score - kLegendScores[i]);
        if (d < best_dist) {
            best_dist = d;
            best = i;
        }
    }
    return best;
}

} /* namespace SemanticHeatmap */
