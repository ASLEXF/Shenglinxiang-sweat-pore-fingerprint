#pragma once

#include <cstring>

// Population presets: pattern and size statistics the generator samples from.
// Percentages are per finger or per palm. Sources and assumptions are listed in HANDOFF.md.
//
// Shared assumptions where a source does not split a class:
// - Twin loops as a share of whorls and tented arches as a share of arches are taken from
//   Buchwald 1981 (north Poland, W2 and At) for every preset: thumb 58% / 3%, index 42% / 30%,
//   middle 44% / 23%, ring 21% / 14%, little 51% / 13%.
// - Ridge period and pore cell by sex use the same cross-population values everywhere (pore
//   cells tuned so 500 dpi fingertips give about 52 / 45 pores per 9 mm2 for women / men); no
//   population-specific pore density was found.
struct Population {
    const char* name;
    const char* note;
    // Per finger, thumb to little: ulnar loop, radial loop, whorl, twin loop, arch, tented arch.
    double finger[5][6];
    // Palms carrying a true pattern in the thenar/I1, I2, I3 and I4 areas and the hypothenar.
    double thenar, i2, i3, i4, hypothenar;
    // Palms with a single transverse (simian) crease, a Sydney line, and a head line that
    // starts apart from the life line.
    double simian, sydney, separate;
    // Index 0 female, 1 male. Hand length (wrist crease to middle fingertip) and breadth at
    // the metacarpal heads, cm; ridge period relative to kRidgePeriodCm; pore cell, cm.
    double hand_mean[2], hand_sd[2];
    double breadth_mean[2];
    double ridge_scale[2];
    double pore_cell[2];
};

inline const Population kPopulations[] = {
    {
        // Fingers: Jiang et al. 2011 (Hebei Han, n=1020). Palms: Zhang et al. 2010 (all Han
        // samples, n=11253). Simian: Jiang 2011, per palm. Sydney: Jiang 2011 "zhong guan"
        // type, not confirmed to equal a Sydney line. Hands: GB 10000-1988 P50, SD from P5-P95.
        "han",
        "Chinese Han adults (default)",
        {
            {43.0, 0.9, 22.4, 30.5, 3.02, 0.08},
            {41.1, 5.6, 27.0, 19.2, 4.95, 2.15},
            {59.6, 1.1, 20.7, 16.0, 2.0, 0.6},
            {36.8, 0.4, 48.7, 12.9, 0.95, 0.15},
            {72.4, 2.1, 11.9, 12.4, 1.04, 0.16},
        },
        7.6, 1.77, 15.05, 68.01, 13.94,
        3.23, 3.0, 30,
        {17.1, 18.3}, {0.73, 0.79},
        {7.6, 8.2},
        {0.92, 1.0},
        {0.0262, 0.0303},
    },
    {
        // Fingers: Buchwald 1981 (north Poland, n=1234). Palms: US whites via Zhang et al.
        // 2010. Simian, Sydney and hand sizes: no European source found; generic rates and
        // ANSUR II (US Army) medians stand in.
        "european",
        "European adults (palms and hand sizes from US white / US Army data)",
        {
            {53.0, 0.2, 18.2, 24.8, 3.8, 0.1},
            {35.1, 17.4, 19.4, 13.8, 9.9, 4.3},
            {72.5, 1.7, 9.7, 7.5, 6.7, 2.0},
            {54.5, 0.7, 33.2, 8.8, 2.5, 0.4},
            {82.5, 0.2, 7.3, 7.6, 2.0, 0.3},
        },
        7.1, 2.5, 37.65, 45.85, 35.2,
        3, 5, 30,
        {18.0, 19.3}, {0.8, 0.85},
        {7.8, 8.8},
        {0.92, 1.0},
        {0.0262, 0.0303},
    },
    {
        // Fingers: Kusaka 1957 (Bizen, Okayama, n=6556). Palms: Tomita 1994 range midpoints.
        // Hands: AIST Japanese hand dimensions (L01, B03). Simian and Sydney: generic rates.
        "japanese",
        "Japanese adults",
        {
            {42.0, 0.4, 23.4, 31.9, 1.95, 0.05},
            {37.0, 11.5, 26.5, 18.8, 4.18, 1.82},
            {60.5, 1.8, 19.3, 15.0, 2.7, 0.8},
            {39.5, 0.25, 47.0, 12.5, 0.65, 0.1},
            {70.8, 0.1, 13.9, 14.4, 0.35, 0.05},
        },
        9.5, 1.0, 21, 79, 24.5,
        3, 5, 30,
        {17.54, 18.96}, {0.73, 0.81},
        {7.4, 8.33},
        {0.92, 1.0},
        {0.0262, 0.0303},
    },
    {
        // Ranges blended across the studies above; not any single group.
        "generic",
        "cross-population blend; not any single group",
        {
            {56, 1, 32, 8, 2, 1},
            {50, 7, 22, 7, 9, 5},
            {70, 2, 17, 4, 5, 2},
            {50, 1, 40, 6, 2, 1},
            {84, 1, 11, 2, 2, 0},
        },
        8, 3, 40, 50, 30,
        3, 5, 30,
        {18.0, 19.3}, {0.9, 1.0},
        {7.8, 8.8},
        {0.92, 1.0},
        {0.0262, 0.0303},
    },
};

inline const Population* find_population(const char* name) {
    for (const auto& p : kPopulations)
        if (std::strcmp(p.name, name) == 0) return &p;
    return nullptr;
}
