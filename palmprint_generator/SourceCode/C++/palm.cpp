#include "palm.h"

#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

constexpr int kFingers = 5;
constexpr int kPalm = 7;
constexpr int kAngles = 32;
constexpr int kPeriods = 8;
constexpr int kIterations = 10;
constexpr double kRidgePeriodCm = 0.048;
constexpr double kMinPeriodPx = 5.0;
// Palm landmarks are written in a design frame and pass through anat() into the model
// frame. anat() widens the hand 1.15x and lengthens the palm below the knuckles 1.31x, so
// the standard hand is about 18 cm long with an 11 cm palm and an 8 cm breadth (ANSUR II
// medians: hand 18.0/19.3 cm, breadth 7.8/8.8 cm for women/men). Fingers are built in the
// model frame directly.
constexpr double kAnatCenterX = 6.2;
constexpr double kAnatWiden = 1.15;
constexpr double kAnatKnuckles = 10.5;
constexpr double kAnatStretch = 1.31;

cv::Point2d anat(const cv::Point2d& p) {
    return {kAnatCenterX + kAnatWiden * (p.x - kAnatCenterX),
        p.y <= kAnatKnuckles ? p.y : kAnatKnuckles + kAnatStretch * (p.y - kAnatKnuckles)};
}

cv::Point2d anat(double x, double y) { return anat(cv::Point2d(x, y)); }

// Palm outline in the design frame, radial wrist corner first, clockwise over the knuckles.
const cv::Point2d kPalmOutline[] = {
    {3.35, 18.7}, {2.35, 16.6}, {1.55, 14.0}, {2.45, 11.5}, {3.55, 9.35}, {4.70, 8.95},
    {6.20, 8.70}, {7.70, 8.95}, {9.20, 9.35}, {10.15, 9.85}, {10.75, 12.6}, {10.45, 15.6},
    {9.45, 18.7}
};

// Interdigital loop core and its triradius, cm below the gap between two finger bases.
constexpr double kLoopCoreDepth = 0.8;
constexpr double kLoopDeltaDepth = 1.4;

// Standard-hand measurements in the model frame with every shape factor at 1 (wrist crease
// to middle fingertip, and breadth at the metacarpal heads).
constexpr double kStandardHand = 18.2;
constexpr double kStandardBreadth = 8.1;

// Centre of the hand-shape warp in the model frame (middle of the palm). The taper acts
// from the knuckles (model y about 9) to the bottom of the palm (about 21.2); fingers and
// wrist take the value at its ends.
const cv::Point2d kShapeCenter(6.2, 14.7);
constexpr double kTaperFrom = -5.7;
constexpr double kTaperTo = 6.6;

cv::Point2d dir_of(double angle) {
    return {std::cos(angle), std::sin(angle)};
}

cv::Point2d rot90(const cv::Point2d& d) {
    return {-d.y, d.x};
}

double norm2d(const cv::Point2d& d) {
    return std::sqrt(d.x * d.x + d.y * d.y);
}

cv::Point2d unit(const cv::Point2d& d) {
    double n = norm2d(d);
    if (n < 1e-8) return {1, 0};
    return {d.x / n, d.y / n};
}

double smoothstep(double t) {
    t = std::clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// splitmix64 finalizer. cv::RNG streams from nearby seeds start out correlated.
uint64_t mix_seed(uint64_t x) {
    x = (x + 0x9E3779B97F4A7C15ull) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 31)) * 0x94D049BB133111EBull;
    return x ^ (x >> 29);
}

const char* pattern_name(int pattern) {
    switch (pattern) {
    case 1: return "arch";
    case 2: return "tented";
    case 3: return "whorl";
    case 4: return "left-loop";
    case 5: return "right-loop";
    case 6: return "twin-loop";
    default: return "?";
    }
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

PalmPrint::PalmPrint(int dpi, unsigned seed, bool right_hand, const Population& population)
    : dpi_(dpi), pop_(&population), seed_(seed), right_hand_(right_hand) {
    rng_ = cv::RNG(stream_seed(2));
    scale_ = dpi_ / 2.54;
    margin_ = std::max(8, static_cast<int>(std::lround(px(0.45))));
}

// Each random stream gets its own salt: lines 0, pads 1, rng_ 2, shape 3, palm patterns 4.
// The population name is folded in too, so one seed gives unrelated hands under different
// presets instead of the same draws against different thresholds.
uint64_t PalmPrint::stream_seed(int stream) const {
    uint64_t name_hash = 1469598103934665603ull;  // FNV-1a
    for (const char* c = pop_->name; *c; ++c) name_hash = (name_hash ^ static_cast<unsigned char>(*c)) * 1099511628211ull;
    return mix_seed(static_cast<uint64_t>(seed_) + (static_cast<uint64_t>(stream) << 32) + name_hash);
}

cv::Point2d PalmPrint::warp(const cv::Point2d& p) const {
    const double dy = p.y - kShapeCenter.y;
    const double taper = 1.0 + shape_taper_ * std::clamp(dy, kTaperFrom, kTaperTo);
    return {kShapeCenter.x + shape_scale_ * shape_ax_ * taper * (p.x - kShapeCenter.x),
        kShapeCenter.y + shape_scale_ * shape_ay_ * dy};
}

cv::Point2d PalmPrint::unwarp(const cv::Point2d& q) const {
    const double dy = (q.y - kShapeCenter.y) / (shape_scale_ * shape_ay_);
    const double taper = 1.0 + shape_taper_ * std::clamp(dy, kTaperFrom, kTaperTo);
    return {kShapeCenter.x + (q.x - kShapeCenter.x) / (shape_scale_ * shape_ax_ * taper),
        kShapeCenter.y + dy};
}

// Ridge direction theta at canonical point p, carried through the warp's Jacobian.
double PalmPrint::warp_angle(const cv::Point2d& p, double theta) const {
    const double dy = p.y - kShapeCenter.y;
    const double taper = 1.0 + shape_taper_ * std::clamp(dy, kTaperFrom, kTaperTo);
    const double slope = dy > kTaperFrom && dy < kTaperTo ? shape_taper_ : 0.0;
    const double c = std::cos(theta), s = std::sin(theta);
    const double vx = shape_ax_ * (taper * c + slope * (p.x - kShapeCenter.x) * s);
    const double vy = shape_ay_ * s;
    return std::atan2(vy, vx);
}

cv::Point PalmPrint::to_px(const cv::Point2d& p) const {
    const cv::Point2d q = warp(p);
    return {
        cvRound((q.x - bounds_.x) * scale_ + margin_),
        cvRound((q.y - bounds_.y) * scale_ + margin_)
    };
}

cv::Point2d PalmPrint::to_cm(int x, int y) const {
    return unwarp({bounds_.x + (x - margin_) / scale_, bounds_.y + (y - margin_) / scale_});
}

void PalmPrint::fill_ellipse(cv::Mat& mask, const cv::Point2d& center, double a, double b, double angle_deg) const {
    // Centre and axes are in the design frame; every point goes through anat() and the warp.
    const double t = angle_deg * CV_PI / 180.0;
    const cv::Point2d u(std::cos(t), std::sin(t)), v(-std::sin(t), std::cos(t));
    std::vector<cv::Point> poly;
    for (int k = 0; k < 180; ++k) {
        double phi = 2 * CV_PI * k / 180;
        poly.push_back(to_px(anat(center + u * (a * std::cos(phi)) + v * (b * std::sin(phi)))));
    }
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(255), cv::LINE_8);
}

std::vector<cv::Point> PalmPrint::palm_outline_px() const {
    std::vector<cv::Point> poly;
    for (const auto& p : kPalmOutline) poly.push_back(to_px(anat(p)));
    return poly;
}

double PalmPrint::singularity(const cv::Point2d& p, const cv::Point2d* cores, int n_core,
    const cv::Point2d* deltas, int n_delta, int pattern) {
    double theta = 0;
    if (n_core > 0) {
        for (int i = 0; i < n_core; ++i)
            theta += std::atan2(cores[i].y - p.y, cores[i].x - p.x) / n_core;
    }
    if (n_delta > 0) {
        double sum = 0;
        for (int i = 0; i < n_delta; ++i)
            sum += std::atan2(deltas[i].y - p.y, deltas[i].x - p.x);
        theta -= sum / n_delta;
        theta *= n_delta / 2.0;
    }
    if (pattern == TENTED_ARCH) theta -= CV_PI;
    return theta;
}

double PalmPrint::sherlock(const cv::Point2d& p, const cv::Point2d* cores, int n_core,
    const cv::Point2d* deltas, int n_delta, double base) {
    double theta = base;
    for (int i = 0; i < n_core; ++i)
        theta += 0.5 * std::atan2(p.y - cores[i].y, p.x - cores[i].x);
    for (int i = 0; i < n_delta; ++i)
        theta -= 0.5 * std::atan2(p.y - deltas[i].y, p.x - deltas[i].x);
    return theta;
}

std::vector<cv::Point2d> PalmPrint::bezier(const cv::Point2d& a, const cv::Point2d& b, const cv::Point2d& c, int n) {
    std::vector<cv::Point2d> pts;
    pts.reserve(n);
    for (int i = 0; i < n; ++i) {
        double t = n == 1 ? 0 : static_cast<double>(i) / (n - 1);
        double u = 1.0 - t;
        pts.push_back(a * (u * u) + b * (2 * u * t) + c * (t * t));
    }
    return pts;
}

// Uniform Catmull-Rom through every control point, sampled about every `step` cm.
std::vector<cv::Point2d> PalmPrint::catmull_rom(const std::vector<cv::Point2d>& ctrl, double step) {
    std::vector<cv::Point2d> pts;
    const int n = static_cast<int>(ctrl.size());
    if (n < 2) return ctrl;
    for (int i = 0; i + 1 < n; ++i) {
        const cv::Point2d& p0 = ctrl[std::max(0, i - 1)];
        const cv::Point2d& p1 = ctrl[i];
        const cv::Point2d& p2 = ctrl[i + 1];
        const cv::Point2d& p3 = ctrl[std::min(n - 1, i + 2)];
        int samples = std::max(2, cvCeil(norm2d(p2 - p1) / step));
        for (int k = 0; k < samples; ++k) {
            double t = static_cast<double>(k) / samples;
            double t2 = t * t, t3 = t2 * t;
            pts.push_back((p1 * 2.0 + (p2 - p0) * t + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * t2
                + (p1 * 3.0 - p0 - p2 * 3.0 + p3) * t3) * 0.5);
        }
    }
    pts.push_back(ctrl.back());
    return pts;
}

void PalmPrint::build_geometry() {
    const char* names[kFingers] = {"thumb", "index", "middle", "ring", "little"};
    // Model frame. Middle finger 7.2 cm from its base crease (literature about 6.8/7.5 cm
    // for women/men); finger breadths 1.4-2.4 cm.
    const cv::Point2d bases[kFingers] = {
        anat(3.15, 12.55), anat(4.55, 9.15), anat(6.20, 8.82), anat(7.75, 9.05), anat(9.15, 9.48)
    };
    const double axes[kFingers] = {-2.12, -1.74, -1.60, -1.46, -1.30};
    const double lengths[kFingers] = {6.3, 6.5, 7.2, 6.65, 5.4};
    const double widths[kFingers] = {2.35, 1.8, 1.93, 1.75, 1.4};
    const int segments[kFingers] = {2, 3, 3, 3, 3};

    // Digital triradii a-d sit about 0.8 cm below the base of each finger, on its axis.
    // The axial triradius t is placed so that atd is about 40 degrees (adults about 41,
    // SD 6-10).
    for (int i = 0; i < 4; ++i) tri_[i] = bases[i + 1] + cv::Point2d(0, 0.8);
    tri_[4] = {6.75, 17.6};

    // Hand shape from its own stream: changing it leaves patterns and creases as they were.
    cv::RNG shape(stream_seed(3));
    // Sex, then hand length and breadth from the population's female/male figures. The
    // standard hand (all factors 1) is kStandardHand long and kStandardBreadth wide.
    sex_ = shape.uniform(0, 2);
    const double hand_cm = pop_->hand_mean[sex_] + pop_->hand_sd[sex_] * std::clamp(shape.gaussian(1.0), -2.5, 2.5);
    shape_scale_ = hand_cm / kStandardHand;
    const double breadth_ratio = (pop_->breadth_mean[sex_] / pop_->hand_mean[sex_]) / (kStandardBreadth / kStandardHand);
    shape_ax_ = breadth_ratio * shape.uniform(0.95, 1.05);
    shape_ay_ = shape.uniform(0.96, 1.04);
    shape_taper_ = shape.uniform(-0.025, 0.025);
    length_scale_ = shape_scale_ * std::sqrt(shape_ax_ * shape_ay_);
    const double spread = shape.uniform(0.75, 1.3);
    // Mostly abduction: turning the thumb towards the palm lays its edge over the life line.
    const double thumb_turn = shape.uniform(-0.15, 0.02);
    double finger_len[kFingers];
    for (int i = 0; i < kFingers; ++i) finger_len[i] = shape.uniform(0.95, 1.05);

    // Ridge period and pore spacing follow sex (women have finer ridges and denser pores).
    base_period_ = std::max(kMinPeriodPx, px(kRidgePeriodCm * pop_->ridge_scale[sex_]));
    period_min_ = base_period_ * 0.85;
    period_max_ = base_period_ * 1.5;
    pore_spacing_cm_ = pop_->pore_cell[sex_];
    {
        char buf[160];
        std::snprintf(buf, sizeof(buf), " sex=%s size=%.2f width=%.2f length=%.2f taper=%+.3f spread=%.2f thumb=%+.2f",
            sex_ ? "M" : "F", shape_scale_, shape_ax_, shape_ay_, shape_taper_, spread, thumb_turn);
        shape_summary_ = buf;
    }

    for (int i = 0; i < kFingers; ++i) {
        Finger& f = fingers_[i];
        f.name = names[i];
        f.base = bases[i];
        f.axis = axes[i] + rng_.uniform(-0.04, 0.04);
        // Fingers fan out around the middle finger's axis; the thumb swings about its base.
        f.axis = i == 0 ? f.axis + thumb_turn : axes[2] + (f.axis - axes[2]) * spread;
        f.length = lengths[i] * rng_.uniform(0.96, 1.04) * finger_len[i];
        f.width = widths[i] * rng_.uniform(0.96, 1.04);
        f.segments = segments[i];
        if (f.segments == 2) {
            f.frac[0] = 0.56;
            f.frac[1] = 0.44;
            f.frac[2] = 0;
        } else {
            f.frac[0] = 0.42;
            f.frac[1] = 0.32;
            f.frac[2] = 0.26;
        }
        f.slope = rng_.uniform(-0.3, 0.3);
        f.prox_bow = rng_.uniform(0.2, 0.5) * (rng_.uniform(0, 2) ? 1.0 : -1.0);
        f.pip_lines = 1 + rng_.uniform(0, 3);
        f.axis_dir = dir_of(f.axis);
        f.across = rot90(f.axis_dir);
        f.radial = i == 0 ? unit({-0.82, -0.25}) : f.across * -1.0;

        // One draw per finger against the population's per-finger table. In the left-hand
        // model LEFT_LOOP opens towards the little finger (ulnar), RIGHT_LOOP towards the thumb.
        {
            const int order[6] = {LEFT_LOOP, RIGHT_LOOP, WHORL, TWIN_LOOP, ARCH, TENTED_ARCH};
            const double* w = pop_->finger[i];
            double total = 0;
            for (int k = 0; k < 6; ++k) total += w[k];
            double roll = rng_.uniform(0.0, total);
            f.pattern = order[0];
            for (int k = 0; k < 6; ++k) {
                if (roll < w[k]) {
                    f.pattern = order[k];
                    break;
                }
                roll -= w[k];
            }
        }

        double acc = 0;
        f.seg_at[0] = 0;
        for (int s = 0; s < f.segments; ++s) {
            acc += f.length * f.frac[s];
            f.seg_at[s + 1] = acc;
        }

        double distal_from = f.seg_at[f.segments - 1];
        double distal_len = f.length - distal_from;
        cv::Point2d distal_base = f.base + f.axis_dir * distal_from;
        cv::Point2d core = distal_base + f.axis_dir * (distal_len * 0.55);
        cv::Point2d base_dir = f.axis_dir * -1.0;
        f.n_core = 0;
        f.n_delta = 0;
        if (f.pattern == WHORL || f.pattern == TWIN_LOOP) {
            // Twin-loop cores sit apart and staggered along the axis; closer they read as a bullseye.
            cv::Point2d offset = f.pattern == TWIN_LOOP
                ? f.across * (f.width * 0.17) + f.axis_dir * (distal_len * 0.10) : cv::Point2d(0, 0);
            f.cores[0] = core + offset;
            f.cores[1] = core - offset;
            f.n_core = f.pattern == TWIN_LOOP ? 2 : 1;
            f.deltas[0] = core + f.across * (f.width * 0.30) + base_dir * (distal_len * 0.18);
            f.deltas[1] = core - f.across * (f.width * 0.30) + base_dir * (distal_len * 0.18);
            f.n_delta = 2;
        } else if (f.pattern == LEFT_LOOP || f.pattern == RIGHT_LOOP) {
            f.cores[0] = core;
            f.n_core = 1;
            double side = f.pattern == LEFT_LOOP ? 1.0 : -1.0;
            f.deltas[0] = core + f.radial * (side * f.width * 0.34) + base_dir * (distal_len * 0.22);
            f.n_delta = 1;
        }
    }

    // Pad shaping draws from its own stream so the rest of the hand keeps its noise.
    cv::RNG tips(stream_seed(1));
    for (auto& f : fingers_) {
        f.tip_arch = tips.uniform(0.55, 0.8);
        f.twist = f.pattern == WHORL ? tips.uniform(0.2, 0.45) * (tips.uniform(0, 2) ? 1.0 : -1.0) : 0.0;
        for (int k = 0; k < 3; ++k) {
            double wavelength = tips.uniform(0.6, 1.2);
            double dir = tips.uniform(0.0, CV_PI);
            f.waves[k] = {std::cos(dir) * 2 * CV_PI / wavelength, std::sin(dir) * 2 * CV_PI / wavelength,
                tips.uniform(0.02, 0.035)};
            f.wave_phase[k] = tips.uniform(0.0, 2 * CV_PI);
        }
    }

    thenar_whorl_ = rng_.uniform(0.0, 100.0) < pop_->thenar;
    thenar_core_ = anat(3.25 + rng_.uniform(-0.2, 0.2), 14.15 + rng_.uniform(-0.3, 0.3));
    hypo_pattern_ = rng_.uniform(0.0, 100.0) < pop_->hypothenar ? LEFT_LOOP : ARCH;
    central_angle_ = rng_.uniform(0.30, 0.52);

    // Interdigital patterns from their own stream.
    cv::RNG palm(stream_seed(4));
    inter_loop_[0] = palm.uniform(0.0, 100.0) < pop_->i2;
    inter_loop_[1] = palm.uniform(0.0, 100.0) < pop_->i3;
    inter_loop_[2] = palm.uniform(0.0, 100.0) < pop_->i4;

    auto consider = [&](const cv::Point2d& p) {
        bounds_.x = std::min(bounds_.x, p.x);
        bounds_.y = std::min(bounds_.y, p.y);
        bounds_.width = std::max(bounds_.width, p.x);
        bounds_.height = std::max(bounds_.height, p.y);
    };
    bounds_ = cv::Rect2d(1e9, 1e9, -1e9, -1e9);
    for (const auto& p : kPalmOutline) consider(warp(anat(p)));
    for (const auto& f : fingers_) {
        consider(warp(f.base));
        consider(warp(f.base + f.axis_dir * f.length));
        consider(warp(f.base + f.across * f.width));
        consider(warp(f.base - f.across * f.width));
        consider(warp(f.base + f.axis_dir * f.length + f.across * f.width));
        consider(warp(f.base + f.axis_dir * f.length - f.across * f.width));
    }
    {
        // Landmarks after both warps: wrist crease to middle fingertip, middle finger base to
        // wrist crease, breadth at the metacarpal heads (design y 10.2) and the atd angle.
        auto edge_x = [&](const cv::Point2d& a, const cv::Point2d& b, double y) {
            return anat(a + (b - a) * ((y - a.y) / (b.y - a.y)));
        };
        const Finger& mid = fingers_[2];
        const cv::Point2d wrist = warp(anat(6.4, 17.6));
        const cv::Point2d tip = warp(mid.base + mid.axis_dir * mid.length);
        const cv::Point2d radial = warp(edge_x({2.45, 11.5}, {3.55, 9.35}, 10.2));
        const cv::Point2d ulnar = warp(edge_x({10.15, 9.85}, {10.75, 12.6}, 10.2));
        const cv::Point2d ta = warp(tri_[0]) - warp(tri_[4]), td = warp(tri_[3]) - warp(tri_[4]);
        const double atd = std::acos(ta.dot(td) / (norm2d(ta) * norm2d(td))) * 180.0 / CV_PI;
        char buf[160];
        std::snprintf(buf, sizeof(buf), " hand=%.1f palm=%.1f breadth=%.1f middle=%.1f atd=%.0f",
            norm2d(tip - wrist), norm2d(warp(mid.base) - wrist), norm2d(ulnar - radial),
            mid.length * length_scale_, atd);
        shape_summary_ += buf;
    }
    double min_x = bounds_.x - 0.7;
    double min_y = bounds_.y - 0.7;
    double max_x = bounds_.width + 0.7;
    double max_y = bounds_.height + 0.7;
    bounds_ = cv::Rect2d(min_x, min_y, max_x - min_x, max_y - min_y);
}

void PalmPrint::add_segment(int finger, int region, const cv::Point2d& a, const cv::Point2d& b,
    double width_a, double width_b, bool cap) {
    cv::Point2d u = unit(b - a);
    cv::Point2d n(-u.y, u.x);
    cv::Point2d ha = n * (width_a * 0.5);
    cv::Point2d hb = n * (width_b * 0.5);
    cv::Point poly[4] = {to_px(a + ha), to_px(a - ha), to_px(b - hb), to_px(b + hb)};
    const int label = pack(region, finger);
    cv::fillConvexPoly(labels_, poly, 4, cv::Scalar(label), cv::LINE_8);
    if (cap) {
        int radius = std::max(1, cvRound(px_len(width_b * 0.5)));
        cv::circle(labels_, to_px(b), radius, cv::Scalar(label), cv::FILLED, cv::LINE_8);
    }
}

void PalmPrint::add_tapered(const cv::Point2d& a, const cv::Point2d& b, const cv::Point2d& c,
    double width_start, double width_mid, double width_end) {
    auto curve = bezier(a, b, c, 42);
    Stroke stroke;
    const double phase = rng_.uniform(0.0, 6.0);
    int max_radius = 1;
    stroke.pts.reserve(curve.size());
    stroke.radii.reserve(curve.size());
    for (int i = 0; i < static_cast<int>(curve.size()); ++i) {
        double t = curve.size() == 1 ? 0.0 : static_cast<double>(i) / (curve.size() - 1);
        double u = 1.0 - t;
        double width = u * u * width_start + 2 * u * t * width_mid + t * t * width_end;
        width *= 1.0 + 0.22 * std::sin(t * 16.0 + phase);
        int radius = std::max(1, cvRound(px_len(width) * 0.5));
        max_radius = std::max(max_radius, radius);
        stroke.pts.push_back(to_px(curve[i]));
        stroke.radii.push_back(radius);
    }
    stroke.ragged = std::max(2, max_radius / 2);
    strokes_.push_back(std::move(stroke));
}

void PalmPrint::add_curve(const std::vector<cv::Point2d>& ctrl, double width_start, double width_mid,
    double width_end, cv::RNG& rng) {
    // One sample per pixel so the stamped circles always overlap.
    auto curve = catmull_rom(ctrl, 1.0 / (scale_ * length_scale_ * 1.15));
    std::vector<double> arc(curve.size(), 0.0);
    for (size_t i = 1; i < curve.size(); ++i) arc[i] = arc[i - 1] + norm2d(curve[i] - curve[i - 1]);
    const double total = std::max(1e-6, arc.back());
    const double phase = rng.uniform(0.0, 6.0);
    Stroke stroke;
    int max_radius = 1;
    stroke.pts.reserve(curve.size());
    stroke.radii.reserve(curve.size());
    for (size_t i = 0; i < curve.size(); ++i) {
        double t = arc[i] / total;
        double u = 1.0 - t;
        double width = u * u * width_start + 2 * u * t * width_mid + t * t * width_end;
        width *= 1.0 + 0.22 * std::sin(arc[i] * 2.7 + phase);
        int radius = std::max(1, cvRound(px_len(width) * 0.5));
        max_radius = std::max(max_radius, radius);
        stroke.pts.push_back(to_px(curve[i]));
        stroke.radii.push_back(radius);
    }
    stroke.ragged = std::max(2, max_radius / 2);
    strokes_.push_back(std::move(stroke));
}

// Flexion creases of the palm, picked from a few anatomical layouts. Coordinates are the
// left-hand model in cm; start points sit just outside the outline and get clipped.
void PalmPrint::add_principal_lines(cv::RNG& rng) {
    auto jitter = [&](const cv::Point2d& p, double amount) {
        return p + cv::Point2d(rng.uniform(-amount, amount), rng.uniform(-amount, amount));
    };
    auto pick = [&](std::initializer_list<int> weights) {
        int total = 0;
        for (int w : weights) total += w;
        int roll = rng.uniform(0, total);
        int index = 0;
        for (int w : weights) {
            if (roll < w) return index;
            roll -= w;
            ++index;
        }
        return index - 1;
    };
    // A short side branch leaving the curve at `from` towards `dir`.
    auto fork = [&](const cv::Point2d& from, const cv::Point2d& dir, double length, double width) {
        cv::Point2d d = unit(dir);
        cv::Point2d side = rot90(d) * rng.uniform(-0.12, 0.12);
        add_curve({from, from + d * (length * 0.5) + side, from + d * length + side * 2.0},
            width, width * 0.8, width * 0.25, rng);
    };

    // Radial border between the index base and the thumb web, pushed 0.25 cm outward. The
    // web is where the thumb's ulnar edge meets the palm's radial edge.
    const Finger& thumb = fingers_[0];
    const Finger& index = fingers_[1];
    const cv::Point2d radial_top = index.base - index.across * (index.width * 0.5);
    cv::Point2d radial_web = anat(3.02, 10.39);
    {
        const cv::Point2d e0 = anat(2.45, 11.5), e1 = anat(3.55, 9.35);
        const cv::Point2d q = thumb.base + thumb.across * (thumb.width * 0.5);
        const cv::Point2d d = thumb.axis_dir, e = e1 - e0;
        const double den = d.x * e.y - d.y * e.x;
        if (std::abs(den) > 1e-9) {
            const double r = (d.x * (q.y - e0.y) - d.y * (q.x - e0.x)) / den;
            if (r >= 0 && r <= 1) radial_web = e0 + e * r;
        }
    }
    cv::Point2d radial_out = unit(rot90(radial_web - radial_top));
    if (radial_out.dot(radial_top - anat(6.2, 13.5)) < 0) radial_out = radial_out * -1.0;
    auto radial_at = [&](double t) { return radial_top + (radial_web - radial_top) * t + radial_out * 0.25; };

    const bool simian = rng.uniform(0.0, 100.0) < pop_->simian;
    const bool separate = !simian && rng.uniform(0.0, 100.0) < pop_->separate;
    const cv::Point2d life_start = jitter(radial_at(separate ? 0.68 : 0.45), 0.05);
    std::string summary;
    auto label_at = [&](const cv::Point2d& p) {
        cv::Point q = to_px(p);
        if (q.x < 0 || q.y < 0 || q.x >= labels_.cols || q.y >= labels_.rows) return 0;
        return static_cast<int>(labels_.at<uint8_t>(q));
    };
    auto check_start = [&](const cv::Point2d& start, const cv::Point2d& next, const char* name) {
        const int a = label_at(start), b = label_at(next);
        if (a != 0 || (b != 0 && finger_of(b) != kPalm))
            std::cout << "warning: " << name << " starts on the hand or runs onto a finger" << std::endl;
    };

    // Life line (thenar crease): arcs around the thenar eminence, widest about a third of
    // the way across the palm, and ends at the wrist, back towards the thumb, or short.
    {
        const double span = rng.uniform(0.0, 1.0);
        const double xw = 4.65 + 0.6 * span;
        std::vector<cv::Point2d> pts = {
            life_start,
            jitter(life_start + cv::Point2d(0.5, 0.8), 0.05),
            jitter(anat(4.25 + 0.3 * span, 11.6), 0.08),
            jitter(anat(xw - 0.1, 12.9), 0.08),
            jitter(anat(xw, 14.3), 0.08),
            jitter(anat(xw - 0.15, 15.6), 0.08)
        };
        const int end = pick({55, 25, 20});
        if (end == 0) {
            pts.push_back(jitter(anat(xw - 0.35, 16.8), 0.08));
            pts.push_back(jitter(anat(xw - 0.55, 17.6), 0.08));
        } else if (end == 1) {
            pts.push_back(jitter(anat(xw - 0.5, 16.6), 0.08));
            pts.push_back(jitter(anat(3.9, 17.2), 0.1));
        } else {
            pts.push_back(jitter(anat(xw - 0.3, 16.0), 0.08));
        }
        check_start(pts[0], pts[1], "life line");
        add_curve(pts, 0.08, 0.2, end == 2 ? 0.03 : 0.06, rng);
        if (rng.uniform(0, 100) < 25)
            fork(pts[5], cv::Point2d(0.6, 1.0), rng.uniform(0.8, 1.4), 0.05);
        summary += std::string(" life=") + (end == 0 ? "wrist" : end == 1 ? "thumb-base" : "short")
            + (span > 0.5 ? "/wide" : "/narrow");
    }

    if (simian) {
        // Single transverse crease replacing heart and head lines.
        add_curve({life_start, jitter(anat(4.4, 10.8), 0.08), jitter(anat(6.3, 11.25), 0.1),
            jitter(anat(8.5, 11.5), 0.1), jitter(anat(10.95, 11.6), 0.08)}, 0.08, 0.17, 0.08, rng);
        summary += " simian";
    } else {
        // Head line (proximal transverse crease): shares its first centimetre with the life
        // line, or starts on its own above it.
        const double tilt = rng.uniform(-0.25, 0.25);
        std::vector<cv::Point2d> head;
        if (separate) {
            // Not higher than about a third of the way down: above that the start can land on
            // the index base when the fingers are spread.
            const cv::Point2d s = jitter(radial_at(0.35), 0.04);
            head = {s, jitter(s + cv::Point2d(0.7, 0.7), 0.05), jitter(anat(5.0, 10.75), 0.06)};
        } else {
            head = {life_start, jitter(life_start + cv::Point2d(0.5, 0.8), 0.03), jitter(anat(4.8, 11.05), 0.06)};
        }
        head.push_back(jitter(anat(6.2, 11.65 + tilt * 0.5), 0.08));
        // Straight and drooping head lines split what the Sydney line leaves.
        const int sydney_w = cvRound(pop_->sydney * 10);
        const int head_end = pick({(1000 - sydney_w) * 53 / 100, (1000 - sydney_w) * 47 / 100, sydney_w});
        if (head_end == 0) {
            head.push_back(jitter(anat(7.4, 12.1 + tilt), 0.08));
            head.push_back(jitter(anat(8.4, 12.45 + tilt), 0.12));
        } else if (head_end == 1) {
            head.push_back(jitter(anat(7.2, 12.35 + tilt * 0.7), 0.08));
            head.push_back(jitter(anat(8.0, 13.2 + tilt * 0.5), 0.1));
            head.push_back(jitter(anat(8.6, 14.1), 0.15));
        } else {
            head.push_back(jitter(anat(7.6, 12.2 + tilt), 0.08));
            head.push_back(jitter(anat(9.3, 12.7 + tilt), 0.08));
            head.push_back(jitter(anat(11.0, 13.0 + tilt), 0.08));
        }
        check_start(head[0], head[1], "head line");
        add_curve(head, 0.07, 0.15, head_end == 2 ? 0.07 : 0.04, rng);
        if (head_end != 2 && rng.uniform(0, 100) < 25)
            fork(head[head.size() - 2], cv::Point2d(1.0, head_end == 1 ? -0.3 : 0.6), rng.uniform(0.6, 1.0), 0.04);

        // Heart line (distal transverse crease): from the ulnar border towards the index.
        // A separate head line passes close under the index, so the heart line stops short.
        const double yh = rng.uniform(11.0, 11.5);
        std::vector<cv::Point2d> heart = {
            anat(10.65, yh), jitter(anat(9.6, yh - 0.25), 0.06), jitter(anat(8.3, yh - 0.55), 0.08)
        };
        const int heart_end = separate ? (pick({40, 20}) == 0 ? 0 : 3) : pick({40, 25, 15, 20});
        if (heart_end == 0) {
            heart.push_back(jitter(anat(6.3, yh - 0.9), 0.08));
            heart.push_back(jitter(anat(5.6, yh - 1.15), 0.06));
            heart.push_back(jitter(anat(5.3, 9.8), 0.05));
        } else if (heart_end == 1) {
            heart.push_back(jitter(anat(6.3, yh - 0.85), 0.08));
            heart.push_back(jitter(anat(5.2, 10.15), 0.06));
            heart.push_back(jitter(anat(4.5, 9.85), 0.05));
        } else if (heart_end == 2) {
            heart.push_back(jitter(anat(6.0, yh - 0.7), 0.08));
            heart.push_back(jitter(anat(4.8, 10.4), 0.06));
            heart.push_back(jitter(anat(4.4, 10.25), 0.04));
        } else {
            heart.push_back(jitter(anat(7.0, yh - 0.75), 0.08));
            heart.push_back(jitter(anat(6.3, 10.05), 0.05));
        }
        add_curve(heart, 0.05, 0.13, 0.04, rng);
        if (rng.uniform(0, 100) < 35) {
            const cv::Point2d& tip = heart[heart.size() - 2];
            fork(tip, cv::Point2d(-0.6, rng.uniform(0, 2) ? 0.5 : -0.4), rng.uniform(0.5, 0.9), 0.04);
        }
        const char* heart_names[] = {"index-middle", "index", "radial", "middle"};
        const char* head_names[] = {"straight", "droop", "sydney"};
        summary = std::string(" heart=") + heart_names[heart_end] + " head=" + head_names[head_end]
            + (separate ? "/separate" : "/joined") + summary;
    }

    // Fate line: thin, often broken, from the wrist towards the middle finger.
    if (rng.uniform(0, 100) < 40) {
        const double x = rng.uniform(6.0, 6.5);
        const double top = simian ? 11.5 : rng.uniform(10.6, 12.0);
        if (rng.uniform(0, 2)) {
            add_curve({anat(x, 17.25), jitter(anat(x - 0.05, 15.5), 0.08), jitter(anat(x - 0.2, 13.8), 0.08),
                jitter(anat(x - 0.4, top), 0.06)}, 0.03, 0.06, 0.02, rng);
        } else {
            add_curve({anat(x, 17.25), jitter(anat(x - 0.05, 16.0), 0.06), jitter(anat(x - 0.1, 14.9), 0.06)}, 0.03, 0.05, 0.02, rng);
            add_curve({jitter(anat(x - 0.15, 14.4), 0.05), jitter(anat(x - 0.25, 13.2), 0.06), jitter(anat(x - 0.4, top), 0.06)},
                0.02, 0.05, 0.02, rng);
        }
        summary += " fate";
    }
    // Sun line under the ring finger.
    if (rng.uniform(0, 100) < 15) {
        add_curve({jitter(anat(8.1, 12.3), 0.1), jitter(anat(8.05, 11.6), 0.06), jitter(anat(7.95, 10.9), 0.06)}, 0.02, 0.04, 0.02, rng);
        summary += " sun";
    }
    line_summary_ = summary;
}

void PalmPrint::add_joint_crease(const Finger& finger, double along, double width_cm, double bulge_cm,
    double reach_across) {
    cv::Point2d mid = finger.base + finger.axis_dir * along;
    cv::Point2d a = mid - finger.across * (finger.width * 0.70);
    cv::Point2d c = mid + finger.across * (finger.width * reach_across);
    cv::Point2d bow = (a + c) * 0.5 + finger.axis_dir * bulge_cm;
    add_tapered(a, bow, c, width_cm * 0.75, width_cm, width_cm * 0.7);
}

void PalmPrint::rasterize() {
    const int width = std::max(1, cvRound(bounds_.width * scale_) + margin_ * 2);
    const int height = std::max(1, cvRound(bounds_.height * scale_) + margin_ * 2);
    labels_ = cv::Mat::zeros(height, width, CV_8U);
    creases_ = cv::Mat::zeros(height, width, CV_8U);

    const std::vector<cv::Point> palm_px = palm_outline_px();
    cv::Mat palm_mask = cv::Mat::zeros(labels_.size(), CV_8U);
    cv::fillPoly(palm_mask, std::vector<std::vector<cv::Point>>{palm_px}, cv::Scalar(255), cv::LINE_8);
    labels_.setTo(pack(CENTRAL, kPalm), palm_mask);

    cv::Mat shape(labels_.size(), CV_8U);
    auto paint_palm = [&](int region) {
        cv::bitwise_and(shape, palm_mask, shape);
        labels_.setTo(pack(region, kPalm), shape);
        shape.setTo(0);
    };
    shape.setTo(0);
    fill_ellipse(shape, {9.05, 14.35}, 2.05, 3.35, 78);
    paint_palm(HYPOTHENAR);
    const cv::Point inter_px[] = {
        to_px(anat(3.70, 9.15)), to_px(anat(9.55, 9.55)), to_px(anat(9.35, 11.25)), to_px(anat(4.15, 11.05))
    };
    cv::fillConvexPoly(shape, inter_px, 4, cv::Scalar(255), cv::LINE_8);
    paint_palm(INTERDIGITAL);
    fill_ellipse(shape, {3.15, 14.25}, 2.25, 3.15, 68);
    paint_palm(THENAR);
    const cv::Point wrist_px[] = {
        to_px(anat(3.15, 16.7)), to_px(anat(9.55, 16.7)), to_px(anat(9.45, 18.7)), to_px(anat(3.35, 18.7))
    };
    cv::fillConvexPoly(shape, wrist_px, 4, cv::Scalar(255), cv::LINE_8);
    paint_palm(WRIST);

    for (int i = 0; i < kFingers; ++i) {
        const Finger& f = fingers_[i];
        auto width_at = [&](double s) {
            return f.width * (1.0 - 0.14 * std::clamp(s / f.length, 0.0, 1.0));
        };
        add_segment(kPalm, i == 0 ? THENAR : INTERDIGITAL, f.base - f.axis_dir * 0.7, f.base,
            width_at(0), width_at(0), false);
        for (int s = 0; s < f.segments; ++s) {
            int region = s == 0 ? PROXIMAL : (s == f.segments - 1 ? DISTAL : MIDDLE);
            cv::Point2d a = f.base + f.axis_dir * f.seg_at[s];
            cv::Point2d b = f.base + f.axis_dir * f.seg_at[s + 1];
            add_segment(i, region, a, b, width_at(f.seg_at[s]), width_at(f.seg_at[s + 1]), region == DISTAL);
        }
        // The thumb's base crease sits on the thenar; its ulnar end stops short of the life line.
        add_joint_crease(f, 0.0, 0.16, rng_.uniform(-0.12, 0.12), i == 0 ? 0.45 : 0.70);
        if (f.segments == 3) {
            double pip = f.seg_at[1];
            add_joint_crease(f, pip - 0.14, 0.11, rng_.uniform(-0.08, 0.08));
            add_joint_crease(f, pip + 0.06, 0.2, rng_.uniform(-0.1, 0.1));
            if (f.pip_lines >= 3)
                add_joint_crease(f, pip + 0.24, 0.07, rng_.uniform(-0.05, 0.05));
            add_joint_crease(f, f.seg_at[2], 0.13, rng_.uniform(-0.08, 0.08));
        } else {
            add_joint_crease(f, f.seg_at[1] - 0.1, 0.12, rng_.uniform(-0.08, 0.08));
            add_joint_crease(f, f.seg_at[1] + 0.12, 0.17, rng_.uniform(-0.08, 0.08));
        }
    }

    // Palm creases draw from their own stream so editing them leaves the ridge noise alone.
    cv::RNG lines(stream_seed(0));
    add_principal_lines(lines);
    auto jitter = [&](const cv::Point2d& p, double amount) {
        return p + cv::Point2d(lines.uniform(-amount, amount), lines.uniform(-amount, amount));
    };
    // Wrist crease.
    add_curve({jitter(anat(3.4, 17.25), 0.08), jitter(anat(5.0, 17.6), 0.08), jitter(anat(6.4, 17.7), 0.08),
        jitter(anat(7.9, 17.55), 0.08), jitter(anat(9.55, 17.2), 0.06)}, 0.04, 0.1, 0.03, lines);

    for (int n = 0; n < 9; ++n) {
        bool thenar = n < 5;
        cv::Point2d a = thenar ? cv::Point2d(lines.uniform(2.1, 4.0), lines.uniform(12.2, 16.2))
                               : cv::Point2d(lines.uniform(4.4, 8.8), lines.uniform(9.5, 11.0));
        cv::Point2d c = a + cv::Point2d(lines.uniform(0.5, 1.6), lines.uniform(-0.45, 0.55));
        cv::Point2d b = (a + c) * 0.5 + cv::Point2d(lines.uniform(-0.25, 0.25), lines.uniform(-0.3, 0.3));
        add_curve({anat(a), anat(b), anat(c)}, 0.02, 0.04, 0.02, lines);
    }

    for (const auto& stroke : strokes_)
        for (size_t i = 0; i < stroke.pts.size(); ++i)
            cv::circle(creases_, stroke.pts[i], stroke.radii[i], cv::Scalar(255), cv::FILLED, cv::LINE_8);
    creases_.setTo(0, labels_ == 0);
}

double PalmPrint::finger_field(const Finger& f, int region, const cv::Point2d& p) const {
    cv::Point2d rel = p - f.base;
    double along = rel.dot(f.axis_dir);
    double u = std::clamp(rel.dot(f.across) / (f.width * 0.5), -1.2, 1.2);
    double across = f.axis + CV_PI / 2.0;
    // Fine waves (wavelength 0.6-1.2 cm, under 0.1 rad) so pads and phalanges are not textbook-clean.
    for (int k = 0; k < 3; ++k)
        across += f.waves[k].z * std::sin(f.waves[k].x * p.x + f.waves[k].y * p.y + f.wave_phase[k]);
    if (region == MIDDLE) return across + f.slope + 0.18 * u;
    if (region == PROXIMAL) return across + 0.5 * f.slope + f.prox_bow * u;

    const double bend = std::sin(std::clamp(u, -1.0, 1.0) * CV_PI / 2.0);
    if (f.pattern == ARCH || f.pattern == TENTED_ARCH) {
        double start = f.seg_at[f.segments - 1];
        double v = (along - start) / std::max(0.2, f.length - start);
        double k = f.pattern == TENTED_ARCH ? 3.2 : 2.0;
        double alpha = std::max(0.0, k - k * (1.0 - v) / 0.8);
        return across + std::atan(alpha) * bend;
    }

    double theta = across + singularity(p, f.cores, f.n_core, f.deltas, f.n_delta, f.pattern);
    // Above the loop head or whorl the ridges follow the rounded tip (same sign as the arch
    // branch). Start 0.2 cm above the core so the head itself keeps its shape.
    cv::Point2d core = f.n_core == 2 ? (f.cores[0] + f.cores[1]) * 0.5 : f.cores[0];
    double core_along = (core - f.base).dot(f.axis_dir) + 0.2;
    double v = smoothstep((along - core_along) / std::max(0.2, f.length - core_along));
    theta += f.tip_arch * v * bend;
    // A whorl is a spiral, not a bullseye: a constant offset on the ring angle near the core.
    if (f.twist != 0) theta += f.twist * std::exp(-norm2d(p - core) / 0.35);
    return theta;
}

double PalmPrint::palm_field(int region, const cv::Point2d& p) const {
    switch (region) {
    case INTERDIGITAL: {
        // Each digital triradius pairs with a virtual core inside its finger. An interdigital
        // loop adds a core just below the finger gap and a triradius under it, so the loop
        // opens towards the gap.
        cv::Point2d cores[7], deltas[7];
        int n = 0;
        for (int i = 0; i < 4; ++i, ++n) {
            cores[n] = tri_[i] + cv::Point2d(0, -3.2);
            deltas[n] = tri_[i];
        }
        for (int k = 0; k < 3; ++k) {
            if (!inter_loop_[k]) continue;
            const cv::Point2d gap = (fingers_[k + 1].base + fingers_[k + 2].base) * 0.5;
            cores[n] = gap + cv::Point2d(0, kLoopCoreDepth);
            deltas[n] = gap + cv::Point2d(0, kLoopDeltaDepth);
            ++n;
        }
        return sherlock(p, cores, n, deltas, n, 0.0);
    }
    case THENAR: {
        if (!thenar_whorl_) {
            cv::Point2d c = fingers_[0].base + cv::Point2d(-0.6, -0.9);
            return std::atan2(p.y - c.y, p.x - c.x) + CV_PI / 2.0;
        }
        const double tilt = 1.05;
        cv::Point2d d = p - thenar_core_;
        double ex = d.x * std::cos(tilt) + d.y * std::sin(tilt);
        double ey = -d.x * std::sin(tilt) + d.y * std::cos(tilt);
        return std::atan2(ey / (0.55 * 0.55), ex) + tilt + CV_PI / 2.0;
    }
    case HYPOTHENAR: {
        cv::Point2d c = hypo_pattern_ == ARCH ? anat(12.0, 19.5) : anat(10.9, 15.2);
        return std::atan2(p.y - c.y, p.x - c.x) + CV_PI / 2.0;
    }
    case WRIST: {
        cv::Point2d core = tri_[4] + cv::Point2d(0, 3.9);
        return sherlock(p, &core, 1, &tri_[4], 1, 0.0);
    }
    default:
        return central_angle_;
    }
}

void PalmPrint::build_orientation() {
    orientation_ = cv::Mat::zeros(labels_.size(), CV_32F);
    cv::Mat warp(7, 5, CV_32F);
    rng_.fill(warp, cv::RNG::UNIFORM, -0.14, 0.14);
    cv::resize(warp, warp, labels_.size(), 0, 0, cv::INTER_CUBIC);

    // Palm regions meet without seams: blend their fields in doubled-angle space
    // with weights from blurred region masks.
    const int palm_regions[] = {INTERDIGITAL, THENAR, HYPOTHENAR, CENTRAL, WRIST};
    constexpr int kPalmRegions = 5;
    cv::Mat weights[kPalmRegions];
    for (int k = 0; k < kPalmRegions; ++k) {
        cv::Mat mask(labels_.size(), CV_32F, cv::Scalar(0));
        for (int y = 0; y < labels_.rows; ++y) {
            const uint8_t* lab = labels_.ptr<uint8_t>(y);
            float* m = mask.ptr<float>(y);
            for (int x = 0; x < labels_.cols; ++x)
                if (lab[x] && region_of(lab[x]) == palm_regions[k]) m[x] = 1.f;
        }
        cv::GaussianBlur(mask, weights[k], cv::Size(0, 0), px(0.5));
    }

    cv::parallel_for_(cv::Range(0, labels_.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const uint8_t* lab = labels_.ptr<uint8_t>(y);
            const float* wp = warp.ptr<float>(y);
            float* ang = orientation_.ptr<float>(y);
            for (int x = 0; x < labels_.cols; ++x) {
                if (!lab[x]) continue;
                const int region = region_of(lab[x]);
                const int finger = finger_of(lab[x]);
                const cv::Point2d p = to_cm(x, y);
                double theta;
                if (region <= PROXIMAL && finger < kFingers) {
                    theta = finger_field(fingers_[finger], region, p);
                } else {
                    double c2 = 0, s2 = 0;
                    for (int k = 0; k < kPalmRegions; ++k) {
                        double w = weights[k].at<float>(y, x);
                        if (w < 0.002) continue;
                        double t = palm_field(palm_regions[k], p);
                        c2 += w * std::cos(2 * t);
                        s2 += w * std::sin(2 * t);
                    }
                    theta = 0.5 * std::atan2(s2, c2);
                }
                ang[x] = static_cast<float>(warp_angle(p, theta) + wp[x]);
            }
        }
    });
}

void PalmPrint::build_period() {
    period_ = cv::Mat::zeros(labels_.size(), CV_32F);
    cv::Mat low(6, 5, CV_32F);
    rng_.fill(low, cv::RNG::UNIFORM, -1.0, 1.0);
    cv::resize(low, low, labels_.size(), 0, 0, cv::INTER_CUBIC);
    // Relative to the fingertip-scale base period. Palm ridges are about 20-30% coarser than
    // fingertip ridges (ridge density about 10-12 vs 13-16 per 5 mm diagonal).
    const double factor[] = {1.0, 0.95, 1.0, 1.05, 1.2, 1.25, 1.25, 1.27, 1.2};

    for (int y = 0; y < labels_.rows; ++y) {
        const uint8_t* lab = labels_.ptr<uint8_t>(y);
        const float* noise = low.ptr<float>(y);
        float* per = period_.ptr<float>(y);
        for (int x = 0; x < labels_.cols; ++x) {
            if (!lab[x]) continue;
            double p = base_period_ * factor[std::clamp(region_of(lab[x]), 0, 8)] * (1.0 + 0.07 * noise[x]);
            per[x] = static_cast<float>(std::clamp(p, period_min_, period_max_));
        }
    }
}

void PalmPrint::render_ridges() {
    // Iterative oriented Gabor filtering from noise (as in fp::generate_ridge), but with
    // kernels stretched along the ridge so lines stay continuous, and a soft nonlinearity
    // so the result keeps gray levels.
    const auto t0 = std::chrono::steady_clock::now();
    const int rows = labels_.rows;
    const int cols = labels_.cols;

    cv::Mat active;
    cv::compare(labels_, 0, active, cv::CMP_GT);
    active.setTo(0, creases_);

    std::vector<cv::Mat> kernels(kAngles * kPeriods);
    std::vector<int> halves(kAngles * kPeriods);
    for (int a = 0; a < kAngles; ++a) {
        for (int b = 0; b < kPeriods; ++b) {
            double period = period_min_ + (period_max_ - period_min_) * (b + 0.5) / kPeriods;
            double ridge = CV_PI * (a + 0.5) / kAngles;
            double sigma = 0.42 * period;
            double gamma = 0.55;
            int half = cvCeil(2.4 * sigma / gamma);
            cv::Mat k = cv::getGaborKernel(cv::Size(2 * half + 1, 2 * half + 1), sigma,
                ridge + CV_PI / 2.0, period, gamma, 0, CV_32F);
            k -= cv::mean(k)[0];
            kernels[a * kPeriods + b] = k;
            halves[a * kPeriods + b] = half;
        }
    }

    cv::Mat bins(labels_.size(), CV_16U, cv::Scalar(0));
    for (int y = 0; y < rows; ++y) {
        const uchar* act = active.ptr<uchar>(y);
        const float* ang = orientation_.ptr<float>(y);
        const float* per = period_.ptr<float>(y);
        uint16_t* bin = bins.ptr<uint16_t>(y);
        for (int x = 0; x < cols; ++x) {
            if (!act[x]) continue;
            double t = std::fmod(static_cast<double>(ang[x]), CV_PI);
            if (t < 0) t += CV_PI;
            int a = std::min(kAngles - 1, static_cast<int>(t / CV_PI * kAngles));
            int b = static_cast<int>((per[x] - period_min_) / (period_max_ - period_min_) * kPeriods);
            bin[x] = static_cast<uint16_t>(a * kPeriods + std::clamp(b, 0, kPeriods - 1));
        }
    }

    cv::Mat cur(labels_.size(), CV_32F);
    rng_.fill(cur, cv::RNG::UNIFORM, -1.0, 1.0);
    cur.setTo(0, active == 0);
    cv::Mat next(labels_.size(), CV_32F, cv::Scalar(0));

    for (int iter = 0; iter < kIterations; ++iter) {
        cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                const uchar* act = active.ptr<uchar>(y);
                const uint16_t* bin = bins.ptr<uint16_t>(y);
                float* out = next.ptr<float>(y);
                for (int x = 0; x < cols; ++x) {
                    if (!act[x]) {
                        out[x] = 0;
                        continue;
                    }
                    const cv::Mat& k = kernels[bin[x]];
                    const int h = halves[bin[x]];
                    const int y0 = std::max(0, y - h), y1 = std::min(rows - 1, y + h);
                    const int x0 = std::max(0, x - h), x1 = std::min(cols - 1, x + h);
                    float sum = 0;
                    for (int yy = y0; yy <= y1; ++yy) {
                        const float* src = cur.ptr<float>(yy) + x0;
                        const float* kr = k.ptr<float>(yy - y + h) + (x0 - x + h);
                        for (int i = 0, n = x1 - x0; i <= n; ++i) sum += src[i] * kr[i];
                    }
                    out[x] = sum;
                }
            }
        });

        double sq = 0;
        long long count = 0;
        for (int y = 0; y < rows; ++y) {
            const uchar* act = active.ptr<uchar>(y);
            const float* v = next.ptr<float>(y);
            for (int x = 0; x < cols; ++x)
                if (act[x]) {
                    sq += static_cast<double>(v[x]) * v[x];
                    ++count;
                }
        }
        const double gain = 1.8 / std::sqrt(std::max(1e-12, sq / std::max(1LL, count)));
        for (int y = 0; y < rows; ++y) {
            const float* v = next.ptr<float>(y);
            float* c = cur.ptr<float>(y);
            for (int x = 0; x < cols; ++x) c[x] = static_cast<float>(std::tanh(v[x] * gain));
        }
        std::cout << "ridge pass " << iter + 1 << "/" << kIterations << " (" << seconds_since(t0) << " s)" << std::endl;
    }
    ridges_ = cur;
}

void PalmPrint::composite() {
    cv::Mat hand;
    cv::compare(labels_, 0, hand, cv::CMP_GT);
    cv::Mat edge_dist;
    cv::distanceTransform(hand, edge_dist, cv::DIST_L2, 3);

    cv::Mat grain(labels_.size(), CV_32F);
    rng_.fill(grain, cv::RNG::UNIFORM, 0.0, 1.0);
    cv::Mat blotch(9, 6, CV_32F);
    rng_.fill(blotch, cv::RNG::UNIFORM, 0.0, 1.0);
    cv::resize(blotch, blotch, labels_.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat rim(labels_.size(), CV_32F);
    rng_.fill(rim, cv::RNG::UNIFORM, 0.0, 1.0);
    cv::GaussianBlur(rim, rim, cv::Size(0, 0), std::max(1.0, px(0.08)));
    cv::normalize(rim, rim, -1.0, 1.0, cv::NORM_MINMAX, -1, hand);

    const double fade = px(0.15);
    cv::Mat ink(labels_.size(), CV_32F, cv::Scalar(0));
    for (int y = 0; y < labels_.rows; ++y) {
        const uchar* h = hand.ptr<uchar>(y);
        const float* v = ridges_.ptr<float>(y);
        const float* g = grain.ptr<float>(y);
        const float* bl = blotch.ptr<float>(y);
        const float* e = edge_dist.ptr<float>(y);
        const float* r = rim.ptr<float>(y);
        float* out = ink.ptr<float>(y);
        for (int x = 0; x < labels_.cols; ++x) {
            if (!h[x]) continue;
            double i = smoothstep((v[x] + 0.15) / 0.65);
            i *= 0.8 + 0.2 * g[x];
            i *= 0.78 + 0.22 * bl[x];
            i *= smoothstep(e[x] / fade + 0.45 * r[x]);
            out[x] = static_cast<float>(i);
        }
    }

    // Ridges stop short of the crease wall in broken, uneven ends.
    cv::Mat band = cv::Mat::zeros(labels_.size(), CV_8U);
    for (const auto& stroke : strokes_)
        for (size_t i = 0; i < stroke.pts.size(); ++i) {
            int reach = stroke.ragged + cvRound(base_period_ * rng_.uniform(0.2, 1.2));
            cv::circle(band, stroke.pts[i], stroke.radii[i] + reach, cv::Scalar(255), cv::FILLED, cv::LINE_8);
        }
    cv::Mat broken(labels_.size(), CV_32F);
    rng_.fill(broken, cv::RNG::UNIFORM, 0.0, 1.0);
    cv::GaussianBlur(broken, broken, cv::Size(0, 0), base_period_ * 0.5);
    cv::Scalar broken_mean = cv::mean(broken);
    cv::Mat ragged = (broken > broken_mean[0]) & band;
    ink.setTo(0, creases_);
    ink.setTo(0, ragged);

    cv::Mat ridge_bin = (ridges_ > 0.3f) & hand;
    ridge_bin.setTo(0, creases_);
    cv::Mat dist;
    cv::distanceTransform(ridge_bin, dist, cv::DIST_L2, 3);
    // Pores: one candidate per cell of the pore spacing, at the cell's most central ridge
    // pixel, skipped when it would crowd an already placed neighbour.
    // Fractional cell size: rounding to whole pixels would merge nearby spacings (at 500 dpi
    // 0.028-0.031 cm all round to 6 px), so cell edges are floor(i * cell).
    const double cell = std::max(2.0, px(pore_spacing_cm_));
    auto edge = [&](int i) { return static_cast<int>(std::floor(i * cell)); };
    const int radius = base_period_ >= 8 ? 1 : 0;
    const float center = static_cast<float>(std::max(1.0, base_period_ * 0.2));
    const int gx = cvCeil(ink.cols / cell), gy = cvCeil(ink.rows / cell);
    std::vector<cv::Point> placed(static_cast<size_t>(gx) * gy, cv::Point(-1, -1));
    const double min_gap2 = std::pow(0.6 * cell, 2);
    int pores = 0, distal_pores = 0;
    for (int cy = 0; cy < gy; ++cy) {
        for (int cx = 0; cx < gx; ++cx) {
            // A random pixel near the ridge centre, not the exact centre, so pores do not
            // line up like stitching.
            float top = center;
            for (int y = edge(cy); y < std::min(ink.rows, edge(cy + 1)); ++y) {
                const float* d = dist.ptr<float>(y);
                for (int x = edge(cx); x < std::min(ink.cols, edge(cx + 1)); ++x) top = std::max(top, d[x]);
            }
            if (top <= center || rng_.uniform(0.0, 1.0) > 0.8) continue;
            const float near_top = std::max(center, 0.7f * top);
            int seen = 0;
            cv::Point best(-1, -1);
            for (int y = edge(cy); y < std::min(ink.rows, edge(cy + 1)); ++y) {
                const float* d = dist.ptr<float>(y);
                for (int x = edge(cx); x < std::min(ink.cols, edge(cx + 1)); ++x)
                    if (d[x] >= near_top && rng_.uniform(0, ++seen) == 0) best = {x, y};
            }
            if (best.x < 0) continue;
            bool crowded = false;
            const int nb[4][2] = {{-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
            for (const auto& o : nb) {
                const int nx = cx + o[0], ny = cy + o[1];
                if (nx < 0 || ny < 0 || nx >= gx) continue;
                const cv::Point q = placed[static_cast<size_t>(ny) * gx + nx];
                if (q.x >= 0 && std::pow(q.x - best.x, 2) + std::pow(q.y - best.y, 2) < min_gap2) crowded = true;
            }
            if (crowded) continue;
            placed[static_cast<size_t>(cy) * gx + cx] = best;
            ++pores;
            if (region_of(labels_.at<uint8_t>(best)) == DISTAL) ++distal_pores;
            if (radius == 0)
                ink.at<float>(best) *= 0.15f;
            else
                cv::circle(ink, best, radius + (rng_.uniform(0, 4) == 0 ? 1 : 0), cv::Scalar(0), cv::FILLED, cv::LINE_8);
        }
    }
    {
        long long distal_px = 0;
        for (int y = 0; y < labels_.rows; ++y) {
            const uint8_t* lab = labels_.ptr<uint8_t>(y);
            const uint8_t* cr = creases_.ptr<uint8_t>(y);
            for (int x = 0; x < labels_.cols; ++x)
                if (lab[x] && region_of(lab[x]) == DISTAL && !cr[x]) ++distal_px;
        }
        const double mm2 = distal_px * std::pow(25.4 / dpi_, 2);
        char buf[120];
        std::snprintf(buf, sizeof(buf), " count=%d distal=%.1f/9mm2", pores, mm2 > 0 ? 9.0 * distal_pores / mm2 : 0.0);
        pore_summary_ = buf;
    }

    ink.convertTo(image_, CV_8U, -235.0, 255.0);

    if (right_hand_) {
        cv::flip(image_, image_, 1);
        cv::flip(labels_, labels_, 1);
        cv::flip(creases_, creases_, 1);
        cv::flip(orientation_, orientation_, 1);
        orientation_ = -orientation_;
        cv::flip(period_, period_, 1);
        cv::flip(ridges_, ridges_, 1);
    }
}

void PalmPrint::generate() {
    std::cout << "palm dpi=" << dpi_ << " seed=" << seed_ << (right_hand_ ? " right" : " left")
              << " population=" << pop_->name << std::endl;
    build_geometry();
    std::cout << "ridge period=" << base_period_ << " px" << std::endl;
    // Pores sit on ridge centres; below about 8 px per ridge (roughly 420 dpi) there are too
    // few centre pixels and fingertip pore density drops to about half the real value.
    if (base_period_ < 8)
        std::cout << "note: at " << dpi_ << " dpi pores come out at about half the real density; use -dpi 500 for pore work" << std::endl;
    rasterize();
    std::cout << "canvas " << labels_.cols << "x" << labels_.rows << std::endl;
    std::cout << "shape" << shape_summary_ << std::endl;
    std::cout << "lines" << line_summary_ << std::endl;
    std::cout << "fingers";
    for (const auto& f : fingers_) std::cout << " " << f.name << "=" << pattern_name(f.pattern);
    std::cout << std::endl;
    std::cout << "palm thenar=" << (thenar_whorl_ ? "whorl" : "arc") << " I2=" << inter_loop_[0]
              << " I3=" << inter_loop_[1] << " I4=" << inter_loop_[2]
              << " hypothenar=" << (hypo_pattern_ == ARCH ? "arch" : "loop") << std::endl;
    build_orientation();
    build_period();
    render_ridges();
    composite();
    std::cout << "pores" << pore_summary_ << std::endl;
}

bool PalmPrint::save(const std::string& path) const {
    std::filesystem::path file(path);
    if (file.has_parent_path()) std::filesystem::create_directories(file.parent_path());
    return !image_.empty() && cv::imwrite(path, image_);
}

bool PalmPrint::save_layers(const std::string& directory) const {
    std::filesystem::create_directories(directory);
    const cv::Vec3b palette[] = {
        {0, 0, 0}, {40, 80, 220}, {40, 180, 80}, {200, 160, 40},
        {180, 80, 200}, {60, 140, 220}, {80, 80, 200}, {160, 160, 160}, {120, 80, 40}
    };
    cv::Mat regions(labels_.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat hsv(labels_.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat period_u8(labels_.size(), CV_8U, cv::Scalar(0));
    cv::Mat ridge_u8(labels_.size(), CV_8U, cv::Scalar(255));
    for (int y = 0; y < labels_.rows; ++y) {
        const uint8_t* lab = labels_.ptr<uint8_t>(y);
        const float* ang = orientation_.ptr<float>(y);
        const float* per = period_.ptr<float>(y);
        const float* ridge = ridges_.ptr<float>(y);
        cv::Vec3b* reg = regions.ptr<cv::Vec3b>(y);
        cv::Vec3b* hrow = hsv.ptr<cv::Vec3b>(y);
        uchar* prow = period_u8.ptr<uchar>(y);
        uchar* rrow = ridge_u8.ptr<uchar>(y);
        for (int x = 0; x < labels_.cols; ++x) {
            if (!lab[x]) continue;
            reg[x] = palette[std::clamp(region_of(lab[x]), 0, 8)];
            double hue = std::fmod(std::fmod(static_cast<double>(ang[x]), CV_PI) + CV_PI, CV_PI) / CV_PI;
            hrow[x] = cv::Vec3b(static_cast<uchar>(hue * 179), 200, 255);
            double t = (per[x] - period_min_) / std::max(1e-6, period_max_ - period_min_);
            prow[x] = static_cast<uchar>(std::clamp(t, 0.0, 1.0) * 255);
            rrow[x] = static_cast<uchar>(std::clamp(127.5 - 127.5 * ridge[x], 0.0, 255.0));
        }
    }
    cv::Mat orient_bgr;
    cv::cvtColor(hsv, orient_bgr, cv::COLOR_HSV2BGR);
    bool ok = true;
    ok = cv::imwrite(directory + "/regions.png", regions) && ok;
    ok = cv::imwrite(directory + "/orientation.png", orient_bgr) && ok;
    ok = cv::imwrite(directory + "/period.png", period_u8) && ok;
    ok = cv::imwrite(directory + "/ridges.png", ridge_u8) && ok;
    ok = cv::imwrite(directory + "/creases.png", creases_) && ok;
    ok = cv::imwrite(directory + "/palmprint.jpg", image_) && ok;
    return ok;
}
