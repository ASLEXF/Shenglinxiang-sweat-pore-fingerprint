#pragma once

#include "population.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

// Whole-hand palmar print. Distal pads use the same core/delta sum as fp::generate_orientation.
// Middle and proximal phalanges, the palm, and flexion creases are separate fields.
class PalmPrint {
public:
    PalmPrint(int dpi, unsigned seed, bool right_hand, const Population& population);

    void generate();
    bool save(const std::string& path) const;
    bool save_layers(const std::string& directory) const;

    const cv::Mat& image() const { return image_; }

private:
    enum Pattern {
        ARCH = 1,
        TENTED_ARCH = 2,
        WHORL = 3,
        LEFT_LOOP = 4,
        RIGHT_LOOP = 5,
        TWIN_LOOP = 6
    };

    enum Region {
        BG = 0,
        DISTAL = 1,
        MIDDLE = 2,
        PROXIMAL = 3,
        INTERDIGITAL = 4,
        THENAR = 5,
        HYPOTHENAR = 6,
        CENTRAL = 7,
        WRIST = 8
    };

    struct Stroke {
        std::vector<cv::Point> pts;
        std::vector<int> radii;
        int ragged = 0;
    };

    struct Finger {
        const char* name = "";
        cv::Point2d base;
        double axis = -CV_PI / 2;
        double length = 7;
        double width = 1.6;
        int segments = 3;
        double frac[3] = {0.42, 0.32, 0.26};
        int pattern = LEFT_LOOP;
        double slope = 0;
        double prox_bow = 0;
        int pip_lines = 1;
        cv::Point2d radial;
        cv::Point2d axis_dir;
        cv::Point2d across;
        cv::Point2d cores[2];
        cv::Point2d deltas[2];
        int n_core = 0;
        int n_delta = 0;
        double seg_at[4] = {};
        // Distal pad shaping on top of singularity(): tip arch, whorl spiral, fine waves.
        double tip_arch = 0;
        double twist = 0;
        cv::Point3d waves[3];
        double wave_phase[3] = {};
    };

    static int pack(int region, int finger) { return region * 8 + finger; }
    static int region_of(int label) { return label >> 3; }
    static int finger_of(int label) { return label & 7; }

    double px(double cm) const { return cm * scale_; }
    // to_px/to_cm take and return canonical cm; the per-seed hand-shape warp sits between.
    cv::Point to_px(const cv::Point2d& p) const;
    cv::Point2d to_cm(int x, int y) const;
    uint64_t stream_seed(int stream) const;
    cv::Point2d warp(const cv::Point2d& p) const;
    cv::Point2d unwarp(const cv::Point2d& q) const;
    double warp_angle(const cv::Point2d& p, double theta) const;
    double px_len(double cm) const { return px(cm * length_scale_); }
    void fill_ellipse(cv::Mat& mask, const cv::Point2d& center, double a, double b, double angle_deg) const;
    std::vector<cv::Point> palm_outline_px() const;

    void build_geometry();
    void rasterize();
    void build_orientation();
    void build_period();
    void render_ridges();
    void composite();

    double finger_field(const Finger& f, int region, const cv::Point2d& p) const;
    double palm_field(int region, const cv::Point2d& p) const;

    void add_segment(int finger, int region, const cv::Point2d& a, const cv::Point2d& b,
        double width_a, double width_b, bool cap);
    void add_joint_crease(const Finger& finger, double along, double width_cm, double bulge_cm,
        double reach_across = 0.70);
    void add_tapered(const cv::Point2d& a, const cv::Point2d& b, const cv::Point2d& c,
        double width_start, double width_mid, double width_end);
    void add_curve(const std::vector<cv::Point2d>& ctrl, double width_start, double width_mid,
        double width_end, cv::RNG& rng);
    void add_principal_lines(cv::RNG& rng);
    static std::vector<cv::Point2d> bezier(const cv::Point2d& a, const cv::Point2d& b, const cv::Point2d& c, int n);
    static std::vector<cv::Point2d> catmull_rom(const std::vector<cv::Point2d>& ctrl, double step);
    static double singularity(const cv::Point2d& p, const cv::Point2d* cores, int n_core,
        const cv::Point2d* deltas, int n_delta, int pattern);
    static double sherlock(const cv::Point2d& p, const cv::Point2d* cores, int n_core,
        const cv::Point2d* deltas, int n_delta, double base);

    int dpi_;
    const Population* pop_;
    int sex_ = 0;  // 0 female, 1 male
    unsigned seed_;
    bool right_hand_;
    double scale_ = 1;
    int margin_ = 0;
    cv::Rect2d bounds_;
    cv::RNG rng_;

    Finger fingers_[5];
    cv::Point2d tri_[5];
    cv::Point2d thenar_core_;
    bool thenar_whorl_ = true;
    // Interdigital loops between index-middle (I2), middle-ring (I3) and ring-little (I4).
    bool inter_loop_[3] = {};
    int hypo_pattern_ = ARCH;
    double central_angle_ = 0.4;
    std::string line_summary_;

    // Hand shape: overall size, palm width and length, and a taper (palm wider at the
    // knuckles or at the wrist). The model is built in a fixed canonical frame and warped.
    double shape_scale_ = 1;
    double shape_ax_ = 1;
    double shape_ay_ = 1;
    double shape_taper_ = 0;
    double length_scale_ = 1;
    std::string shape_summary_;

    // Ridge period in pixels. Never below 5 px so a ridge and a furrow both stay visible.
    double base_period_ = 5;
    double period_min_ = 4;
    double period_max_ = 7;
    // Pore placement cell, set per sex from the population preset (centre spacing is
    // measured at 210-650 um).
    double pore_spacing_cm_ = 0.030;
    std::string pore_summary_;

    cv::Mat labels_;
    cv::Mat creases_;
    cv::Mat orientation_;
    cv::Mat period_;
    cv::Mat ridges_;
    cv::Mat image_;
    std::vector<Stroke> strokes_;
};
