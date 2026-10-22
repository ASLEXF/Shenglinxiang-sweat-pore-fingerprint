#include "palm.h"

#include <opencv2/imgcodecs.hpp>

#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void show_help() {
    std::cout
        << "usage: palmprint -o <image> [-dpi <n>] [-seed <n>] [-population <name>] [-right] [-layers <dir>]\n"
        << "                 [-height <png>] [-relief <mm>] [-ridge-scale <f>] [-crease-scale <f>]\n"
        << "                 [-no-pores]\n"
        << "  -o <image>     final palmar image (jpg or png)\n"
        << "  -dpi <n>       pixels per inch (default 250). Below ~265 dpi ridges are drawn\n"
        << "                 at 5 px and come out wider than life. Below ~600 dpi fingertip pores\n"
        << "                 reach only about half the real density. Use 1000 for pore work.\n"
        << "  -seed <n>      pattern seed (default 1)\n"
        << "  -population <name>  statistics to sample patterns and sizes from (default han):\n";
    for (const auto& p : kPopulations) std::cout << "                 " << p.name << ": " << p.note << "\n";
    std::cout
        << "  -right         mirror the left-hand model into a right hand\n"
        << "  -layers <dir>  also write region, orientation, ridge and crease layers\n"
        << "  -ridge-scale <f>  multiply the ridge period (default 0.8, range 0.4-1.5); 1 is the\n"
        << "                 preset's own period, lower gives finer, denser ridges. The 5 px\n"
        << "                 floor still applies at low dpi.\n"
        << "  -crease-scale <f>  multiply the width of all creases: palm lines, finger joints, wrist\n"
        << "                 (default 0.5, range 0.2-1.5)\n"
        << "  -dump-creases <json>  write the crease centre lines and outline as JSON (for the depth editor)\n"
        << "  -no-pores      leave out the sweat pores (image and height map)\n"
        << "  -height <png>  write a 16 bit grey height map for a 3D model (mid grey 32768 = flat)\n"
        << "  -relief <mm>   ridge crest-to-furrow height the height map stands for (default 0.1,\n"
        << "                 an art-direction value, not a measurement); only changes the mm\n"
        << "                 figures printed, not the image\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string output;
    std::string layers;
    std::string height;
    double relief_mm = 0.1;
    double ridge_scale = 0.8;
    double crease_scale = 0.5;
    bool pores = true;
    std::string dump_creases;
    int dpi = 250;
    unsigned seed = 1;
    bool right = false;
    const Population* population = &kPopulations[0];

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << flag << "\n";
                show_help();
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            show_help();
            return 0;
        } else if (arg == "-o") {
            output = need("-o");
        } else if (arg == "-dpi") {
            dpi = std::atoi(need("-dpi"));
        } else if (arg == "-seed") {
            // atoi saturates at INT_MAX, which silently merged every larger seed into one.
            const char* text = need("-seed");
            char* end = nullptr;
            errno = 0;
            unsigned long long value = std::strtoull(text, &end, 10);
            if (*text == '-' || end == text || *end != '\0' || errno == ERANGE || value > 0xFFFFFFFFull) {
                std::cerr << "seed must be an integer in 0..4294967295: " << text << "\n";
                return 1;
            }
            seed = static_cast<unsigned>(value);
        } else if (arg == "-population") {
            const char* name = need("-population");
            population = find_population(name);
            if (!population) {
                std::cerr << "unknown population: " << name << "\n";
                show_help();
                return 1;
            }
        } else if (arg == "-right") {
            right = true;
        } else if (arg == "-layers") {
            layers = need("-layers");
        } else if (arg == "-height") {
            height = need("-height");
        } else if (arg == "-crease-scale") {
            const char* text = need("-crease-scale");
            char* end = nullptr;
            errno = 0;
            const double f = std::strtod(text, &end);
            if (end == text || *end != '\0' || errno == ERANGE || !(f >= 0.2 && f <= 1.5)) {
                std::cerr << "crease-scale must be a number in [0.2, 1.5]: " << text << "\n";
                return 1;
            }
            crease_scale = f;
        } else if (arg == "-dump-creases") {
            dump_creases = need("-dump-creases");
        } else if (arg == "-no-pores") {
            pores = false;
        } else if (arg == "-ridge-scale") {
            const char* text = need("-ridge-scale");
            char* end = nullptr;
            errno = 0;
            const double f = std::strtod(text, &end);
            if (end == text || *end != '\0' || errno == ERANGE || !(f >= 0.4 && f <= 1.5)) {
                std::cerr << "ridge-scale must be a number in [0.4, 1.5]: " << text << "\n";
                return 1;
            }
            ridge_scale = f;
        } else if (arg == "-relief") {
            const char* text = need("-relief");
            char* end = nullptr;
            errno = 0;
            const double mm = std::strtod(text, &end);
            if (end == text || *end != '\0' || errno == ERANGE || !(mm > 0.0 && mm <= 10.0)) {
                std::cerr << "relief must be a number in (0, 10] mm: " << text << "\n";
                return 1;
            }
            relief_mm = mm;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            show_help();
            return 1;
        }
    }

    if (output.empty() || dpi < 40) {
        show_help();
        return 1;
    }

    // Reject an unsupported extension before spending seconds on generation.
    if (!cv::haveImageWriter(output)) {
        std::cerr << "no image writer for " << output << " (use .jpg or .png)\n";
        return 1;
    }

    try {
        PalmPrint palm(dpi, seed, right, *population);
        palm.set_ridge_scale(ridge_scale);
        palm.set_crease_scale(crease_scale);
        palm.set_pores(pores);
        palm.generate();
        if (!palm.save(output)) {
            std::cerr << "failed to write " << output << "\n";
            return 1;
        }
        std::cout << "wrote " << output << "\n";
        if (!layers.empty() && !palm.save_layers(layers)) {
            std::cerr << "failed to write layers in " << layers << "\n";
            return 1;
        }
        if (!dump_creases.empty() && !palm.dump_creases(dump_creases)) {
            std::cerr << "failed to write " << dump_creases << "\n";
            return 1;
        }
        if (!height.empty() && !palm.save_height(height, relief_mm)) {
            std::cerr << "failed to write " << height << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
