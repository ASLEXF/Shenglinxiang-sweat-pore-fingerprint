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
        << "  -o <image>     final palmar image (jpg or png)\n"
        << "  -dpi <n>       pixels per inch (default 250). Below ~265 dpi ridges are drawn\n"
        << "                 at 5 px and come out wider than life. Below ~420 dpi fingertip pores\n"
        << "                 reach only about half the real density. Use 500 for pore work.\n"
        << "  -seed <n>      pattern seed (default 1)\n"
        << "  -population <name>  statistics to sample patterns and sizes from (default han):\n";
    for (const auto& p : kPopulations) std::cout << "                 " << p.name << ": " << p.note << "\n";
    std::cout
        << "  -right         mirror the left-hand model into a right hand\n"
        << "  -layers <dir>  also write region, orientation, ridge and crease layers\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string output;
    std::string layers;
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
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
