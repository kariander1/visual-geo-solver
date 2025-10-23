#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <limits>
#include <random>
#include <stdexcept>
#include <numeric>
#include <set>
#include <string>

constexpr double EPS = 1e-12;

struct Point {
    double x, y;
    Point(double x = 0, double y = 0) : x(x), y(y) {}
};

class MaxAreaPolygonSolver {
private:
    std::vector<Point> points;
    std::vector<std::vector<std::vector<std::vector<bool>>>> cross_table;
    std::vector<bool> used;
    std::vector<int> path;
    std::vector<int> best_order;
    std::vector<int> angle_order;
    double best_area;
    int anchor;
    double cx, cy; // centroid
    std::chrono::steady_clock::time_point start_time;
    double time_limit;
    bool use_time_limit;

    inline double orientation(double ax, double ay, double bx, double by, double cx, double cy) const {
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    }

    inline bool on_segment(double ax, double ay, double bx, double by, double cx, double cy) const {
        return (std::min(ax, bx) - EPS <= cx && cx <= std::max(ax, bx) + EPS &&
                std::min(ay, by) - EPS <= cy && cy <= std::max(ay, by) + EPS);
    }

    bool segments_properly_intersect(const Point& a, const Point& b, const Point& c, const Point& d) const {
        double o1 = orientation(a.x, a.y, b.x, b.y, c.x, c.y);
        double o2 = orientation(a.x, a.y, b.x, b.y, d.x, d.y);
        double o3 = orientation(c.x, c.y, d.x, d.y, a.x, a.y);
        double o4 = orientation(c.x, c.y, d.x, d.y, b.x, b.y);
        
        if ((o1 * o2 < -EPS) && (o3 * o4 < -EPS)) return true;
        
        if (std::abs(o1) <= EPS && on_segment(a.x, a.y, b.x, b.y, c.x, c.y)) {
            if ((std::abs(c.x - a.x) > EPS || std::abs(c.y - a.y) > EPS) && 
                (std::abs(c.x - b.x) > EPS || std::abs(c.y - b.y) > EPS)) return true;
        }
        if (std::abs(o2) <= EPS && on_segment(a.x, a.y, b.x, b.y, d.x, d.y)) {
            if ((std::abs(d.x - a.x) > EPS || std::abs(d.y - a.y) > EPS) && 
                (std::abs(d.x - b.x) > EPS || std::abs(d.y - b.y) > EPS)) return true;
        }
        if (std::abs(o3) <= EPS && on_segment(c.x, c.y, d.x, d.y, a.x, a.y)) {
            if ((std::abs(a.x - c.x) > EPS || std::abs(a.y - c.y) > EPS) && 
                (std::abs(a.x - d.x) > EPS || std::abs(a.y - d.y) > EPS)) return true;
        }
        if (std::abs(o4) <= EPS && on_segment(c.x, c.y, d.x, d.y, b.x, b.y)) {
            if ((std::abs(b.x - c.x) > EPS || std::abs(b.y - c.y) > EPS) && 
                (std::abs(b.x - d.x) > EPS || std::abs(b.y - d.y) > EPS)) return true;
        }
        return false;
    }

    void precompute_crossings() {
        int n = points.size();
        cross_table.assign(n, std::vector<std::vector<std::vector<bool>>>(
            n, std::vector<std::vector<bool>>(n, std::vector<bool>(n, false))));
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                for (int k = 0; k < n; k++) {
                    for (int l = 0; l < n; l++) {
                        if (k == l || i == k || i == l || j == k || j == l) continue;
                        if (segments_properly_intersect(points[i], points[j], points[k], points[l])) {
                            cross_table[i][j][k][l] = true;
                        }
                    }
                }
            }
        }
    }

    int choose_anchor() const {
        int best = 0;
        double bx = points[0].x, by = points[0].y;
        for (int i = 0; i < points.size(); i++) {
            double x = points[i].x, y = points[i].y;
            if ((y < by - EPS) || (std::abs(y - by) <= EPS && x < bx - EPS)) {
                best = i;
                bx = x;
                by = y;
            }
        }
        return best;
    }

    void compute_angle_order() {
        int n = points.size();
        cx = cy = 0;
        for (const auto& p : points) {
            cx += p.x;
            cy += p.y;
        }
        cx /= n;
        cy /= n;

        std::vector<std::pair<double, int>> angles;
        for (int i = 0; i < n; i++) {
            double angle = std::atan2(points[i].y - cy, points[i].x - cx);
            angles.emplace_back(angle, i);
        }
        std::sort(angles.begin(), angles.end());
        
        angle_order.clear();
        for (const auto& a : angles) {
            angle_order.push_back(a.second);
        }
    }

    double polygon_area_signed(const std::vector<int>& order) const {
        double area2 = 0.0;
        int n = order.size();
        for (int i = 0; i < n; i++) {
            const Point& p1 = points[order[i]];
            const Point& p2 = points[order[(i + 1) % n]];
            area2 += p1.x * p2.y - p2.x * p1.y;
        }
        return area2 * 0.5;
    }

    bool last_edge_intersects(int v) const {
        if (path.size() < 2) return false;
        int u = path.back();
        
        for (int i = 0; i < (int)path.size() - 2; i++) {
            int p = path[i];
            int q = path[i + 1];
            if (cross_table[u][v][p][q]) return true;
        }
        return false;
    }

    bool closing_edge_intersects() const {
        int u = path.back();
        int v = path[0];
        
        for (int i = 0; i < (int)path.size() - 1; i++) {
            int p = path[i];
            int q = path[i + 1];
            if (p == u || p == v || q == u || q == v) continue;
            if (cross_table[u][v][p][q]) return true;
        }
        return false;
    }

    bool check_time_limit() const {
        if (!use_time_limit) return false;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - start_time).count();
        return elapsed > time_limit;
    }

    void dfs() {
        if (check_time_limit()) return;
        
        int n = points.size();
        if (path.size() == n) {
            if (closing_edge_intersects()) return;
            
            double s = polygon_area_signed(path);
            if (s <= EPS) return;
            
            double area = std::abs(s);
            if (area > best_area + 1e-15) {
                best_area = area;
                best_order = path;
            }
            return;
        }

        for (int v : angle_order) {
            if (used[v]) continue;
            if (path.size() == n - 1 && v == anchor) continue;
            if (last_edge_intersects(v)) continue;
            
            used[v] = true;
            path.push_back(v);
            dfs();
            path.pop_back();
            used[v] = false;
        }
    }

public:
    std::pair<std::vector<int>, double> solve(const std::vector<Point>& input_points, 
                                            double time_limit_seconds = -1,
                                            bool precompute_cross = true) {
        points = input_points;
        int n = points.size();
        
        if (n < 3) {
            return {std::vector<int>(), -1.0};
        }

        start_time = std::chrono::steady_clock::now();
        time_limit = time_limit_seconds;
        use_time_limit = (time_limit_seconds > 0);

        anchor = choose_anchor();
        if (precompute_cross) {
            precompute_crossings();
        }
        compute_angle_order();

        used.assign(n, false);
        used[anchor] = true;
        path.clear();
        path.push_back(anchor);
        best_order.clear();
        best_area = -1.0;

        dfs();
        
        return {best_order, best_area};
    }

    // Static convenience function
    static std::pair<std::vector<int>, double> max_area_polygon(
        const std::vector<Point>& points,
        double time_limit = -1,
        bool precompute_cross = true) {
        
        MaxAreaPolygonSolver solver;
        return solver.solve(points, time_limit, precompute_cross);
    }
};

class RandomPointGenerator {
private:
    std::mt19937 rng;
    
public:
    RandomPointGenerator(uint32_t seed = 0) : rng(seed) {}
    
    std::vector<Point> generate_random_points(
        int num_points,
        double min_dist = 0.15,
        double node_radius = 7.0,
        double edge_width = 2.0,
        double image_size = 128.0,
        int max_attempts = 1000) {
        
        // Calculate padding to keep nodes away from image edges
        double pad = (node_radius + edge_width) / image_size;
        
        std::vector<Point> points;
        points.reserve(num_points);
        
        std::uniform_real_distribution<double> dist(pad, 1.0 - pad);
        
        int attempts = 0;
        while (points.size() < static_cast<size_t>(num_points) && attempts < max_attempts) {
            Point candidate(dist(rng), dist(rng));
            
            // Check minimum distance constraint
            bool valid = true;
            for (const auto& p : points) {
                double dx = candidate.x - p.x;
                double dy = candidate.y - p.y;
                double distance = std::sqrt(dx*dx + dy*dy);
                if (distance <= min_dist) {
                    valid = false;
                    break;
                }
            }
            
            if (valid) {
                points.push_back(candidate);
            }
            attempts++;
        }
        
        if (points.size() < static_cast<size_t>(num_points)) {
            throw std::runtime_error("Could not place " + std::to_string(num_points) + 
                                   " non-overlapping points after " + std::to_string(max_attempts) + 
                                   " attempts");
        }
        
        return points;
    }
    
    // Static convenience function
    static std::vector<Point> generate_points(
        int num_points,
        uint32_t seed,
        double min_dist = 0.15,
        double node_radius = 7.0,
        double edge_width = 2.0,
        double image_size = 128.0,
        int max_attempts = 1000) {
        
        RandomPointGenerator generator(seed);
        return generator.generate_random_points(
            num_points, min_dist, node_radius, edge_width, image_size, max_attempts);
    }
};

class RandomPermutationGenerator {
private:
    std::mt19937 rng;
    
    // Check if a polygon defined by order is simple (non-self-intersecting)
    bool is_simple_polygon(const std::vector<Point>& points, const std::vector<int>& order) const {
        if (order.size() < 3) return false;
        
        int n = order.size();
        for (int i = 0; i < n; i++) {
            Point edge1_start = points[order[i]];
            Point edge1_end = points[order[(i + 1) % n]];
            
            for (int j = i + 2; j < n; j++) {
                if (i == 0 && j == n - 1) continue;
                
                Point edge2_start = points[order[j]];
                Point edge2_end = points[order[(j + 1) % n]];
                
                if (segments_properly_intersect(edge1_start, edge1_end, edge2_start, edge2_end)) {
                    return false;
                }
            }
        }
        return true;
    }
    
    // Check if segments properly intersect (reuse from MaxAreaPolygonSolver)
    bool segments_properly_intersect(const Point& a, const Point& b, const Point& c, const Point& d) const {
        auto orientation = [](double ax, double ay, double bx, double by, double cx, double cy) -> double {
            return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        };
        
        auto on_segment = [](double ax, double ay, double bx, double by, double cx, double cy) -> bool {
            return (std::min(ax, bx) - EPS <= cx && cx <= std::max(ax, bx) + EPS &&
                    std::min(ay, by) - EPS <= cy && cy <= std::max(ay, by) + EPS);
        };
        
        double o1 = orientation(a.x, a.y, b.x, b.y, c.x, c.y);
        double o2 = orientation(a.x, a.y, b.x, b.y, d.x, d.y);
        double o3 = orientation(c.x, c.y, d.x, d.y, a.x, a.y);
        double o4 = orientation(c.x, c.y, d.x, d.y, b.x, b.y);
        
        if ((o1 * o2 < -EPS) && (o3 * o4 < -EPS)) return true;
        
        if (std::abs(o1) <= EPS && on_segment(a.x, a.y, b.x, b.y, c.x, c.y)) {
            if ((std::abs(c.x - a.x) > EPS || std::abs(c.y - a.y) > EPS) && 
                (std::abs(c.x - b.x) > EPS || std::abs(c.y - b.y) > EPS)) return true;
        }
        if (std::abs(o2) <= EPS && on_segment(a.x, a.y, b.x, b.y, d.x, d.y)) {
            if ((std::abs(d.x - a.x) > EPS || std::abs(d.y - a.y) > EPS) && 
                (std::abs(d.x - b.x) > EPS || std::abs(d.y - b.y) > EPS)) return true;
        }
        if (std::abs(o3) <= EPS && on_segment(c.x, c.y, d.x, d.y, a.x, a.y)) {
            if ((std::abs(a.x - c.x) > EPS || std::abs(a.y - c.y) > EPS) && 
                (std::abs(a.x - d.x) > EPS || std::abs(a.y - d.y) > EPS)) return true;
        }
        if (std::abs(o4) <= EPS && on_segment(c.x, c.y, d.x, d.y, b.x, b.y)) {
            if ((std::abs(b.x - c.x) > EPS || std::abs(b.y - c.y) > EPS) && 
                (std::abs(b.x - d.x) > EPS || std::abs(b.y - d.y) > EPS)) return true;
        }
        return false;
    }
    
    // Calculate polygon area
    double polygon_area(const std::vector<Point>& points, const std::vector<int>& order) const {
        double area2 = 0.0;
        int n = order.size();
        for (int i = 0; i < n; i++) {
            const Point& p1 = points[order[i]];
            const Point& p2 = points[order[(i + 1) % n]];
            area2 += p1.x * p2.y - p2.x * p1.y;
        }
        return std::abs(area2) * 0.5;
    }
    
    // Calculate polygon perimeter
    double polygon_perimeter(const std::vector<Point>& points, const std::vector<int>& order) const {
        double perimeter = 0.0;
        int n = order.size();
        for (int i = 0; i < n; i++) {
            const Point& p1 = points[order[i]];
            const Point& p2 = points[order[(i + 1) % n]];
            double dx = p1.x - p2.x;
            double dy = p1.y - p2.y;
            perimeter += std::sqrt(dx*dx + dy*dy);
        }
        return perimeter;
    }
    
public:
    struct PolygonVariation {
        std::vector<int> order;
        double area;
        double perimeter;
        double compactness;
        std::string strategy;
        
        PolygonVariation(const std::vector<int>& o, double a, double p, const std::string& s)
            : order(o), area(a), perimeter(p), strategy(s) {
            compactness = (p > 0) ? (4.0 * M_PI * a) / (p * p) : 0.0;
        }
    };
    
    RandomPermutationGenerator(uint32_t seed = 0) : rng(seed) {}
    
    // Generate a single random permutation polygon
    std::vector<int> generate_single_random_permutation(
        const std::vector<Point>& points,
        int max_attempts = 1000) {
        
        int n = points.size();
        if (n < 3) return {};
        
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        
        for (int attempt = 0; attempt < max_attempts; attempt++) {
            std::shuffle(order.begin(), order.end(), rng);
            
            if (is_simple_polygon(points, order)) {
                return order;
            }
        }
        
        return {}; // Failed to find valid polygon
    }
    
    // Generate multiple random permutation polygons
    std::vector<PolygonVariation> generate_random_permutations(
        const std::vector<Point>& points,
        int num_variations = 5,
        double min_area_ratio = 0.1,
        int max_attempts_per_variation = 100) {
        
        std::vector<PolygonVariation> variations;
        std::set<std::vector<int>> seen_normalized_orders;
        
        auto normalize_order = [](const std::vector<int>& order) -> std::vector<int> {
            if (order.empty()) return order;
            
            // Find minimum element and rotate to start from there
            auto min_it = std::min_element(order.begin(), order.end());
            int min_idx = std::distance(order.begin(), min_it);
            
            std::vector<int> normalized;
            normalized.insert(normalized.end(), order.begin() + min_idx, order.end());
            normalized.insert(normalized.end(), order.begin(), order.begin() + min_idx);
            
            // Also check reverse direction
            std::vector<int> reversed = {normalized[0]};
            for (int i = normalized.size() - 1; i >= 1; i--) {
                reversed.push_back(normalized[i]);
            }
            
            // Return lexicographically smaller one
            return std::min(normalized, reversed);
        };
        
        int total_attempts = 0;
        int max_total_attempts = num_variations * max_attempts_per_variation;
        
        while (variations.size() < static_cast<size_t>(num_variations) && 
               total_attempts < max_total_attempts) {
            
            total_attempts++;
            
            auto order = generate_single_random_permutation(points, max_attempts_per_variation);
            if (order.empty()) continue;
            
            // Check for duplicates
            auto normalized = normalize_order(order);
            if (seen_normalized_orders.count(normalized)) continue;
            
            double area = polygon_area(points, order);
            double perimeter = polygon_perimeter(points, order);
            
            // Apply area filter
            if (area < min_area_ratio * 0.1) continue;
            
            variations.emplace_back(order, area, perimeter, "random_permutation");
            seen_normalized_orders.insert(normalized);
        }
        
        // Aggressive mode if not enough variations
        while (variations.size() < static_cast<size_t>(num_variations) && 
               total_attempts < max_total_attempts * 2) {
            
            total_attempts++;
            
            auto order = generate_single_random_permutation(points, 200);
            if (order.empty()) continue;
            
            auto normalized = normalize_order(order);
            if (seen_normalized_orders.count(normalized)) continue;
            
            double area = polygon_area(points, order);
            double perimeter = polygon_perimeter(points, order);
            
            // More lenient area filter
            if (area < min_area_ratio * 0.05) continue;
            
            variations.emplace_back(order, area, perimeter, "random_permutation_aggressive");
            seen_normalized_orders.insert(normalized);
        }
        
        // Sort by area (largest first)
        std::sort(variations.begin(), variations.end(), 
                  [](const PolygonVariation& a, const PolygonVariation& b) {
                      return a.area > b.area;
                  });
        
        return variations;
    }
    
    // Static convenience function
    static std::vector<PolygonVariation> generate_random_polygon_variations(
        const std::vector<Point>& points,
        uint32_t seed = 0,
        int num_variations = 5,
        double min_area_ratio = 0.1,
        int max_attempts_per_variation = 100) {
        
        RandomPermutationGenerator generator(seed);
        return generator.generate_random_permutations(
            points, num_variations, min_area_ratio, max_attempts_per_variation);
    }
};

// C-style interface for Python binding
extern "C" {
    struct PolygonResult {
        int* order;
        int order_length;
        double area;
        int success;
    };

    PolygonResult* solve_max_area_polygon_c(double* points_x, double* points_y, 
                                           int num_points, double time_limit) {
        std::vector<Point> points;
        points.reserve(num_points);
        
        for (int i = 0; i < num_points; i++) {
            points.emplace_back(points_x[i], points_y[i]);
        }

        auto result = MaxAreaPolygonSolver::max_area_polygon(points, time_limit, true);
        
        PolygonResult* c_result = new PolygonResult;
        
        if (result.second > 0 && !result.first.empty()) {
            c_result->order_length = result.first.size();
            c_result->order = new int[c_result->order_length];
            for (int i = 0; i < c_result->order_length; i++) {
                c_result->order[i] = result.first[i];
            }
            c_result->area = result.second;
            c_result->success = 1;
        } else {
            c_result->order = nullptr;
            c_result->order_length = 0;
            c_result->area = -1.0;
            c_result->success = 0;
        }
        
        return c_result;
    }

    void free_polygon_result(PolygonResult* result) {
        if (result) {
            if (result->order) {
                delete[] result->order;
            }
            delete result;
        }
    }
}

#ifdef PYTHON_BINDING
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

std::pair<std::vector<int>, double> solve_max_area_polygon_py(
    py::array_t<double> points_array, 
    double time_limit = -1,
    bool precompute_cross = true) {
    
    py::buffer_info buf = points_array.request();
    
    if (buf.ndim != 2 || buf.shape[1] != 2) {
        throw std::runtime_error("Input array must be N x 2");
    }
    
    int num_points = buf.shape[0];
    double* ptr = (double*) buf.ptr;
    
    std::vector<Point> points;
    points.reserve(num_points);
    
    for (int i = 0; i < num_points; i++) {
        points.emplace_back(ptr[i * 2], ptr[i * 2 + 1]);
    }
    
    return MaxAreaPolygonSolver::max_area_polygon(points, time_limit, precompute_cross);
}

py::array_t<double> generate_random_points_py(
    int num_points,
    uint32_t seed = 0,
    double min_dist = 0.15,
    double node_radius = 7.0,
    double edge_width = 2.0,
    double image_size = 128.0,
    int max_attempts = 1000) {
    
    auto points = RandomPointGenerator::generate_points(
        num_points, seed, min_dist, node_radius, edge_width, image_size, max_attempts);
    
    // Create numpy array with correct shape
    std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(points.size()), 2};
    auto result = py::array_t<double>(shape);
    
    py::buffer_info buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    
    for (size_t i = 0; i < points.size(); i++) {
        ptr[i * 2] = points[i].x;
        ptr[i * 2 + 1] = points[i].y;
    }
    
    return result;
}

std::vector<py::dict> generate_random_polygons_py(
    py::array_t<double> points_array,
    uint32_t seed = 0,
    int num_variations = 5,
    double min_area_ratio = 0.1,
    int max_attempts_per_variation = 100) {
    
    py::buffer_info buf = points_array.request();
    
    if (buf.ndim != 2 || buf.shape[1] != 2) {
        throw std::runtime_error("Input array must be N x 2");
    }
    
    int num_points = buf.shape[0];
    double* ptr = (double*) buf.ptr;
    
    std::vector<Point> points;
    points.reserve(num_points);
    
    for (int i = 0; i < num_points; i++) {
        points.emplace_back(ptr[i * 2], ptr[i * 2 + 1]);
    }
    
    auto variations = RandomPermutationGenerator::generate_random_polygon_variations(
        points, seed, num_variations, min_area_ratio, max_attempts_per_variation);
    
    std::vector<py::dict> result;
    for (const auto& var : variations) {
        py::dict py_var;
        py_var["order"] = var.order;
        py_var["area"] = var.area;
        py_var["perimeter"] = var.perimeter;
        py_var["compactness"] = var.compactness;
        py_var["strategy"] = var.strategy;
        result.push_back(py_var);
    }
    
    return result;
}

PYBIND11_MODULE(max_area_polygon_cpp, m) {
    m.doc() = "High-performance C++ implementation of maximum area polygonalization";
    
    py::class_<Point>(m, "Point")
        .def(py::init<double, double>())
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y);
    
    py::class_<MaxAreaPolygonSolver>(m, "MaxAreaPolygonSolver")
        .def(py::init<>())
        .def("solve", &MaxAreaPolygonSolver::solve);
    
    py::class_<RandomPointGenerator>(m, "RandomPointGenerator")
        .def(py::init<uint32_t>(), py::arg("seed") = 0)
        .def("generate_random_points", &RandomPointGenerator::generate_random_points,
             py::arg("num_points"), py::arg("min_dist") = 0.15, 
             py::arg("node_radius") = 7.0, py::arg("edge_width") = 2.0,
             py::arg("image_size") = 128.0, py::arg("max_attempts") = 1000);
    
    py::class_<RandomPermutationGenerator::PolygonVariation>(m, "PolygonVariation")
        .def_readonly("order", &RandomPermutationGenerator::PolygonVariation::order)
        .def_readonly("area", &RandomPermutationGenerator::PolygonVariation::area)
        .def_readonly("perimeter", &RandomPermutationGenerator::PolygonVariation::perimeter)
        .def_readonly("compactness", &RandomPermutationGenerator::PolygonVariation::compactness)
        .def_readonly("strategy", &RandomPermutationGenerator::PolygonVariation::strategy);
    
    py::class_<RandomPermutationGenerator>(m, "RandomPermutationGenerator")
        .def(py::init<uint32_t>(), py::arg("seed") = 0)
        .def("generate_random_permutations", &RandomPermutationGenerator::generate_random_permutations,
             py::arg("points"), py::arg("num_variations") = 5, 
             py::arg("min_area_ratio") = 0.1, py::arg("max_attempts_per_variation") = 100);
    
    m.def("max_area_polygon", &solve_max_area_polygon_py, 
          "Solve maximum area polygonalization problem",
          py::arg("points"), py::arg("time_limit") = -1, py::arg("precompute_cross") = true);
    
    m.def("generate_random_points", &generate_random_points_py,
          "Generate random points with distance constraints",
          py::arg("num_points"), py::arg("seed") = 0, py::arg("min_dist") = 0.15,
          py::arg("node_radius") = 7.0, py::arg("edge_width") = 2.0,
          py::arg("image_size") = 128.0, py::arg("max_attempts") = 1000);
    
    m.def("generate_random_polygon_variations", &generate_random_polygons_py,
          "Generate random polygon variations using permutation strategy",
          py::arg("points"), py::arg("seed") = 0, py::arg("num_variations") = 5,
          py::arg("min_area_ratio") = 0.1, py::arg("max_attempts_per_variation") = 100);
}
#endif