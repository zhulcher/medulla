#ifndef CUTS_KAON
#define CUTS_KAON

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <cassert>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "utilities.h"
#include "framework.h"

// ---------------------------
// C++ structs
// ---------------------------
struct Particle
{
    int pdg_code;
    double reco_length;
    int shape;
    int pid;
    bool is_matched;
    bool is_primary;
    double calo_ke;
    double reco_ke;
    long id;
    
    std::vector<long> match_ids;


    std::array<double, 3> start_point_; // store raw data in C++
    std::array<double, 3> end_point_;   // store raw data in C++


    std::array<double, 3> momentum_;   // store raw data in C++
    std::array<double, 3> start_dir_;   // store raw data in C++
    std::array<double, 3> end_dir_;   // store raw data in C++

    // Python sees these as np.array via property
};

struct Interaction
{
    std::vector<double> reco_vertex;
    std::vector<Particle> particles;
    bool is_flash_matched;
};


// ---------------------------
// Function to register types
// ---------------------------
inline void register_bindings(py::module_ &m)
{
    static bool done = false;
    if (done)
        return;
    done = true;

    py::class_<Particle>(m, "Particle")
        .def(py::init<>())
        .def_readwrite("pdg_code", &Particle::pdg_code)
        .def_readwrite("reco_length", &Particle::reco_length)
        .def_readwrite("shape", &Particle::shape)
        .def_readwrite("pid", &Particle::pid)
        .def_readwrite("is_matched", &Particle::is_matched)
        .def_readwrite("is_primary", &Particle::is_primary)
        .def_readwrite("calo_ke", &Particle::calo_ke)
        .def_readwrite("reco_ke", &Particle::reco_ke)
        .def_readwrite("match_ids", &Particle::match_ids)
        .def_readwrite("id", &Particle::id)

        // start_point as numpy array
        .def_property("start_point", [](Particle &self)
                      { return py::array_t<double>(
                            {3},                      // shape
                            {sizeof(double)},         // stride
                            self.start_point_.data(), // pointer
                            py::cast(&self)           // base object ensures lifetime
                        ); }, [](Particle &self, py::array_t<double> arr)
                      { 
                if (arr.size() != 3) throw std::runtime_error("start_point must have length 3");
                auto a = arr.unchecked<1>();
                for (ssize_t i = 0; i < 3; ++i) self.start_point_[i] = a(i); })

        // end_point as numpy array
        .def_property("end_point", [](Particle &self)
                      { return py::array_t<double>(
                            {3},                    // shape
                            {sizeof(double)},       // stride
                            self.end_point_.data(), // pointer to the C++ array
                            py::cast(&self)         // base object to keep memory alive
                        ); }, [](Particle &self, py::array_t<double> arr)
                      {
                if (arr.size() != 3) throw std::runtime_error("end_point must have length 3");
                auto a = arr.unchecked<1>();
                for (int i = 0; i < 3; ++i) self.end_point_[i] = a(i); })

        // start_dir as numpy array
        .def_property("start_dir", [](Particle &self)
                      { return py::array_t<double>(
                            {3},                    // shape
                            {sizeof(double)},       // stride
                            self.start_dir_.data(), // pointer to the C++ array
                            py::cast(&self)         // base object to keep memory alive
                        ); }, [](Particle &self, py::array_t<double> arr)
                      {
                if (arr.size() != 3) throw std::runtime_error("start_dir must have length 3");
                auto a = arr.unchecked<1>();
                for (int i = 0; i < 3; ++i) self.start_dir_[i] = a(i); })

        // end_dir as numpy array
        .def_property("end_dir", [](Particle &self)
                      { return py::array_t<double>(
                            {3},                  // shape
                            {sizeof(double)},     // stride
                            self.end_dir_.data(), // pointer to the C++ array
                            py::cast(&self)       // base object to keep memory alive
                        ); }, [](Particle &self, py::array_t<double> arr)
                      {
                if (arr.size() != 3) throw std::runtime_error("end_dir must have length 3");
                auto a = arr.unchecked<1>();
                for (int i = 0; i < 3; ++i) self.end_dir_[i] = a(i); })

        // momentum as numpy array
        .def_property("momentum", [](Particle &self)
                      { return py::array_t<double>(
                            {3},                   // shape
                            {sizeof(double)},      // stride
                            self.momentum_.data(), // pointer to the C++ array
                            py::cast(&self)        // base object to keep memory alive
                        ); }, [](Particle &self, py::array_t<double> arr)
                      {
                if (arr.size() != 3) throw std::runtime_error("momentum must have length 3");
                auto a = arr.unchecked<1>();
                for (int i = 0; i < 3; ++i) self.momentum_[i] = a(i); });

    py::class_<Interaction>(m, "Interaction")
        .def(py::init<>())
        .def_readwrite("reco_vertex", &Interaction::reco_vertex)
        .def_readwrite("particles", &Interaction::particles)
        .def_readwrite("is_flash_matched", &Interaction::is_flash_matched);
}
// ---------------------------
// Pass cuts function
// ---------------------------
template <class T>
bool pass_cuts(const T &obj, std::vector<double> cut_params = {})
{

    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, "../../2x2_Strange");

    static py::module_ analysis = py::module_::import("analysis.analysis_cuts");
    static bool bindings_registered = false;
    if (!bindings_registered) {
        register_bindings(analysis);
        bindings_registered = true;
    }

    std::vector<std::string> strings = {
        "Fiducialization", "Valid Interaction", "Primary $K^+$",
        "Initial HIP", "Valid MIP Len", "Michel Child",
        "Connected Non-Primary MIP", "Low MIP len $\\pi^0$ Tag",
        "Come to Rest", "Kaon Len", "Close to Vertex",
        "MIP Child At Most 1 Michel", "No HIP Deltas",
        "Come to Rest MIP", "Single MIP Decay", "Bragg Peak HIP"};

    assert(cut_params.size() == 16);
    assert(strings.size() == 16);

    std::map<std::string, double> the_map;
    for (size_t i = 0; i < strings.size(); ++i)
        the_map[strings[i]] = cut_params[i];
    the_map[""] = 1.0;

    // Create Python-exposed Interaction
    Interaction py_obj;
    py_obj.reco_vertex = {obj.vertex[0], obj.vertex[1], obj.vertex[2]};
    py_obj.is_flash_matched=obj.is_flash_matched;
    for (const auto &p : obj.particles)
    {
        Particle py_p;
        py_p.pdg_code = p.pdg_code;
        py_p.reco_length=p.length;
        py_p.shape=p.shape;
        py_p.pid=p.pid;
        py_p.is_matched=p.is_matched;
        py_p.is_primary=p.is_primary;
        py_p.calo_ke=p.calo_ke;
        py_p.reco_ke=p.ke;
        py_p.id=p.id;

        py_p.match_ids.clear();
        for (auto id : p.match_ids) // assuming p.match_ids exists in your original object
            py_p.match_ids.push_back(id);

        // Fill the data
        for (int i = 0; i < 3; ++i)
        {
            py_p.start_point_[i] = p.start_point[i]; // raw C++ array
            py_p.end_point_[i] = p.end_point[i];

            py_p.start_dir_[i] = p.start_dir[i];
            py_p.end_dir_[i] = p.end_dir[i];

            py_p.momentum_[i] = p.momentum[i];
        }

        // py_p.ancestor_pdg_code = p.ancestor_pdg_code;
        // py_p.energy = pvars::ke(p);
        py_obj.particles.push_back(py_p);
    }

    for (const auto &particle : py_obj.particles)
    {
        // Now pybind11 knows about Interaction and Particle
        try {
    if (analysis.attr("K_plus_cut_cascade")(py_obj, the_map, particle).template cast<bool>())
        return true;
        } catch (py::error_already_set &e) {
            std::cerr << "Python error: " << e.what() << std::endl;
            return false; // or handle appropriately
}
    }

    return false;
}

REGISTER_CUT_SCOPE(RegistrationScope::Both, pass_cuts, pass_cuts);

// ---------------------------
// Single particle K+ multiplicity cut
// ---------------------------
template <class T>
bool at_least_one_kaon_plus(const T &obj, std::vector<double> params = {40.0})
{
    for (const auto &p : obj.particles)
    {
        if (p.pdg_code == 321 && p.ancestor_pdg_code == 321 && pvars::ke(p) >= params[0])
            return true;
    }
    return false;
}

REGISTER_CUT_SCOPE(RegistrationScope::True, at_least_one_kaon_plus, at_least_one_kaon_plus);

#endif // CUTS_KAON