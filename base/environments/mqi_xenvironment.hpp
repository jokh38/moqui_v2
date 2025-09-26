/// \file
///
/// \brief This file defines the x_environment class, a virtual base class for creating Monte Carlo simulation environments.
///
/// The x_environment class provides a framework for setting up and running Monte Carlo simulations.
/// It defines the common interface and data structures required for any simulation environment,
/// such as the world geometry, materials, and particle beam source.

#ifndef MQI_XENVIRONMENT_HPP
#define MQI_XENVIRONMENT_HPP

#include <chrono>
#include <strings.h>
#include <sys/stat.h>

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_node.hpp>
#include <moqui/base/materials/mqi_material.hpp>
#include <moqui/base/mqi_beamsource.hpp>
#include <moqui/base/mqi_cli.hpp>
#include <moqui/base/mqi_error_check.hpp>
#include <moqui/base/mqi_fippel_physics.hpp>
#include <moqui/base/mqi_io.hpp>
#include <moqui/base/mqi_scorer.hpp>
#include <moqui/base/mqi_threads.hpp>
#include <moqui/base/mqi_track.hpp>
#include <moqui/kernel_functions/mqi_kernel_functions.hpp>

namespace mqi
{

/// \class x_environment
/// \brief A virtual base class for creating Monte Carlo simulation environments.
/// \tparam R The floating-point type used for simulation (e.g., float or double).
///
/// This class defines the fundamental structure and interface for a Monte Carlo simulation environment.
/// It includes pointers to the world geometry, materials, and beam source, as well as methods
/// for initialization, execution, and finalization of the simulation. Derived classes must
/// implement the pure virtual methods to define the specific details of the environment.
template<typename R>
class x_environment
{

public:
    mqi::node_t<R>* world;   ///< Host pointer to the top-level geometry node (the world).
    mqi::node_t<R>* d_world; ///< Device pointer to the world geometry node.

    uint16_t            n_materials; ///< Number of materials in the simulation.
    mqi::material_t<R>* materials;   ///< Host pointer to the array of materials.
    mqi::material_t<R>* d_materials; ///< Device pointer to the array of materials.

    mqi::beamsource<R> beamsource; ///< The particle beam source for the simulation.

    mqi::vertex_t<R>* vertices;   ///< Host pointer to the primary particle vertices.
    mqi::vertex_t<R>* d_vertices; ///< Device pointer to the primary particle vertices.

    thrd_t* worker_threads; ///< Pointer to the worker threads for multi-threaded execution.

    int         num_total_threads; ///< The total number of threads to be used.
    int         gpu_id;            ///< The ID of the GPU to be used.
    uint32_t    num_spots;         ///< The number of beam spots.
    std::string output_path;       ///< The path to the output directory.
    std::string output_format;     ///< The format of the output files.

    std::default_random_engine beam_rng; ///< Random number engine for the beam source.

    /// \brief Default constructor for the x_environment class.
    CUDA_HOST
    x_environment() {
        ;
    }

    /// \brief Destructor for the x_environment class.
    CUDA_HOST
    ~x_environment() {
        ;
    }

    /// \brief Pure virtual method to set up the world geometry.
    CUDA_HOST
    virtual void
    setup_world() = 0;

    /// \brief Pure virtual method to set up the materials.
    CUDA_HOST
    virtual void
    setup_materials() = 0;

    /// \brief Pure virtual method to set up the beam source.
    CUDA_HOST
    virtual void
    setup_beamsource() = 0;

    /// \brief Pure virtual method to run the simulation.
    CUDA_HOST
    virtual void
    run() = 0;

    /// \brief Initializes the simulation environment.
    CUDA_HOST
    virtual void
    initialize() {
        auto start = std::chrono::high_resolution_clock::now();
        ///< call user defined method to setup world
        std::cout << "---------------------------------------------------------------------------" << std::endl;
        std::cout << "Starting process of creating world geometry and beam models.." << std::endl;
        std::cout << "---------------------------------------------------------------------------" << std::endl;
        this->setup_world();
        check_cuda_last_error("(setup world)");
        std::cout << "Creating world geometry complete!" << std::endl;

        this->setup_beamsource();

        check_cuda_last_error("(setup beamsource)");

#if defined(__CUDACC__)
        cudaMalloc(&mc::mc_world, sizeof(mqi::node_t<R>));
        std::cout << "---------------------------------------------------------------------------" << std::endl;
        std::cout << "Starting process of creating and uploading node to GPU.." << std::endl;
        std::cout << "---------------------------------------------------------------------------" << std::endl;
        mc::upload_node<R>(this->world, mc::mc_world);
        cudaDeviceSynchronize();
        check_cuda_last_error("(upload node)");
        // printf("world: %p, mc_world: %p, size(node_t): %lu\n",
        //        this->world,
        //        mc::mc_world,
        //        sizeof(mqi::node_t<R>));
#else

#endif
        ///< print upload results
#if defined(__CUDACC__)
        mc::print_node_specification<R><<<1, 1>>>(mc::mc_world);
        cudaDeviceSynchronize();
        check_cuda_last_error("print node");
#else
        mc::print_node_specification<R>(this->world);
        mc::print_materials<R>(this->materials, this->n_materials);
#endif
        auto                                      stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = stop - start;
        //printf("Initialization for geometry done %f ms\n", duration.count());
    }

    /// \brief Finalizes the simulation and cleans up resources.
    CUDA_HOST
    virtual void
    finalize() {
        printf("Starting to download node data from GPU..\n");
        //post_process_scorer(); JW: Jan12,2021
#if defined(__CUDACC__)
        check_cuda_last_error("(finalize)");
        mc::download_node<R>(this->world, mc::mc_world);
        check_cuda_last_error("(after download)");
        gpu_err_chk(cudaFree(mc::mc_world));
#endif
        //printf("finalizing\n");
        printf("Simlulation finalizing..\n");
    }   ///write output

    ///<virtual void update() =  0;

    /// \brief Reshapes the scored data from a sparse format to a dense 3D grid.
    /// \param[in] c_ind The index of the child node.
    /// \param[in] s_ind The index of the scorer.
    /// \param[in] dim The dimensions of the output grid.
    /// \return A pointer to the reshaped data.
    CUDA_HOST
    double*
    reshape_data(int c_ind, int s_ind, mqi::vec3<ijk_t> dim) {
        double* reshaped_data = new double[dim.x * dim.y * dim.z];
        int ind_x = 0, ind_y = 0, ind_z = 0, lin = 0;
        for (int i = 0; i < dim.x * dim.y * dim.z; i++) {
            reshaped_data[i] = 0;
        }
        //printf("max capacity %d\n", this->world->children[c_ind]->scorers[s_ind]->max_capacity_);
        for (int ind = 0; ind < this->world->children[c_ind]->scorers[s_ind]->max_capacity_;
             ind++) {
            if (this->world->children[c_ind]->scorers[s_ind]->data_[ind].key1 != mqi::empty_pair &&
                this->world->children[c_ind]->scorers[s_ind]->data_[ind].key2 != mqi::empty_pair) {
                reshaped_data[this->world->children[c_ind]->scorers[s_ind]->data_[ind].key1] +=
                  this->world->children[c_ind]->scorers[s_ind]->data_[ind].value;
            }
        }
        return reshaped_data;
    }

    /// \brief Saves the reshaped simulation results to files.
    CUDA_HOST
    void
    save_reshaped_files() {
        //auto             start = std::chrono::high_resolution_clock::now();
        uint32_t         vol_size;
        mqi::vec3<ijk_t> dim;
        double*          reshaped_data;
        std::string      filename;
        for (int c_ind = 0; c_ind < this->world->n_children; c_ind++) {
            for (int s_ind = 0; s_ind < this->world->children[c_ind]->n_scorers; s_ind++) {
                filename =
                  std::to_string(c_ind) + "_" + this->world->children[c_ind]->scorers[s_ind]->name_;
                dim           = this->world->children[c_ind]->geo->get_nxyz();
                vol_size      = dim.x * dim.y * dim.z;
                reshaped_data = this->reshape_data(c_ind, s_ind, dim);
                if (!this->output_format.compare("mhd")) {
                    mqi::io::save_to_mhd<R>(this->world->children[c_ind],
                                            reshaped_data,
                                            1.0,
                                            this->output_path,
                                            filename,
                                            vol_size);
                } else if (!this->output_format.compare("mha")) {
                    mqi::io::save_to_mha<R>(this->world->children[c_ind],
                                            reshaped_data,
                                            1.0,
                                            this->output_path,
                                            filename,
                                            vol_size);
                } else {
                    mqi::io::save_to_bin<double>(
                      reshaped_data, 1.0, this->output_path, filename, vol_size);
                }

                delete[] reshaped_data;
            }
        }
        //auto                                      stop = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double, std::milli> duration = stop - start;
        //printf("Reshape and save done %f ms\n", duration.count());
        std::cout << "Simulation result file reshape and saving process complete!" << std::endl;
    }

    /// \brief Saves the simulation results in a sparse format (npz).
    CUDA_HOST
    virtual void
    save_sparse_file() {
        auto start = std::chrono::high_resolution_clock::now();
        mqi::vec3<ijk_t> dim;
        std::string      filename;
        printf("num spots \n");
        printf("%d\n", this->num_spots);
        for (int c_ind = 0; c_ind < this->world->n_children; c_ind++) {
            for (int s_ind = 0; s_ind < this->world->children[c_ind]->n_scorers; s_ind++) {
                dim = this->world->children[c_ind]->geo->get_nxyz();
                filename =
                  std::to_string(c_ind) + "_" + this->world->children[c_ind]->scorers[s_ind]->name_;
#if defined(__CUDACC__)
                mqi::io::save_to_npz<R>(this->world->children[c_ind]->scorers[s_ind],
                                        1.0,
                                        this->output_path,
                                        filename,
                                        dim,
                                        this->num_spots);
#else
                mqi::io::save_to_npz<R>(this->world->children[c_ind]->scorers[s_ind],
                                        1.0,
                                        this->output_path,
                                        filename,
                                        dim,
                                        this->num_spots);
#endif
            }
        }
        auto                                      stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = stop - start;
        printf("Reshape and save to npz done %f ms\n", duration.count());
    }

    /// \brief Saves the simulation results to binary files.
    CUDA_HOST
    virtual void
    save_files() {
        ///< Get node pointer for phantom as we know it's id w.r.t world
        const mqi::node_t<R>* phantom = this->world->children[0];
#if defined(__CUDACC__)
        mqi::io::save_to_bin<R>(phantom->scorers[0], 1.0, this->output_path, "/table_gpu_test");
#else
        mqi::io::save_to_bin<R>(phantom->scorers[0], 1.0, this->output_path, "/table_cpu_test");
#endif
    }
    ///< a loop for beam-sources &

#if defined(__CUDACC__)
    /// \brief Calculates the expected memory usage on the GPU.
    CUDA_HOST
    void
    calculate_memory_expectation() {
        size_t free, total, mem = 0;
        cudaMemGetInfo(&free, &total);
        printf(
          "total memory %lu MB Free memory %lu MB\n", total / (1024 * 1024), free / (1024 * 1024));
        mqi::vec3<ijk_t> dim = this->world->geo->get_nxyz();
        mem += sizeof(mqi::node_t<R>);
        mem += sizeof(R*) * dim.x;
        mem += sizeof(R*) * dim.y;
        mem += sizeof(R*) * dim.z;
        mem += sizeof(density_t*) * dim.x * dim.y * dim.z;   // size of data_
        for (int s_ind = 0; s_ind < this->world->n_scorers; s_ind++) {
            mem += sizeof(R);
            mem += sizeof(mqi::scorer_t);
            mem += sizeof(R*) * this->world->scorers[s_ind]->max_capacity_;
            if (this->world->scorers[s_ind]->score_variance) {
                mem += 3 * sizeof(R*) * this->world->scorers[s_ind]->max_capacity_;
            }   // for the member variables
            mem += sizeof(R**) * this->world->scorers[s_ind]->max_capacity_;
            if (this->world->scorers[s_ind]->score_variance) {
                mem += 3 * sizeof(R**) * this->world->scorers[s_ind]->max_capacity_;
            }   // for data download variables
        }
        mem += sizeof(node_t<R>**) * this->world->n_children;
        for (int node_ind = 0; node_ind < this->world->n_children; node_ind++) {
            mqi::vec3<ijk_t> dim = this->world->children[node_ind]->geo->get_nxyz();
            mem += sizeof(R*) * dim.x;
            mem += sizeof(R*) * dim.y;
            mem += sizeof(R*) * dim.z;
            mem += sizeof(density_t*) * dim.x * dim.y * dim.z;
            mem += sizeof(mqi::scorer<R>**) * this->world->children[node_ind]->n_scorers;
            for (int s_ind = 0; s_ind < this->world->children[node_ind]->n_scorers; s_ind++) {
                mem += sizeof(R);
                mem += sizeof(uint32_t*) *
                       this->world->children[node_ind]->scorers[s_ind]->max_capacity_;
                mem += sizeof(mqi::scorer_t);
                mem += sizeof(R*) * this->world->children[node_ind]->scorers[s_ind]->max_capacity_;
                if (this->world->children[node_ind]->scorers[s_ind]->score_variance) {
                    mem += 3 * sizeof(R*) *
                           this->world->children[node_ind]->scorers[s_ind]->max_capacity_;
                }   // for the member variables
                mem += sizeof(R**) * this->world->children[node_ind]->scorers[s_ind]->max_capacity_;
                if (this->world->children[node_ind]->scorers[s_ind]->score_variance) {
                    mem += 3 * sizeof(R**) *
                           this->world->children[node_ind]->scorers[s_ind]->max_capacity_;
                }   // for data download variables
            }
        }

        mem += sizeof(mqi::vertex_t<R>*) * this->beamsource.total_histories();
        printf("Required memory %lu MB\n", mem / (1024 * 1024));
    }
#endif
};

}   // namespace mqi

#endif
