#ifndef MQI_UPLOAD_DATA_HPP
#define MQI_UPLOAD_DATA_HPP

#include <moqui/base/mqi_error_check.hpp>
#include <moqui/base/mqi_roi.hpp>
#include <moqui/base/mqi_scorer.hpp>

namespace mc
{
#if defined(__CUDACC__)

/*!
 * @brief Uploads a node and its hierarchy from CPU to GPU.
 * @tparam R The floating-point type (e.g., float, double).
 * @param c_node Pointer to the source node on the CPU.
 * @param g_node Reference to the pointer for the destination node on the GPU.
*/
template<typename R>
void
upload_node(mqi::node_t<R>* c_node, mqi::node_t<R>*& g_node);

/*!
 * @brief Uploads material data from CPU to GPU.
 * @tparam R The floating-point type (e.g., float, double).
 * @param c_materials Pointer to the source material data on the CPU.
 * @param g_materials Reference to the pointer for the destination material data on the GPU.
 * @param n_materials The number of materials to upload.
*/
template<typename R>
void
upload_materials(mqi::material_t<R>*  c_materials,
                 mqi::material_t<R>*& g_materials,
                 uint16_t             n_materials);

/*!
 * @brief Uploads vertex data from CPU to GPU.
 * @tparam R The floating-point type (e.g., float, double).
 * @param src Pointer to the source vertex data on the CPU.
 * @param dest Reference to the pointer for the destination vertex data on the GPU.
 * @param h0 The starting index of the vertices to upload.
 * @param h1 The ending index of the vertices to upload.
*/
template<typename R>
void
upload_vertices(mqi::vertex_t<R>* src, mqi::vertex_t<R>*& dest, size_t h0, size_t h1);

/*!
 * @brief Uploads CT data from CPU to GPU.
 * @param c_ct Pointer to the source CT data on the CPU.
 * @param g_ct Reference to the pointer for the destination CT data on the GPU.
 * @param size_ The number of CT voxels to upload.
*/
void
upload_ct(int16_t* c_ct, int16_t*& g_ct, uint32_t size_);

/*!
 * @brief Uploads density data (float) from CPU to GPU.
 * @param c_density Pointer to the source density data on the CPU.
 * @param g_density Reference to the pointer for the destination density data on the GPU.
 * @param size_ The number of density voxels to upload.
*/
void
upload_density(float* c_density, float*& g_density, uint32_t size_);

/*!
 * @brief Uploads density data (__half) from CPU to GPU.
 * @param c_density Pointer to the source density data on the CPU.
 * @param g_density Reference to the pointer for the destination density data on the GPU.
 * @param size_ The number of density voxels to upload.
*/
void
upload_density(__half* c_density, __half*& g_density, uint32_t size_);

/*!
 * @brief Uploads scorer index data from CPU to GPU.
 * @tparam R The floating-point type (e.g., float, double).
 * @param c_scorer_idx Pointer to the source scorer index data on the CPU.
 * @param g_scorer_idx Reference to the pointer for the destination scorer index data on the GPU.
 * @param n_scorer_idx The number of scorer indices to upload.
*/
template<typename R>
void
upload_scorer_idx(int32_t* c_scorer_idx, int32_t*& g_scorer_idx, uint32_t n_scorer_idx) {
    gpu_err_chk(cudaMalloc(&g_scorer_idx, n_scorer_idx * sizeof(int32_t)));
    gpu_err_chk(cudaMemcpy(
      g_scorer_idx, c_scorer_idx, n_scorer_idx * sizeof(int32_t), cudaMemcpyHostToDevice));
}

/*!
 * @brief CUDA kernel to print beam source data for debugging.
 * @tparam R The floating-point type (e.g., float, double).
 * @param beamsource Pointer to the beam source object on the GPU.
*/
template<typename R>
CUDA_GLOBAL void
print_vertex(mqi::beamsource<R>* beamsource) {
    for (int i = 0; i < beamsource->beamlet_size_; i++) {
        printf("cdf %d %lu %lu\n", i, beamsource->array_cdf_[i], beamsource->array_beamlet_[i]);
    }
    printf("beamsizes %d\n", beamsource->beamlet_size_);
}

/*!
 * @brief CUDA kernel to add beamlet data to a beam source on the GPU.
 * @tparam R The floating-point type (e.g., float, double).
 * @param beamsource Pointer to the beam source object on the GPU.
 * @param cdf Pointer to the CDF array for beamlet sampling.
 * @param beamlet Pointer to the beamlet index array.
 * @param beamlet_size The number of beamlets.
*/
template<typename R>
CUDA_GLOBAL void
add_beamlet(mqi::beamsource<R>* beamsource, size_t* cdf, size_t* beamlet, size_t beamlet_size) {
    beamsource->array_cdf_     = cdf;
    beamsource->array_beamlet_ = beamlet;
    beamsource->beamlet_size_  = beamlet_size;
    //    for (int i = 0; i < beamlet_size; i++) {
    //        printf("gpu %d %lu %lu\n", i, cdf[i], beamlet[i]);
    //    }
}

/*!
 * @brief Uploads a beam source object from CPU to GPU.
 * @tparam R The floating-point type (e.g., float, double).
 * @param src The source beam source object on the CPU.
 * @param dest Reference to the pointer for the destination beam source object on the GPU.
*/
template<typename R>
void
upload_beamsource(mqi::beamsource<R> src, mqi::beamsource<R>*& dest) {
    gpu_err_chk(cudaMalloc(&dest, sizeof(mqi::beamsource<R>)));
    gpu_err_chk(cudaMemcpy(dest, &src, sizeof(mqi::beamsource<R>), cudaMemcpyHostToDevice));
    size_t* d_cdf;
    size_t* d_beamlet;
    gpu_err_chk(cudaMalloc(&d_cdf, src.beamlet_size_ * sizeof(size_t)));
    gpu_err_chk(cudaMemcpy(
      d_cdf, src.array_cdf_, src.beamlet_size_ * sizeof(size_t), cudaMemcpyHostToDevice));
    gpu_err_chk(cudaMalloc(&d_beamlet, src.beamlet_size_ * sizeof(size_t)));
    gpu_err_chk(cudaMemcpy(
      d_beamlet, src.array_beamlet_, src.beamlet_size_ * sizeof(size_t), cudaMemcpyHostToDevice));
    add_beamlet<R><<<1, 1>>>(dest, d_cdf, d_beamlet, src.beamlet_size_);
    mqi::check_cuda_last_error("(add beamlet)");
    //    for (int i = 0; i < src.beamlet_size_; i++) {
    //        printf("cdf cpu %d %d %d\n", i, src.array_cdf_[i], src.array_beamlet_[i]);
    //    }
}

/*!
 * @brief CUDA kernel to add geometry data to a node on the GPU.
 * @tparam R The floating-point type (e.g., float, double).
 * @param node Pointer to the node on the GPU.
 * @param xe Pointer to the x-edges of the geometry.
 * @param nxe The number of x-edges.
 * @param ye Pointer to the y-edges of the geometry.
 * @param nye The number of y-edges.
 * @param ze Pointer to the z-edges of the geometry.
 * @param nze The number of z-edges.
 * @param data Pointer to the density data.
 * @param rotation_matrix_inv Pointer to the inverse rotation matrix.
 * @param rotation_matrix_fwd Pointer to the forward rotation matrix.
 * @param translation_vector Pointer to the translation vector.
 * @param n_children The number of child nodes.
 * @param children Pointer to the array of child nodes.
*/
template<typename R>
CUDA_GLOBAL void
add_node_geometry(mqi::node_t<R>*  node,
                  R*               xe,
                  mqi::ijk_t       nxe,
                  R*               ye,
                  mqi::ijk_t       nye,
                  R*               ze,
                  mqi::ijk_t       nze,
                  mqi::density_t*  data,
                  mqi::mat3x3<R>*  rotation_matrix_inv,
                  mqi::mat3x3<R>*  rotation_matrix_fwd,
                  mqi::vec3<R>*    translation_vector,
                  uint16_t         n_children = 0,
                  mqi::node_t<R>** children   = nullptr) {

    //std::cout << "Adding geometry node .. : Node --> " << node << ", number of children --> " << n_children << std::endl;

    node->geo = new mqi::grid3d<mqi::density_t, R>;
    node->geo->set_edges(xe, nxe, ye, nye, ze, nze);
    node->geo->rotation_matrix_inv = rotation_matrix_inv[0];
    node->geo->rotation_matrix_fwd = rotation_matrix_fwd[0];
    node->geo->translation_vector  = translation_vector[0];

    node->geo->set_data(data);
    node->n_children   = n_children;
    node->children     = children;
    node->n_scorers    = 0;
    node->scorers_data = nullptr;
    //if (n_children >= 1) { printf("children:%d\n", n_children); }
    printf("Adding geometry node.. : Node --> %p, number of children --> %d\n", node, n_children);

    //printf("Adding geometry node complete! \n");
    //std::cout << "Adding node geometry complete!" << std::endl;
}

/*!
 * @brief CUDA kernel to add scorer data to a node on the GPU.
 * @tparam R The floating-point type (e.g., float, double).
 * @param node Pointer to the node on the GPU.
 * @param n_scorers The number of scorers to add.
 * @param scorers_data Pointer to the array of scorer data pointers.
 * @param scorers_count Pointer to the array of scorer count pointers (for variance scoring).
 * @param scorers_mean Pointer to the array of scorer mean pointers (for variance scoring).
 * @param scorers_variance Pointer to the array of scorer variance pointers (for variance scoring).
 * @param scorer_types Pointer to the array of scorer types.
 * @param scorer_sizes Pointer to the array of scorer sizes.
 * @param scorer_names Pointer to the array of scorer names.
 * @param fp Pointer to the array of compute_hit function pointers.
 * @param roi_method Pointer to the array of ROI mapping methods.
 * @param roi_original_length Pointer to the array of original ROI lengths.
 * @param roi_length Pointer to the array of ROI lengths.
 * @param roi_start Pointer to the array of ROI start pointers.
 * @param roi_stride Pointer to the array of ROI stride pointers.
 * @param roi_acc_stride Pointer to the array of ROI accumulated stride pointers.
*/
template<typename R>
CUDA_GLOBAL void
add_node_scorers(mqi::node_t<R>*         node,
                 uint16_t                n_scorers           = 0,
                 mqi::key_value**        scorers_data        = nullptr,
                 mqi::key_value**        scorers_count       = nullptr,
                 mqi::key_value**        scorers_mean        = nullptr,
                 mqi::key_value**        scorers_variance    = nullptr,
                 mqi::scorer_t*          scorer_types        = nullptr,
                 uint32_t*               scorer_sizes        = nullptr,
                 std::string*            scorer_names        = nullptr,
                 mqi::fp_compute_hit<R>* fp                  = nullptr,
                 mqi::roi_mapping_t*     roi_method          = nullptr,
                 uint32_t*               roi_original_length = nullptr,
                 uint32_t*               roi_length          = nullptr,
                 uint32_t**              roi_start           = nullptr,
                 uint32_t**              roi_stride          = nullptr,
                 uint32_t**              roi_acc_stride      = nullptr) {

    //std::cout << "Adding scorers node .. : Node --> " << node << ", number of children --> " << n_scorers << std::endl;

    node->n_scorers    = n_scorers;
    node->scorers_data = scorers_data;
    if (scorers_count) {
        node->scorers_count    = scorers_count;
        node->scorers_mean     = scorers_mean;
        node->scorers_variance = scorers_variance;
    }

    if (n_scorers >= 1) node->scorers = new mqi::scorer<R>*[n_scorers];

    for (int i = 0; i < n_scorers; i++) {
        //printf("scorer size[%d] %d\n", i, scorer_sizes[i]);
        node->scorers[i] = new mqi::scorer<R>("", scorer_sizes[i], fp[i]);
        //printf("compute hit %p\n", node->scorers[i]->compute_hit_);

        if (scorers_count) {
            node->scorers[i]->score_variance_ = true;
        } else {
            node->scorers[i]->score_variance_ = false;
        }
        //printf("scorer data[i] %p\n", scorers_data[i]);
        node->scorers[i]->data_ = scorers_data[i];
        //        node->scorers[i]->roi_  = roi[i];
        node->scorers[i]->roi_ = new mqi::roi_t(roi_method[i],
                                                roi_original_length[i],
                                                roi_length[i],
                                                roi_start[i],
                                                roi_stride[i],
                                                roi_acc_stride[i]);
        //        printf("scorer[i] mask %p\n", node->scorers[i]->roi_mask_);
        if (scorers_count) {
            node->scorers[i]->count_    = scorers_count[i];
            node->scorers[i]->mean_     = scorers_mean[i];
            node->scorers[i]->variance_ = scorers_variance[i];
        }
    }
    printf("Adding scorers node.. : Node --> %p, number of children --> %d\n", node, n_scorers);
    //printf("Adding scorers node complete!\n");
    //std::cout << "Adding scorers node complete!" << std::endl;
}

///< Upload nodes in CPU to GPU
///< recursive operation
template<typename R>
void
upload_node(mqi::node_t<R>* c_node, mqi::node_t<R>*& g_node) {
    printf("Uploading data to node.. : Allocated node --> %p\n", g_node);
    //std::cout << "Uploading data to node.. : Allocated node --> " << g_node << std::endl;
    ///< edge data for geometries

    R*              x_edges = nullptr;
    R*              y_edges = nullptr;
    R*              z_edges = nullptr;
    mqi::mat3x3<R>* rotation_matrix_inv;
    mqi::mat3x3<R>* rotation_matrix_fwd;
    mqi::vec3<R>*   translation_vector;
    mqi::density_t* density = nullptr;

    mqi::vec3<mqi::ijk_t> dim = c_node->geo->get_nxyz();

    gpu_err_chk(cudaMalloc(&x_edges, (dim.x + 1) * sizeof(R)));
    gpu_err_chk(cudaMalloc(&y_edges, (dim.y + 1) * sizeof(R)));
    gpu_err_chk(cudaMalloc(&z_edges, (dim.z + 1) * sizeof(R)));
    gpu_err_chk(cudaMalloc(&density, (dim.x * dim.y * dim.z) * sizeof(mqi::density_t)));
    gpu_err_chk(cudaMalloc(&rotation_matrix_fwd, 1 * sizeof(mqi::mat3x3<R>)));
    gpu_err_chk(cudaMalloc(&rotation_matrix_inv, 1 * sizeof(mqi::mat3x3<R>)));
    gpu_err_chk(cudaMalloc(&translation_vector, 1 * sizeof(mqi::vec3<R>)));

    gpu_err_chk(cudaMemcpy(
      x_edges, c_node->geo->get_x_edges(), (dim.x + 1) * sizeof(R), cudaMemcpyHostToDevice));
    gpu_err_chk(cudaMemcpy(
      y_edges, c_node->geo->get_y_edges(), (dim.y + 1) * sizeof(R), cudaMemcpyHostToDevice));
    gpu_err_chk(cudaMemcpy(
      z_edges, c_node->geo->get_z_edges(), (dim.z + 1) * sizeof(R), cudaMemcpyHostToDevice));
    gpu_err_chk(cudaMemcpy(density,
                           c_node->geo->get_data(),
                           (dim.x * dim.y * dim.z) * sizeof(mqi::density_t),
                           cudaMemcpyHostToDevice));
    gpu_err_chk(cudaMemcpy(rotation_matrix_fwd,
                           &(c_node->geo[0].rotation_matrix_fwd),
                           sizeof(mqi::mat3x3<R>),
                           cudaMemcpyHostToDevice));
    gpu_err_chk(cudaMemcpy(rotation_matrix_inv,
                           &(c_node->geo[0].rotation_matrix_inv),
                           sizeof(mqi::mat3x3<R>),
                           cudaMemcpyHostToDevice));
    gpu_err_chk(cudaMemcpy(translation_vector,
                           &(c_node->geo[0].translation_vector),
                           sizeof(mqi::vec3<R>),
                           cudaMemcpyHostToDevice));

    mqi::key_value** h_scorers_data  = nullptr;
    mqi::key_value** h_scorers_count = nullptr;
    mqi::key_value** h_scorers_mean  = nullptr;
    mqi::key_value** h_scorers_var   = nullptr;

    mqi::key_value** d_scorers_data     = nullptr;
    mqi::key_value** d_scorers_count    = nullptr;
    mqi::key_value** d_scorers_mean     = nullptr;
    mqi::key_value** d_scorers_variance = nullptr;

    mqi::scorer_t* scorers_types   = nullptr;
    mqi::scorer_t* d_scorers_types = nullptr;

    //    mqi::roi_t** roi   = nullptr;
    //    mqi::roi_t** d_roi = nullptr;
    //    mqi::roi_t** roi   = nullptr;

    uint32_t**          h_roi_start         = nullptr;
    uint32_t**          h_roi_stride        = nullptr;
    uint32_t**          h_roi_acc_stride    = nullptr;
    uint32_t*           roi_length          = nullptr;
    uint32_t*           roi_original_length = nullptr;
    mqi::roi_mapping_t* roi_method          = nullptr;

    uint32_t**          d_roi_start           = nullptr;
    uint32_t**          d_roi_stride          = nullptr;
    uint32_t**          d_roi_acc_stride      = nullptr;
    uint32_t*           d_roi_length          = nullptr;
    uint32_t*           d_roi_original_length = nullptr;
    mqi::roi_mapping_t* d_roi_method          = nullptr;

    uint32_t* scorers_size   = nullptr;
    uint32_t* d_scorers_size = nullptr;

    mqi::fp_compute_hit<R>* fp   = nullptr;
    mqi::fp_compute_hit<R>* d_fp = nullptr;

    std::string* scorers_name   = nullptr;
    std::string* d_scorers_name = nullptr;
    if (c_node->n_scorers > 0) {
        h_scorers_data      = new mqi::key_value*[c_node->n_scorers];
        scorers_types       = new mqi::scorer_t[c_node->n_scorers];
        scorers_size        = new uint32_t[c_node->n_scorers];
        scorers_name        = new std::string[c_node->n_scorers];
        fp                  = new mqi::fp_compute_hit<R>[c_node->n_scorers];
        h_roi_start         = new uint32_t*[c_node->n_scorers];
        h_roi_stride        = new uint32_t*[c_node->n_scorers];
        h_roi_acc_stride    = new uint32_t*[c_node->n_scorers];
        roi_length          = new uint32_t[c_node->n_scorers];
        roi_original_length = new uint32_t[c_node->n_scorers];
        roi_method          = new mqi::roi_mapping_t[c_node->n_scorers];
        //        roi            = new mqi::roi_t*[c_node->n_scorers];
        gpu_err_chk(cudaMalloc(&d_scorers_types, c_node->n_scorers * sizeof(mqi::scorer_t)));
        gpu_err_chk(cudaMalloc(&d_scorers_size, c_node->n_scorers * sizeof(uint32_t)));
        gpu_err_chk(cudaMalloc(&d_scorers_data, c_node->n_scorers * sizeof(mqi::key_value*)));
        gpu_err_chk(cudaMalloc(&d_fp, c_node->n_scorers * sizeof(mqi::fp_compute_hit<R>)));
        gpu_err_chk(cudaMalloc(&d_roi_start, c_node->n_scorers * sizeof(uint32_t*)));
        gpu_err_chk(cudaMalloc(&d_roi_stride, c_node->n_scorers * sizeof(uint32_t*)));
        gpu_err_chk(cudaMalloc(&d_roi_acc_stride, c_node->n_scorers * sizeof(uint32_t*)));
        gpu_err_chk(cudaMalloc(&d_roi_length, c_node->n_scorers * sizeof(uint32_t)));
        gpu_err_chk(cudaMalloc(&d_roi_original_length, c_node->n_scorers * sizeof(uint32_t)));
        gpu_err_chk(cudaMalloc(&d_roi_method, c_node->n_scorers * sizeof(mqi::roi_mapping_t)));

        if (c_node->scorers[0]->score_variance_) {
            printf("Score variance turn on\n");
            h_scorers_count = new mqi::key_value*[c_node->n_scorers];
            h_scorers_mean  = new mqi::key_value*[c_node->n_scorers];
            h_scorers_var   = new mqi::key_value*[c_node->n_scorers];
            gpu_err_chk(cudaMalloc(&d_scorers_count, c_node->n_scorers * sizeof(mqi::key_value*)));
            gpu_err_chk(cudaMalloc(&d_scorers_mean, c_node->n_scorers * sizeof(mqi::key_value*)));
            gpu_err_chk(
              cudaMalloc(&d_scorers_variance, c_node->n_scorers * sizeof(mqi::key_value*)));
        }

        for (int i = 0; i < c_node->n_scorers; i++) {
            ///< pointer initialization in GPU
            //printf("ind %d n_scorer %d size_ %lu\n",i,c_node->n_scorers,c_node->scorers[i]->size_);

            //printf("max capacity %d\n", c_node->scorers[i]->max_capacity_);
            gpu_err_chk(cudaMalloc(&h_scorers_data[i],
                                   c_node->scorers[i]->max_capacity_ * sizeof(mqi::key_value)));
            cudaMemset(
              h_scorers_data[i], 0xff, sizeof(mqi::key_value) * c_node->scorers[i]->max_capacity_);
            //            init_table_cuda<<<1, 1>>>(h_scorers_data[i], c_node->scorers[i]->max_capacity_);
            gpu_err_chk(cudaMemcpy(h_scorers_data[i],
                                   c_node->scorers[i]->data_,
                                   c_node->scorers[i]->max_capacity_ * sizeof(mqi::key_value),
                                   cudaMemcpyHostToDevice));
            gpu_err_chk(
              cudaMalloc(&h_roi_start[i], c_node->scorers[i]->roi_->length_ * sizeof(uint32_t)));
            gpu_err_chk(
              cudaMalloc(&h_roi_stride[i], c_node->scorers[i]->roi_->length_ * sizeof(uint32_t)));
            gpu_err_chk(cudaMalloc(&h_roi_acc_stride[i],
                                   c_node->scorers[i]->roi_->length_ * sizeof(uint32_t)));
            if (c_node->scorers[i]->roi_->length_ > 0) {
                gpu_err_chk(cudaMemcpy(h_roi_start[i],
                                       c_node->scorers[i]->roi_->start_,
                                       c_node->scorers[i]->roi_->length_ * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice));
                gpu_err_chk(cudaMemcpy(h_roi_stride[i],
                                       c_node->scorers[i]->roi_->stride_,
                                       c_node->scorers[i]->roi_->length_ * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice));
                gpu_err_chk(cudaMemcpy(h_roi_acc_stride[i],
                                       c_node->scorers[i]->roi_->acc_stride_,
                                       c_node->scorers[i]->roi_->length_ * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice));
            } else {
                h_roi_start[i]      = nullptr;
                h_roi_stride[i]     = nullptr;
                h_roi_acc_stride[i] = nullptr;
            }
            if (c_node->scorers[i]->score_variance_) {
                gpu_err_chk(cudaMalloc(&h_scorers_count[i],
                                       c_node->scorers[i]->max_capacity_ * sizeof(mqi::key_value)));
                cudaMemset(h_scorers_count[i],
                           0xff,
                           sizeof(mqi::key_value) * c_node->scorers[i]->max_capacity_);
                //                init_table_cuda<<<1, 1>>>(h_scorers_count[i], c_node->scorers[i]->max_capacity_);

                gpu_err_chk(cudaMemcpy(h_scorers_count[i],
                                       c_node->scorers[i]->count_,
                                       c_node->scorers[i]->max_capacity_ * sizeof(mqi::key_value),
                                       cudaMemcpyHostToDevice));

                gpu_err_chk(cudaMalloc(&h_scorers_mean[i],
                                       c_node->scorers[i]->max_capacity_ * sizeof(mqi::key_value)));

                cudaMemset(h_scorers_mean[i],
                           0xff,
                           sizeof(mqi::key_value) * c_node->scorers[i]->max_capacity_);
                //                init_table_cuda<<<1, 1>>>(h_scorers_mean[i], c_node->scorers[i]->max_capacity_);

                gpu_err_chk(cudaMemcpy(h_scorers_mean[i],
                                       c_node->scorers[i]->mean_,
                                       c_node->scorers[i]->max_capacity_ * sizeof(mqi::key_value),
                                       cudaMemcpyHostToDevice));

                gpu_err_chk(cudaMalloc(&h_scorers_var[i],
                                       c_node->scorers[i]->max_capacity_ * sizeof(mqi::key_value)));
                cudaMemset(h_scorers_var[i],
                           0xff,
                           sizeof(mqi::key_value) * c_node->scorers[i]->max_capacity_);
                //                init_table_cuda<<<1, 1>>>(h_scorers_var[i], c_node->scorers[i]->max_capacity_);

                gpu_err_chk(cudaMemcpy(h_scorers_var[i],
                                       c_node->scorers[i]->variance_,
                                       c_node->scorers[i]->max_capacity_ * sizeof(mqi::key_value),
                                       cudaMemcpyHostToDevice));
            }

            scorers_types[i]       = c_node->scorers[i]->type_;
            scorers_size[i]        = c_node->scorers[i]->max_capacity_;
            scorers_name[i]        = c_node->scorers[i]->name_;
            fp[i]                  = c_node->scorers[i]->compute_hit_;
            roi_method[i]          = c_node->scorers[i]->roi_->method_;
            roi_original_length[i] = c_node->scorers[i]->roi_->original_length_;
            roi_length[i]          = c_node->scorers[i]->roi_->length_;
        }
        gpu_err_chk(cudaMemcpy(d_scorers_data,
                               h_scorers_data,
                               c_node->n_scorers * sizeof(mqi::key_value*),
                               cudaMemcpyHostToDevice));

        gpu_err_chk(cudaMemcpy(
          d_roi_start, h_roi_start, c_node->n_scorers * sizeof(uint32_t*), cudaMemcpyHostToDevice));
        gpu_err_chk(cudaMemcpy(d_roi_stride,
                               h_roi_stride,
                               c_node->n_scorers * sizeof(uint32_t*),
                               cudaMemcpyHostToDevice));
        gpu_err_chk(cudaMemcpy(d_roi_acc_stride,
                               h_roi_acc_stride,
                               c_node->n_scorers * sizeof(uint32_t*),
                               cudaMemcpyHostToDevice));
        if (h_scorers_count) {
            gpu_err_chk(cudaMemcpy(d_scorers_count,
                                   h_scorers_count,
                                   c_node->n_scorers * sizeof(mqi::key_value*),
                                   cudaMemcpyHostToDevice));

            gpu_err_chk(cudaMemcpy(d_scorers_mean,
                                   h_scorers_mean,
                                   c_node->n_scorers * sizeof(mqi::key_value*),
                                   cudaMemcpyHostToDevice));

            gpu_err_chk(cudaMemcpy(d_scorers_variance,
                                   h_scorers_var,
                                   c_node->n_scorers * sizeof(mqi::key_value*),
                                   cudaMemcpyHostToDevice));
        }
        gpu_err_chk(cudaMemcpy(d_scorers_types,
                               scorers_types,
                               c_node->n_scorers * sizeof(mqi::scorer_t),
                               cudaMemcpyHostToDevice));
        gpu_err_chk(cudaMemcpy(d_scorers_size,
                               scorers_size,
                               c_node->n_scorers * sizeof(uint32_t),
                               cudaMemcpyHostToDevice));
        gpu_err_chk(cudaMemcpy(
          d_fp, fp, c_node->n_scorers * sizeof(mqi::fp_compute_hit<R>), cudaMemcpyHostToDevice));
        gpu_err_chk(cudaMemcpy(
          d_roi_length, roi_length, c_node->n_scorers * sizeof(uint32_t), cudaMemcpyHostToDevice));
        gpu_err_chk(cudaMemcpy(d_roi_original_length,
                               roi_original_length,
                               c_node->n_scorers * sizeof(uint32_t),
                               cudaMemcpyHostToDevice));
        gpu_err_chk(cudaMemcpy(d_roi_method,
                               roi_method,
                               c_node->n_scorers * sizeof(mqi::roi_mapping_t),
                               cudaMemcpyHostToDevice));
    }

    mqi::node_t<R>** h_children = nullptr;
    mqi::node_t<R>** d_children = nullptr;
    if (c_node->n_children >= 1) {
        h_children = new mqi::node_t<R>*[c_node->n_children];
        gpu_err_chk(cudaMalloc(&d_children, c_node->n_children * sizeof(mqi::node_t<R>*)));

        for (int i = 0; i < c_node->n_children; ++i) {
            gpu_err_chk(cudaMalloc(&h_children[i], sizeof(mqi::node_t<R>)));
            printf("Uploading data to node.. : g_node --> %p, d_children --> %p, h_children[%d] --> %p\n",
                   g_node,
                   d_children,
                   i,
                   h_children[i]);
            //std::cout << "Uploading data to node.. : Allocated d_children --> " << d_children << std::endl;
            //std::cout << "Uploading data to node.. : Allocated h_children " << "[" + std::to_string(i) << "] --> " << h_children[i] << std::endl;
        }
        gpu_err_chk(cudaMemcpy(d_children,
                               h_children,
                               c_node->n_children * sizeof(mqi::node_t<R>*),
                               cudaMemcpyHostToDevice));
    }

    ///< create node object using transfered data
    ///< copy GPU memory of g_node to use re-map down below
    mc::add_node_geometry<R><<<1, 1>>>(g_node,
                                       x_edges,
                                       dim.x + 1,
                                       y_edges,
                                       dim.y + 1,
                                       z_edges,
                                       dim.z + 1,
                                       density,
                                       rotation_matrix_inv,
                                       rotation_matrix_fwd,
                                       translation_vector,
                                       c_node->n_children,
                                       d_children);
    cudaDeviceSynchronize();
    if (c_node->n_scorers > 0) {
        mc::add_node_scorers<R><<<1, 1>>>(g_node,
                                          c_node->n_scorers,
                                          d_scorers_data,
                                          d_scorers_count,
                                          d_scorers_mean,
                                          d_scorers_variance,
                                          d_scorers_types,
                                          d_scorers_size,
                                          d_scorers_name,
                                          d_fp,
                                          d_roi_method,
                                          d_roi_original_length,
                                          d_roi_length,
                                          d_roi_start,
                                          d_roi_stride,
                                          d_roi_acc_stride);
    } else {
        mc::add_node_scorers<R><<<1, 1>>>(g_node);
    }
    cudaDeviceSynchronize();
    mqi::check_cuda_last_error("(add_node_geometry)");
    //printf("n_children %d\n", c_node->n_children);
    ///< Repeat uploading node recursively
    for (int i = 0; i < c_node->n_children; ++i) {
        upload_node(c_node->children[i], h_children[i]);
    }

    delete[] h_scorers_data;
    delete[] h_scorers_count;
    delete[] h_scorers_mean;
    delete[] h_scorers_var;
    delete[] h_children;
    delete[] scorers_types;
    delete[] scorers_size;

    gpu_err_chk(cudaFree(d_scorers_types));   // it's working, but not sure it is required
    gpu_err_chk(cudaFree(d_scorers_size));    // it's working, but not sure it is required
    gpu_err_chk(cudaFree(d_roi_method));
    gpu_err_chk(cudaFree(d_roi_original_length));
    gpu_err_chk(cudaFree(d_roi_length));
    gpu_err_chk(cudaFree(d_roi_start));
    gpu_err_chk(cudaFree(d_roi_stride));
    gpu_err_chk(cudaFree(d_roi_acc_stride));
    //    gpu_err_chk(cudaFree(d_roi));             // it's working, but not sure it is required
}   //upload_node

///< Upload nodes in CPU to GPU
///< recursive operation
template<typename R>
void
upload_materials(mqi::material_t<R>*  c_materials,
                 mqi::material_t<R>*& g_materials,
                 uint16_t             n_materials) {
    ///< map c_node's data to GPU memory
    gpu_err_chk(cudaMalloc(&g_materials, n_materials * sizeof(mqi::material_t<R>)));
    gpu_err_chk(cudaMemcpy(
      g_materials, c_materials, n_materials * sizeof(mqi::material_t<R>), cudaMemcpyHostToDevice));
}

template<typename R>
void
upload_vertices(mqi::vertex_t<R>* src, mqi::vertex_t<R>*& dest, size_t h0, size_t h1) {
    //printf("h0 %d h1 %d %lu\n", h0, h1, sizeof(mqi::vertex_t<R>));
    gpu_err_chk(cudaMalloc(&dest, (h1 - h0) * sizeof(mqi::vertex_t<R>)));
    //Send histories to GPU
    gpu_err_chk(
      cudaMemcpy(dest, src, (h1 - h0) * sizeof(mqi::vertex_t<R>), cudaMemcpyHostToDevice));
}

void
upload_ct(int16_t* c_ct, int16_t*& g_ct, uint32_t size_) {
    gpu_err_chk(cudaMalloc(&g_ct, size_ * sizeof(int16_t)));
    gpu_err_chk(cudaMemcpy(g_ct, c_ct, size_ * sizeof(int16_t), cudaMemcpyHostToDevice));
}

void
upload_density(float* c_density, float*& g_density, uint32_t size_) {
    gpu_err_chk(cudaMalloc(&g_density, size_ * sizeof(float)));
    gpu_err_chk(cudaMemcpy(g_density, c_density, size_ * sizeof(float), cudaMemcpyHostToDevice));
}

void
upload_density(__half* c_density, __half*& g_density, uint32_t size_) {
    gpu_err_chk(cudaMalloc(&g_density, size_ * sizeof(__half)));
    gpu_err_chk(cudaMemcpy(g_density, c_density, size_ * sizeof(__half), cudaMemcpyHostToDevice));
}

/*!
 * @brief Uploads a scorer offset vector (uint16_t) from CPU to GPU.
 * @param src Pointer to the source vector on the CPU.
 * @param dest Reference to the pointer for the destination vector on the GPU.
 * @param size_ The number of elements in the vector.
*/
void
upload_scorer_offset_vector(uint16_t* src, uint16_t*& dest, size_t size_) {
    gpu_err_chk(cudaMalloc(&dest, size_ * sizeof(uint16_t)));
    gpu_err_chk(cudaMemcpy(dest, src, size_ * sizeof(uint16_t), cudaMemcpyHostToDevice));
}

/*!
 * @brief Uploads a scorer offset vector (uint32_t) from CPU to GPU.
 * @param src Pointer to the source vector on the CPU.
 * @param dest Reference to the pointer for the destination vector on the GPU.
 * @param size_ The number of elements in the vector.
*/
void
upload_scorer_offset_vector(uint32_t* src, uint32_t*& dest, size_t size_) {
    gpu_err_chk(cudaMalloc(&dest, size_ * sizeof(uint32_t)));
    gpu_err_chk(cudaMemcpy(dest, src, size_ * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

/*!
 * @brief Uploads a scorer offset vector (unsigned long) from CPU to GPU.
 * @param src Pointer to the source vector on the CPU.
 * @param dest Reference to the pointer for the destination vector on the GPU.
 * @param size_ The number of elements in the vector.
*/
void
upload_scorer_offset_vector(unsigned long* src, unsigned long*& dest, size_t size_) {
    gpu_err_chk(cudaMalloc(&dest, size_ * sizeof(uint32_t)));
    gpu_err_chk(cudaMemcpy(dest, src, size_ * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

/*!
 * @brief Uploads a scorer offset vector (unsigned long long int) from CPU to GPU.
 * @param src Pointer to the source vector on the CPU.
 * @param dest Reference to the pointer for the destination vector on the GPU.
 * @param size_ The number of elements in the vector.
*/
void
upload_scorer_offset_vector(unsigned long long int*  src,
                            unsigned long long int*& dest,
                            size_t                   size_) {
    gpu_err_chk(cudaMalloc(&dest, size_ * sizeof(uint32_t)));
    gpu_err_chk(cudaMemcpy(dest, src, size_ * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

#endif
}   // namespace mc

#endif   // UPLOAD_DATA_CPP
