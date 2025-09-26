```
.
├── CMakeLists.txt
├── base
│   ├── distributions
│   │   ├── mqi_const_1d.hpp
│   │   ├── mqi_norm_1d.hpp
│   │   ├── mqi_pdfMd.hpp
│   │   ├── mqi_phsp6d.hpp
│   │   ├── mqi_phsp6d_fanbeam.hpp
│   │   ├── mqi_phsp6d_ray.hpp
│   │   ├── mqi_phsp6d_uniform.hpp
│   │   └── mqi_uni_1d.hpp
│   ├── environments
│   │   ├── mqi_phantom_env.hpp
│   │   ├── mqi_tps_env.hpp
│   │   └── mqi_xenvironment.hpp
│   ├── materials
│   │   ├── material_table_data.hpp
│   │   ├── mqi_material.hpp
│   │   ├── mqi_material_ct.hpp
│   │   └── mqi_patient_materials.hpp
│   ├── scorers
│   │   └── mqi_scorer_energy_deposit.hpp
│   ├── mqi_aperture.hpp
│   ├── mqi_aperture3d.hpp
│   ├── mqi_beam_module.hpp
│   ├── mqi_beam_module_ion.hpp
│   ├── mqi_beam_module_rt.hpp
│   ├── mqi_beamlet.hpp
│   ├── mqi_beamline.hpp
│   ├── mqi_beamsource.hpp
│   ├── mqi_cli.hpp
│   ├── mqi_common.hpp
│   ├── mqi_coordinate_transform.hpp
│   ├── mqi_ct.hpp
│   ├── mqi_dataset.hpp
│   ├── mqi_distributions.hpp
│   ├── mqi_error_check.hpp
│   ├── mqi_file_handler.hpp
│   ├── mqi_fippel_physics.hpp
│   ├── mqi_geometry.hpp
│   ├── mqi_grid3d.hpp
│   ├── mqi_hash_table.hpp
│   ├── mqi_interaction.hpp
│   ├── mqi_io.hpp
│   ├── mqi_material.hpp
│   ├── mqi_math.hpp
│   ├── mqi_matrix.hpp
│   ├── mqi_node.hpp
│   ├── mqi_p_ionization.hpp
│   ├── mqi_physics_constants.hpp
│   ├── mqi_physics_list.hpp
│   ├── mqi_po_elastic.hpp
│   ├── mqi_po_inelastic.hpp
│   ├── mqi_pp_elastic.hpp
│   ├── mqi_rangeshifter.hpp
│   ├── mqi_rect3d.hpp
│   ├── mqi_relativistic_quantities.hpp
│   ├── mqi_roi.hpp
│   ├── mqi_scorer.hpp
│   ├── mqi_sparse_io.hpp
│   ├── mqi_threads.hpp
│   ├── mqi_track.hpp
│   ├── mqi_track_stack.hpp
│   ├── mqi_treatment_machine.hpp
│   ├── mqi_treatment_machine_ion.hpp
│   ├── mqi_treatment_machine_pbs.hpp
│   ├── mqi_treatment_session.hpp
│   ├── mqi_utils.hpp
│   ├── mqi_vec.hpp
│   └── mqi_vertex.hpp
├── kernel_functions
│   ├── mqi_download_data.hpp
│   ├── mqi_kernel_functions.hpp
│   ├── mqi_print_data.hpp
│   ├── mqi_transport.hpp
│   ├── mqi_upload_data.hpp
│   └── mqi_variables.hpp
├── tps_env.cpp
└── treatment_machines
    ├── README.md
    ├── mqi_treatment_machine_smc_gtr1.hpp
    ├── mqi_treatment_machine_smc_gtr2.hpp
    └── spline_interp.hpp
```
