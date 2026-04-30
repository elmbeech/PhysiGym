////////
// title: PhyiCell/custom_modules/custom.cpp
//
// language: C++
// date: 2015-2024
// license: BSD-3-Clause
// author: Elmar Bucher, Paul Macklin
//
// original source code: https://github.com/MathCancer/PhysiCell
// modified source code: https://github.com/elmbeech/physicellembedding
////////


// load library
#include "custom.h"

#include "../BioFVM/BioFVM.h"
using namespace BioFVM;

#include <algorithm> // for std::find
std::vector<int> arrival_voxel_indexes;  //= {0,1,2,3}; // vector to store the indexes of each voxel to be an arrival spot
std::vector<int> departure_voxel_indexes;  //= {4092,4093,4094,4095}; // vector to store the indexes of each voxel to be an departure spot

// constantes variables
//static const double ZERO = 0;
//static const std::vector<double> VECTOR_ZERO (4, ZERO);  // generate a 4 character long vector of zeros.

// development
//extern std::ofstream myfile;

///////////////
// functions //
///////////////

// bue 20250606: modified standard_volume_update function
// solid debris export implementation

// setup debris function pointer
static void (*debris_function)(Cell* pCell, std::vector<double>* export_rates) = NULL;

// set this function to export apoptotic and necrotic debris
void debris_function_apoptotic_necrotic(Cell* pCell, std::vector<double>* export_rates) {
    set_single_behavior(pCell, "debris_apoptotic export", (*export_rates)[1]);
    set_single_behavior(pCell, "debris_necrotic export", (*export_rates)[2]);
    return;
}

// set this function to export apoptotic debris only
void debris_function_apoptotic(Cell* pCell, std::vector<double>* export_rates) {
    set_single_behavior(pCell, "debris_apoptotic export", (*export_rates)[1]);
    return;
}

// set this function to export necrotic debris only
void debris_function_necrotic(Cell* pCell, std::vector<double>* export_rates) {
    set_single_behavior(pCell, "debris_necrotic export", (*export_rates)[2]);
    return;
}

// set function
void standard_volume_update_and_debris_release(Cell* pCell, Phenotype& phenotype, double dt) {
    // fluid
    phenotype.volume.fluid += dt * phenotype.volume.fluid_change_rate * (phenotype.volume.target_fluid_fraction * phenotype.volume.total - phenotype.volume.fluid);
    if (phenotype.volume.fluid < 0.0) { phenotype.volume.fluid = 0.0; }
    phenotype.volume.nuclear_fluid = (phenotype.volume.nuclear / (phenotype.volume.total + 1e-16)) * phenotype.volume.fluid;
    phenotype.volume.cytoplasmic_fluid = phenotype.volume.fluid - phenotype.volume.nuclear_fluid;

    // solid
    // bue 20250606: nuclear_solid
    double delta_volume_nuclear_solid = dt * phenotype.volume.nuclear_biomass_change_rate * (phenotype.volume.target_solid_nuclear - phenotype.volume.nuclear_solid);

    phenotype.volume.nuclear_solid += delta_volume_nuclear_solid;
    if (phenotype.volume.nuclear_solid < 0.0) { phenotype.volume.nuclear_solid = 0.0; }
    phenotype.volume.target_solid_cytoplasmic = phenotype.volume.target_cytoplasmic_to_nuclear_ratio * phenotype.volume.target_solid_nuclear;

    // bue 20250606: cytoplasmic_solid
    double delta_volume_cytoplasmic_solid = dt * phenotype.volume.cytoplasmic_biomass_change_rate * (phenotype.volume.target_solid_cytoplasmic - phenotype.volume.cytoplasmic_solid);
    phenotype.volume.cytoplasmic_solid += delta_volume_cytoplasmic_solid;
    if(phenotype.volume.cytoplasmic_solid < 0.0) { phenotype.volume.cytoplasmic_solid = 0.0; }

    // bue 20250606: delta solid
    double delta_volume_solid = delta_volume_nuclear_solid + delta_volume_cytoplasmic_solid;
    phenotype.volume.solid = phenotype.volume.nuclear_solid + phenotype.volume.cytoplasmic_solid;
    phenotype.volume.nuclear = phenotype.volume.nuclear_solid + phenotype.volume.nuclear_fluid;
    phenotype.volume.cytoplasmic = phenotype.volume.cytoplasmic_solid + phenotype.volume.cytoplasmic_fluid;
    phenotype.volume.calcified_fraction += dt * phenotype.volume.calcification_rate * (1 - phenotype.volume.calcified_fraction);
    phenotype.volume.total = phenotype.volume.cytoplasmic + phenotype.volume.nuclear;
    phenotype.volume.fluid_fraction = phenotype.volume.fluid / (1e-16 + phenotype.volume.total);
    phenotype.geometry.update( pCell,phenotype,dt );

    // bue 20250606: export debris
    if (debris_function && get_single_signal( pCell, "dead") == true ) {
        // calculate debris release rate
        double net_export_rate = - (delta_volume_solid / (cell_defaults.phenotype.volume.solid * dt));
        // populate net_export_rates vector
        std::vector<double> net_export_rates(3,0);
        net_export_rates[0] = net_export_rate;  // total
        if (get_single_signal(pCell, "apoptotic")) {
            net_export_rates[1] = net_export_rate;
        }
        else if (get_single_signal(pCell, "necrotic")) {
            net_export_rates[2] = net_export_rate;
        }
        else {
           std::cout << "Warning: cell in a unknowen dead model state detected!" << std::endl;
        }
        // call debris release function
        debris_function( pCell, &net_export_rates );

        // output to file for development
        //myfile << pCell->ID << "," << PhysiCell_globals.current_time << "," << pCell->phenotype.death.current_death_model_index<< "," << pCell->phenotype.volume.nuclear_solid << "," << pCell->phenotype.volume.cytoplasmic_solid << "," << pCell->phenotype.volume.fluid << "," << delta_volume_solid << "," << net_export_rate <<  "," << cell_defaults.phenotype.volume.solid << std::endl;
    }
    return;
}

// bue 20260217: modified standard_domain_edge_avoidance_interactions impelentation
void prototype_domain_edge_avoidance_interactions(Cell* pCell, Phenotype& phenotype, double dt) {
    if( pCell->functions.calculate_distance_to_membrane == NULL ) {
        pCell->functions.calculate_distance_to_membrane = distance_to_domain_edge;
    }
    phenotype.mechanics.cell_BM_repulsion_strength = 100;

    // note that the distance_to_membrane function must set displacement values (as a normal vector)
    double distance = pCell->functions.calculate_distance_to_membrane(pCell,phenotype,dt);
    double radius = phenotype.geometry.radius;
    if (radius < 9) { radius = 9; }

    // repulsion from basement membrane
    double temp_r = 0;
    if (distance < radius) {
        temp_r = (1 - distance / radius);
        temp_r *= temp_r;
        temp_r *= phenotype.mechanics.cell_BM_repulsion_strength;
    }
    if (fabs( temp_r) < 1e-16) { return; }

    axpy(&(pCell->velocity), temp_r, pCell->displacement);
    return;
}


// modified standard custom cpp function
void create_cell_types(void) {
    std::cout << "generate cell types ..." << std::endl;
    std::cout << "cell types can only be defined the first episode of the runtime!" << std::endl;

    // put any modifications to default cell definition here if you
    // want to have "inherited" by other cell types.
    // this is a good place to set default functions.

    // cell_default initial definition
    initialize_default_cell_definition();
    cell_defaults.phenotype.secretion.sync_to_microenvironment(&microenvironment);

    cell_defaults.functions.volume_update_function = standard_volume_update_and_debris_release; // bue 20250606
    cell_defaults.functions.update_velocity = standard_update_cell_velocity;

    cell_defaults.functions.update_migration_bias = NULL;
    cell_defaults.functions.custom_cell_rule = NULL;
    cell_defaults.functions.contact_function = NULL;

    cell_defaults.functions.add_cell_basement_membrane_interactions = prototype_domain_edge_avoidance_interactions; // bue 20260217
    cell_defaults.functions.calculate_distance_to_membrane = NULL;

    // this is a good place to set custom functions.
    cell_defaults.functions.update_phenotype = phenotype_function; // bue 20250911
    cell_defaults.functions.custom_cell_rule = custom_function;
    cell_defaults.functions.contact_function = contact_function;

    // parse the cell definitions in the XML config file
    initialize_cell_definitions_from_pugixml();

    // put any modifications to individual cell definitions here.

    // generate the maps of cell definitions.
    build_cell_definitions_maps();

    // intializes cell signal and response dictionaries
    setup_signal_behavior_dictionaries();

    // initializ cell rule definitions
    setup_cell_rules();

    // summarize the cell defintion setup.
    display_cell_definitions(std::cout);

    return;
}


// heber 20250000: cell arrival departure implementation

// inspiration: fan brun bryne 2025 exploring the relationship between vascular remodelling and tumour growth using agent-based modelling.
// https://doi.org/10.1101/2025.03.17.643670

void cell_arrival_function(double dt) {
    std::string arrival_rate_name, arrival_substrate_name, arrival_half_max_name, arrival_signal_power_name;
    std::vector<double> arrival_voxel_probabilities(arrival_voxel_indexes.size(), 0.0);
    bool no_signal = false;
    unsigned int index_substrate;
    double total_weight = 0.0;
    double substrate_at_voxel;
    Cell* pC = NULL;

    // Find cells to arrive at the domain
    for (int k=0; k < cell_definitions_by_index.size() ; k++) {
        Cell_Definition* pCD = cell_definitions_by_index[k];

        // check: arrival_rate_{cell type X} : maximum rate of arrival of cell type X
        arrival_rate_name = "max_arrival_rate_" + pCD->name;

        // no arrival rate for this cell type
        if (parameters.doubles.find_index(arrival_rate_name) <= -1) { continue; }

        arrival_substrate_name = "arrival_signal_substrate_" + pCD->name;
        arrival_half_max_name = "arrival_signal_half_max_" + pCD->name;
        arrival_signal_power_name = "arrival_signal_power_" + pCD->name;

        // no arrival substrate, half max or signal power defined for this cell type
        if ((parameters.strings.find_index(arrival_substrate_name) <= -1 ) || (parameters.doubles.find_index(arrival_half_max_name) <= -1) || (parameters.ints.find_index(arrival_signal_power_name) <= -1)) {
            std::cout << "Error: no arrival substrate, half max or signal power defined for this cell type!" << pCD->name << std::endl;
            std::cout << "Add in user parameters: arrival_signal_half_max_" << pCD->name << " (type double)." << std::endl;
            std::cout << "Add in user parameters: arrival_signal_substrate_" << pCD->name << " (type string)." << std::endl;
            std::cout << "Add in user parameters: arrival_signal_power_" << pCD->name << " (type int)." << std::endl;
            exit(-1);
        }
        else { no_signal = true; }  // no arrival substrate defined for this cell type
        if (!no_signal) {
            // update the probabilities of arriving in each voxel selected according to the signal of substrate
            index_substrate = microenvironment.find_density_index(parameters.strings(arrival_substrate_name));
            total_weight = 0.0;
            for (unsigned int n = 0; n < arrival_voxel_indexes.size(); n++) {
                substrate_at_voxel = microenvironment.density_vector(arrival_voxel_indexes[n])[index_substrate];
                arrival_voxel_probabilities[n] =  Hill_response_function(substrate_at_voxel, parameters.doubles(arrival_half_max_name), parameters.ints(arrival_signal_power_name));
                total_weight += arrival_voxel_probabilities[n];
            }
        }

        // normalize the probabilities, update the microenvironment and generate cells according to the arrival rate (0, max_arrival_rate)
        // note that sum of individual poisson events is a poisson event with rate equal to the sum of the individual rates
        for (unsigned int n = 0; n < arrival_voxel_indexes.size(); n++) {
            if (total_weight > 1e-16) { arrival_voxel_probabilities[n] /= total_weight; }  // avoid division by zero
            else {
                // no signal, all voxels have the same probability
                if (no_signal) { arrival_voxel_probabilities[n] = 1.0 / arrival_voxel_indexes.size(); }
                // no signal because no substrate at the voxel
                else { arrival_voxel_probabilities[n] = 0.0; }
            }
            // store the probability in the microenvironment
            int arrival_prob_idx = microenvironment.find_density_index(pCD->name+"_arrival_prob");
            microenvironment.density_vector(arrival_voxel_indexes[n])[arrival_prob_idx] = parameters.doubles(arrival_rate_name) * arrival_voxel_probabilities[n] * dt;
            // std::cout << "Voxel " << arrival_voxel_indexes[n] << " has probability " << arrival_voxel_probabilities[n] << std::endl;
            // cell arrival
            if (UniformRandom() < microenvironment.density_vector(arrival_voxel_indexes[n])[arrival_prob_idx]){
                //std::cout << "Cell of type " << pCD->name << " arrived at voxel " << arrival_voxel_indexes[n] << std::endl;
                pC = create_cell(*pCD);
                pC->assign_position(microenvironment.mesh.voxels[arrival_voxel_indexes[n]].center);
            }
        }
    }
}


void cell_departure_function(double dt) {
    std::string departure_rate_name, departure_substrate_name, departure_half_max_name, departure_signal_power_name;
    std::vector<double> departure_voxel_probabilities(departure_voxel_indexes.size(), 0.0);
    bool no_signal = false;
    unsigned int index_substrate;
    double total_weight = 0.0;
    double substrate_at_voxel;
    //Cell* pC = NULL;

    // find cells to depart from the domain
    for (int k=0; k < cell_definitions_by_index.size() ; k++) {
        Cell_Definition* pCD = cell_definitions_by_index[k];

        // check: departure_rate_{cell type X} : maximum rate of departure of cell type X
        departure_rate_name = "max_departure_rate_" + pCD->name;

        // no departure rate for this cell type
        if (parameters.doubles.find_index(departure_rate_name) <= -1) { continue; }
        else {
            departure_substrate_name = "departure_signal_substrate_" + pCD->name;
            departure_half_max_name = "departure_signal_half_max_" + pCD->name;
            departure_signal_power_name = "departure_signal_power_" + pCD->name;
        }

        // no departure substrate, half max or signal power defined for this cell type
        if ((parameters.strings.find_index(departure_substrate_name) <= -1 ) || (parameters.doubles.find_index(departure_half_max_name) <= -1) || (parameters.ints.find_index(departure_signal_power_name) <= -1)) {
            std::cout << "Error: no departure substrate, half max or signal power defined for this cell type!" << pCD->name << std::endl;
            std::cout << "Add in user parameters: departure_signal_half_max_" << pCD->name << " (type double)." << std::endl;
            std::cout << "Add in user parameters: departure_signal_substrate_" << pCD->name << " (type string)." << std::endl;
            std::cout << "Add in user parameters: departure_signal_power_" << pCD->name << " (type int)." << std::endl;
            exit(-1);
        }
        else { no_signal = true; } // no departure substrate defined for this cell type
        if (!no_signal) {
            // update the probabilities of departing in each voxel selected according to the signal of substrate
            index_substrate = microenvironment.find_density_index(parameters.strings(departure_substrate_name));
            total_weight = 0.0;
            for (unsigned int n = 0; n < departure_voxel_indexes.size(); n++) {
                substrate_at_voxel = microenvironment.density_vector(departure_voxel_indexes[n])[index_substrate];
                departure_voxel_probabilities[n] = Hill_response_function(substrate_at_voxel, parameters.doubles(departure_half_max_name), parameters.ints(departure_signal_power_name));
                total_weight += departure_voxel_probabilities[n];
            }
        }

        // normalize the probabilities, update the microenvironment and generate cells according to the departure rate (0, max_departure_rate)
        // note that sum of individual poisson events is a poisson event with rate equal to the sum of the individual rates
        for (unsigned int n = 0; n < departure_voxel_indexes.size();  n++) {
            if (total_weight > 1e-16) { departure_voxel_probabilities[n] /= total_weight; } // avoid division by zero
            else {
                // no signal, all voxels have the same probability
                if (no_signal) { departure_voxel_probabilities[n] = 1.0 / departure_voxel_indexes.size(); }
                // no signal because no substrate at the voxel
                else { departure_voxel_probabilities[n] = 0.0; }
            }

            // store the probability in the microenvironment
            int departure_prob_idx = microenvironment.find_density_index(pCD->name + "_departure_prob");
            microenvironment.density_vector(departure_voxel_indexes[n])[departure_prob_idx] = parameters.doubles(departure_rate_name) * departure_voxel_probabilities[n] * dt;

            // find the mechanic voxel index corresponding to the microenvironment voxel index
            int mechanic_voxel_index = ((Cell_Container *)microenvironment.agent_container)->underlying_mesh.nearest_voxel_index(microenvironment.mesh.voxels[departure_voxel_indexes[n]].center);
            auto &cells = ((Cell_Container *)microenvironment.agent_container)->agent_grid[mechanic_voxel_index];
            for (BioFVM::Basic_Agent* p_agent : cells) {
                if (!p_agent) { continue; }
                Cell* p_cell = dynamic_cast< Cell* >(p_agent);
                if (p_cell == nullptr) { continue; }
                if (p_cell->type == pCD->type & p_agent->get_current_voxel_index() == departure_voxel_indexes[n]) {
                    // cell departure
                    if (UniformRandom() < microenvironment.density_vector(departure_voxel_indexes[n])[departure_prob_idx]) {
                        // std::cout << "Cell of type " << pCD->name << " departed from voxel " << departure_voxel_indexes[n] << std::endl;
                        p_cell->lyse_cell();
                        break;  // only one cell can depart at a time (preserves probability sum)
                    }
                }
            }
        }
    }
}


// modified standard custom cpp function
void setup_microenvironment(void) {
    // set domain parameters

    // put any custom code to set non-homogeneous initial conditions or
    // extra Dirichlet nodes here.

    // initialize BioFVM
    initialize_microenvironment();

    // heber 20250000
    // initialize voxels that allow cell arrival
    unsigned int num_voxels_arriving_cells = parameters.doubles("fraction_of_voxels_arriving_cells") * microenvironment.mesh.voxels.size();
    for (unsigned int n = 0; n < num_voxels_arriving_cells; n++) {
        //sample between 0 and microenvironment.mesh.voxels.size() without repetition
        unsigned int sampled_voxel;
        do {
            sampled_voxel = (unsigned int) (UniformRandom() * (microenvironment.mesh.voxels.size()-1));
        } while (std::find(arrival_voxel_indexes.begin(), arrival_voxel_indexes.end(), static_cast<int>(sampled_voxel)) != arrival_voxel_indexes.end());
        arrival_voxel_indexes.push_back(static_cast<int>(sampled_voxel));
    }

    // initialize voxels that allow cell departure
    unsigned int num_voxels_departing_cells = parameters.doubles("fraction_of_voxels_departing_cells") * microenvironment.mesh.voxels.size();
    for (unsigned int n = 0; n < num_voxels_departing_cells; n++) {
        //sample between 0 and microenvironment.mesh.voxels.size() without repetition
        unsigned int sampled_voxel;
        do {
            sampled_voxel = (unsigned int) (UniformRandom() * (microenvironment.mesh.voxels.size()-1));
        } while (std::find(departure_voxel_indexes.begin(), departure_voxel_indexes.end(), static_cast<int>(sampled_voxel)) != departure_voxel_indexes.end());
        departure_voxel_indexes.push_back(static_cast<int>(sampled_voxel));
    }

    // create substrates to track the probability of arrival and departure of each cell type
    std::string arrival_rate_name;
    std::string departure_rate_name;

    return;
}


// modified standard custom cpp function
void setup_tissue(void) {
    double Xmin = microenvironment.mesh.bounding_box[0];
    double Ymin = microenvironment.mesh.bounding_box[1];
    double Zmin = microenvironment.mesh.bounding_box[2];

    double Xmax = microenvironment.mesh.bounding_box[3];
    double Ymax = microenvironment.mesh.bounding_box[4];
    double Zmax = microenvironment.mesh.bounding_box[5];

    if (default_microenvironment_options.simulate_2D == true) {
        Zmin = 0.0;
        Zmax = 0.0;
    }

    double Xrange = Xmax - Xmin;
    double Yrange = Ymax - Ymin;
    double Zrange = Zmax - Zmin;

    // create some of each type of cell
    Cell* pC;

    for (int k=0; k < cell_definitions_by_index.size(); k++) {
        Cell_Definition* pCD = cell_definitions_by_index[k];
        std::cout << "Placing cells of type " << pCD->name << " ... " << std::endl;
        for (int n = 0; n < parameters.ints("number_of_cells"); n++) {
            std::vector<double> position = {0,0,0};
            position[0] = Xmin + UniformRandom() * Xrange;
            position[1] = Ymin + UniformRandom() * Yrange;
            position[2] = Zmin + UniformRandom() * Zrange;

            pC = create_cell(*pCD);
            pC->assign_position(position);
        }
    }
    std::cout << std::endl;

    // load cells from your CSV file (if enabled)
    load_cells_from_pugixml();
    set_parameters_from_distributions();

    // add custom data vector
    //for (int i = 0 ; i < all_cells->size(); i++) {
    //    std::vector<double> vector_double = VECTOR_ZERO;
    //    (*all_cells)[i]->custom_data.add_vector_variable("my_vector", vector_double);
    //}

    // heber 20250000
    // check for each cell type if a substrates of the probability of arrival exist.
    std::string arrival_rate_name;
    for (int k=0; k < cell_definitions_by_index.size() ; k++) {
        Cell_Definition* pCD = cell_definitions_by_index[k];
        arrival_rate_name = "max_arrival_rate_" + pCD->name;
        // if no arrival rate is defined do not check for the substrate
        if (parameters.doubles.find_index(arrival_rate_name) <= -1) { continue; }
        if (microenvironment.find_density_index(pCD->name+"_arrival_prob") == -1) {
            std::cout << "Error: no substrate of the probability of arrival defined!" << std::endl;
            std::cout << "Add a substrate " << pCD->name << "_arrival_prob." << std::endl;
            exit(-1);
        }
    }

    // heber 20250000
    // check for each cell type if a substrates of the probability of departure exist.
    std::string departure_rate_name;
    for (int k=0; k < cell_definitions_by_index.size() ; k++) {
        Cell_Definition* pCD = cell_definitions_by_index[k];
        departure_rate_name = "max_departure_rate_" + pCD->name;
        // if no departure rate is defined do not check for the substrate
        if (parameters.doubles.find_index(departure_rate_name) <= -1) { continue; }
        if (microenvironment.find_density_index(pCD->name+"_departure_prob") == -1) {
            std::cout << "Error: no substrate of the probability of departure defined!" << std::endl;
            std::cout << "Add a substrate " << pCD->name << "_departure_prob." << std::endl;
            exit(-1);
        }
    }

    return;
}


// standard custom cpp function
std::vector<std::string> my_coloring_function(Cell* pCell) {
    return paint_by_number_cell_coloring(pCell);
}

// standard custom cpp function
// bue 20250911: possibly plug in delay
void phenotype_function(Cell* pCell, Phenotype& phenotype, double dt) {
    return;
}

// standard custom cpp function
void custom_function(Cell* pCell, Phenotype& phenotype, double dt) {
    return;
}

// standard custom cpp function
void contact_function(Cell* pMe, Phenotype& phenoMe, Cell* pOther, Phenotype& phenoOther, double dt) {
    return;
}


// bue 20251014: pwn efferocytosis impelentation (updated in intracellular function)
void efferocytosis(double dt) {
    // description:
    //   function digest first necrotic engulfed dying bodies and ingested debris then apoptotic the the rest.
    //   realted literature from zent and elliott 2016/17 https://doi.org/10.1111/febs.13961

    // digestin rate constant
    double hunger = (2.5 / 60) * dt;  // 10/4[cell_solid/h] = 2.5[cell_solid/h] = 0.041666[cell_solid/min] executed every dt [min]

    // BioFVM Indices
    static int debris_apoptotic_index = microenvironment.find_density_index("debris_apoptotic");
    static int debris_necrotic_index = microenvironment.find_density_index("debris_necrotic");

    #pragma omp parallel for
    for (Cell* pCell: (*all_cells)) {
        // extract values
        double food_debris_apoptotic = 0;
        double food_debris_necrotic = 0;
        if (debris_apoptotic_index > -1) { food_debris_apoptotic = pCell->phenotype.molecular.internalized_total_substrates[debris_apoptotic_index]; }
        if (debris_necrotic_index > -1) { food_debris_necrotic = pCell->phenotype.molecular.internalized_total_substrates[debris_necrotic_index]; }

        // digest engulfed_necrotic
        if (hunger > pCell->custom_data["engulfed_necrotic"]) {
            hunger = hunger - pCell->custom_data["engulfed_necrotic"];
            pCell->custom_data["engulfed_necrotic"] = 0;
        }
        else {
           pCell->custom_data["engulfed_necrotic"] = pCell->custom_data["engulfed_necrotic"] - hunger;
           hunger = 0;
        }
        // digest debris_necrotic
        if (debris_necrotic_index > -1) {
            if (hunger > food_debris_necrotic) {
                hunger = hunger - food_debris_necrotic;
                food_debris_necrotic = 0;
            }
            else {
                food_debris_necrotic = food_debris_necrotic - hunger;
                hunger = 0;
            }
            pCell->phenotype.molecular.internalized_total_substrates[debris_necrotic_index] = food_debris_necrotic;
        }

        // digest engulfed_apoptotic
        if (hunger > pCell->custom_data["engulfed_apoptotic"]) {
            hunger = hunger - pCell->custom_data["engulfed_apoptotic"];
            pCell->custom_data["engulfed_apoptotic"] = 0;
        }
        else {
           pCell->custom_data["engulfed_apoptotic"] = pCell->custom_data["engulfed_apoptotic"] - hunger;
           hunger = 0;
        }
        // digest debris apoptotic
        if (debris_apoptotic_index > -1) {
            if (hunger > food_debris_apoptotic) {
                hunger = hunger - food_debris_apoptotic;
                food_debris_apoptotic = 0;
            }
            else {
                food_debris_apoptotic = food_debris_apoptotic - hunger;
                hunger = 0;
            }
            pCell->phenotype.molecular.internalized_total_substrates[debris_apoptotic_index] = food_debris_apoptotic;
        }

        // update debris_total
        pCell->custom_data["debris_total"] = pCell->custom_data["engulfed_necrotic"] + food_debris_necrotic + pCell->custom_data["engulfed_apoptotic"] + food_debris_apoptotic;

        // output
        //if (pCell->custom_data["debris_total"] > 9) {
        //    std::cout << "cell ID:" << pCell->ID << ",  type:"<< pCell->type_name << ", debris_total variable:" << pCell->custom_data["debris_total"] << std::endl;
        //}
    }
    return;
}

// 20260423: physigym
int add_substrate(std::string s_substrate, double r_dose) {
    // update substrate concentration
    int k = microenvironment.find_density_index(s_substrate);
    for (unsigned int n=0; n < microenvironment.number_of_voxels(); n++) {
        microenvironment(n)[k] += r_dose;
    }
    return 0;
}
