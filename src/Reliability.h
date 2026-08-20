#ifndef MULTINEAT_RELIABILITY_H
#define MULTINEAT_RELIABILITY_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <limits>

namespace NEAT
{

enum EvolutionMode
{
    LEGACY_V2 = 0,
    RELIABLE_NEAT,
    ROBUST_QD_NEAT
};

enum FitnessAggregation
{
    FITNESS_MEAN = 0,
    BOOTSTRAP_LOWER_BOUND
};

enum EvaluationPhase
{
    TRAINING_EVALUATION = 0,
    VALIDATION_EVALUATION,
    CERTIFICATION_EVALUATION
};

enum OperatorKind
{
    WEIGHT_MUTATION_OPERATOR = 0,
    STRUCTURAL_ADD_OPERATOR,
    STRUCTURAL_REMOVE_OPERATOR,
    PARAMETER_MUTATION_OPERATOR,
    CROSSOVER_OPERATOR,
    WEIGHT_EMITTER_OPERATOR,
    OPERATOR_KIND_COUNT
};

struct EvaluationSample
{
    std::uint64_t case_id = 0;
    double score = 0.0;
    bool has_success = false;
    bool success = false;
    std::vector<double> behavior_descriptor;
};

struct EvaluationSummary
{
    std::size_t sample_count = 0;
    std::size_t success_count = 0;
    std::size_t success_sample_count = 0;
    double mean = 0.0;
    double standard_deviation = 0.0;
    double standard_error = 0.0;
    double lower_confidence_bound = 0.0;
    double lower_tail_mean = 0.0;
    double success_rate = 0.0;
    double success_lower_bound = 0.0;
};

struct EvaluationRequest
{
    int genome_id = -1;
    std::uint64_t case_id = 0;
    unsigned int stage = 0;
    EvaluationPhase phase = TRAINING_EVALUATION;
    bool archive_evaluation = false;
    std::size_t archive_index = std::numeric_limits<std::size_t>::max();
};

struct LineageRecord
{
    std::vector<int> parent_ids;
    std::vector<OperatorKind> operators;
    double best_parent_score = 0.0;
    int origin_species_id = 0;
    bool archive_replacement = false;
    std::vector<double> emitter_parameters;
};

EvaluationSummary SummarizeEvaluations(
    const std::vector<EvaluationSample>& samples,
    double confidence = 0.90,
    unsigned int bootstrap_samples = 2048);

double WilsonLowerBound(
    std::size_t successes,
    std::size_t samples,
    double z = 1.6448536269514722);

} // namespace NEAT

#endif
