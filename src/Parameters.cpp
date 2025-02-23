#include "Parameters.h"
#include <iostream>
#include <unordered_map>
#include <functional>
#include <fstream>
#include <string>

namespace NEAT {

    Parameters::Parameters() {
        Reset();
    }

    void Parameters::Reset() {
        // Basic parameters
        PopulationSize = 300;
        Speciation = true;
        DynamicCompatibility = true;
        MinSpecies = 5;
        MaxSpecies = 10;
        InnovationsForever = true;
        AllowClones = true;
        ArchiveEnforcement = false;
        DontUseBiasNeuron = false;
        AllowLoops = false;
        NormalizeGenomeSize = false;
        CustomConstraints = nullptr;

        // GA Parameters
        YoungAgeTreshold = 5;
        YoungAgeFitnessBoost = 1.1;
        SpeciesMaxStagnation = 25000;
        StagnationDelta = 0.0;
        OldAgeTreshold = 30;
        OldAgePenalty = 0.5;
        DetectCompetetiveCoevolutionStagnation = false;
        KillWorstSpeciesEach = 15;
        KillWorstAge = 10;
        SurvivalRate = 0.2;
        CrossoverRate = 0.7;
        OverallMutationRate = 0.75;
        InterspeciesCrossoverRate = 0.0001;
        MultipointCrossoverRate = 0.75;
        PreferFitterParentRate = 0.25;
        RouletteWheelSelection = false;
        TournamentSelection = true;
        TournamentSize = 5;
        EliteFraction = 0.000001;

        // Phased Search parameters
        PhasedSearching = false;
        DeltaCoding = false;
        SimplifyingPhaseMPCTreshold = 20;
        SimplifyingPhaseStagnationTreshold = 30;
        ComplexityFloorGenerations = 40;

        // Novelty Search parameters
        NoveltySearch_K = 15;
        NoveltySearch_P_min = 0.5;
        NoveltySearch_Dynamic_Pmin = true;
        NoveltySearch_No_Archiving_Stagnation_Treshold = 150;
        NoveltySearch_Pmin_lowering_multiplier = 0.9;
        NoveltySearch_Pmin_min = 0.05;
        NoveltySearch_Quick_Archiving_Min_Evaluations = 8;
        NoveltySearch_Pmin_raising_multiplier = 1.1;
        NoveltySearch_Recompute_Sparseness_Each = 25;

        // Mutation parameters
        MutateAddNeuronProb = 0.01;
        SplitRecurrent = false;
        SplitLoopedRecurrent = false;
        NeuronTries = 64;
        MutateAddLinkProb = 0.03;
        MutateAddLinkFromBiasProb = 0.0;
        MutateRemLinkProb = 0.0;
        MutateRemSimpleNeuronProb = 0.0;
        LinkTries = 64;
        MaxLinks = -1;
        MaxNeurons = -1;
        RecurrentProb = 0.25;
        RecurrentLoopProb = 0.25;
        MutateWeightsProb = 0.90;
        MutateWeightsSevereProb = 0.25;
        WeightMutationRate = 1.0;
        WeightMutationMaxPower = 1.0;
        WeightReplacementRate = 0.2;
        WeightReplacementMaxPower = 1.0;
        MaxWeight = 8.0;
        MinWeight = -8.0;
        MutateActivationAProb = 0.0;
        MutateActivationBProb = 0.0;
        ActivationAMutationMaxPower = 0.0;
        ActivationBMutationMaxPower = 0.0;
        TimeConstantMutationMaxPower = 0.0;
        BiasMutationMaxPower = WeightMutationMaxPower;
        MinActivationA = 1.0;
        MaxActivationA = 1.0;
        MinActivationB = 0.0;
        MaxActivationB = 0.0;
        MutateNeuronActivationTypeProb = 0.0;
        ActivationFunction_SignedSigmoid_Prob = 0.0;
        ActivationFunction_UnsignedSigmoid_Prob = 1.0;
        ActivationFunction_Tanh_Prob = 0.0;
        ActivationFunction_TanhCubic_Prob = 0.0;
        ActivationFunction_SignedStep_Prob = 0.0;
        ActivationFunction_UnsignedStep_Prob = 0.0;
        ActivationFunction_SignedGauss_Prob = 0.0;
        ActivationFunction_UnsignedGauss_Prob = 0.0;
        ActivationFunction_Abs_Prob = 0.0;
        ActivationFunction_SignedSine_Prob = 0.0;
        ActivationFunction_UnsignedSine_Prob = 0.0;
        ActivationFunction_Linear_Prob = 0.0;
        ActivationFunction_Relu_Prob = 0.0;
        ActivationFunction_Softplus_Prob = 0.0;
        MutateNeuronTimeConstantsProb = 0.0;
        MutateNeuronBiasesProb = 0.0;
        MinNeuronTimeConstant = 0.0;
        MaxNeuronTimeConstant = 0.0;
        MinNeuronBias = 0.0;
        MaxNeuronBias = 0.0;

        // Speciation parameters
        DisjointCoeff = 1.0;
        ExcessCoeff = 1.0;
        ActivationADiffCoeff = 0.0;
        ActivationBDiffCoeff = 0.0;
        WeightDiffCoeff = 0.5;
        TimeConstantDiffCoeff = 0.0;
        BiasDiffCoeff = 0.0;
        ActivationFunctionDiffCoeff = 0.0;
        CompatTreshold = 3.0;
        MinCompatTreshold = 0.0;
        CompatTresholdModifier = 0.1;
        CompatTreshChangeInterval_Generations = 1;
        CompatTreshChangeInterval_Evaluations = 1;
        MinDeltaCompatEqualGenomes = 1e-7;
        ConstraintTrials = 2000000;

        DontUseBiasNeuron = false;
        AllowLoops = false;

        // ES-HyperNEAT parameters
        DivisionThreshold = 0.03;
        VarianceThreshold = 0.03;
        BandThreshold = 0.3;
        InitialDepth = 3;
        MaxDepth = 3;
        IterationLevel = 1;
        CPPN_Bias = 1.0;
        Width = 2.0;
        Height = 2.0;
        Qtree_X = 0.0;
        Qtree_Y = 0.0;
        Leo = false;
        LeoThreshold = 0.1;
        LeoSeed = false;
        GeometrySeed = false;

        EliteFraction = 0.0;

        // Universal traits (cleared by default)
        NeuronTraits.clear();
        LinkTraits.clear();
        GenomeTraits.clear();
        MutateNeuronTraitsProb = 0.0;
        MutateLinkTraitsProb = 0.0;
        MutateGenomeTraitsProb = 0.0;
    }

    int Parameters::Load(std::ifstream& a_DataFile) {
        std::string s, tf;
    
        // Move to the start of the parameters block
        do {
            a_DataFile >> s;
        } while (s != "NEAT_ParametersStart" && !a_DataFile.eof());
    
        // Mapping parameter names to lambdas that update the correct member variable
        std::unordered_map<std::string, std::function<void()>> param_map = {
            // Basic parameters
            {"PopulationSize", [&]() { a_DataFile >> PopulationSize; }},
            {"Speciation", [&]() { a_DataFile >> tf; Speciation = (tf == "true" || tf == "1"); }},
            {"DynamicCompatibility", [&]() { a_DataFile >> tf; DynamicCompatibility = (tf == "true" || tf == "1"); }},
            {"MinSpecies", [&]() { a_DataFile >> MinSpecies; }},
            {"MaxSpecies", [&]() { a_DataFile >> MaxSpecies; }},
            {"InnovationsForever", [&]() { a_DataFile >> tf; InnovationsForever = (tf == "true" || tf == "1"); }},
            {"AllowClones", [&]() { a_DataFile >> tf; AllowClones = (tf == "true" || tf == "1"); }},
            {"ArchiveEnforcement", [&]() { a_DataFile >> tf; ArchiveEnforcement = (tf == "true" || tf == "1"); }},
            {"DontUseBiasNeuron", [&]() { a_DataFile >> tf; DontUseBiasNeuron = (tf == "true" || tf == "1"); }},
            {"AllowLoops", [&]() { a_DataFile >> tf; AllowLoops = (tf == "true" || tf == "1"); }},
            {"NormalizeGenomeSize", [&]() { a_DataFile >> tf; NormalizeGenomeSize = (tf == "true" || tf == "1"); }},
            {"ConstraintTrials", [&]() { a_DataFile >> ConstraintTrials; }},
            
            // GA Parameters
            {"YoungAgeTreshold", [&]() { a_DataFile >> YoungAgeTreshold; }},
            {"YoungAgeFitnessBoost", [&]() { a_DataFile >> YoungAgeFitnessBoost; }},
            {"SpeciesMaxStagnation", [&]() { a_DataFile >> SpeciesMaxStagnation; }},
            {"StagnationDelta", [&]() { a_DataFile >> StagnationDelta; }},
            {"OldAgeTreshold", [&]() { a_DataFile >> OldAgeTreshold; }},
            {"OldAgePenalty", [&]() { a_DataFile >> OldAgePenalty; }},
            {"DetectCompetetiveCoevolutionStagnation", [&]() { a_DataFile >> tf; DetectCompetetiveCoevolutionStagnation = (tf == "true" || tf == "1"); }},
            {"KillWorstSpeciesEach", [&]() { a_DataFile >> KillWorstSpeciesEach; }},
            {"KillWorstAge", [&]() { a_DataFile >> KillWorstAge; }},
            {"SurvivalRate", [&]() { a_DataFile >> SurvivalRate; }},
            {"CrossoverRate", [&]() { a_DataFile >> CrossoverRate; }},
            {"OverallMutationRate", [&]() { a_DataFile >> OverallMutationRate; }},
            {"InterspeciesCrossoverRate", [&]() { a_DataFile >> InterspeciesCrossoverRate; }},
            {"MultipointCrossoverRate", [&]() { a_DataFile >> MultipointCrossoverRate; }},
            {"PreferFitterParentRate", [&]() { a_DataFile >> PreferFitterParentRate; }},
            {"RouletteWheelSelection", [&]() { a_DataFile >> tf; RouletteWheelSelection = (tf == "true" || tf == "1"); }},
            {"TournamentSelection", [&]() { a_DataFile >> tf; TournamentSelection = (tf == "true" || tf == "1"); }},
            {"TournamentSize", [&]() { a_DataFile >> TournamentSize; }},
            {"EliteFraction", [&]() { a_DataFile >> EliteFraction; }},
    
            // Phased Search parameters
            {"PhasedSearching", [&]() { a_DataFile >> tf; PhasedSearching = (tf == "true" || tf == "1"); }},
            {"DeltaCoding", [&]() { a_DataFile >> tf; DeltaCoding = (tf == "true" || tf == "1"); }},
            {"SimplifyingPhaseMPCTreshold", [&]() { a_DataFile >> SimplifyingPhaseMPCTreshold; }},
            {"SimplifyingPhaseStagnationTreshold", [&]() { a_DataFile >> SimplifyingPhaseStagnationTreshold; }},
            {"ComplexityFloorGenerations", [&]() { a_DataFile >> ComplexityFloorGenerations; }},
    
            // Novelty Search parameters
            {"NoveltySearch_K", [&]() { a_DataFile >> NoveltySearch_K; }},
            {"NoveltySearch_P_min", [&]() { a_DataFile >> NoveltySearch_P_min; }},
            {"NoveltySearch_Dynamic_Pmin", [&]() { a_DataFile >> tf; NoveltySearch_Dynamic_Pmin = (tf == "true" || tf == "1"); }},
            {"NoveltySearch_No_Archiving_Stagnation_Treshold", [&]() { a_DataFile >> NoveltySearch_No_Archiving_Stagnation_Treshold; }},
            {"NoveltySearch_Pmin_lowering_multiplier", [&]() { a_DataFile >> NoveltySearch_Pmin_lowering_multiplier; }},
            {"NoveltySearch_Pmin_min", [&]() { a_DataFile >> NoveltySearch_Pmin_min; }},
            {"NoveltySearch_Quick_Archiving_Min_Evaluations", [&]() { a_DataFile >> NoveltySearch_Quick_Archiving_Min_Evaluations; }},
            {"NoveltySearch_Pmin_raising_multiplier", [&]() { a_DataFile >> NoveltySearch_Pmin_raising_multiplier; }},
            {"NoveltySearch_Recompute_Sparseness_Each", [&]() { a_DataFile >> NoveltySearch_Recompute_Sparseness_Each; }},
    
            // Speciation parameters
            {"DisjointCoeff", [&]() { a_DataFile >> DisjointCoeff; }},
            {"ExcessCoeff", [&]() { a_DataFile >> ExcessCoeff; }},
            {"ActivationADiffCoeff", [&]() { a_DataFile >> ActivationADiffCoeff; }},
            {"ActivationBDiffCoeff", [&]() { a_DataFile >> ActivationBDiffCoeff; }},
            {"WeightDiffCoeff", [&]() { a_DataFile >> WeightDiffCoeff; }},
            {"TimeConstantDiffCoeff", [&]() { a_DataFile >> TimeConstantDiffCoeff; }},
            {"BiasDiffCoeff", [&]() { a_DataFile >> BiasDiffCoeff; }},
            {"ActivationFunctionDiffCoeff", [&]() { a_DataFile >> ActivationFunctionDiffCoeff; }},
            {"CompatTreshold", [&]() { a_DataFile >> CompatTreshold; }},
            {"MinCompatTreshold", [&]() { a_DataFile >> MinCompatTreshold; }},
            {"CompatTresholdModifier", [&]() { a_DataFile >> CompatTresholdModifier; }},
            {"CompatTreshChangeInterval_Generations", [&]() { a_DataFile >> CompatTreshChangeInterval_Generations; }},
            {"CompatTreshChangeInterval_Evaluations", [&]() { a_DataFile >> CompatTreshChangeInterval_Evaluations; }},
            {"MinDeltaCompatEqualGenomes", [&]() { a_DataFile >> MinDeltaCompatEqualGenomes; }},
        };
    
        while (s != "NEAT_ParametersEnd" && !a_DataFile.eof()) {
            a_DataFile >> s;
            auto it = param_map.find(s);
            if (it != param_map.end()) {
                it->second(); // Call the associated lambda function
            } else {
                std::cerr << "Unknown parameter: " << s << std::endl;
            }
        }
    
        return 0;
    }
    

    int Parameters::Load(const char* a_FileName) {
        std::ifstream data(a_FileName);
        if (!data.is_open())
            return 0;
        int result = Load(data);
        data.close();
        return result;
    }

    void Parameters::Save(const char* filename) {
        FILE* f = fopen(filename, "w");
        Save(f);
        fclose(f);
    }

    void Parameters::Save(FILE* a_fstream) {
        fprintf(a_fstream, "NEAT_ParametersStart\n");
        fprintf(a_fstream, "PopulationSize %d\n", PopulationSize);
        fprintf(a_fstream, "Speciation %s\n", Speciation ? "true" : "false");
        fprintf(a_fstream, "DynamicCompatibility %s\n", DynamicCompatibility ? "true" : "false");
        fprintf(a_fstream, "MinSpecies %d\n", MinSpecies);
        fprintf(a_fstream, "MaxSpecies %d\n", MaxSpecies);
        fprintf(a_fstream, "InnovationsForever %s\n", InnovationsForever ? "true" : "false");
        fprintf(a_fstream, "AllowClones %s\n", AllowClones ? "true" : "false");
        fprintf(a_fstream, "NormalizeGenomeSize %s\n", NormalizeGenomeSize ? "true" : "false");
        fprintf(a_fstream, "ConstraintTrials %d\n", ConstraintTrials);
        fprintf(a_fstream, "YoungAgeTreshold %d\n", YoungAgeTreshold);
        fprintf(a_fstream, "YoungAgeFitnessBoost %3.20f\n", YoungAgeFitnessBoost);
        fprintf(a_fstream, "SpeciesMaxStagnation %d\n", SpeciesMaxStagnation);
        fprintf(a_fstream, "StagnationDelta %3.20f\n", StagnationDelta);
        fprintf(a_fstream, "OldAgeTreshold %d\n", OldAgeTreshold);
        fprintf(a_fstream, "OldAgePenalty %3.20f\n", OldAgePenalty);
        fprintf(a_fstream, "DetectCompetetiveCoevolutionStagnation %s\n", DetectCompetetiveCoevolutionStagnation ? "true" : "false");
        fprintf(a_fstream, "KillWorstSpeciesEach %d\n", KillWorstSpeciesEach);
        fprintf(a_fstream, "KillWorstAge %d\n", KillWorstAge);
        fprintf(a_fstream, "SurvivalRate %3.20f\n", SurvivalRate);
        fprintf(a_fstream, "CrossoverRate %3.20f\n", CrossoverRate);
        fprintf(a_fstream, "OverallMutationRate %3.20f\n", OverallMutationRate);
        fprintf(a_fstream, "InterspeciesCrossoverRate %3.20f\n", InterspeciesCrossoverRate);
        fprintf(a_fstream, "MultipointCrossoverRate %3.20f\n", MultipointCrossoverRate);
        fprintf(a_fstream, "PreferFitterParentRate %3.20f\n", PreferFitterParentRate);
        fprintf(a_fstream, "RouletteWheelSelection %s\n", RouletteWheelSelection ? "true" : "false");
        fprintf(a_fstream, "TournamentSelection %s\n", TournamentSelection ? "true" : "false");
        fprintf(a_fstream, "TournamentSize %d\n", TournamentSize);
        fprintf(a_fstream, "PhasedSearching %s\n", PhasedSearching ? "true" : "false");
        fprintf(a_fstream, "DeltaCoding %s\n", DeltaCoding ? "true" : "false");
        fprintf(a_fstream, "SimplifyingPhaseMPCTreshold %d\n", SimplifyingPhaseMPCTreshold);
        fprintf(a_fstream, "SimplifyingPhaseStagnationTreshold %d\n", SimplifyingPhaseStagnationTreshold);
        fprintf(a_fstream, "ComplexityFloorGenerations %d\n", ComplexityFloorGenerations);
        fprintf(a_fstream, "NoveltySearch_K %d\n", NoveltySearch_K);
        fprintf(a_fstream, "NoveltySearch_P_min %3.20f\n", NoveltySearch_P_min);
        fprintf(a_fstream, "NoveltySearch_Dynamic_Pmin %s\n", NoveltySearch_Dynamic_Pmin ? "true" : "false");
        fprintf(a_fstream, "NoveltySearch_No_Archiving_Stagnation_Treshold %d\n", NoveltySearch_No_Archiving_Stagnation_Treshold);
        fprintf(a_fstream, "NoveltySearch_Pmin_lowering_multiplier %3.20f\n", NoveltySearch_Pmin_lowering_multiplier);
        fprintf(a_fstream, "NoveltySearch_Pmin_min %3.20f\n", NoveltySearch_Pmin_min);
        fprintf(a_fstream, "NoveltySearch_Quick_Archiving_Min_Evaluations %d\n", NoveltySearch_Quick_Archiving_Min_Evaluations);
        fprintf(a_fstream, "NoveltySearch_Pmin_raising_multiplier %3.20f\n", NoveltySearch_Pmin_raising_multiplier);
        fprintf(a_fstream, "NoveltySearch_Recompute_Sparseness_Each %d\n", NoveltySearch_Recompute_Sparseness_Each);
        fprintf(a_fstream, "MutateAddNeuronProb %3.20f\n", MutateAddNeuronProb);
        fprintf(a_fstream, "SplitRecurrent %s\n", SplitRecurrent ? "true" : "false");
        fprintf(a_fstream, "SplitLoopedRecurrent %s\n", SplitLoopedRecurrent ? "true" : "false");
        fprintf(a_fstream, "NeuronTries %d\n", NeuronTries);
        fprintf(a_fstream, "MutateAddLinkProb %3.20f\n", MutateAddLinkProb);
        fprintf(a_fstream, "MutateAddLinkFromBiasProb %3.20f\n", MutateAddLinkFromBiasProb);
        fprintf(a_fstream, "MutateRemLinkProb %3.20f\n", MutateRemLinkProb);
        fprintf(a_fstream, "MutateRemSimpleNeuronProb %3.20f\n", MutateRemSimpleNeuronProb);
        fprintf(a_fstream, "LinkTries %d\n", LinkTries);
        fprintf(a_fstream, "MaxLinks %d\n", MaxLinks);
        fprintf(a_fstream, "MaxNeurons %d\n", MaxNeurons);
        fprintf(a_fstream, "RecurrentProb %3.20f\n", RecurrentProb);
        fprintf(a_fstream, "RecurrentLoopProb %3.20f\n", RecurrentLoopProb);
        fprintf(a_fstream, "MutateWeightsProb %3.20f\n", MutateWeightsProb);
        fprintf(a_fstream, "MutateWeightsSevereProb %3.20f\n", MutateWeightsSevereProb);
        fprintf(a_fstream, "WeightMutationRate %3.20f\n", WeightMutationRate);
        fprintf(a_fstream, "WeightMutationMaxPower %3.20f\n", WeightMutationMaxPower);
        fprintf(a_fstream, "WeightReplacementRate %3.20f\n", WeightReplacementRate);
        fprintf(a_fstream, "WeightReplacementMaxPower %3.20f\n", WeightReplacementMaxPower);
        fprintf(a_fstream, "MaxWeight %3.20f\n", MaxWeight);
        fprintf(a_fstream, "MinWeight %3.20f\n", MinWeight);
        fprintf(a_fstream, "MutateActivationAProb %3.20f\n", MutateActivationAProb);
        fprintf(a_fstream, "MutateActivationBProb %3.20f\n", MutateActivationBProb);
        fprintf(a_fstream, "ActivationAMutationMaxPower %3.20f\n", ActivationAMutationMaxPower);
        fprintf(a_fstream, "ActivationBMutationMaxPower %3.20f\n", ActivationBMutationMaxPower);
        fprintf(a_fstream, "TimeConstantMutationMaxPower %3.20f\n", TimeConstantMutationMaxPower);
        fprintf(a_fstream, "BiasMutationMaxPower %3.20f\n", BiasMutationMaxPower);
        fprintf(a_fstream, "MinActivationA %3.20f\n", MinActivationA);
        fprintf(a_fstream, "MaxActivationA %3.20f\n", MaxActivationA);
        fprintf(a_fstream, "MinActivationB %3.20f\n", MinActivationB);
        fprintf(a_fstream, "MaxActivationB %3.20f\n", MaxActivationB);
        fprintf(a_fstream, "MutateNeuronActivationTypeProb %3.20f\n", MutateNeuronActivationTypeProb);
        fprintf(a_fstream, "ActivationFunction_SignedSigmoid_Prob %3.20f\n", ActivationFunction_SignedSigmoid_Prob);
        fprintf(a_fstream, "ActivationFunction_UnsignedSigmoid_Prob %3.20f\n", ActivationFunction_UnsignedSigmoid_Prob);
        fprintf(a_fstream, "ActivationFunction_Tanh_Prob %3.20f\n", ActivationFunction_Tanh_Prob);
        fprintf(a_fstream, "ActivationFunction_TanhCubic_Prob %3.20f\n", ActivationFunction_TanhCubic_Prob);
        fprintf(a_fstream, "ActivationFunction_SignedStep_Prob %3.20f\n", ActivationFunction_SignedStep_Prob);
        fprintf(a_fstream, "ActivationFunction_UnsignedStep_Prob %3.20f\n", ActivationFunction_UnsignedStep_Prob);
        fprintf(a_fstream, "ActivationFunction_SignedGauss_Prob %3.20f\n", ActivationFunction_SignedGauss_Prob);
        fprintf(a_fstream, "ActivationFunction_UnsignedGauss_Prob %3.20f\n", ActivationFunction_UnsignedGauss_Prob);
        fprintf(a_fstream, "ActivationFunction_Abs_Prob %3.20f\n", ActivationFunction_Abs_Prob);
        fprintf(a_fstream, "ActivationFunction_SignedSine_Prob %3.20f\n", ActivationFunction_SignedSine_Prob);
        fprintf(a_fstream, "ActivationFunction_UnsignedSine_Prob %3.20f\n", ActivationFunction_UnsignedSine_Prob);
        fprintf(a_fstream, "ActivationFunction_Linear_Prob %3.20f\n", ActivationFunction_Linear_Prob);
        fprintf(a_fstream, "ActivationFunction_Relu_Prob %3.20f\n", ActivationFunction_Relu_Prob);
        fprintf(a_fstream, "ActivationFunction_Softplus_Prob %3.20f\n", ActivationFunction_Softplus_Prob);
        fprintf(a_fstream, "MutateNeuronTimeConstantsProb %3.20f\n", MutateNeuronTimeConstantsProb);
        fprintf(a_fstream, "MutateNeuronBiasesProb %3.20f\n", MutateNeuronBiasesProb);
        fprintf(a_fstream, "MinNeuronTimeConstant %3.20f\n", MinNeuronTimeConstant);
        fprintf(a_fstream, "MaxNeuronTimeConstant %3.20f\n", MaxNeuronTimeConstant);
        fprintf(a_fstream, "MinNeuronBias %3.20f\n", MinNeuronBias);
        fprintf(a_fstream, "MaxNeuronBias %3.20f\n", MaxNeuronBias);
        fprintf(a_fstream, "DontUseBiasNeuron %s\n", DontUseBiasNeuron ? "true" : "false");
        fprintf(a_fstream, "ArchiveEnforcement %s\n", ArchiveEnforcement ? "true" : "false");
        fprintf(a_fstream, "AllowLoops %s\n", AllowLoops ? "true" : "false");
        fprintf(a_fstream, "DisjointCoeff %3.20f\n", DisjointCoeff);
        fprintf(a_fstream, "ExcessCoeff %3.20f\n", ExcessCoeff);
        fprintf(a_fstream, "ActivationADiffCoeff %3.20f\n", ActivationADiffCoeff);
        fprintf(a_fstream, "ActivationBDiffCoeff %3.20f\n", ActivationBDiffCoeff);
        fprintf(a_fstream, "WeightDiffCoeff %3.20f\n", WeightDiffCoeff);
        fprintf(a_fstream, "TimeConstantDiffCoeff %3.20f\n", TimeConstantDiffCoeff);
        fprintf(a_fstream, "BiasDiffCoeff %3.20f\n", BiasDiffCoeff);
        fprintf(a_fstream, "ActivationFunctionDiffCoeff %3.20f\n", ActivationFunctionDiffCoeff);
        fprintf(a_fstream, "CompatTreshold %3.20f\n", CompatTreshold);
        fprintf(a_fstream, "MinCompatTreshold %3.20f\n", MinCompatTreshold);
        fprintf(a_fstream, "CompatTresholdModifier %3.20f\n", CompatTresholdModifier);
        fprintf(a_fstream, "CompatTreshChangeInterval_Generations %d\n", CompatTreshChangeInterval_Generations);
        fprintf(a_fstream, "CompatTreshChangeInterval_Evaluations %d\n", CompatTreshChangeInterval_Evaluations);
        fprintf(a_fstream, "MinDeltaCompatEqualGenomes %3.20f\n", MinDeltaCompatEqualGenomes);
        fprintf(a_fstream, "DivisionThreshold %3.20f\n", DivisionThreshold);
        fprintf(a_fstream, "VarianceThreshold %3.20f\n", VarianceThreshold);
        fprintf(a_fstream, "BandThreshold %3.20f\n", BandThreshold);
        fprintf(a_fstream, "InitialDepth %d\n", InitialDepth);
        fprintf(a_fstream, "MaxDepth %d\n", MaxDepth);
        fprintf(a_fstream, "IterationLevel %d\n", IterationLevel);
        fprintf(a_fstream, "TournamentSize %d\n", TournamentSize);
        fprintf(a_fstream, "CPPN_Bias %3.20f\n", CPPN_Bias);
        fprintf(a_fstream, "Width %3.20f\n", Width);
        fprintf(a_fstream, "Height %3.20f\n", Height);
        fprintf(a_fstream, "Qtree_X %3.20f\n", Qtree_X);
        fprintf(a_fstream, "Qtree_Y %3.20f\n", Qtree_Y);
        fprintf(a_fstream, "Leo %s\n", Leo ? "true" : "false");
        fprintf(a_fstream, "LeoThreshold %3.20f\n", LeoThreshold);
        fprintf(a_fstream, "TournamentSize %d\n", TournamentSize);
        fprintf(a_fstream, "LeoSeed %s\n", LeoSeed ? "true" : "false");
        fprintf(a_fstream, "GeometrySeed %s\n", GeometrySeed ? "true" : "false");
        fprintf(a_fstream, "Elitism %3.20f\n", EliteFraction);
        fprintf(a_fstream, "NEAT_ParametersEnd\n");
    }

} // namespace NEAT
