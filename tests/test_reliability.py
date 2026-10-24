"""Regression coverage for evidence handling and asynchronous evaluation."""

import math
import sys
import unittest

import pymultineat as neat


def sample(case_id, score, descriptor=()):
    result = neat.EvaluationSample()
    result.case_id = case_id
    result.score = score
    result.behavior_descriptor = list(descriptor)
    return result


def population(qd=False):
    params = neat.Parameters()
    params.PopulationSize = 4
    params.Speciation = False
    params.EvaluationMinSamples = 4
    params.EvaluationMaxSamples = 16
    params.EvaluationBootstrapSamples = 128
    params.ValidationSamples = 4
    if qd:
        params.EvolutionModeSetting = neat.ROBUST_QD_NEAT
        params.DeterministicEvaluation = True
        params.QDDescriptorMinimums = [0.0]
        params.QDDescriptorMaximums = [1.0]
        params.QDDescriptorBins = [1]
        params.QDCellDepth = 8
        params.QDArchiveReevaluationFraction = 1.0
    init = neat.GenomeInitStruct()
    init.NumInputs = 2
    init.NumOutputs = 1
    return neat.Population(neat.Genome(params, init), params, True, 1.0, 42)


class ReliabilityTests(unittest.TestCase):
    def test_summary_is_independent_of_submission_order(self):
        samples = [sample(i, float(i * i % 17)) for i in range(16)]
        forward = neat.SummarizeEvaluations(samples, 0.9, 128)
        backward = neat.SummarizeEvaluations(list(reversed(samples)), 0.9, 128)
        for field in ("mean", "standard_deviation", "standard_error",
                      "lower_confidence_bound", "lower_tail_mean"):
            self.assertEqual(getattr(forward, field), getattr(backward, field))

    def test_extreme_finite_scores_do_not_overflow(self):
        maximum = sys.float_info.max
        summary = neat.SummarizeEvaluations([sample(i, maximum) for i in range(10)])
        self.assertEqual(summary.mean, maximum)
        self.assertEqual(summary.lower_confidence_bound, maximum)
        self.assertEqual(summary.lower_tail_mean, maximum)
        self.assertEqual(summary.standard_deviation, 0.0)
        summary = neat.SummarizeEvaluations([sample(0, -1e200), sample(1, 1e200)])
        self.assertTrue(math.isclose(summary.standard_deviation, math.sqrt(2) * 1e200))

    def test_empty_evidence_still_validates_options(self):
        for confidence, count in ((0.0, 128), (math.nan, 128), (0.9, 0)):
            with self.subTest(confidence=confidence, count=count):
                with self.assertRaises(ValueError):
                    neat.SummarizeEvaluations([], confidence, count)
        with self.assertRaises(ValueError):
            neat.WilsonLowerBound(1, 0)

    def test_wilson_bounds_remain_finite_for_extreme_confidence(self):
        for successes in (0, 1, 64):
            for z in (sys.float_info.min, sys.float_info.max):
                with self.subTest(successes=successes, z=z):
                    bound = neat.WilsonLowerBound(successes, 64, z)
                    self.assertTrue(math.isfinite(bound))
                    self.assertGreaterEqual(bound, 0.0)
                    self.assertLessEqual(bound, successes / 64)

    def test_large_finite_descriptors_can_finish_evaluation(self):
        pop = population()
        maximum = sys.float_info.max
        while requests := pop.GetEvaluationRequests():
            for request in requests:
                pop.SubmitEvaluation(request.genome_id, request.case_id, 1.0,
                                     False, False, [maximum])
        self.assertTrue(pop.ReadyForEpoch())
        for index in range(pop.NumGenomes()):
            self.assertEqual(pop.AccessGenomeByIndex(index).GetBehaviorDescriptor(), [maximum])

    def test_rejected_evidence_batch_preserves_genome(self):
        genome = population().AccessGenomeByIndex(0)
        genome.SetEvaluationSamples([sample(0, 1.0)])
        before = genome.Serialize()
        with self.assertRaises(ValueError):
            genome.SetEvaluationSamples([sample(1, 2.0), sample(2, math.nan)])
        self.assertEqual(genome.Serialize(), before)

    def test_descriptor_dimensions_are_checked_before_accepting_evidence(self):
        pop = population()
        requests = pop.GetEvaluationRequests()
        first, second = requests[:2]
        pop.SubmitEvaluation(first.genome_id, first.case_id, 1.0, False, False, [0.5])
        before = pop.Serialize()
        with self.assertRaises(ValueError):
            pop.SubmitEvaluation(second.genome_id, second.case_id, 1.0,
                                 False, False, [0.5, 0.5])
        self.assertEqual(pop.Serialize(), before)
        pop.SubmitEvaluation(second.genome_id, second.case_id, 1.0,
                             False, False, [0.5])

    def test_racing_and_resume_ignore_worker_completion_order(self):
        forward = population()
        backward = neat.Population.Deserialize(forward.Serialize())
        for generation in range(2):
            while requests := forward.GetEvaluationRequests():
                other = backward.GetEvaluationRequests()
                self.assertEqual([(r.genome_id, r.case_id) for r in requests],
                                 [(r.genome_id, r.case_id) for r in other])
                for pop, batch in ((forward, requests), (backward, reversed(other))):
                    for request in batch:
                        score = float((request.case_id * 7 + request.genome_id) % 19)
                        pop.SubmitEvaluation(request.genome_id, request.case_id, score)
                backward = neat.Population.Deserialize(backward.Serialize())
            self.assertEqual(forward.Serialize(), backward.Serialize())
            forward.Epoch()
            backward.Epoch()

    def test_archive_reevaluation_does_not_duplicate_candidates(self):
        pop = population(qd=True)
        archive_requests = 0
        while requests := pop.GetEvaluationRequests():
            for request in requests:
                archive_requests += int(request.archive_evaluation)
                pop.SubmitEvaluation(request.genome_id, request.case_id, 1.0,
                                     False, False, [0.5])
        archive = pop.GetQDArchive()
        self.assertEqual(archive_requests, 1)
        self.assertEqual(len(archive), pop.NumGenomes())
        self.assertEqual(len({entry.genome.GetID() for entry in archive}), len(archive))
        self.assertEqual(sum(entry.reevaluations for entry in archive), 1)
        self.assertEqual(sum(entry.summary.sample_count for entry in archive), 5)

    def test_invalid_validation_evidence_can_be_retried(self):
        pop = population()
        while requests := pop.GetEvaluationRequests():
            for request in requests:
                pop.SubmitEvaluation(request.genome_id, request.case_id, 1.0)
        pop.ValidateCandidates(1)
        for phase in (neat.VALIDATION_EVALUATION, neat.CERTIFICATION_EVALUATION):
            requests = pop.GetEvaluationRequests()
            self.assertEqual(requests[0].phase, phase)
            before = pop.Serialize()
            with self.assertRaises(ValueError):
                pop.SubmitEvaluation(requests[0].genome_id, requests[0].case_id,
                                     1.0, True, True, [math.nan])
            self.assertEqual(pop.Serialize(), before)
            for request in requests:
                pop.SubmitEvaluation(request.genome_id, request.case_id, 1.0, True, True)
        self.assertTrue(pop.GetValidatedChampion().validated)


if __name__ == "__main__":
    unittest.main()
