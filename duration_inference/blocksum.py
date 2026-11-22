from decimal import Decimal, localcontext, MAX_EMAX, MIN_EMIN
import math
from duration_inference.evidence import (EvidenceValue, ZeroEvidenceError,
                                         InferenceNumericalError)
from duration_inference.initial import equilibrium_initial_mass
from duration_inference.segments import phase_space, outgoing_edges, emission_weight

SUMMARY_FORMAT = "shift-block/v1"
PRECISION = 70

def _widen(context):
    context.prec, context.Emax, context.Emin = PRECISION, MAX_EMAX, MIN_EMIN

def equilibrium_vector(model, phases):
    mass = equilibrium_initial_mass(model)
    return [Decimal(mass[phase].numerator) / Decimal(mass[phase].denominator)
            for phase in phases]

def _beat_summary(model, phases, symbol, initial):
    count = len(phases)
    dims = len(model.states) + 1
    index = {phase: i for i, phase in enumerate(phases)}
    kernel = [[Decimal(0)] * count for _ in range(count)]
    renewal = [[False] * count for _ in range(count)]
    if initial:
        for i, phase in enumerate(phases):
            kernel[i][i] = emission_weight(model, phase, symbol)
    else:
        for i, phase in enumerate(phases):
            for edge in outgoing_edges(model, phase):
                target = index[edge.target]
                kernel[i][target] += (edge.probability
                                      * emission_weight(model, edge.target, symbol))
                if edge.renewal:
                    renewal[i][target] = True
    first = [[[Decimal(0)] * dims for _ in range(count)] for _ in range(count)]
    second = [[[[Decimal(0)] * dims for _ in range(dims)]
               for _ in range(count)] for _ in range(count)]
    for i in range(count):
        for j in range(count):
            weight = kernel[i][j]
            if not weight:
                continue
            reward = [Decimal(0)] * dims
            reward[phases[j].state] = Decimal(1)
            if renewal[i][j]:
                reward[dims - 1] = Decimal(1)
            for a in range(dims):
                first[i][j][a] = weight * reward[a]
                for b in range(dims):
                    second[i][j][a][b] = weight * reward[a] * reward[b]
    return {"kernel": kernel, "first": first, "second": second}

def compose_summaries(left, right, dims):
    count = len(left["kernel"])
    kernel = [[Decimal(0)] * count for _ in range(count)]
    first = [[[Decimal(0)] * dims for _ in range(count)] for _ in range(count)]
    second = [[[[Decimal(0)] * dims for _ in range(dims)]
               for _ in range(count)] for _ in range(count)]
    for p in range(count):
        for r in range(count):
            left_kernel = left["kernel"][p][r]
            if not left_kernel:
                continue
            left_first = left["first"][p][r]
            left_second = left["second"][p][r]
            for q in range(count):
                right_kernel = right["kernel"][r][q]
                if not right_kernel:
                    continue
                right_first = right["first"][r][q]
                right_second = right["second"][r][q]
                kernel[p][q] += left_kernel * right_kernel
                for a in range(dims):
                    first[p][q][a] += (left_first[a] * right_kernel
                                       + left_kernel * right_first[a])
                    for b in range(dims):
                        second[p][q][a][b] += (
                            left_second[a][b] * right_kernel
                            + left_first[a] * right_first[b]
                            + left_first[b] * right_first[a]
                            + left_kernel * right_second[a][b])
    return {"kernel": kernel, "first": first, "second": second}

def block_summary(model, symbols, mode):
    symbols = tuple(symbols)
    if not symbols:
        raise ValueError("block requires at least one observation")
    if mode not in ("head", "tail"):
        raise ValueError("unknown block summary mode")
    phases = phase_space(model)
    dims = len(model.states) + 1
    with localcontext() as context:
        _widen(context)
        summary = _beat_summary(model, phases, symbols[0], mode == "head")
        for symbol in symbols[1:]:
            summary = compose_summaries(
                summary, _beat_summary(model, phases, symbol, False), dims)
    return phases, summary

def evidence_from_total(total):
    if not total.is_finite() or total < 0:
        raise InferenceNumericalError("invalid evidence total")
    if not total:
        raise ZeroEvidenceError("observations have zero evidence")
    shift = total.adjusted()
    mantissa = total.scaleb(-shift)
    log_evidence = float(mantissa.ln()) + shift * math.log(10)
    if not math.isfinite(log_evidence):
        raise InferenceNumericalError("nonfinite log evidence")
    probability = float(total)
    if probability == 0 or not math.isfinite(probability):
        probability = None
    return EvidenceValue(str(mantissa), shift, log_evidence, probability)

def summary_evidence(model, phases, summary):
    with localcontext() as context:
        _widen(context)
        initial = equilibrium_vector(model, phases)
        total = Decimal(0)
        for p, weight in enumerate(initial):
            if not weight:
                continue
            total += weight * sum(summary["kernel"][p], Decimal(0))
        return evidence_from_total(total)

def joint_shift_moments(model, phases, left, right):
    count = len(phases)
    dims = len(model.states) + 1
    total_dims = 2 * dims
    with localcontext() as context:
        _widen(context)
        initial = equilibrium_vector(model, phases)
        weights = [Decimal(0)] * count
        first = [[Decimal(0)] * dims for _ in range(count)]
        second = [[[Decimal(0)] * dims for _ in range(dims)] for _ in range(count)]
        for p in range(count):
            mass = initial[p]
            if not mass:
                continue
            for q in range(count):
                weight = left["kernel"][p][q]
                if not weight:
                    continue
                weights[q] += mass * weight
                for a in range(dims):
                    first[q][a] += mass * left["first"][p][q][a]
                    for b in range(dims):
                        second[q][a][b] += mass * left["second"][p][q][a][b]
        joint_weights = [Decimal(0)] * count
        joint_first = [[Decimal(0)] * total_dims for _ in range(count)]
        joint_second = [[[Decimal(0)] * total_dims for _ in range(total_dims)]
                        for _ in range(count)]
        for p in range(count):
            mass = weights[p]
            if not mass:
                continue
            for q in range(count):
                weight = right["kernel"][p][q]
                if not weight:
                    continue
                joint_weights[q] += mass * weight
                for a in range(dims):
                    joint_first[q][a] += first[p][a] * weight
                    joint_first[q][dims + a] += mass * right["first"][p][q][a]
                    for b in range(dims):
                        joint_second[q][a][b] += second[p][a][b] * weight
                        cross = first[p][a] * right["first"][p][q][b]
                        joint_second[q][a][dims + b] += cross
                        joint_second[q][dims + b][a] += cross
                        joint_second[q][dims + a][dims + b] += (
                            mass * right["second"][p][q][a][b])
        total = sum(joint_weights, Decimal(0))
        evidence = evidence_from_total(total)
        mean = [sum(joint_first[q][a] for q in range(count)) / total
                for a in range(total_dims)]
        raw = [[sum(joint_second[q][a][b] for q in range(count)) / total
                for b in range(total_dims)] for a in range(total_dims)]
        covariance = [[float(raw[a][b] - mean[a] * mean[b])
                       for b in range(total_dims)] for a in range(total_dims)]
        return evidence, [float(x) for x in mean], covariance
