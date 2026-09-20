"""Resource-bounded HTTP adapters for Phase 19 regime analytics."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.config import Settings, get_settings
from backend.app.models.regime import (
    HMMDecodeRequest,
    HMMDecodeResult,
    HMMFilterRequest,
    HMMFilterResult,
    HMMFitRequest,
    HMMFitResult,
    HMMForecastRequest,
    HMMForecastResult,
    HMMModelOrderRequest,
    HMMModelOrderResult,
    MarkovDiagnosticsRequest,
    MarkovDiagnosticsResult,
    RegimeRiskRequest,
    RegimeRiskResult,
)
from backend.app.regime.hmm import (
    decode_hmm,
    filter_step,
    fit_gaussian_hmm,
    forecast_hmm,
    model_order_comparison,
)
from backend.app.regime.markov import diagnostics_with_forecasts
from backend.app.regime.risk import regime_conditioned_risk

router = APIRouter(prefix="/api/regime", tags=["regime-hmm"])


def _reject(message: str) -> None:
    raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, message)


def _enforce_dataset(request: HMMFitRequest | HMMDecodeRequest, settings: Settings) -> None:
    dataset = request.dataset
    if dataset.observation_count > settings.max_hmm_observations:
        _reject("HMM observation count exceeds API limit")
    if dataset.feature_dimension > settings.max_hmm_features:
        _reject("HMM feature count exceeds API limit")
    states = (
        request.config.num_states
        if isinstance(request, HMMFitRequest)
        else request.parameters.num_states
    )
    if states > settings.max_hmm_states:
        _reject("HMM state count exceeds API limit")
    if dataset.observation_count * states > settings.max_hmm_observations * settings.max_hmm_states:
        _reject("HMM forward/backward allocation exceeds API limit")
    if (
        dataset.observation_count * states * states
        > settings.max_hmm_observations * settings.max_hmm_states * settings.max_hmm_states
    ):
        _reject("HMM transition work exceeds API limit")


def _enforce_fit(request: HMMFitRequest, settings: Settings) -> None:
    _enforce_dataset(request, settings)
    if request.config.max_iterations > settings.max_hmm_iterations:
        _reject("HMM iteration count exceeds API limit")
    if request.config.num_starts > settings.max_hmm_starts:
        _reject("HMM start count exceeds API limit")


@router.post("/hmm/fit", response_model=HMMFitResult)
def fit_hmm(
    request: HMMFitRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HMMFitResult:
    _enforce_fit(request, settings)
    return fit_gaussian_hmm(
        request.dataset,
        request.config,
        max_probability_points=settings.max_hmm_probability_output_points,
    )


@router.post("/hmm/decode", response_model=HMMDecodeResult)
def decode(
    request: HMMDecodeRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HMMDecodeResult:
    _enforce_dataset(request, settings)
    return decode_hmm(request, max_probability_points=settings.max_hmm_probability_output_points)


@router.post("/hmm/forecast", response_model=HMMForecastResult)
def forecast(
    request: HMMForecastRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HMMForecastResult:
    if request.parameters.num_states > settings.max_hmm_states:
        _reject("HMM state count exceeds API limit")
    if request.parameters.feature_dimension > settings.max_hmm_features:
        _reject("HMM feature count exceeds API limit")
    return forecast_hmm(request.parameters, request.current_probabilities, request.horizons)


@router.post("/hmm/filter", response_model=HMMFilterResult)
def one_step_filter(
    request: HMMFilterRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HMMFilterResult:
    if request.parameters.num_states > settings.max_hmm_states:
        _reject("HMM state count exceeds API limit")
    if request.parameters.feature_dimension > settings.max_hmm_features:
        _reject("HMM feature count exceeds API limit")
    return filter_step(request.parameters, request.prior_probabilities, request.observation)


@router.post("/hmm/model-order", response_model=HMMModelOrderResult)
def model_order(
    request: HMMModelOrderRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HMMModelOrderResult:
    fit_request = HMMFitRequest(dataset=request.dataset, config=request.config)
    _enforce_fit(fit_request, settings)
    if len(request.state_counts) > settings.max_hmm_model_order_candidates:
        _reject("HMM model-order candidate count exceeds API limit")
    if max(request.state_counts) > settings.max_hmm_states:
        _reject("HMM model-order state count exceeds API limit")
    return model_order_comparison(request)


@router.post("/risk", response_model=RegimeRiskResult)
def risk(
    request: RegimeRiskRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> RegimeRiskResult:
    if request.parameters.num_states > settings.max_hmm_states:
        _reject("HMM state count exceeds API limit")
    if request.parameters.feature_dimension > settings.max_hmm_features:
        _reject("HMM feature count exceeds API limit")
    if request.num_simulations > settings.max_regime_risk_simulations:
        _reject("regime risk simulation count exceeds API limit")
    return regime_conditioned_risk(request)


@router.post("/markov/diagnostics", response_model=MarkovDiagnosticsResult)
def markov(
    request: MarkovDiagnosticsRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> MarkovDiagnosticsResult:
    if request.chain.num_states > settings.max_hmm_states:
        _reject("Markov state count exceeds API limit")
    return diagnostics_with_forecasts(
        request.chain, request.horizons, request.current_probabilities
    )
