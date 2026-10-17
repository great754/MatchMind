"""Allowlisted provider failures. Never retain raw exceptions, HTTP bodies or headers."""
from enum import Enum
import json
import logging
from .base import ProviderError

class MicrosoftFailure(str,Enum):
    CONFIGURATION='configuration_error'
    REQUEST_INVALID='request_invalid'
    TLS='tls_failure'
    NETWORK='network_failure'
    TIMEOUT='timeout'
    BAD_REQUEST='http_400'
    UNAUTHORIZED='http_401'
    FORBIDDEN='http_403'
    NOT_FOUND='http_404'
    RATE_LIMITED='http_429'
    SERVER_ERROR='http_5xx'
    HTTP_ERROR='http_error'
    MALFORMED_RESPONSE='malformed_response'
    INVALID_TOOL_PLAN='invalid_tool_plan'
    INVALID_SELECTION='invalid_selection'
    INTERNAL='internal_error'

MESSAGES={
    MicrosoftFailure.CONFIGURATION:'Microsoft provider configuration is missing or invalid.',
    MicrosoftFailure.REQUEST_INVALID:'Microsoft request exceeds its bounds or cannot be serialized.',
    MicrosoftFailure.TLS:'Microsoft HTTPS certificate verification failed.',
    MicrosoftFailure.NETWORK:'Microsoft network connection failed.',
    MicrosoftFailure.TIMEOUT:'Microsoft request timed out.',
    MicrosoftFailure.BAD_REQUEST:'Microsoft rejected the request (HTTP 400). Check model/API compatibility.',
    MicrosoftFailure.UNAUTHORIZED:'Microsoft authentication failed (HTTP 401).',
    MicrosoftFailure.FORBIDDEN:'Microsoft access was denied (HTTP 403).',
    MicrosoftFailure.NOT_FOUND:'Microsoft endpoint or deployment was not found (HTTP 404).',
    MicrosoftFailure.RATE_LIMITED:'Microsoft rate limit was reached (HTTP 429).',
    MicrosoftFailure.SERVER_ERROR:'Microsoft service failed (HTTP 5xx).',
    MicrosoftFailure.HTTP_ERROR:'Microsoft returned an unexpected HTTP status.',
    MicrosoftFailure.MALFORMED_RESPONSE:'Microsoft returned a malformed or unexpected response.',
    MicrosoftFailure.INVALID_TOOL_PLAN:'Microsoft returned an invalid tool plan.',
    MicrosoftFailure.INVALID_SELECTION:'Microsoft returned an invalid grounded selection.',
    MicrosoftFailure.INTERNAL:'Microsoft provider failed unexpectedly.'}

class MicrosoftProviderError(ProviderError):
    def __init__(self,code,stage='request',http_status=None):
        self.code=MicrosoftFailure(code)
        self.stage=stage if stage in ('configuration','request','tool_plan','selection','commentary') else 'request'
        self.http_status=http_status if type(http_status) is int and 100<=http_status<=599 else None
        super().__init__(MESSAGES[self.code])
    def diagnostic(self):
        return dict(provider='microsoft_foundry',code=self.code.value,stage=self.stage,http_status=self.http_status)

def failure(code,stage='request',http_status=None):
    error=MicrosoftProviderError(code,stage,http_status)
    logging.getLogger('tactiqai.providers').warning(json.dumps(dict(event='provider_failure',**error.diagnostic())))
    return error
