use axum::{http::StatusCode, response::IntoResponse, response::Response, Json};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub(crate) struct ApiError {
    pub(crate) message: String,
    #[serde(rename = "type")]
    pub(crate) error_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) code: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ErrorResponse {
    pub(crate) error: ApiError,
}

impl ErrorResponse {
    pub(crate) fn bad_request(message: impl Into<String>, param: Option<String>) -> Self {
        Self {
            error: ApiError {
                message: message.into(),
                error_type: "invalid_request_error".to_string(),
                param,
                code: None,
            },
        }
    }

    pub(crate) fn internal_error(message: impl Into<String>) -> Self {
        Self {
            error: ApiError {
                message: message.into(),
                error_type: "server_error".to_string(),
                param: None,
                code: Some("internal_error".to_string()),
            },
        }
    }

    pub(crate) fn too_many_requests(message: impl Into<String>) -> Self {
        Self {
            error: ApiError {
                message: message.into(),
                error_type: "rate_limit_error".to_string(),
                param: None,
                code: Some("rate_limit_exceeded".to_string()),
            },
        }
    }

    pub(crate) fn service_unavailable(message: impl Into<String>) -> Self {
        Self {
            error: ApiError {
                message: message.into(),
                error_type: "service_unavailable".to_string(),
                param: None,
                code: Some("service_unavailable".to_string()),
            },
        }
    }
}

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> Response {
        let status = match self.error.error_type.as_str() {
            "invalid_request_error" => StatusCode::BAD_REQUEST,
            "rate_limit_error" => StatusCode::TOO_MANY_REQUESTS,
            "service_unavailable" => StatusCode::SERVICE_UNAVAILABLE,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(self)).into_response()
    }
}
