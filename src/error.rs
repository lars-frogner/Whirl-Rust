//! Error handling.

use std::{borrow::Cow, error::Error, fmt};

#[derive(Clone, Debug)]
pub struct WhirlError {
    message: Cow<'static, str>,
}

pub type WhirlResult<T> = Result<T, WhirlError>;

impl WhirlError {
    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn into_string(self) -> String {
        self.message.to_owned().to_string()
    }

    pub fn from_error<E: fmt::Display>(front_message: &'static str, error: E) -> Self {
        Self {
            message: Cow::from(format!("{}{}", front_message, error)),
        }
    }
}

impl fmt::Display for WhirlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl Error for WhirlError {}

impl From<&'static str> for WhirlError {
    fn from(s: &'static str) -> Self {
        Self {
            message: Cow::from(s),
        }
    }
}

impl From<String> for WhirlError {
    fn from(s: String) -> Self {
        Self {
            message: Cow::from(s),
        }
    }
}
