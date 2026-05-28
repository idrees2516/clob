use std::hint;

/// Branch prediction hints for hot path optimization
pub struct BranchHints;

impl BranchHints {
    /// Hint that the condition is likely to be true
    #[inline(always)]
    pub fn likely(condition: bool) -> bool {
        if condition {
            hint::black_box(true)
        } else {
            hint::black_box(false)
        }
    }

    /// Hint that the condition is unlikely to be true
    #[inline(always)]
    pub fn unlikely(condition: bool) -> bool {
        if !condition {
            hint::black_box(false)
        } else {
            hint::black_box(true)
        }
    }

    /// Cold path marker for rarely executed code
    #[cold]
    #[inline(never)]
    pub fn cold_path<T, F: FnOnce() -> T>(f: F) -> T {
        f()
    }

    /// Hot path marker for frequently executed code
    #[inline(always)]
    pub fn hot_path<T, F: FnOnce() -> T>(f: F) -> T {
        f()
    }
}

/// Macro for likely branch prediction
#[macro_export]
macro_rules! likely {
    ($cond:expr) => {
        $crate::performance::branch_prediction::BranchHints::likely($cond)
    };
}

/// Macro for unlikely branch prediction
#[macro_export]
macro_rules! unlikely {
    ($cond:expr) => {
        $crate::performance::branch_prediction::BranchHints::unlikely($cond)
    };
}

/// Optimized conditional execution with branch hints
pub struct ConditionalExecution;

impl ConditionalExecution {
    /// Execute function if condition is likely true
    #[inline(always)]
    pub fn if_likely<T, F>(condition: bool, f: F) -> Option<T>
    where
        F: FnOnce() -> T,
    {
        if likely!(condition) {
            Some(f())
        } else {
            None
        }
    }

    /// Execute function if condition is unlikely true
    #[inline(always)]
    pub fn if_unlikely<T, F>(condition: bool, f: F) -> Option<T>
    where
        F: FnOnce() -> T,
    {
        if unlikely!(condition) {
            Some(f())
        } else {
            None
        }
    }

    /// Execute one of two functions based on condition with hints
    #[inline(always)]
    pub fn if_else_likely<T, F1, F2>(condition: bool, if_true: F1, if_false: F2) -> T
    where
        F1: FnOnce() -> T,
        F2: FnOnce() -> T,
    {
        if likely!(condition) {
            if_true()
        } else {
            if_false()
        }
    }

    /// Execute one of two functions based on condition with unlikely hint
    #[inline(always)]
    pub fn if_else_unlikely<T, F1, F2>(condition: bool, if_true: F1, if_false: F2) -> T
    where
        F1: FnOnce() -> T,
        F2: FnOnce() -> T,
    {
        if unlikely!(condition) {
            if_true()
        } else {
            if_false()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branch_hints() {
        assert!(likely!(true));
        assert!(!unlikely!(true));
        assert!(!likely!(false));
        assert!(unlikely!(false));
    }

    #[test]
    fn test_conditional_execution() {
        let result = ConditionalExecution::if_likely(true, || 42);
        assert_eq!(result, Some(42));

        let result = ConditionalExecution::if_likely(false, || 42);
        assert_eq!(result, None);

        let result = ConditionalExecution::if_else_likely(true, || 1, || 2);
        assert_eq!(result, 1);

        let result = ConditionalExecution::if_else_likely(false, || 1, || 2);
        assert_eq!(result, 2);
    }
}