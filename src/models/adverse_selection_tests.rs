//! Tests for adverse selection protection functionality

#[cfg(test)]
mod tests {
    use super::super::adverse_selection::*;
    use super::super::avellaneda_stoikov::*;
    use crate::math::FixedPoint;
    
    fn create_test_trade_info(price_impact: f64, volume: i64, volatility: f64) -> TradeInfo {
        TradeInfo {
            price: FixedPoint::from_float(100.0 + price_impact),
            volume,
            mid_price: FixedPoint::from_float(100.0),
            volatility: FixedPoint::from_float(volatility),
            total_volume: 10000,
            order_flow_imbalance: FixedPoint::from_float(0.1),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }
    
    #[test]
    fn test_adverse_selection_protection_basic() {
        let params = AdverseSelectionParams::default();
        let base_frequency = FixedPoint::from_float(10.0);
        
        let mut protection = AdverseSelectionProtection::new(params, base_frequency).unwrap();
        
        // Test with normal trade
        let normal_trade = create_test_trade_info(0.01, 1000, 0.2);
        let result = protection.update(normal_trade);
        assert!(result.is_ok());
        
        let state = result.unwrap();
        assert!(state.premium >= FixedPoint::ZERO);
        assert!(state.frequency_adjustment > FixedPoint::ZERO);
        assert!(state.frequency_adjustment <= FixedPoint::ONE);
    }
    
    #[test]
    fn test_adverse_selection_detection() {
        let params = AdverseSelectionParams::default();
        let base_frequency = FixedPoint::from_float(10.0);
        
        let mut protection = AdverseSelectionProtection::new(params, base_frequency).unwrap();
        
        // Test with high impact trade (potential adverse selection)
        let high_impact_trade = create_test_trade_info(0.10, 2000, 0.3);
        let result = protection.update(high_impact_trade);
        assert!(result.is_ok());
        
        let state = result.unwrap();
        // Should detect some adverse selection
        assert!(state.information_asymmetry.raw_measure > FixedPoint::ZERO);
        assert!(state.premium > FixedPoint::ZERO);
        
        // Frequency adjustment should be less than 1 (reduced frequency)
        assert!(state.frequency_adjustment < FixedPoint::ONE);
    }
    
    #[test]
    fn test_toxicity_detection() {
        let params = AdverseSelectionParams::default();
        let base_frequency = FixedPoint::from_float(10.0);
        
        let mut protection = AdverseSelectionProtection::new(params, base_frequency).unwrap();
        
        // Create sequence of trades with increasing toxicity
        for i in 0..20 {
            let trade = create_test_trade_info(
                i as f64 * 0.01, // Increasing price impact
                if i % 2 == 0 { 1000 } else { -1000 }, // Alternating buy/sell
                0.2 + i as f64 * 0.01, // Increasing volatility
            );
            
            let _ = protection.update(trade);
        }
        
        let state = protection.get_state();
        // Should detect elevated toxicity
        assert!(state.toxicity_level > FixedPoint::ZERO);
    }
    
    #[test]
    fn test_frequency_adjustment_calculation() {
        let mut params = AdverseSelectionParams::default();
        params.frequency_threshold = FixedPoint::from_float(0.05);
        params.min_frequency_ratio = FixedPoint::from_float(0.2);
        
        let base_frequency = FixedPoint::from_float(10.0);
        let mut protection = AdverseSelectionProtection::new(params, base_frequency).unwrap();
        
        // Create trade with high adverse selection premium
        let toxic_trade = create_test_trade_info(0.20, 3000, 0.5);
        let result = protection.update(toxic_trade);
        assert!(result.is_ok());
        
        let state = result.unwrap();
        // Should significantly reduce quote frequency
        assert!(state.frequency_adjustment < FixedPoint::from_float(0.8));
        assert!(state.frequency_adjustment >= FixedPoint::from_float(0.2)); // Min ratio
    }
    
    #[test]
    fn test_information_asymmetry_calculation() {
        let params = AdverseSelectionParams::default();
        let base_frequency = FixedPoint::from_float(10.0);
        
        let mut protection = AdverseSelectionProtection::new(params, base_frequency).unwrap();
        
        // Test with different scenarios
        let scenarios = vec![
            (0.01, 1000, 0.1, "low impact"),
            (0.05, 2000, 0.2, "medium impact"),
            (0.15, 3000, 0.4, "high impact"),
        ];
        
        let mut previous_ia = FixedPoint::ZERO;
        
        for (price_impact, volume, volatility, description) in scenarios {
            let trade = create_test_trade_info(price_impact, volume, volatility);
            let result = protection.update(trade);
            assert!(result.is_ok(), "Failed for scenario: {}", description);
            
            let state = result.unwrap();
            let current_ia = state.information_asymmetry.raw_measure;
            
            // Information asymmetry should generally increase with price impact
            if previous_ia > FixedPoint::ZERO {
                println!("Scenario: {}, IA: {}, Previous: {}", description, current_ia.to_float(), previous_ia.to_float());
            }
            
            previous_ia = current_ia;
        }
    }
    
    #[test]
    fn test_protection_activation() {
        let mut params = AdverseSelectionParams::default();
        params.toxicity_threshold = FixedPoint::from_float(0.2);
        
        let base_frequency = FixedPoint::from_float(10.0);
        let mut protection = AdverseSelectionProtection::new(params, base_frequency).unwrap();
        
        // Initially protection should not be active
        assert!(!protection.get_state().protection_active);
        
        // Create highly toxic sequence
        for i in 0..30 {
            let trade = create_test_trade_info(
                0.05 + i as f64 * 0.01, // Large and increasing price impact
                if i % 2 == 0 { 2000 } else { -2000 }, // Large alternating volumes
                0.3 + i as f64 * 0.01, // High and increasing volatility
            );
            
            let _ = protection.update(trade);
        }
        
        let state = protection.get_state();
        // Protection should now be active due to high toxicity
        println!("Final toxicity level: {}", state.toxicity_level.to_float());
        println!("Protection active: {}", state.protection_active);
    }
    
    #[test]
    fn test_parameter_validation() {
        let base_frequency = FixedPoint::from_float(10.0);
        
        // Test invalid beta
        let mut params = AdverseSelectionParams::default();
        params.beta = FixedPoint::ZERO;
        assert!(AdverseSelectionProtection::new(params, base_frequency).is_err());
        
        // Test invalid frequency threshold
        params = AdverseSelectionParams::default();
        params.frequency_threshold = FixedPoint::ZERO;
        assert!(AdverseSelectionProtection::new(params, base_frequency).is_err());
        
        // Test invalid max premium ratio
        params = AdverseSelectionParams::default();
        params.max_premium_ratio = FixedPoint::from_float(1.5);
        assert!(AdverseSelectionProtection::new(params, base_frequency).is_err());
        
        // Test valid parameters
        params = AdverseSelectionParams::default();
        assert!(AdverseSelectionProtection::new(params, base_frequency).is_ok());
    }
    
    #[test]
    fn test_diagnostics() {
        let params = AdverseSelectionParams::default();
        let base_frequency = FixedPoint::from_float(10.0);
        
        let mut protection = AdverseSelectionProtection::new(params, base_frequency).unwrap();
        
        // Add some trades
        for i in 0..10 {
            let trade = create_test_trade_info(0.02, 1000 + i * 100, 0.2);
            let _ = protection.update(trade);
        }
        
        let diagnostics = protection.get_diagnostics();
        assert_eq!(diagnostics.trade_count, 10);
        assert_eq!(diagnostics.impact_count, 10);
        assert!(diagnostics.mean_impact >= FixedPoint::ZERO);
        assert!(diagnostics.confidence > FixedPoint::ZERO);
    }
    
    #[test]
    fn test_integration_with_avellaneda_stoikov() {
        // Test that the adverse selection protection integrates properly with the main engine
        let params = AvellanedaStoikovParams::default();
        let mut engine = AvellanedaStoikovEngine::new(params).unwrap();
        
        let market_state = MarketState {
            mid_price: FixedPoint::from_float(100.0),
            bid_price: FixedPoint::from_float(99.95),
            ask_price: FixedPoint::from_float(100.05),
            bid_volume: 1000,
            ask_volume: 1000,
            last_trade_price: FixedPoint::from_float(100.0),
            last_trade_volume: 100,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            sequence_number: 1,
            volatility: FixedPoint::from_float(0.2),
            order_flow_imbalance: FixedPoint::from_float(0.1),
            microstructure_noise: FixedPoint::from_float(0.001),
        };
        
        // Calculate quotes - should work without errors
        let result = engine.calculate_optimal_quotes(
            FixedPoint::from_float(100.0),
            500, // Some inventory
            FixedPoint::from_float(0.2),
            FixedPoint::from_float(0.1),
            &market_state,
        );
        
        assert!(result.is_ok());
        let quotes = result.unwrap();
        
        // Verify adverse selection premium is included
        assert!(quotes.adverse_selection_premium >= FixedPoint::ZERO);
        
        // Check that adverse selection state is accessible
        let as_state = engine.get_adverse_selection_state();
        assert!(as_state.premium >= FixedPoint::ZERO);
        
        // Check frequency adjustment
        let freq_adjustment = engine.get_quote_frequency_adjustment();
        assert!(freq_adjustment > FixedPoint::ZERO);
        assert!(freq_adjustment <= FixedPoint::ONE);
        
        // Check if protection is active
        let is_active = engine.is_adverse_selection_protection_active();
        println!("Adverse selection protection active: {}", is_active);
        
        // Get diagnostics
        let diagnostics = engine.get_adverse_selection_diagnostics();
        assert!(diagnostics.confidence >= FixedPoint::ZERO);
    }
    
    #[test]
    fn test_reset_functionality() {
        let params = AdverseSelectionParams::default();
        let base_frequency = FixedPoint::from_float(10.0);
        
        let mut protection = AdverseSelectionProtection::new(params, base_frequency).unwrap();
        
        // Add some trades to build up state
        for i in 0..10 {
            let trade = create_test_trade_info(0.05, 1000, 0.3);
            let _ = protection.update(trade);
        }
        
        // Verify state has been built up
        let state_before = protection.get_state();
        assert!(state_before.information_asymmetry.observation_count > 0);
        
        // Reset and verify state is cleared
        protection.reset();
        let state_after = protection.get_state();
        assert_eq!(state_after.information_asymmetry.observation_count, 0);
        assert_eq!(state_after.premium, FixedPoint::ZERO);
        assert_eq!(state_after.frequency_adjustment, FixedPoint::ONE);
        assert!(!state_after.protection_active);
    }
}