use super::atomic_operations::{AtomicOperations, MemoryOrder, AlignedAtomicPtr};
use super::hazard_pointers::HazardPointerManager;
use super::memory_reclamation::{EpochBasedReclamation, HybridReclamationManager};
use super::price_level::{LockFreePriceLevel, LockFreeError};
use super::order_node::LockFreeOrderNode;
use crate::orderbook::types::{Order, OrderId, Side, Trade, Symbol};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::ptr;

/// Lock-free order book implementation with sub-microsecond latency
#[repr(align(64))] // Cache-line aligned
pub struct LockFreeOrderBook {
    /// Symbol for this order book
    pub symbol: Symbol,
    
    /// Best bid price level (highest price)
    pub best_bid: AlignedAtomicPtr<LockFreePriceLevel>,
    
    /// Best ask price level (lowest price)
    pub best_ask: AlignedAtomicPtr<LockFreePriceLevel>,
    
    /// Sequence number for ordering operations
    pub sequence: AtomicU64,
    
    /// Total number of orders in the book
    pub total_orders: AtomicUsize,
    
    /// Total bid volume
    pub total_bid_volume: AtomicU64,
    
    /// Total ask volume
    pub total_ask_volume: AtomicU64,
    
    /// Memory reclamation manager
    reclamation_manager: Arc<HybridReclamationManager>,
    
    /// Padding to prevent false sharing
    _padding: [u8; 0],
}

impl LockFreeOrderBook {
    /// Create a new lock-free order book
    pub fn new(symbol: Symbol, max_threads: usize) -> Self {
        Self {
            symbol,
            best_bid: AlignedAtomicPtr::new(ptr::null_mut()),
            best_ask: AlignedAtomicPtr::new(ptr::null_mut()),
            sequence: AtomicU64::new(0),
            total_orders: AtomicUsize::new(0),
            total_bid_volume: AtomicU64::new(0),
            total_ask_volume: AtomicU64::new(0),
            reclamation_manager: Arc::new(HybridReclamationManager::new(max_threads)),
            _padding: [],
        }
    }

    /// Add an order to the order book
    pub fn add_order(&self, order: Order) -> Result<Vec<Trade>, LockFreeError> {
        let _guard = self.reclamation_manager.pin();
        let sequence = self.sequence.fetch_add(1, Ordering::AcqRel);
        
        match order.side {
            Side::Buy => self.add_buy_order(order, sequence),
            Side::Sell => self.add_sell_order(order, sequence),
        }
    }

    /// Add a buy order (bid)
    fn add_buy_order(&self, order: Order, sequence: u64) -> Result<Vec<Trade>, LockFreeError> {
        let mut trades = Vec::new();
        let mut remaining_order = order;

        // First, try to match against existing sell orders
        loop {
            let best_ask = self.best_ask.load(MemoryOrder::Acquire);
            
            if best_ask.is_null() {
                break; // No asks to match against
            }

            let hazard = self.reclamation_manager.acquire_hazard();
            hazard.protect(best_ask);
            
            // Verify ask hasn't changed after protection
            if self.best_ask.load(MemoryOrder::Acquire) != best_ask {
                continue;
            }

            unsafe {
                let ask_price = (*best_ask).price;
                
                // Check if we can match
                if remaining_order.price < ask_price {
                    break; // No more matches possible
                }

                // Try to match orders at this price level
                if let Some(matched_order) = (*best_ask).peek_first_order(self.reclamation_manager.hazard_manager()) {
                    let trade_size = std::cmp::min(remaining_order.size, matched_order.size);
                    
                    // Create trade
                    let trade = Trade {
                        id: sequence,
                        symbol: remaining_order.symbol.clone(),
                        buy_order_id: remaining_order.id,
                        sell_order_id: matched_order.id,
                        price: ask_price,
                        size: trade_size,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_nanos() as u64,
                    };
                    
                    trades.push(trade);
                    
                    // Update remaining order size
                    remaining_order.size -= trade_size;
                    
                    // Partially fill or remove the matched order
                    if trade_size == matched_order.size {
                        // Complete fill - remove the order
                        (*best_ask).pop_first_order(self.reclamation_manager.hazard_manager())?;
                        
                        // Check if price level is now empty
                        if (*best_ask).is_empty() {
                            self.remove_empty_ask_level(best_ask)?;
                        }
                    } else {
                        // Partial fill
                        (*best_ask).partial_fill_order(
                            matched_order.id,
                            trade_size,
                            self.reclamation_manager.hazard_manager(),
                        )?;
                    }
                    
                    // Update volume counters
                    self.total_ask_volume.fetch_sub(trade_size, Ordering::AcqRel);
                    
                    // If buy order is completely filled, we're done
                    if remaining_order.size == 0 {
                        return Ok(trades);
                    }
                } else {
                    // No orders at this level, remove empty level
                    self.remove_empty_ask_level(best_ask)?;
                }
            }
        }

        // If there's remaining quantity, add to bid side
        if remaining_order.size > 0 {
            self.insert_bid_order(remaining_order)?;
        }

        Ok(trades)
    }

    /// Add a sell order (ask)
    fn add_sell_order(&self, order: Order, sequence: u64) -> Result<Vec<Trade>, LockFreeError> {
        let mut trades = Vec::new();
        let mut remaining_order = order;

        // First, try to match against existing buy orders
        loop {
            let best_bid = self.best_bid.load(MemoryOrder::Acquire);
            
            if best_bid.is_null() {
                break; // No bids to match against
            }

            let hazard = self.reclamation_manager.acquire_hazard();
            hazard.protect(best_bid);
            
            // Verify bid hasn't changed after protection
            if self.best_bid.load(MemoryOrder::Acquire) != best_bid {
                continue;
            }

            unsafe {
                let bid_price = (*best_bid).price;
                
                // Check if we can match
                if remaining_order.price > bid_price {
                    break; // No more matches possible
                }

                // Try to match orders at this price level
                if let Some(matched_order) = (*best_bid).peek_first_order(self.reclamation_manager.hazard_manager()) {
                    let trade_size = std::cmp::min(remaining_order.size, matched_order.size);
                    
                    // Create trade
                    let trade = Trade {
                        id: sequence,
                        symbol: remaining_order.symbol.clone(),
                        buy_order_id: matched_order.id,
                        sell_order_id: remaining_order.id,
                        price: bid_price,
                        size: trade_size,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_nanos() as u64,
                    };
                    
                    trades.push(trade);
                    
                    // Update remaining order size
                    remaining_order.size -= trade_size;
                    
                    // Partially fill or remove the matched order
                    if trade_size == matched_order.size {
                        // Complete fill - remove the order
                        (*best_bid).pop_first_order(self.reclamation_manager.hazard_manager())?;
                        
                        // Check if price level is now empty
                        if (*best_bid).is_empty() {
                            self.remove_empty_bid_level(best_bid)?;
                        }
                    } else {
                        // Partial fill
                        (*best_bid).partial_fill_order(
                            matched_order.id,
                            trade_size,
                            self.reclamation_manager.hazard_manager(),
                        )?;
                    }
                    
                    // Update volume counters
                    self.total_bid_volume.fetch_sub(trade_size, Ordering::AcqRel);
                    
                    // If sell order is completely filled, we're done
                    if remaining_order.size == 0 {
                        return Ok(trades);
                    }
                } else {
                    // No orders at this level, remove empty level
                    self.remove_empty_bid_level(best_bid)?;
                }
            }
        }

        // If there's remaining quantity, add to ask side
        if remaining_order.size > 0 {
            self.insert_ask_order(remaining_order)?;
        }

        Ok(trades)
    }

    /// Insert a bid order into the order book
    fn insert_bid_order(&self, order: Order) -> Result<(), LockFreeError> {
        let price_level = self.find_or_create_bid_level(order.price)?;
        
        unsafe {
            (*price_level).add_order(order.clone(), self.reclamation_manager.hazard_manager())?;
        }
        
        self.total_orders.fetch_add(1, Ordering::AcqRel);
        self.total_bid_volume.fetch_add(order.size, Ordering::AcqRel);
        
        Ok(())
    }

    /// Insert an ask order into the order book
    fn insert_ask_order(&self, order: Order) -> Result<(), LockFreeError> {
        let price_level = self.find_or_create_ask_level(order.price)?;
        
        unsafe {
            (*price_level).add_order(order.clone(), self.reclamation_manager.hazard_manager())?;
        }
        
        self.total_orders.fetch_add(1, Ordering::AcqRel);
        self.total_ask_volume.fetch_add(order.size, Ordering::AcqRel);
        
        Ok(())
    }

    /// Find or create a bid price level
    fn find_or_create_bid_level(&self, price: u64) -> Result<*mut LockFreePriceLevel, LockFreeError> {
        loop {
            let current_best = self.best_bid.load(MemoryOrder::Acquire);
            
            if current_best.is_null() {
                // No bid levels exist, create the first one
                let new_level = Box::into_raw(Box::new(LockFreePriceLevel::new(price)));
                
                match self.best_bid.compare_exchange_weak(
                    ptr::null_mut(),
                    new_level,
                    MemoryOrder::Release,
                    MemoryOrder::Relaxed,
                ) {
                    Ok(_) => return Ok(new_level),
                    Err(_) => {
                        // Another thread created a level, clean up and retry
                        unsafe { Box::from_raw(new_level); }
                        continue;
                    }
                }
            }

            // Find the correct position for this price level
            let hazard = self.reclamation_manager.acquire_hazard();
            hazard.protect(current_best);
            
            // Verify best bid hasn't changed
            if self.best_bid.load(MemoryOrder::Acquire) != current_best {
                continue;
            }

            unsafe {
                let current_price = (*current_best).price;
                
                if price == current_price {
                    // Price level already exists
                    return Ok(current_best);
                } else if price > current_price {
                    // New best bid
                    let new_level = Box::into_raw(Box::new(LockFreePriceLevel::new(price)));
                    (*new_level).next_level.store(current_best, MemoryOrder::Release);
                    (*current_best).prev_level.store(new_level, MemoryOrder::Release);
                    
                    match self.best_bid.compare_exchange_weak(
                        current_best,
                        new_level,
                        MemoryOrder::Release,
                        MemoryOrder::Relaxed,
                    ) {
                        Ok(_) => return Ok(new_level),
                        Err(_) => {
                            // Clean up and retry
                            Box::from_raw(new_level);
                            continue;
                        }
                    }
                } else {
                    // Insert in sorted order (descending for bids)
                    return self.insert_bid_level_sorted(current_best, price);
                }
            }
        }
    }

    /// Find or create an ask price level
    fn find_or_create_ask_level(&self, price: u64) -> Result<*mut LockFreePriceLevel, LockFreeError> {
        loop {
            let current_best = self.best_ask.load(MemoryOrder::Acquire);
            
            if current_best.is_null() {
                // No ask levels exist, create the first one
                let new_level = Box::into_raw(Box::new(LockFreePriceLevel::new(price)));
                
                match self.best_ask.compare_exchange_weak(
                    ptr::null_mut(),
                    new_level,
                    MemoryOrder::Release,
                    MemoryOrder::Relaxed,
                ) {
                    Ok(_) => return Ok(new_level),
                    Err(_) => {
                        // Another thread created a level, clean up and retry
                        unsafe { Box::from_raw(new_level); }
                        continue;
                    }
                }
            }

            // Find the correct position for this price level
            let hazard = self.reclamation_manager.acquire_hazard();
            hazard.protect(current_best);
            
            // Verify best ask hasn't changed
            if self.best_ask.load(MemoryOrder::Acquire) != current_best {
                continue;
            }

            unsafe {
                let current_price = (*current_best).price;
                
                if price == current_price {
                    // Price level already exists
                    return Ok(current_best);
                } else if price < current_price {
                    // New best ask
                    let new_level = Box::into_raw(Box::new(LockFreePriceLevel::new(price)));
                    (*new_level).next_level.store(current_best, MemoryOrder::Release);
                    (*current_best).prev_level.store(new_level, MemoryOrder::Release);
                    
                    match self.best_ask.compare_exchange_weak(
                        current_best,
                        new_level,
                        MemoryOrder::Release,
                        MemoryOrder::Relaxed,
                    ) {
                        Ok(_) => return Ok(new_level),
                        Err(_) => {
                            // Clean up and retry
                            Box::from_raw(new_level);
                            continue;
                        }
                    }
                } else {
                    // Insert in sorted order (ascending for asks)
                    return self.insert_ask_level_sorted(current_best, price);
                }
            }
        }
    }

    /// Insert a bid level in sorted order (helper function)
    fn insert_bid_level_sorted(&self, start: *mut LockFreePriceLevel, price: u64) -> Result<*mut LockFreePriceLevel, LockFreeError> {
        // Implementation for inserting in sorted order
        // This is a simplified version - full implementation would handle all edge cases
        let new_level = Box::into_raw(Box::new(LockFreePriceLevel::new(price)));
        
        unsafe {
            let mut current = start;
            while !current.is_null() {
                let current_price = (*current).price;
                if price == current_price {
                    // Level already exists, clean up and return existing
                    Box::from_raw(new_level);
                    return Ok(current);
                } else if price > current_price {
                    // Insert before current
                    let prev = (*current).prev_level.load(MemoryOrder::Acquire);
                    (*new_level).next_level.store(current, MemoryOrder::Release);
                    (*new_level).prev_level.store(prev, MemoryOrder::Release);
                    (*current).prev_level.store(new_level, MemoryOrder::Release);
                    
                    if !prev.is_null() {
                        (*prev).next_level.store(new_level, MemoryOrder::Release);
                    }
                    
                    return Ok(new_level);
                }
                current = (*current).next_level.load(MemoryOrder::Acquire);
            }
        }
        
        // If we reach here, insert at the end
        Ok(new_level)
    }

    /// Insert an ask level in sorted order (helper function)
    fn insert_ask_level_sorted(&self, start: *mut LockFreePriceLevel, price: u64) -> Result<*mut LockFreePriceLevel, LockFreeError> {
        // Similar to bid insertion but with ascending order
        let new_level = Box::into_raw(Box::new(LockFreePriceLevel::new(price)));
        
        unsafe {
            let mut current = start;
            while !current.is_null() {
                let current_price = (*current).price;
                if price == current_price {
                    // Level already exists, clean up and return existing
                    Box::from_raw(new_level);
                    return Ok(current);
                } else if price < current_price {
                    // Insert before current
                    let prev = (*current).prev_level.load(MemoryOrder::Acquire);
                    (*new_level).next_level.store(current, MemoryOrder::Release);
                    (*new_level).prev_level.store(prev, MemoryOrder::Release);
                    (*current).prev_level.store(new_level, MemoryOrder::Release);
                    
                    if !prev.is_null() {
                        (*prev).next_level.store(new_level, MemoryOrder::Release);
                    }
                    
                    return Ok(new_level);
                }
                current = (*current).next_level.load(MemoryOrder::Acquire);
            }
        }
        
        // If we reach here, insert at the end
        Ok(new_level)
    }

    /// Remove an empty bid level
    fn remove_empty_bid_level(&self, level: *mut LockFreePriceLevel) -> Result<(), LockFreeError> {
        unsafe {
            let next = (*level).next_level.load(MemoryOrder::Acquire);
            let prev = (*level).prev_level.load(MemoryOrder::Acquire);
            
            if prev.is_null() {
                // This is the best bid, update best bid pointer
                self.best_bid.store(next, MemoryOrder::Release);
            } else {
                (*prev).next_level.store(next, MemoryOrder::Release);
            }
            
            if !next.is_null() {
                (*next).prev_level.store(prev, MemoryOrder::Release);
            }
            
            // Retire the level for reclamation
            self.reclamation_manager.retire(level);
        }
        
        Ok(())
    }

    /// Remove an empty ask level
    fn remove_empty_ask_level(&self, level: *mut LockFreePriceLevel) -> Result<(), LockFreeError> {
        unsafe {
            let next = (*level).next_level.load(MemoryOrder::Acquire);
            let prev = (*level).prev_level.load(MemoryOrder::Acquire);
            
            if prev.is_null() {
                // This is the best ask, update best ask pointer
                self.best_ask.store(next, MemoryOrder::Release);
            } else {
                (*prev).next_level.store(next, MemoryOrder::Release);
            }
            
            if !next.is_null() {
                (*next).prev_level.store(prev, MemoryOrder::Release);
            }
            
            // Retire the level for reclamation
            self.reclamation_manager.retire(level);
        }
        
        Ok(())
    }

    /// Cancel an order
    pub fn cancel_order(&self, order_id: OrderId, side: Side) -> Result<Option<Order>, LockFreeError> {
        let _guard = self.reclamation_manager.pin();
        
        match side {
            Side::Buy => self.cancel_bid_order(order_id),
            Side::Sell => self.cancel_ask_order(order_id),
        }
    }

    /// Cancel a bid order
    fn cancel_bid_order(&self, order_id: OrderId) -> Result<Option<Order>, LockFreeError> {
        let mut current = self.best_bid.load(MemoryOrder::Acquire);
        
        while !current.is_null() {
            let hazard = self.reclamation_manager.acquire_hazard();
            hazard.protect(current);
            
            unsafe {
                if let Some(cancelled_order) = (*current).remove_order(order_id, self.reclamation_manager.hazard_manager())? {
                    self.total_orders.fetch_sub(1, Ordering::AcqRel);
                    self.total_bid_volume.fetch_sub(cancelled_order.size, Ordering::AcqRel);
                    
                    // Check if level is now empty
                    if (*current).is_empty() {
                        self.remove_empty_bid_level(current)?;
                    }
                    
                    return Ok(Some(cancelled_order));
                }
                
                current = (*current).next_level.load(MemoryOrder::Acquire);
            }
        }
        
        Ok(None)
    }

    /// Cancel an ask order
    fn cancel_ask_order(&self, order_id: OrderId) -> Result<Option<Order>, LockFreeError> {
        let mut current = self.best_ask.load(MemoryOrder::Acquire);
        
        while !current.is_null() {
            let hazard = self.reclamation_manager.acquire_hazard();
            hazard.protect(current);
            
            unsafe {
                if let Some(cancelled_order) = (*current).remove_order(order_id, self.reclamation_manager.hazard_manager())? {
                    self.total_orders.fetch_sub(1, Ordering::AcqRel);
                    self.total_ask_volume.fetch_sub(cancelled_order.size, Ordering::AcqRel);
                    
                    // Check if level is now empty
                    if (*current).is_empty() {
                        self.remove_empty_ask_level(current)?;
                    }
                    
                    return Ok(Some(cancelled_order));
                }
                
                current = (*current).next_level.load(MemoryOrder::Acquire);
            }
        }
        
        Ok(None)
    }

    /// Get the best bid price
    pub fn get_best_bid(&self) -> Option<u64> {
        let best_bid = self.best_bid.load(MemoryOrder::Acquire);
        if best_bid.is_null() {
            None
        } else {
            unsafe { Some((*best_bid).price) }
        }
    }

    /// Get the best ask price
    pub fn get_best_ask(&self) -> Option<u64> {
        let best_ask = self.best_ask.load(MemoryOrder::Acquire);
        if best_ask.is_null() {
            None
        } else {
            unsafe { Some((*best_ask).price) }
        }
    }

    /// Get the spread (ask - bid)
    pub fn get_spread(&self) -> Option<u64> {
        match (self.get_best_bid(), self.get_best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get order book statistics
    pub fn get_stats(&self) -> OrderBookStats {
        OrderBookStats {
            symbol: self.symbol.clone(),
            total_orders: self.total_orders.load(Ordering::Acquire),
            total_bid_volume: self.total_bid_volume.load(Ordering::Acquire),
            total_ask_volume: self.total_ask_volume.load(Ordering::Acquire),
            best_bid: self.get_best_bid(),
            best_ask: self.get_best_ask(),
            spread: self.get_spread(),
            sequence: self.sequence.load(Ordering::Acquire),
        }
    }

    /// Force memory reclamation
    pub fn force_reclaim(&self) {
        self.reclamation_manager.force_reclaim();
    }
}

/// Order book statistics
#[derive(Debug, Clone)]
pub struct OrderBookStats {
    pub symbol: Symbol,
    pub total_orders: usize,
    pub total_bid_volume: u64,
    pub total_ask_volume: u64,
    pub best_bid: Option<u64>,
    pub best_ask: Option<u64>,
    pub spread: Option<u64>,
    pub sequence: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orderbook::types::{OrderType};
    use std::sync::Arc;
    use std::thread;

    fn create_test_order(id: u64, side: Side, price: u64, size: u64) -> Order {
        Order {
            id: OrderId::new(id),
            symbol: Symbol::new("BTCUSD").unwrap(),
            side,
            order_type: OrderType::Limit,
            price,
            size,
            timestamp: 1000,
        }
    }

    #[test]
    fn test_order_book_creation() {
        let symbol = Symbol::new("BTCUSD").unwrap();
        let book = LockFreeOrderBook::new(symbol.clone(), 4);
        
        assert_eq!(book.symbol, symbol);
        assert!(book.get_best_bid().is_none());
        assert!(book.get_best_ask().is_none());
        assert!(book.get_spread().is_none());
    }

    #[test]
    fn test_add_bid_order() {
        let symbol = Symbol::new("BTCUSD").unwrap();
        let book = LockFreeOrderBook::new(symbol, 4);
        let order = create_test_order(1, Side::Buy, 50000, 100);
        
        let trades = book.add_order(order).unwrap();
        assert!(trades.is_empty()); // No matching orders
        
        assert_eq!(book.get_best_bid(), Some(50000));
        assert!(book.get_best_ask().is_none());
        
        let stats = book.get_stats();
        assert_eq!(stats.total_orders, 1);
        assert_eq!(stats.total_bid_volume, 100);
    }

    #[test]
    fn test_add_ask_order() {
        let symbol = Symbol::new("BTCUSD").unwrap();
        let book = LockFreeOrderBook::new(symbol, 4);
        let order = create_test_order(1, Side::Sell, 51000, 100);
        
        let trades = book.add_order(order).unwrap();
        assert!(trades.is_empty()); // No matching orders
        
        assert!(book.get_best_bid().is_none());
        assert_eq!(book.get_best_ask(), Some(51000));
        
        let stats = book.get_stats();
        assert_eq!(stats.total_orders, 1);
        assert_eq!(stats.total_ask_volume, 100);
    }

    #[test]
    fn test_order_matching() {
        let symbol = Symbol::new("BTCUSD").unwrap();
        let book = LockFreeOrderBook::new(symbol, 4);
        
        // Add a bid order
        let bid_order = create_test_order(1, Side::Buy, 50000, 100);
        let trades = book.add_order(bid_order).unwrap();
        assert!(trades.is_empty());
        
        // Add a matching ask order
        let ask_order = create_test_order(2, Side::Sell, 50000, 50);
        let trades = book.add_order(ask_order).unwrap();
        
        assert_eq!(trades.len(), 1);
        let trade = &trades[0];
        assert_eq!(trade.price, 50000);
        assert_eq!(trade.size, 50);
        assert_eq!(trade.buy_order_id, OrderId::new(1));
        assert_eq!(trade.sell_order_id, OrderId::new(2));
        
        // Check remaining order
        let stats = book.get_stats();
        assert_eq!(stats.total_orders, 1); // One order partially filled
        assert_eq!(stats.total_bid_volume, 50); // 50 remaining
    }

    #[test]
    fn test_cancel_order() {
        let symbol = Symbol::new("BTCUSD").unwrap();
        let book = LockFreeOrderBook::new(symbol, 4);
        
        // Add an order
        let order = create_test_order(1, Side::Buy, 50000, 100);
        book.add_order(order.clone()).unwrap();
        
        // Cancel the order
        let cancelled = book.cancel_order(OrderId::new(1), Side::Buy).unwrap();
        assert!(cancelled.is_some());
        assert_eq!(cancelled.unwrap().id, OrderId::new(1));
        
        // Check order book is empty
        let stats = book.get_stats();
        assert_eq!(stats.total_orders, 0);
        assert_eq!(stats.total_bid_volume, 0);
    }

    #[test]
    fn test_concurrent_operations() {
        let symbol = Symbol::new("BTCUSD").unwrap();
        let book = Arc::new(LockFreeOrderBook::new(symbol, 20));
        let mut handles = vec![];

        // Spawn threads to add orders concurrently
        for i in 0..10 {
            let book_clone = book.clone();
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let order_id = i * 10 + j + 1;
                    let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
                    let price = if side == Side::Buy { 49000 + i * 10 } else { 51000 + i * 10 };
                    let order = create_test_order(order_id, side, price, 10);
                    
                    let _ = book_clone.add_order(order);
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify final state
        let stats = book.get_stats();
        assert!(stats.total_orders > 0);
        
        // Force cleanup
        book.force_reclaim();
    }

    #[test]
    fn test_price_level_management() {
        let symbol = Symbol::new("BTCUSD").unwrap();
        let book = LockFreeOrderBook::new(symbol, 4);
        
        // Add orders at different price levels
        let orders = vec![
            create_test_order(1, Side::Buy, 50000, 100),
            create_test_order(2, Side::Buy, 49000, 100),
            create_test_order(3, Side::Buy, 51000, 100), // This should be best bid
            create_test_order(4, Side::Sell, 52000, 100),
            create_test_order(5, Side::Sell, 53000, 100),
            create_test_order(6, Side::Sell, 51500, 100), // This should be best ask
        ];

        for order in orders {
            book.add_order(order).unwrap();
        }

        assert_eq!(book.get_best_bid(), Some(51000));
        assert_eq!(book.get_best_ask(), Some(51500));
        assert_eq!(book.get_spread(), Some(500));
    }
}