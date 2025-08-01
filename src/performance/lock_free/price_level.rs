use super::atomic_operations::{AtomicOperations, MemoryOrder, AlignedAtomicPtr, AlignedAtomicU64};
use super::hazard_pointers::{HazardPointer, HazardPointerManager};
use super::order_node::LockFreeOrderNode;
use crate::orderbook::types::{Order, OrderId};
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicU32, Ordering};
use std::ptr;

/// Lock-free price level using atomic linked list operations
/// Cache-line aligned to prevent false sharing
#[repr(align(64))]
pub struct LockFreePriceLevel {
    /// Price for this level (immutable after creation)
    pub price: u64,
    
    /// Total volume at this price level
    pub total_volume: AlignedAtomicU64,
    
    /// Number of orders at this price level
    pub order_count: AtomicU32,
    
    /// Head of the order linked list
    pub orders_head: AlignedAtomicPtr<LockFreeOrderNode>,
    
    /// Tail of the order linked list for efficient insertion
    pub orders_tail: AlignedAtomicPtr<LockFreeOrderNode>,
    
    /// Next price level in the sorted list
    pub next_level: AlignedAtomicPtr<LockFreePriceLevel>,
    
    /// Previous price level in the sorted list
    pub prev_level: AlignedAtomicPtr<LockFreePriceLevel>,
    
    /// Epoch for memory reclamation
    pub epoch: AtomicU64,
    
    /// Padding to ensure cache-line alignment
    _padding: [u8; 0],
}

impl LockFreePriceLevel {
    /// Create a new lock-free price level
    pub fn new(price: u64) -> Self {
        Self {
            price,
            total_volume: AlignedAtomicU64::new(0),
            order_count: AtomicU32::new(0),
            orders_head: AlignedAtomicPtr::new(ptr::null_mut()),
            orders_tail: AlignedAtomicPtr::new(ptr::null_mut()),
            next_level: AlignedAtomicPtr::new(ptr::null_mut()),
            prev_level: AlignedAtomicPtr::new(ptr::null_mut()),
            epoch: AtomicU64::new(0),
            _padding: [],
        }
    }

    /// Add an order to this price level using lock-free operations
    pub fn add_order(
        &self,
        order: Order,
        hazard_manager: &HazardPointerManager,
    ) -> Result<(), LockFreeError> {
        let new_node = Box::into_raw(Box::new(LockFreeOrderNode::new(order)));
        
        // Protect the new node with hazard pointer
        let hazard = hazard_manager.acquire_hazard_pointer();
        hazard.protect(new_node);

        loop {
            let tail = self.orders_tail.load(MemoryOrder::Acquire);
            
            if tail.is_null() {
                // Empty list - try to set both head and tail
                match self.orders_head.compare_exchange_weak(
                    ptr::null_mut(),
                    new_node,
                    MemoryOrder::Release,
                    MemoryOrder::Relaxed,
                ) {
                    Ok(_) => {
                        // Successfully set head, now set tail
                        self.orders_tail.store(new_node, MemoryOrder::Release);
                        break;
                    }
                    Err(_) => {
                        // Another thread beat us, retry
                        continue;
                    }
                }
            } else {
                // Non-empty list - append to tail
                unsafe {
                    // Set the next pointer of current tail
                    if (*tail).next.compare_exchange_weak(
                        ptr::null_mut(),
                        new_node,
                        MemoryOrder::Release,
                        MemoryOrder::Relaxed,
                    ).is_ok() {
                        // Successfully linked, now update tail
                        let _ = self.orders_tail.compare_exchange_weak(
                            tail,
                            new_node,
                            MemoryOrder::Release,
                            MemoryOrder::Relaxed,
                        );
                        break;
                    }
                }
            }
        }

        // Update counters
        self.total_volume.fetch_add(unsafe { (*new_node).order.size }, MemoryOrder::AcqRel);
        self.order_count.fetch_add(1, Ordering::AcqRel);

        hazard_manager.release_hazard_pointer(hazard);
        Ok(())
    }

    /// Remove an order from this price level
    pub fn remove_order(
        &self,
        order_id: OrderId,
        hazard_manager: &HazardPointerManager,
    ) -> Result<Option<Order>, LockFreeError> {
        let hazard = hazard_manager.acquire_hazard_pointer();
        
        loop {
            let mut prev: *mut LockFreeOrderNode = ptr::null_mut();
            let mut current = self.orders_head.load(MemoryOrder::Acquire);
            
            while !current.is_null() {
                hazard.protect(current);
                
                unsafe {
                    if (*current).order.id == order_id {
                        // Found the order to remove
                        let next = (*current).next.load(MemoryOrder::Acquire);
                        
                        if prev.is_null() {
                            // Removing head
                            if self.orders_head.compare_exchange_weak(
                                current,
                                next,
                                MemoryOrder::Release,
                                MemoryOrder::Relaxed,
                            ).is_ok() {
                                // Update tail if we removed the last node
                                if next.is_null() {
                                    self.orders_tail.store(ptr::null_mut(), MemoryOrder::Release);
                                }
                                
                                let removed_order = (*current).order.clone();
                                
                                // Update counters
                                self.total_volume.fetch_sub(removed_order.size, MemoryOrder::AcqRel);
                                self.order_count.fetch_sub(1, Ordering::AcqRel);
                                
                                // Schedule for reclamation
                                hazard_manager.retire_pointer(current);
                                hazard_manager.release_hazard_pointer(hazard);
                                
                                return Ok(Some(removed_order));
                            }
                        } else {
                            // Removing from middle or end
                            if (*prev).next.compare_exchange_weak(
                                current,
                                next,
                                MemoryOrder::Release,
                                MemoryOrder::Relaxed,
                            ).is_ok() {
                                // Update tail if we removed the last node
                                if next.is_null() {
                                    self.orders_tail.store(prev, MemoryOrder::Release);
                                }
                                
                                let removed_order = (*current).order.clone();
                                
                                // Update counters
                                self.total_volume.fetch_sub(removed_order.size, MemoryOrder::AcqRel);
                                self.order_count.fetch_sub(1, Ordering::AcqRel);
                                
                                // Schedule for reclamation
                                hazard_manager.retire_pointer(current);
                                hazard_manager.release_hazard_pointer(hazard);
                                
                                return Ok(Some(removed_order));
                            }
                        }
                        
                        // CAS failed, restart from beginning
                        break;
                    }
                    
                    prev = current;
                    current = (*current).next.load(MemoryOrder::Acquire);
                }
            }
            
            if current.is_null() {
                // Order not found
                hazard_manager.release_hazard_pointer(hazard);
                return Ok(None);
            }
        }
    }

    /// Get the first order (FIFO) without removing it
    pub fn peek_first_order(
        &self,
        hazard_manager: &HazardPointerManager,
    ) -> Option<Order> {
        let hazard = hazard_manager.acquire_hazard_pointer();
        let head = self.orders_head.load(MemoryOrder::Acquire);
        
        if head.is_null() {
            hazard_manager.release_hazard_pointer(hazard);
            return None;
        }
        
        hazard.protect(head);
        let order = unsafe { (*head).order.clone() };
        hazard_manager.release_hazard_pointer(hazard);
        
        Some(order)
    }

    /// Remove and return the first order (FIFO)
    pub fn pop_first_order(
        &self,
        hazard_manager: &HazardPointerManager,
    ) -> Result<Option<Order>, LockFreeError> {
        let hazard = hazard_manager.acquire_hazard_pointer();
        
        loop {
            let head = self.orders_head.load(MemoryOrder::Acquire);
            
            if head.is_null() {
                hazard_manager.release_hazard_pointer(hazard);
                return Ok(None);
            }
            
            hazard.protect(head);
            
            unsafe {
                let next = (*head).next.load(MemoryOrder::Acquire);
                
                if self.orders_head.compare_exchange_weak(
                    head,
                    next,
                    MemoryOrder::Release,
                    MemoryOrder::Relaxed,
                ).is_ok() {
                    // Successfully removed head
                    if next.is_null() {
                        // List is now empty, update tail
                        self.orders_tail.store(ptr::null_mut(), MemoryOrder::Release);
                    }
                    
                    let removed_order = (*head).order.clone();
                    
                    // Update counters
                    self.total_volume.fetch_sub(removed_order.size, MemoryOrder::AcqRel);
                    self.order_count.fetch_sub(1, Ordering::AcqRel);
                    
                    // Schedule for reclamation
                    hazard_manager.retire_pointer(head);
                    hazard_manager.release_hazard_pointer(hazard);
                    
                    return Ok(Some(removed_order));
                }
            }
            
            // CAS failed, retry
        }
    }

    /// Check if this price level is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.order_count.load(Ordering::Acquire) == 0
    }

    /// Get the total volume at this price level
    #[inline(always)]
    pub fn get_total_volume(&self) -> u64 {
        self.total_volume.load(MemoryOrder::Acquire)
    }

    /// Get the number of orders at this price level
    #[inline(always)]
    pub fn get_order_count(&self) -> u32 {
        self.order_count.load(Ordering::Acquire)
    }

    /// Partially fill an order at this price level
    pub fn partial_fill_order(
        &self,
        order_id: OrderId,
        fill_size: u64,
        hazard_manager: &HazardPointerManager,
    ) -> Result<Option<Order>, LockFreeError> {
        let hazard = hazard_manager.acquire_hazard_pointer();
        let mut current = self.orders_head.load(MemoryOrder::Acquire);
        
        while !current.is_null() {
            hazard.protect(current);
            
            unsafe {
                if (*current).order.id == order_id {
                    let mut order = (*current).order.clone();
                    
                    if order.size <= fill_size {
                        // Complete fill - remove the order
                        hazard_manager.release_hazard_pointer(hazard);
                        return self.remove_order(order_id, hazard_manager);
                    } else {
                        // Partial fill - update the order size
                        order.size -= fill_size;
                        (*current).order = order.clone();
                        
                        // Update total volume
                        self.total_volume.fetch_sub(fill_size, MemoryOrder::AcqRel);
                        
                        hazard_manager.release_hazard_pointer(hazard);
                        return Ok(Some(order));
                    }
                }
                
                current = (*current).next.load(MemoryOrder::Acquire);
            }
        }
        
        hazard_manager.release_hazard_pointer(hazard);
        Ok(None)
    }

    /// Iterator over orders in this price level (for debugging/monitoring)
    pub fn iter_orders(&self, hazard_manager: &HazardPointerManager) -> LockFreePriceLevelIterator {
        LockFreePriceLevelIterator::new(self, hazard_manager)
    }
}

/// Error types for lock-free price level operations
#[derive(Debug, Clone)]
pub enum LockFreeError {
    MemoryAllocationFailed,
    HazardPointerExhausted,
    ConcurrentModification,
    InvalidOperation,
}

/// Iterator for lock-free price level orders
pub struct LockFreePriceLevelIterator<'a> {
    current: *mut LockFreeOrderNode,
    hazard: HazardPointer<'a>,
    _level: &'a LockFreePriceLevel,
}

impl<'a> LockFreePriceLevelIterator<'a> {
    fn new(level: &'a LockFreePriceLevel, hazard_manager: &'a HazardPointerManager) -> Self {
        let hazard = hazard_manager.acquire_hazard_pointer();
        let current = level.orders_head.load(MemoryOrder::Acquire);
        
        Self {
            current,
            hazard,
            _level: level,
        }
    }
}

impl<'a> Iterator for LockFreePriceLevelIterator<'a> {
    type Item = Order;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_null() {
            return None;
        }

        self.hazard.protect(self.current);
        
        unsafe {
            let order = (*self.current).order.clone();
            self.current = (*self.current).next.load(MemoryOrder::Acquire);
            Some(order)
        }
    }
}

impl<'a> Drop for LockFreePriceLevelIterator<'a> {
    fn drop(&mut self) {
        // Hazard pointer will be automatically released when dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orderbook::types::{Order, OrderId, Symbol, Side, OrderType};
    use std::sync::Arc;
    use std::thread;

    fn create_test_order(id: u64, size: u64) -> Order {
        Order {
            id: OrderId::new(id),
            symbol: Symbol::new("BTCUSD").unwrap(),
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: 50000,
            size,
            timestamp: 1000,
        }
    }

    #[test]
    fn test_price_level_creation() {
        let level = LockFreePriceLevel::new(50000);
        assert_eq!(level.price, 50000);
        assert_eq!(level.get_total_volume(), 0);
        assert_eq!(level.get_order_count(), 0);
        assert!(level.is_empty());
    }

    #[test]
    fn test_add_single_order() {
        let level = LockFreePriceLevel::new(50000);
        let hazard_manager = HazardPointerManager::new(10);
        let order = create_test_order(1, 100);

        let result = level.add_order(order.clone(), &hazard_manager);
        assert!(result.is_ok());
        
        assert_eq!(level.get_total_volume(), 100);
        assert_eq!(level.get_order_count(), 1);
        assert!(!level.is_empty());

        let first_order = level.peek_first_order(&hazard_manager);
        assert!(first_order.is_some());
        assert_eq!(first_order.unwrap().id, order.id);
    }

    #[test]
    fn test_add_multiple_orders() {
        let level = LockFreePriceLevel::new(50000);
        let hazard_manager = HazardPointerManager::new(10);

        for i in 1..=5 {
            let order = create_test_order(i, 100);
            level.add_order(order, &hazard_manager).unwrap();
        }

        assert_eq!(level.get_total_volume(), 500);
        assert_eq!(level.get_order_count(), 5);
    }

    #[test]
    fn test_remove_order() {
        let level = LockFreePriceLevel::new(50000);
        let hazard_manager = HazardPointerManager::new(10);
        let order = create_test_order(1, 100);

        level.add_order(order.clone(), &hazard_manager).unwrap();
        
        let removed = level.remove_order(OrderId::new(1), &hazard_manager).unwrap();
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, order.id);
        
        assert_eq!(level.get_total_volume(), 0);
        assert_eq!(level.get_order_count(), 0);
        assert!(level.is_empty());
    }

    #[test]
    fn test_pop_first_order() {
        let level = LockFreePriceLevel::new(50000);
        let hazard_manager = HazardPointerManager::new(10);

        // Add orders in sequence
        for i in 1..=3 {
            let order = create_test_order(i, 100);
            level.add_order(order, &hazard_manager).unwrap();
        }

        // Pop orders - should come out in FIFO order
        let first = level.pop_first_order(&hazard_manager).unwrap();
        assert!(first.is_some());
        assert_eq!(first.unwrap().id, OrderId::new(1));

        let second = level.pop_first_order(&hazard_manager).unwrap();
        assert!(second.is_some());
        assert_eq!(second.unwrap().id, OrderId::new(2));

        assert_eq!(level.get_order_count(), 1);
        assert_eq!(level.get_total_volume(), 100);
    }

    #[test]
    fn test_partial_fill() {
        let level = LockFreePriceLevel::new(50000);
        let hazard_manager = HazardPointerManager::new(10);
        let order = create_test_order(1, 100);

        level.add_order(order, &hazard_manager).unwrap();

        // Partial fill of 30
        let result = level.partial_fill_order(OrderId::new(1), 30, &hazard_manager).unwrap();
        assert!(result.is_some());
        
        let remaining_order = result.unwrap();
        assert_eq!(remaining_order.size, 70);
        assert_eq!(level.get_total_volume(), 70);
        assert_eq!(level.get_order_count(), 1);

        // Complete fill
        let result = level.partial_fill_order(OrderId::new(1), 70, &hazard_manager).unwrap();
        assert!(result.is_some());
        assert!(level.is_empty());
    }

    #[test]
    fn test_concurrent_operations() {
        let level = Arc::new(LockFreePriceLevel::new(50000));
        let hazard_manager = Arc::new(HazardPointerManager::new(100));
        let mut handles = vec![];

        // Spawn threads to add orders concurrently
        for i in 0..10 {
            let level_clone = level.clone();
            let hazard_clone = hazard_manager.clone();
            
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let order_id = i * 10 + j + 1;
                    let order = create_test_order(order_id, 10);
                    level_clone.add_order(order, &hazard_clone).unwrap();
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify final state
        assert_eq!(level.get_order_count(), 100);
        assert_eq!(level.get_total_volume(), 1000);
    }

    #[test]
    fn test_iterator() {
        let level = LockFreePriceLevel::new(50000);
        let hazard_manager = HazardPointerManager::new(10);

        // Add some orders
        for i in 1..=5 {
            let order = create_test_order(i, 100);
            level.add_order(order, &hazard_manager).unwrap();
        }

        // Iterate and collect orders
        let orders: Vec<Order> = level.iter_orders(&hazard_manager).collect();
        assert_eq!(orders.len(), 5);

        // Verify FIFO order
        for (i, order) in orders.iter().enumerate() {
            assert_eq!(order.id, OrderId::new((i + 1) as u64));
        }
    }
}